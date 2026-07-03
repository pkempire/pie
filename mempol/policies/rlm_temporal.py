"""RLM-style temporal reader.

This policy is the read-side teacher we were missing:

  1. retrieve broadly from the backend,
  2. expand to local chronological neighborhoods when possible,
  3. reconstruct a dated state/event timeline,
  4. answer from that reconstructed timeline.

It is not trained yet. It is intentionally a policy with logged steps so its
trajectories can later become GEPA/SFT/GRPO data.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict

from mempol import config, llm
from mempol.backends.base import Backend, Hit, Unit
from mempol.policies.base import ReadPolicy, Step, Trace
from mempol.policies.v0_naive import answer_with_context


_ROUTE_SYS = (
    "Decide if answering this memory question requires temporal state "
    "reconstruction: dates, relative time, changes, current-vs-past state, "
    "or what was true at a specific time. Return strict JSON: "
    '{"needs_timeline": bool, "reason": string}.'
)

_TIMELINE_SYS = (
    "You reconstruct a timeline from conversation evidence. Extract only "
    "events, state changes, decisions, plans, preferences, commitments, and "
    "facts whose date/order/validity may matter later. Resolve relative times "
    "against the session date when possible. Return strict JSON: "
    '{"items":[{"date":"ISO/date/range if known","what":"plain English",'
    '"source_ids":["..."],"validity":"current|past|planned|unknown"}]}.'
)

_ANSWER_TIMELINE_SYS = (
    "Answer using the reconstructed timeline and source evidence. If the "
    "question asks what was true at time T, use the latest relevant state at "
    "or before T, not the newest fact overall. If evidence is insufficient, "
    "say 'not in context'. Be concise."
)


def _stable_key(parts: list[str]) -> str:
    raw = "\n".join(parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _route(question: str) -> dict:
    try:
        raw = llm.chat(
            [
                {"role": "system", "content": _ROUTE_SYS},
                {"role": "user", "content": question},
            ],
            model=config.REFORMULATE_MODEL,
            json_mode=True,
        )
        obj = json.loads(raw)
        return {
            "needs_timeline": bool(obj.get("needs_timeline")),
            "reason": str(obj.get("reason", "")),
        }
    except Exception:
        q = question.lower()
        needs = any(w in q for w in ("when", "where was", "as of", "before", "after", "current", "now", "then"))
        return {"needs_timeline": needs, "reason": "fallback_keyword_route"}


class TemporalRLMPolicy(ReadPolicy):
    """Read-time temporal reconstruction over retrieved evidence.

    This is deliberately backend-light. It works best with FlatBackend because
    FlatBackend exposes adjacent-turn expansion, but it can still run against
    any Backend that supports retrieve().
    """

    name = "rlm_temporal"

    def __init__(
        self,
        first_k: int = 24,
        final_k: int = 12,
        expand_seed_k: int = 8,
        force_timeline: bool = False,
    ) -> None:
        self.first_k = first_k
        self.final_k = final_k
        self.expand_seed_k = expand_seed_k
        self.force_timeline = force_timeline
        self._timeline_cache: dict[str, list[dict]] = {}

    def run(self, question: str, backend: Backend) -> Trace:
        trace = Trace(qid="", question=question, backend=backend.name, policy=self.name)

        route = _route(question)
        if self.force_timeline:
            route["needs_timeline"] = True
            route["reason"] = "force_timeline"
        trace.steps.append(Step("route_temporal", {}, json.dumps(route)[:180]))

        hits = backend.retrieve(question, k=self.first_k, source="hybrid")
        trace.n_retrievals += 1
        trace.steps.append(Step("retrieve_broad", {"k": self.first_k}, f"{len(hits)} hits"))

        seed_uids = [h.unit.uid for h in hits[: self.expand_seed_k]]
        expanded = backend.expand(seed_uids, k_per=2) if seed_uids else []
        if expanded:
            trace.steps.append(Step("expand_chronological", {"seed_k": self.expand_seed_k}, f"+{len(expanded)} hits"))

        evidence = _dedupe_hits(hits + expanded)
        if not route.get("needs_timeline"):
            out = evidence[: self.final_k]
            trace.final_hits = out
            trace.answer = answer_with_context(question, out)
            trace.steps.append(Step("answer_raw", {"k": len(out)}, trace.answer[:160]))
            return trace

        timeline = self._timeline_for_evidence(evidence)
        trace.steps.append(Step("reconstruct_timeline", {"evidence": len(evidence)}, f"{len(timeline)} timeline items"))

        out = evidence[: self.final_k]
        trace.final_hits = out
        trace.answer = self._answer_from_timeline(question, timeline, out)
        trace.steps.append(Step("answer_from_timeline", {"timeline_items": len(timeline), "evidence_k": len(out)}, trace.answer[:160]))
        return trace

    def _timeline_for_evidence(self, hits: list[Hit]) -> list[dict]:
        # Group by session/date so the LLM can resolve relative expressions
        # against a coherent local time anchor.
        groups: dict[tuple[str, str], list[Hit]] = defaultdict(list)
        for h in hits:
            md = h.unit.metadata
            groups[(str(md.get("session", "")), str(md.get("session_date", "")))].append(h)

        key = _stable_key(sorted(h.unit.uid for h in hits))
        if key in self._timeline_cache:
            return self._timeline_cache[key]

        items: list[dict] = []
        for (_session, session_date), ghits in sorted(groups.items(), key=lambda kv: kv[0]):
            block = "\n".join(_format_unit_for_timeline(h.unit) for h in ghits)
            try:
                raw = llm.chat(
                    [
                        {"role": "system", "content": _TIMELINE_SYS},
                        {
                            "role": "user",
                            "content": (
                                f"Session date/time anchor: {session_date or 'unknown'}\n\n"
                                f"Evidence:\n{block}"
                            ),
                        },
                    ],
                    model=config.REFORMULATE_MODEL,
                    json_mode=True,
                )
                obj = json.loads(raw)
                for item in obj.get("items", []) or []:
                    if not isinstance(item, dict):
                        continue
                    item.setdefault("date", session_date)
                    item.setdefault("source_ids", [h.unit.metadata.get("dia_id", h.unit.uid) for h in ghits])
                    items.append(item)
            except Exception:
                continue

        items.sort(key=lambda x: str(x.get("date", "")))
        self._timeline_cache[key] = items
        return items

    def _answer_from_timeline(self, question: str, timeline: list[dict], hits: list[Hit]) -> str:
        timeline_text = "\n".join(
            f"- date={it.get('date','?')} validity={it.get('validity','unknown')} "
            f"source={','.join(str(s) for s in it.get('source_ids', [])[:4])}: {it.get('what','')}"
            for it in timeline
        ) or "(no timeline items reconstructed)"
        evidence_text = "\n".join(_format_unit_for_timeline(h.unit) for h in hits)
        return llm.chat(
            [
                {"role": "system", "content": _ANSWER_TIMELINE_SYS},
                {
                    "role": "user",
                    "content": (
                        f"Timeline:\n{timeline_text}\n\n"
                        f"Source evidence:\n{evidence_text}\n\n"
                        f"Question: {question}\nAnswer:"
                    ),
                },
            ],
            model=config.ANSWER_MODEL,
        ).strip()


def _format_unit_for_timeline(unit: Unit) -> str:
    md = unit.metadata
    return (
        f"[source={md.get('dia_id', unit.uid)} | date={md.get('session_date','')} "
        f"| speaker={md.get('speaker','?')}] {unit.text}"
    )


def _dedupe_hits(hits: list[Hit]) -> list[Hit]:
    out: list[Hit] = []
    seen: set[str] = set()
    for h in hits:
        if h.unit.uid in seen:
            continue
        seen.add(h.unit.uid)
        out.append(h)
    return out
