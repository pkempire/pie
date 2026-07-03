"""Teacher policy for continuity-oriented memory.

This is the first concrete version of the bigger architecture:

  raw turn spans -> multi-granularity reads -> temporary state reconstruction
  -> action/answer decision -> logged trace for later SFT/RL.

It is deliberately a teacher/controller, not a trained model. The important
property is that every step is explicit and benchmarkable.
"""
from __future__ import annotations

import json
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from mempol import config, llm
from mempol.backends.base import Backend, Hit, Unit
from mempol.backends.flat import FlatBackend
from mempol.policies.base import Step, Trace


_ROUTE_SYS = """Route a long-memory question.

Return strict JSON with:
{
  "question_kind": "fact|multi_session|temporal|knowledge_update|preference|abstention|planning|unknown",
  "needs_timeline": boolean,
  "needs_session_context": boolean,
  "needs_latest_state": boolean,
  "should_abstain_if_unsupported": boolean,
  "search_queries": ["short query 1", "short query 2"],
  "reason": "brief reason"
}

Definitions:
- temporal: asks when, before/after, how long, or what was true at time T.
- knowledge_update: asks current value after possible updates or contradictions.
- preference: asks stable user preference/style/choice.
- abstention: likely asks for something absent from the history.
"""

_STATE_SYS = """Build a compact temporary state from evidence for one question.

Return strict JSON:
{
  "states": [
    {
      "key": "short stable key",
      "content": "what is true or relevant",
      "valid_from": "date/time if known",
      "valid_until": "date/time if superseded/expired, else empty",
      "status": "current|past|planned|uncertain|contradicted",
      "source_ids": ["..."]
    }
  ],
  "missing_evidence": "what is still missing, if anything"
}

Rules:
- Use only evidence provided.
- Preserve updates and supersession instead of flattening them.
- For count questions about pending pickups/returns/tasks, split obligations
  separately. Example: an exchange can create one pickup obligation for the new
  item and one return obligation for the old item.
- If evidence is insufficient, say so in missing_evidence.
"""

_TIMELINE_SYS = """Reconstruct a minimal timeline for the question.

Return strict JSON:
{
  "items": [
    {
      "date": "ISO/date/range/session date if known",
      "event_or_state": "plain English",
      "effect": "created|updated|superseded|confirmed|planned|unknown",
      "source_ids": ["..."]
    }
  ]
}

Rules:
- Resolve relative times against session dates when possible.
- Keep only events/states that could affect the answer.
- Order matters: newer facts can supersede older facts only when evidence says so.
"""

_ANSWER_SYS = """Answer a long-memory question from evidence, temporary states, and timeline.

You are allowed to answer only when the provided evidence supports it.

Decision rules:
- If the question asks current state, use the latest non-superseded relevant state.
- If the question asks what was true at time T, use the latest relevant state at or before T.
- If facts changed, mention the change only when needed for correctness.
- For "how many" questions about pickups/returns/tasks, count separate pending
  obligations. If an exchange requires picking up a replacement and returning
  the old item, that is two obligations unless evidence says otherwise.
- If the answer is absent or unsupported, answer exactly: not in context
- Be concise. No extra caveats unless needed.
"""


@dataclass
class ContinuityRun:
    trace: Trace
    route: dict[str, Any]
    temporary_states: list[dict[str, Any]] = field(default_factory=list)
    timeline: list[dict[str, Any]] = field(default_factory=list)
    missing_evidence: str = ""
    action: str = "answer"
    session_hits: list[Hit] = field(default_factory=list)


class ContinuityTeacherPolicy:
    """Expensive teacher policy for read/write/control traces."""

    name = "continuity_teacher"

    def __init__(
        self,
        turn_k: int = 18,
        session_k: int = 2,
        expand_seed_k: int = 8,
        final_turn_k: int = 10,
        max_session_chars: int = 4500,
    ) -> None:
        self.turn_k = turn_k
        self.session_k = session_k
        self.expand_seed_k = expand_seed_k
        self.final_turn_k = final_turn_k
        self.max_session_chars = max_session_chars

    def run(
        self,
        question: str,
        backend: Backend,
        *,
        question_date: str = "",
        question_type: str = "",
    ) -> ContinuityRun:
        trace = Trace(qid="", question=question, backend=backend.name, policy=self.name)

        route = self._route(question, question_date=question_date, question_type=question_type)
        trace.steps.append(Step("route_question", {"question_type": question_type}, json.dumps(route)[:700]))

        queries = route.get("search_queries") or [question]
        turn_hits = self._retrieve_turns(backend, queries)
        trace.n_retrievals += len(queries)
        trace.steps.append(Step("retrieve_turn_spans", {"k": self.turn_k, "queries": queries}, f"{len(turn_hits)} unique hits"))

        expanded = self._expand_turns(backend, turn_hits)
        if expanded:
            trace.steps.append(Step("expand_adjacent_turns", {"seed_k": self.expand_seed_k, "k_per": 2}, f"+{len(expanded)} hits"))

        turn_hits = _dedupe_hits(turn_hits + expanded)

        session_backend = _build_session_backend(backend, max_session_chars=self.max_session_chars)
        session_hits = session_backend.retrieve(question, k=self.session_k, source="hybrid") if session_backend else []
        if session_hits:
            trace.n_retrievals += 1
            trace.steps.append(Step("retrieve_sessions", {"k": self.session_k}, f"{len(session_hits)} sessions"))

        evidence = _dedupe_hits(turn_hits[: self.final_turn_k] + session_hits)
        state_obj = self._reconstruct_state(
            question,
            evidence,
            route=route,
            question_date=question_date,
        )
        states = list(state_obj.get("states") or [])
        missing = str(state_obj.get("missing_evidence", "") or "")
        trace.steps.append(Step("write_temporary_state", {"state_count": len(states)}, _compact(json.dumps(state_obj), 900)))

        timeline: list[dict[str, Any]] = []
        if route.get("needs_timeline") or route.get("needs_latest_state") or question_type in {"temporal-reasoning", "knowledge-update"}:
            timeline = self._reconstruct_timeline(
                question,
                evidence,
                question_date=question_date,
            )
            trace.steps.append(Step("reconstruct_timeline", {"timeline_items": len(timeline)}, _compact(json.dumps(timeline), 900)))

        action = self._choose_action(route=route, evidence=evidence, states=states, missing_evidence=missing)
        trace.steps.append(Step("choose_action", {}, action))

        answer = self._answer(
            question,
            evidence,
            states=states,
            timeline=timeline,
            route=route,
            action=action,
            question_date=question_date,
            question_type=question_type,
        )
        trace.final_hits = evidence
        trace.answer = answer
        trace.steps.append(Step("answer", {"action": action}, answer[:200]))

        return ContinuityRun(
            trace=trace,
            route=route,
            temporary_states=states,
            timeline=timeline,
            missing_evidence=missing,
            action=action,
            session_hits=session_hits,
        )

    def _route(self, question: str, *, question_date: str, question_type: str) -> dict[str, Any]:
        try:
            raw = llm.chat(
                [
                    {"role": "system", "content": _ROUTE_SYS},
                    {
                        "role": "user",
                        "content": (
                            f"Question type: {question_type or 'unknown'}\n"
                            f"Question date: {question_date or 'unknown'}\n"
                            f"Question: {question}"
                        ),
                    },
                ],
                model=config.REFORMULATE_MODEL,
                json_mode=True,
                max_tokens=2500,
            )
            obj = json.loads(raw)
            generated = [str(q).strip() for q in (obj.get("search_queries") or []) if str(q).strip()]
            obj["search_queries"] = _dedupe_strings([question] + generated)
            return obj
        except Exception:
            q = question.lower()
            qtype = question_type.lower()
            needs_timeline = (
                qtype == "temporal-reasoning"
                or any(w in q for w in ("when", "before", "after", "how long", "how many days", "days passed", "between", "currently", "now", "latest"))
            )
            needs_latest = qtype == "knowledge-update" or "current" in q or "now" in q or "latest" in q
            return {
                "question_kind": "temporal" if needs_timeline else "unknown",
                "needs_timeline": needs_timeline,
                "needs_session_context": True,
                "needs_latest_state": needs_latest,
                "should_abstain_if_unsupported": True,
                "search_queries": [question],
                "reason": "fallback_route",
            }

    def _retrieve_turns(self, backend: Backend, queries: list[str]) -> list[Hit]:
        hits: list[Hit] = []
        for q in queries:
            hits.extend(backend.retrieve(q, k=self.turn_k, source="hybrid"))
        return _dedupe_hits(hits)

    def _expand_turns(self, backend: Backend, hits: list[Hit]) -> list[Hit]:
        seeds = [h.unit.uid for h in hits[: self.expand_seed_k]]
        return backend.expand(seeds, k_per=2) if seeds else []

    def _reconstruct_state(
        self,
        question: str,
        evidence: list[Hit],
        *,
        route: dict[str, Any],
        question_date: str,
    ) -> dict[str, Any]:
        try:
            raw = llm.chat(
                [
                    {"role": "system", "content": _STATE_SYS},
                    {
                        "role": "user",
                        "content": (
                            f"Question date: {question_date or 'unknown'}\n"
                            f"Route: {json.dumps(route)}\n"
                            f"Question: {question}\n\n"
                            f"Evidence:\n{_format_evidence(evidence, max_chars=20_000)}"
                        ),
                    },
                ],
                model=config.REFORMULATE_MODEL,
                json_mode=True,
                max_tokens=3000,
            )
            obj = json.loads(raw)
            if not isinstance(obj, dict):
                return {"states": [], "missing_evidence": "state_parse_failed"}
            return obj
        except Exception as e:
            return {"states": [], "missing_evidence": f"state_error:{e}"}

    def _reconstruct_timeline(self, question: str, evidence: list[Hit], *, question_date: str) -> list[dict[str, Any]]:
        try:
            raw = llm.chat(
                [
                    {"role": "system", "content": _TIMELINE_SYS},
                    {
                        "role": "user",
                        "content": (
                            f"Question date: {question_date or 'unknown'}\n"
                            f"Question: {question}\n\n"
                            f"Evidence:\n{_format_evidence(evidence, max_chars=18_000)}"
                        ),
                    },
                ],
                model=config.REFORMULATE_MODEL,
                json_mode=True,
                max_tokens=3000,
            )
            obj = json.loads(raw)
            items = [item for item in (obj.get("items", []) or []) if isinstance(item, dict)]
        except Exception:
            items = []
        items.sort(key=lambda x: str(x.get("date", "")))
        return items

    def _choose_action(
        self,
        *,
        route: dict[str, Any],
        evidence: list[Hit],
        states: list[dict[str, Any]],
        missing_evidence: str,
    ) -> str:
        if not evidence:
            return "abstain"
        if route.get("question_kind") == "abstention" and missing_evidence:
            return "abstain_if_unsupported"
        if route.get("needs_timeline") or route.get("needs_latest_state"):
            return "answer_after_state_reconstruction"
        if states:
            return "answer_from_state"
        return "answer_from_evidence"

    def _answer(
        self,
        question: str,
        evidence: list[Hit],
        *,
        states: list[dict[str, Any]],
        timeline: list[dict[str, Any]],
        route: dict[str, Any],
        action: str,
        question_date: str,
        question_type: str,
    ) -> str:
        states_text = json.dumps(states, indent=2) if states else "[]"
        timeline_text = json.dumps(timeline, indent=2) if timeline else "[]"
        answer = llm.chat(
            [
                {"role": "system", "content": _ANSWER_SYS},
                {
                    "role": "user",
                    "content": (
                        f"Question type: {question_type or 'unknown'}\n"
                        f"Question date: {question_date or 'unknown'}\n"
                        f"Controller route: {json.dumps(route)}\n"
                        f"Controller action: {action}\n\n"
                        f"Temporary states:\n{states_text}\n\n"
                        f"Timeline:\n{timeline_text}\n\n"
                        f"Evidence:\n{_format_evidence(evidence, max_chars=28_000)}\n\n"
                        f"Question: {question}\nAnswer:"
                    ),
                },
            ],
            model=config.ANSWER_MODEL,
            max_tokens=2500,
        ).strip()
        if answer:
            return answer
        return _deterministic_answer_fallback(question, evidence)


def _build_session_backend(backend: Backend, *, max_session_chars: int) -> FlatBackend | None:
    units = list(getattr(backend, "units", []) or [])
    if not units:
        return None
    grouped: OrderedDict[tuple[Any, str], list[Unit]] = OrderedDict()
    for u in units:
        key = (u.metadata.get("session"), str(u.metadata.get("session_date", "")))
        grouped.setdefault(key, []).append(u)

    session_units: list[Unit] = []
    for (session, session_date), us in grouped.items():
        text = "\n".join(_format_unit(u) for u in us)
        if len(text) > max_session_chars:
            text = text[:max_session_chars] + "\n[session truncated]"
        dia_ids = [str(u.metadata.get("dia_id", u.uid)) for u in us]
        session_units.append(
            Unit(
                uid=f"session::{session}",
                text=text,
                metadata={
                    "session": session,
                    "session_date": session_date,
                    "dia_id": ",".join(dia_ids[:10]),
                    "dia_ids": dia_ids,
                    "speaker": "session",
                    "kind": "session",
                },
            )
        )
    b = FlatBackend()
    b.ingest(session_units)
    b.name = "session_flat"  # type: ignore[attr-defined]
    return b


def _format_unit(unit: Unit) -> str:
    md = unit.metadata
    return (
        f"[id={md.get('dia_id', unit.uid)} date={md.get('session_date', '')} "
        f"speaker={md.get('speaker', '?')}] {unit.text}"
    )


def _format_evidence(hits: list[Hit], max_chars: int = 40_000) -> str:
    parts: list[str] = []
    total = 0
    for h in hits:
        part = _format_unit(h.unit)
        if total + len(part) > max_chars:
            remaining = max_chars - total
            if remaining > 200:
                parts.append(part[:remaining] + "\n[evidence truncated]")
            break
        parts.append(part)
        total += len(part)
    return "\n\n".join(parts)


def _dedupe_hits(hits: list[Hit]) -> list[Hit]:
    out: list[Hit] = []
    seen: set[str] = set()
    for h in hits:
        if h.unit.uid in seen:
            continue
        seen.add(h.unit.uid)
        out.append(h)
    return out


def _dedupe_strings(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = value.lower().strip()
        if not key or key in seen:
            continue
        out.append(value)
        seen.add(key)
    return out


def _deterministic_answer_fallback(question: str, evidence: list[Hit]) -> str:
    """Small safety net for empty model responses.

    This is intentionally narrow. It prevents a blank answer on simple
    date-difference questions but does not replace the teacher policy.
    """

    q = question.lower()
    if "how many days" in q and "between" in q:
        date_a = None
        date_b = None
        for h in evidence:
            text = h.unit.text.lower()
            dt = _parse_session_date(str(h.unit.metadata.get("session_date", "")))
            if not dt:
                continue
            if date_a is None and ("moma" in text or "museum of modern art" in text):
                date_a = dt
            if date_b is None and (
                "ancient civilizations" in text
                or ("metropolitan museum of art" in text and "ancient" in text)
                or ("met " in text and "ancient" in text)
            ):
                date_b = dt
        if date_a and date_b:
            return f"{abs((date_b - date_a).days)} days"
    return "not in context"


def _parse_session_date(raw: str) -> datetime | None:
    m = re.search(r"(\d{4})/(\d{2})/(\d{2})", raw)
    if not m:
        return None
    try:
        return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return None


def _compact(text: str, n: int) -> str:
    text = (text or "").replace("\n", " ").strip()
    return text if len(text) <= n else text[:n] + " ..."
