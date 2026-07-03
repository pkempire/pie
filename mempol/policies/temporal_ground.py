"""Temporal Ground: egocentric worldline reader.

The successor to timestamp injection. Instead of prefixing raw date strings on
evidence chunks and hoping the model does date arithmetic + ordering +
validity resolution implicitly, this policy compiles time deterministically
(mempol/temporal/worldline.py) and hands the model:

  1. a chronological WORLDLINE with egocentric offsets and explicit
     "( N weeks pass )" gap markers — elapsed time as tokens, not subtraction;
  2. a BELIEF LEDGER — keyed facts with valid-time windows, CURRENT vs
     superseded resolved in Python by recency-within-key;
  3. a TIME ARITHMETIC card — every relevant duration precomputed exactly;
  4. an operator-specific answering rule chosen by deterministic question
     anchoring (locate / duration / point_in_time / order / current_state /
     change / frequency).

Cost: 1 LLM call for non-temporal questions, 2 for temporal ones — cheaper
than rlm_temporal (which spends a routing call plus one timeline call per
session group).
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from mempol import config, llm
from mempol.backends.base import Backend, Hit
from mempol.data.locomo import parse_locomo_date
from mempol.policies.base import ReadPolicy, Step, Trace
from mempol.policies.v0_naive import answer_with_context
from mempol.temporal.worldline import (
    Event,
    TemporalFrame,
    arithmetic_card,
    classify_question,
    compile_worldline,
    fmt_day,
)


_BELIEF_SYS = (
    "You read a chronological worldline of one or two people's lives and "
    "extract keyed facts whose validity can change over time: states, "
    "preferences, plans, possessions, jobs, locations, relationships, habits. "
    "Use a stable dotted key per fact family (e.g. 'caroline.pet', "
    "'melanie.job') so later values supersede earlier ones. Dates must come "
    "from the worldline. Return strict JSON: "
    '{"facts":[{"key":"person.topic","value":"plain English",'
    '"date":"YYYY-MM-DD","kind":"state|event|plan","source_ids":["..."]}]}.'
)

_ANSWER_SYS = (
    "You answer a question about people's lives using a WORLDLINE "
    "reconstructed from their conversations.\n"
    "How to read it:\n"
    "- Events are in true chronological order. Each line starts with the "
    "offset relative to NOW (e.g. '5 months, 12 days ago').\n"
    "- '( N weeks pass )' markers show elapsed time between events. A fact "
    "stated before a long gap may have changed after it unless restated.\n"
    "- The BELIEF LEDGER lists, per fact key, the CURRENT value and the "
    "superseded history with validity windows. For 'current/still' questions "
    "trust the ledger, not the most salient mention.\n"
    "- The TIME ARITHMETIC table has precomputed exact durations. NEVER "
    "compute date arithmetic yourself; read durations from the table.\n"
    "Be concise. If the evidence is insufficient, reply 'not in context'."
)

_OPERATOR_RULES = {
    "locate": (
        "This question asks WHEN something happened. Find the event on the "
        "worldline and answer with its date (and offset if helpful)."
    ),
    "duration": (
        "This question asks about elapsed time. Identify the two endpoints on "
        "the worldline, then read the duration from the TIME ARITHMETIC table. "
        "Do not subtract dates yourself."
    ),
    "point_in_time": (
        "This question is anchored to a REFERENCE TIME. Use the latest state "
        "at or before that reference time — not the newest fact overall. "
        "Facts after the reference time are the future relative to the "
        "question and must not be used as the answer."
    ),
    "order": (
        "This question asks about ordering. Use worldline order: earlier "
        "lines happened first. Check the gap markers for how far apart events are."
    ),
    "current_state": (
        "This question asks what is true NOW. Answer from the BELIEF LEDGER's "
        "CURRENT values; superseded values are history, not the answer."
    ),
    "change": (
        "This question asks about a change. Compare superseded vs CURRENT "
        "values in the ledger and describe what changed and when."
    ),
    "frequency": (
        "This question asks how often/how many times. Count distinct dated "
        "events on the worldline; do not double-count restatements of one event."
    ),
}


def _stable_key(parts: list[str]) -> str:
    return hashlib.sha1("\n".join(parts).encode("utf-8")).hexdigest()


def _dedupe_hits(hits: list[Hit]) -> list[Hit]:
    out: list[Hit] = []
    seen: set[str] = set()
    for h in hits:
        if h.unit.uid in seen:
            continue
        seen.add(h.unit.uid)
        out.append(h)
    return out


def _event_from_hit(h: Hit) -> Event:
    md = h.unit.metadata
    return Event(
        ts=parse_locomo_date(str(md.get("session_date", ""))),
        text=h.unit.text,
        source_id=str(md.get("dia_id", h.unit.uid)),
        speaker=str(md.get("speaker", "")),
        session=str(md.get("session", "")),
    )


class TemporalGroundPolicy(ReadPolicy):
    """Read-time temporal grounding via a deterministic worldline compiler."""

    name = "temporal_ground"

    def __init__(
        self,
        first_k: int = 24,
        final_k: int = 12,
        expand_seed_k: int = 8,
        use_belief_ledger: bool = True,
        force_temporal: bool = False,
    ) -> None:
        self.first_k = first_k
        self.final_k = final_k
        self.expand_seed_k = expand_seed_k
        self.use_belief_ledger = use_belief_ledger
        self.force_temporal = force_temporal
        self._ledger_cache: dict[str, list[dict]] = {}
        self._now_cache: dict[int, float] = {}

    # -- time horizon ------------------------------------------------------

    def _now_ts(self, backend: Backend, events: list[Event]) -> float:
        """NOW = the latest dated moment the agent has ever observed.

        Prefer the full ingested corpus (so NOW doesn't shrink to whatever this
        query happened to retrieve); fall back to the retrieved events.
        """
        cache_key = id(backend)
        if cache_key in self._now_cache:
            return self._now_cache[cache_key]
        best = 0.0
        units = getattr(backend, "units", None)
        if units:
            for u in units:
                ts = parse_locomo_date(str(u.metadata.get("session_date", "")))
                best = max(best, ts)
        if best <= 0.0:
            best = max((e.ts for e in events if e.dated), default=0.0)
        if best > 0.0:
            self._now_cache[cache_key] = best
        return best

    # -- main --------------------------------------------------------------

    def run(self, question: str, backend: Backend) -> Trace:
        trace = Trace(qid="", question=question, backend=backend.name, policy=self.name)

        hits = backend.retrieve(question, k=self.first_k, source="hybrid")
        trace.n_retrievals += 1
        trace.steps.append(Step("retrieve_broad", {"k": self.first_k}, f"{len(hits)} hits"))

        seed_uids = [h.unit.uid for h in hits[: self.expand_seed_k]]
        expanded = backend.expand(seed_uids, k_per=2) if seed_uids else []
        if expanded:
            trace.steps.append(Step("expand_chronological", {"seed_k": self.expand_seed_k}, f"+{len(expanded)} hits"))

        evidence = _dedupe_hits(hits + expanded)
        events = [_event_from_hit(h) for h in evidence]
        now_ts = self._now_ts(backend, events)

        frame = classify_question(question, event_ts=[e.ts for e in events], now_ts=now_ts or None)
        if self.force_temporal:
            frame.is_temporal = True
            if frame.operator == "none":
                frame.operator = "locate"
        trace.steps.append(Step(
            "anchor_question",
            {},
            f"operator={frame.operator} ref={fmt_day(frame.reference_ts) if frame.reference_ts else '-'} signals={frame.signals[:3]}",
        ))

        trace.final_hits = evidence[: self.final_k]

        if now_ts <= 0.0:
            # No dates anywhere — degrade to the plain reader.
            trace.answer = answer_with_context(question, trace.final_hits)
            trace.steps.append(Step("answer_undated_fallback", {}, trace.answer[:160]))
            return trace

        worldline = compile_worldline(events, now_ts)
        trace.steps.append(Step("compile_worldline", {"events": len(events)}, f"{len(worldline)} chars"))

        if not frame.is_temporal:
            # Even non-temporal questions read better off an ordered worldline
            # with gaps than off a bag of chunks — and it costs nothing extra.
            trace.answer = self._answer(question, worldline, ledger_text="", card_text="", frame=frame)
            trace.steps.append(Step("answer_worldline", {}, trace.answer[:160]))
            return trace

        card_text = arithmetic_card(
            events, now_ts,
            reference_ts=frame.reference_ts,
            reference_label=frame.reference_raw,
        )

        ledger_text = ""
        if self.use_belief_ledger and frame.operator in ("current_state", "change", "point_in_time"):
            ledger = self._belief_ledger(worldline, [e.source_id for e in events])
            ledger_text = self._render_ledger(ledger, now_ts)
            trace.steps.append(Step("belief_ledger", {"facts": len(ledger)}, ledger_text[:160]))

        trace.answer = self._answer(question, worldline, ledger_text, card_text, frame)
        trace.steps.append(Step("answer_grounded", {"operator": frame.operator}, trace.answer[:160]))
        return trace

    # -- belief ledger -------------------------------------------------------

    def _belief_ledger(self, worldline: str, source_ids: list[str]) -> list[dict]:
        key = _stable_key(sorted(set(source_ids)))
        if key in self._ledger_cache:
            return self._ledger_cache[key]
        facts: list[dict] = []
        try:
            raw = llm.chat(
                [
                    {"role": "system", "content": _BELIEF_SYS},
                    {"role": "user", "content": worldline},
                ],
                model=config.REFORMULATE_MODEL,
                json_mode=True,
            )
            obj = json.loads(raw)
            for f in obj.get("facts", []) or []:
                if isinstance(f, dict) and f.get("key") and f.get("value"):
                    facts.append(f)
        except Exception:
            pass
        self._ledger_cache[key] = facts
        return facts

    @staticmethod
    def _render_ledger(facts: list[dict], now_ts: float) -> str:
        """Resolve supersession in Python: within a key, the latest dated
        state/plan is CURRENT; earlier ones get explicit validity windows."""
        if not facts:
            return ""
        by_key: dict[str, list[dict]] = {}
        for f in facts:
            by_key.setdefault(str(f["key"]), []).append(f)
        lines = ["## BELIEF LEDGER (current vs superseded, resolved by validity)", ""]
        for k in sorted(by_key):
            chain = sorted(by_key[k], key=lambda f: str(f.get("date", "")))
            states = [f for f in chain if f.get("kind") != "event"]
            events = [f for f in chain if f.get("kind") == "event"]
            lines.append(f"- {k}:")
            if states:
                cur = states[-1]
                lines.append(f"    CURRENT (since {cur.get('date', '?')}): {cur['value']}")
                for old, nxt in zip(states, states[1:]):
                    lines.append(
                        f"    superseded ({old.get('date', '?')} -> {nxt.get('date', '?')}): {old['value']}"
                    )
            for ev in events:
                lines.append(f"    event ({ev.get('date', '?')}): {ev['value']}")
        return "\n".join(lines)

    # -- answering -----------------------------------------------------------

    def _answer(
        self,
        question: str,
        worldline: str,
        ledger_text: str,
        card_text: str,
        frame: TemporalFrame,
    ) -> str:
        rule = _OPERATOR_RULES.get(frame.operator, "")
        blocks = [worldline]
        if ledger_text:
            blocks.append(ledger_text)
        if card_text:
            blocks.append(card_text)
        if rule:
            blocks.append(f"## QUESTION-TYPE RULE\n{rule}")
        blocks.append(f"Question: {question}\nAnswer:")
        return llm.chat(
            [
                {"role": "system", "content": _ANSWER_SYS},
                {"role": "user", "content": "\n\n".join(blocks)},
            ],
            model=config.ANSWER_MODEL,
        ).strip()
