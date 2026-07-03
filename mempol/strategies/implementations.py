"""All strategy implementations.

Each class is a concrete MemoryStrategy plugin. Import order:
  A. Repo-backed strategies (runnable=True)  — 1..5
  B. SOTA stubs (runnable=False)             — 6..8

Adding a new strategy: subclass MemoryStrategy, set name/label/paper/tags,
implement build_backend() and run(), register it in registry.py.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict
from math import ceil
from typing import Any

from mempol import config, llm
from mempol.backends.base import Backend, Hit, Unit
from mempol.backends.flat import FlatBackend
from mempol.data.locomo import Conversation, QA
from mempol.eval.runner import conv_to_units
from mempol.policies.continuity import ContinuityTeacherPolicy
from mempol.policies.rlm_temporal import TemporalRLMPolicy
from mempol.policies.v0_naive import NaivePolicy, answer_with_context
from mempol.policies.v1_heuristic import HeuristicPolicy
from mempol.policies.base import Trace

from .base import MemoryStrategy, PaperRef


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _token_est(chars: int) -> int:
    return int(ceil(chars / 4))


def _storage_metrics(units: list[Unit], *, raw_chars: int | None = None) -> dict[str, Any]:
    stored_chars = sum(len(u.text or "") for u in units)
    vector_bytes = len(units) * 3072 * 4
    out: dict[str, Any] = {
        "stored_units": len(units),
        "stored_chars": stored_chars,
        "stored_tokens_est": _token_est(stored_chars),
        "vector_dim": 3072,
        "vector_bytes_est": vector_bytes,
        "vector_mb_est": round(vector_bytes / (1024 * 1024), 3),
    }
    if raw_chars:
        out["storage_compression_ratio"] = stored_chars / max(raw_chars, 1)
    return out


def _flat_backend(conv: Conversation) -> FlatBackend:
    b = FlatBackend()
    b.ingest(conv_to_units(conv))
    return b


def _trace_to_dict(trace: Trace) -> dict:
    return {
        "policy": trace.policy,
        "backend": trace.backend,
        "n_steps": len(trace.steps),
        "n_retrievals": trace.n_retrievals,
        "steps": [asdict(s) for s in trace.steps],
        "retrieved": [
            {
                "uid": h.unit.uid,
                "source": h.source,
                "score": h.score,
                "text": (h.unit.text or "")[:700],
                "session": h.unit.metadata.get("session"),
                "session_date": h.unit.metadata.get("session_date"),
                "speaker": h.unit.metadata.get("speaker"),
                "dia_id": h.unit.metadata.get("dia_id"),
            }
            for h in trace.final_hits
        ],
    }


def _run_policy(
    conv: Conversation,
    qa: QA,
    backend: FlatBackend,
    policy,
    *,
    raw_chars: int | None = None,
) -> tuple[str, dict]:
    trace = policy.run(qa.question, backend)
    context_chars = sum(len(h.unit.text) for h in trace.final_hits)
    rc = raw_chars or sum(len(t.text or "") for t in conv.turns)
    stored_chars = sum(len(u.text or "") for u in backend.units)
    trace_dict: dict[str, Any] = {
        "trace": _trace_to_dict(trace),
        "context_chars": context_chars,
        "retrieved_tokens_est": _token_est(context_chars),
        "retrieval_count": len(trace.final_hits),
        "n_steps": len(trace.steps),
        "n_retrievals": trace.n_retrievals,
        "stored_units": len(backend.units),
        "stored_chars": stored_chars,
        "storage_compression_ratio": stored_chars / max(rc, 1),
        "retrieval_to_storage_ratio": context_chars / max(stored_chars, 1),
    }
    return trace.answer, trace_dict


# ---------------------------------------------------------------------------
# 1. TurnRAG — naive turn-level hybrid retrieval
# ---------------------------------------------------------------------------

class TurnRAG(MemoryStrategy):
    """
    Naive RAG baseline: concatenate all turns chronologically, retrieve
    top-10 via BM25+dense hybrid, answer once. This is the "naive_rag_turn"
    baseline from the LongMemEval paper (Wu et al., 2024, arXiv:2410.10813,
    Table 3). No temporal reasoning; no expansion; no reranking.

    This is the floor for any memory system.
    """

    name = "turn_rag"
    label = "Turn RAG"
    paper = PaperRef(
        arxiv_id="2410.10813",
        title="LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory",
    )
    tags = ["RAG", "baseline"]
    runnable = True

    def __init__(self) -> None:
        self._policy = NaivePolicy(k=10)

    def build_backend(self, conv: Conversation) -> tuple[Backend, dict]:
        b = _flat_backend(conv)
        raw_chars = sum(len(t.text or "") for t in conv.turns)
        return b, _storage_metrics(b.units, raw_chars=raw_chars)

    def run(self, question: str, backend: Backend, qa: QA) -> tuple[str, dict]:
        assert isinstance(backend, FlatBackend)
        answer, trace = _run_policy(
            # conv is not needed here; pass a dummy for the helper
            type("_C", (), {"turns": []})(),  # type: ignore[arg-type]
            qa,
            backend,
            self._policy,
        )
        return answer, trace


# ---------------------------------------------------------------------------
# 2. TimelineSynthesis — read-time dated timeline reconstruction
# ---------------------------------------------------------------------------

class TimelineSynthesis(MemoryStrategy):
    """
    Read-time temporal reconstruction. Retrieve top-24 turns, expand adjacent
    turns, then ask the LLM to extract a dated event timeline per session.
    Answer from that timeline. All computation at query time (per question).

    This is our read-time method. Closer reference for the timeline
    reconstruction step: arXiv:2603.16862 (Chronos, Tabrizi et al., 2026)
    which does the same extraction but at WRITE time. Ours does it at READ
    time — more expensive per question, cheaper to build the index.

    Benchmark reference: arXiv:2410.10813 (LongMemEval paper, Wu et al., 2024)
    which defines the evaluation protocol this is tested on.
    """

    name = "timeline_synthesis"
    label = "Timeline Synthesis"
    paper = PaperRef(
        arxiv_id="2410.10813",
        title="LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory",
    )
    tags = ["temporal-aware", "read-time", "timeline"]
    runnable = True

    def __init__(self) -> None:
        self._policy = TemporalRLMPolicy(
            first_k=24,
            final_k=12,
            expand_seed_k=8,
            force_timeline=True,
        )

    def build_backend(self, conv: Conversation) -> tuple[Backend, dict]:
        b = _flat_backend(conv)
        raw_chars = sum(len(t.text or "") for t in conv.turns)
        return b, _storage_metrics(b.units, raw_chars=raw_chars)

    def run(self, question: str, backend: Backend, qa: QA) -> tuple[str, dict]:
        assert isinstance(backend, FlatBackend)
        trace = self._policy.run(question, backend)
        context_chars = sum(len(h.unit.text) for h in trace.final_hits)
        stored_chars = sum(len(u.text or "") for u in backend.units)
        return trace.answer, {
            "trace": _trace_to_dict(trace),
            "context_chars": context_chars,
            "retrieved_tokens_est": _token_est(context_chars),
            "retrieval_count": len(trace.final_hits),
            "n_steps": len(trace.steps),
            "n_retrievals": trace.n_retrievals,
            "stored_units": len(backend.units),
            "stored_chars": stored_chars,
            "retrieval_to_storage_ratio": context_chars / max(stored_chars, 1),
        }


# ---------------------------------------------------------------------------
# 3. ContinuityTeacher — 7-step agentic read with SFT trace logging
# ---------------------------------------------------------------------------

class ContinuityTeacher(MemoryStrategy):
    """
    Seven-step agentic teacher controller:
      route -> multi-query retrieve -> expand -> session retrieve ->
      state reconstruction -> timeline -> action -> answer.

    Each step is logged explicitly for SFT/RL data collection. This is the
    most expensive runnable strategy (~7 LLM calls per question).

    Primary reference: arXiv:2512.24601 (Zhang et al., 2024) — recursive
    language model pattern for multi-step reads. Theoretical framing from
    arXiv:2309.02427 (CoALA, Sumers et al., 2024) which formalises cognitive
    architectures for language agents with episodic/semantic memory distinction.
    """

    name = "continuity_teacher"
    label = "Continuity Teacher"
    paper = PaperRef(
        arxiv_id="2512.24601",
        title="Recursive Language Models",
    )
    tags = ["agentic", "multi-step", "temporal-aware", "read-time"]
    runnable = True

    def __init__(self) -> None:
        self._policy = ContinuityTeacherPolicy(
            turn_k=18,
            session_k=2,
            expand_seed_k=8,
            final_turn_k=10,
            max_session_chars=4500,
        )

    def build_backend(self, conv: Conversation) -> tuple[Backend, dict]:
        b = _flat_backend(conv)
        raw_chars = sum(len(t.text or "") for t in conv.turns)
        return b, _storage_metrics(b.units, raw_chars=raw_chars)

    def run(self, question: str, backend: Backend, qa: QA) -> tuple[str, dict]:
        run_result = self._policy.run(
            question,
            backend,
            question_date=str(getattr(qa, "question_date", "")),
            question_type=str(qa.category_name),
        )
        trace = run_result.trace
        context_chars = sum(len(h.unit.text) for h in trace.final_hits)
        stored_chars = sum(
            len(u.text or "") for u in getattr(backend, "units", [])
        )
        return trace.answer, {
            "trace": _trace_to_dict(trace),
            "context_chars": context_chars,
            "retrieved_tokens_est": _token_est(context_chars),
            "retrieval_count": len(trace.final_hits),
            "n_steps": len(trace.steps),
            "n_retrievals": trace.n_retrievals,
            "continuity_route": run_result.route,
            "continuity_action": run_result.action,
            "temporary_state_count": len(run_result.temporary_states),
            "timeline_item_count": len(run_result.timeline),
            "missing_evidence": run_result.missing_evidence,
            "session_retrieval_count": len(run_result.session_hits),
            "stored_units": len(getattr(backend, "units", [])),
            "stored_chars": stored_chars,
            "retrieval_to_storage_ratio": context_chars / max(stored_chars, 1),
        }


# ---------------------------------------------------------------------------
# 4. HybridSearchBaseline — BM25+dense RRF with expansion and rerank
# ---------------------------------------------------------------------------

class HybridSearchBaseline(MemoryStrategy):
    """
    Hybrid retrieval baseline: BM25 + dense RRF, with adjacent-turn
    expansion and dense reranking. This is the "hybrid search" baseline
    described in the LongMemEval paper (Wu et al., 2024, arXiv:2410.10813).

    Implements HeuristicPolicy(do_expand=True) over FlatBackend. The standard
    RAG baseline to beat before adding temporal reasoning.
    """

    name = "hybrid_search"
    label = "Hybrid Search"
    paper = PaperRef(
        arxiv_id="2410.10813",
        title="LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory",
    )
    tags = ["RAG", "hybrid", "baseline"]
    runnable = True

    def __init__(self) -> None:
        self._policy = HeuristicPolicy(
            do_reformulate=False,
            do_route=False,
            do_expand=True,
        )

    def build_backend(self, conv: Conversation) -> tuple[Backend, dict]:
        b = _flat_backend(conv)
        raw_chars = sum(len(t.text or "") for t in conv.turns)
        return b, _storage_metrics(b.units, raw_chars=raw_chars)

    def run(self, question: str, backend: Backend, qa: QA) -> tuple[str, dict]:
        assert isinstance(backend, FlatBackend)
        trace = self._policy.run(question, backend)
        context_chars = sum(len(h.unit.text) for h in trace.final_hits)
        stored_chars = sum(len(u.text or "") for u in backend.units)
        return trace.answer, {
            "trace": _trace_to_dict(trace),
            "context_chars": context_chars,
            "retrieved_tokens_est": _token_est(context_chars),
            "retrieval_count": len(trace.final_hits),
            "n_steps": len(trace.steps),
            "n_retrievals": trace.n_retrievals,
            "stored_units": len(backend.units),
            "stored_chars": stored_chars,
            "retrieval_to_storage_ratio": context_chars / max(stored_chars, 1),
        }


# ---------------------------------------------------------------------------
# 5. ChronosStrategy — write-time structured event extraction
# ---------------------------------------------------------------------------

_CHRONOS_EXTRACT_SYS = """\
Extract temporal events from these conversation turns as JSON.
Return: {"events": [{"entity": str, "event": str, "date": str,
"validity": "current|past|planned", "dia_ids": [str]}]}

Guidelines:
- Extract facts, preferences, plans, state changes, decisions, commitments.
- For each event, resolve relative dates (yesterday, next week) against
  session_date when possible. If unresolvable, use the session_date itself.
- "validity": current = still true, past = was true, planned = will be true.
- Include dia_ids from the source turns.
- One event per atomic fact change. Do not merge unrelated events.
"""

_CHRONOS_TEMPORAL_SYS = """\
Answer this temporal/knowledge-update question using ordered dated events.
The events are extracted from a personal conversation log, ordered chronologically.
For "what is X now" or "what changed" questions: use the most recent non-superseded
event for entity X. For "what was X at time T" questions: use the latest event
for X at or before time T.
If insufficient evidence, say: not in context.
Be concise.
"""

_CHRONOS_FACTUAL_SYS = """\
Answer this question using the provided conversation excerpts.
Only use the provided context. If the answer is not present, say: not in context.
Be concise.
"""

_CHRONOS_ROUTE_SYS = """\
Classify this memory question into exactly one category. Return JSON:
{"category": "temporal|knowledge_update|preference|factual",
 "reason": "one sentence"}

Definitions:
- temporal: asks when something happened, before/after comparisons, durations.
- knowledge_update: asks current state of something that may have changed.
- preference: asks stable user preferences, habits, likes/dislikes.
- factual: asks a specific fact that was mentioned and unlikely to change.
"""


def _chronos_extract_events(
    turns_by_session: dict[int, list[Unit]],
    session_dates: dict[int, str],
) -> list[Unit]:
    """Extract structured events from each session. Returns event Units."""
    event_units: list[Unit] = []
    event_idx = 0
    for session_num in sorted(turns_by_session.keys()):
        turns = turns_by_session[session_num]
        session_date = session_dates.get(session_num, "")
        block = "\n".join(
            f"[{u.metadata.get('dia_id', u.uid)} | "
            f"speaker={u.metadata.get('speaker', '?')}] {u.text}"
            for u in turns
        )
        try:
            raw = llm.chat(
                [
                    {"role": "system", "content": _CHRONOS_EXTRACT_SYS},
                    {
                        "role": "user",
                        "content": (
                            f"Session date: {session_date or 'unknown'}\n\n"
                            f"Turns:\n{block}"
                        ),
                    },
                ],
                model=config.REFORMULATE_MODEL,
                json_mode=True,
                max_tokens=2000,
            )
            obj = json.loads(raw)
            events = obj.get("events") or []
        except Exception:
            events = []

        for ev in events:
            if not isinstance(ev, dict):
                continue
            entity = str(ev.get("entity", ""))
            event_text = str(ev.get("event", ""))
            date_str = str(ev.get("date", session_date or ""))
            validity = str(ev.get("validity", "unknown"))
            dia_ids = [str(d) for d in (ev.get("dia_ids") or [])]

            if not entity or not event_text:
                continue

            text = f"[{date_str}] {entity}: {event_text}"
            uid = f"chronos_event::{session_num}::{event_idx}"
            event_units.append(
                Unit(
                    uid=uid,
                    text=text,
                    metadata={
                        "entity": entity,
                        "event": event_text,
                        "date": date_str,
                        "validity": validity,
                        "dia_ids": dia_ids,
                        "session": session_num,
                        "session_date": session_date,
                        "speaker": "extracted_event",
                        "dia_id": dia_ids[0] if dia_ids else uid,
                        "kind": "chronos_event",
                    },
                )
            )
            event_idx += 1
    return event_units


def _chronos_route(question: str) -> str:
    """Return question category: temporal|knowledge_update|preference|factual."""
    try:
        raw = llm.chat(
            [
                {"role": "system", "content": _CHRONOS_ROUTE_SYS},
                {"role": "user", "content": question},
            ],
            model=config.REFORMULATE_MODEL,
            json_mode=True,
        )
        obj = json.loads(raw)
        cat = str(obj.get("category", "factual")).lower()
        if cat in {"temporal", "knowledge_update", "preference", "factual"}:
            return cat
        return "factual"
    except Exception:
        q = question.lower()
        if any(w in q for w in ("when", "before", "after", "how long", "days")):
            return "temporal"
        if any(w in q for w in ("current", "now", "latest", "changed", "update")):
            return "knowledge_update"
        if any(w in q for w in ("prefer", "like", "favorite", "usually", "habit")):
            return "preference"
        return "factual"


class _ChronosBackendBundle:
    """Holds both event and raw backends for Chronos."""
    def __init__(self, event_backend: FlatBackend, raw_backend: FlatBackend) -> None:
        self.event_backend = event_backend
        self.raw_backend = raw_backend
        # expose .units for storage metrics
        self.units = event_backend.units
        self.name = "chronos_dual"


class ChronosStrategy(MemoryStrategy):
    """
    Chronos (Tabrizi et al., PricewaterhouseCoopers, 2026 — arXiv:2603.16862).
    LongMemEval-S result: 95.6% (High), 92.6% (Low) — #3 overall.

    KEY INNOVATION vs TimelineSynthesis:
    Event extraction happens at WRITE TIME (once per conversation) rather than
    READ TIME (once per question). This makes query answering cheap once the
    index is built, at the cost of upfront extraction calls.

    WRITE TIME (build_backend):
    1. Group turns by session.
    2. For each session, call LLM to extract structured temporal event tuples:
         {"events": [{"entity", "event", "date", "validity", "dia_ids"}]}
    3. Build a FlatBackend over event Units (text = "[date] entity: event").
    4. Also keep a raw-turn FlatBackend for fallback on factual questions.

    READ TIME (run):
    1. Route question into: temporal | knowledge_update | preference | factual.
    2. For temporal/knowledge_update: retrieve from EVENT backend, build
       structured prompt that orders events chronologically.
    3. For factual/preference: fall back to RAW backend (preference questions
       often need exact wording; factual questions may not have been extracted).
    4. Answer with category-conditioned prompt.

    This dual-backend design with category-conditioned prompting is the core
    of the Chronos paper's contribution. The paper shows that this structured
    retrieval approach significantly outperforms plain RAG on temporal and
    knowledge-update question types.
    """

    name = "chronos"
    label = "Chronos"
    paper = PaperRef(
        arxiv_id="2603.16862",
        title=(
            "Chronos: Temporal-Aware Conversational Agents with Structured "
            "Event Retrieval for Long-Term Memory"
        ),
    )
    tags = ["temporal-aware", "write-time", "structured-extraction", "agentic"]
    runnable = True

    def build_backend(
        self, conv: Conversation
    ) -> tuple[_ChronosBackendBundle, dict]:
        raw_units = conv_to_units(conv)
        raw_chars = sum(len(t.text or "") for t in conv.turns)

        # Group turns by session for extraction
        turns_by_session: dict[int, list[Unit]] = defaultdict(list)
        session_dates: dict[int, str] = {}
        for u in raw_units:
            s = int(u.metadata.get("session", 0))
            turns_by_session[s].append(u)
            if s not in session_dates:
                session_dates[s] = str(u.metadata.get("session_date", ""))

        print(
            f"  [chronos] extracting events from "
            f"{len(turns_by_session)} sessions ({len(raw_units)} turns)...",
            flush=True,
        )
        event_units = _chronos_extract_events(turns_by_session, session_dates)
        print(
            f"  [chronos] extracted {len(event_units)} events", flush=True
        )

        # Build event backend
        event_backend = FlatBackend()
        if event_units:
            event_backend.ingest(event_units)

        # Build raw backend for fallback
        raw_backend = FlatBackend()
        raw_backend.ingest(raw_units)

        bundle = _ChronosBackendBundle(event_backend, raw_backend)

        stored_chars = sum(len(u.text or "") for u in event_units)
        storage_metrics: dict[str, Any] = {
            "stored_unit_kind": "chronos_event",
            "stored_units": len(event_units),
            "stored_chars": stored_chars,
            "stored_tokens_est": _token_est(stored_chars),
            "raw_units": len(raw_units),
            "raw_chars": raw_chars,
            "storage_compression_ratio": stored_chars / max(raw_chars, 1),
            "vector_dim": 3072,
            "vector_bytes_est": len(event_units) * 3072 * 4,
            "vector_mb_est": round(len(event_units) * 3072 * 4 / (1024 * 1024), 3),
        }
        return bundle, storage_metrics

    def run(
        self, question: str, backend: Any, qa: QA
    ) -> tuple[str, dict]:
        assert isinstance(backend, _ChronosBackendBundle), (
            f"ChronosStrategy.run() expects _ChronosBackendBundle, got {type(backend)}"
        )

        category = _chronos_route(question)
        n_retrievals = 1  # route call

        if category in {"temporal", "knowledge_update"}:
            # Use event backend for structured temporal retrieval
            hits = backend.event_backend.retrieve(question, k=16, source="hybrid")
            n_retrievals += 1
            # Sort events chronologically by date metadata
            hits.sort(key=lambda h: str(h.unit.metadata.get("date", "")))
            events_text = "\n".join(
                f"[date={h.unit.metadata.get('date','?')} "
                f"validity={h.unit.metadata.get('validity','?')}] "
                f"{h.unit.text}"
                for h in hits
            )
            if not events_text.strip():
                # Fall back to raw if no events extracted
                hits = backend.raw_backend.retrieve(question, k=10, source="hybrid")
                n_retrievals += 1
                answer = answer_with_context(question, hits)
                used_backend = "raw_fallback"
            else:
                answer = llm.chat(
                    [
                        {"role": "system", "content": _CHRONOS_TEMPORAL_SYS},
                        {
                            "role": "user",
                            "content": (
                                f"Dated events (chronological):\n{events_text}\n\n"
                                f"Question: {question}\nAnswer:"
                            ),
                        },
                    ],
                    model=config.ANSWER_MODEL,
                ).strip()
                used_backend = "event"
        else:
            # factual / preference — use raw turns
            hits = backend.raw_backend.retrieve(question, k=10, source="hybrid")
            n_retrievals += 1
            answer = answer_with_context(question, hits)
            used_backend = "raw"

        context_chars = sum(len(h.unit.text) for h in hits)
        event_count = len(backend.event_backend.units)
        raw_count = len(backend.raw_backend.units)

        return answer, {
            "category_routed": category,
            "used_backend": used_backend,
            "context_chars": context_chars,
            "retrieved_tokens_est": _token_est(context_chars),
            "retrieval_count": len(hits),
            "n_steps": 2,
            "n_retrievals": n_retrievals,
            "chronos_event_count": event_count,
            "chronos_raw_count": raw_count,
            "retrieved": [
                {
                    "uid": h.unit.uid,
                    "source": h.source,
                    "score": h.score,
                    "text": (h.unit.text or "")[:700],
                    "date": h.unit.metadata.get("date"),
                    "validity": h.unit.metadata.get("validity"),
                    "entity": h.unit.metadata.get("entity"),
                }
                for h in hits
            ],
        }


# ---------------------------------------------------------------------------
# 6. HindsightStrategy — stub
# ---------------------------------------------------------------------------

class HindsightStrategy(MemoryStrategy):
    """
    Hindsight (Vectorize.io, 2025 — arXiv:2512.12818).
    "Hindsight is 20/20: Building Agent Memory that Retains, Recalls, and
    Reflects"
    LongMemEval-S: 91.4%   LoCoMo: 89.61%

    ARCHITECTURE (four-network memory):
    1. Semantic network: entity-level facts stored as a knowledge graph.
       Nodes = entities (person, place, concept). Edges = relationships.
       Updated at write time via LLM-based entity extraction + resolution.

    2. Episodic network: temporally-ordered event records. Each episode stores:
       - what happened, when, who was involved, what changed.
       Linked to semantic nodes by entity reference. Enables "when did X"
       queries without scanning all turns.

    3. Procedural network: reusable action templates extracted from repeated
       patterns. E.g. "user's weekly grocery routine" → can predict what
       to infer without re-reading old turns.

    4. Social network: models of the people the user mentioned — their
       relationships, traits, roles, how they relate to each other.

    REFLECTION mechanism:
    After N turns (configurable), a reflection LLM pass scans all four
    networks for contradictions, redundancies, and consolidation opportunities.
    It produces "meta-memories" — higher-order observations about the user's
    patterns that are then indexed separately. This is what the paper calls
    the "hindsight" step.

    TO MAKE RUNNABLE:
    Implement _build_hindsight_graph() that takes conv_to_units() output and
    maintains four in-memory graphs (can use NetworkX or dicts). The retrieval
    step must dispatch by question type to the right sub-graph. The hardest
    part is the reflection step, which needs a separate LLM pass that can
    handle contradictions in the semantic network.

    STATUS: Not runnable. Requires multi-network memory architecture that has
    not yet been implemented in this repo.
    """

    name = "hindsight"
    label = "Hindsight"
    paper = PaperRef(
        arxiv_id="2512.12818",
        title="Hindsight is 20/20: Building Agent Memory that Retains, Recalls, and Reflects",
    )
    tags = ["write-time-compression", "temporal-KG", "reflection"]
    runnable = False

    def build_backend(self, conv: Conversation) -> tuple[Backend, dict]:
        raise NotImplementedError(
            "HindsightStrategy is not yet implemented. "
            "See arXiv:2512.12818 (Vectorize.io, 2025) for the full architecture. "
            "To implement: build four memory networks (semantic, episodic, "
            "procedural, social) plus a reflection pass. "
            "See the class docstring for details."
        )

    def run(self, question: str, backend: Backend, qa: QA) -> tuple[str, dict]:
        raise NotImplementedError(
            "HindsightStrategy is not yet implemented. "
            "See arXiv:2512.12818 and the class docstring."
        )


# ---------------------------------------------------------------------------
# 7. MnemisStrategy — stub
# ---------------------------------------------------------------------------

class MnemisStrategy(MemoryStrategy):
    """
    Mnemis (Microsoft, 2026 — arXiv:2602.15313).
    "Mnemis: Dual-Route Retrieval on Hierarchical Graphs for Long-Term LLM
    Memory"
    LongMemEval-S: 91.6%   Rank: #9 on LongMemEval leaderboard.

    ARCHITECTURE (dual-process hierarchical graph):
    Inspired by the cognitive science distinction between System-1 (fast,
    associative) and System-2 (slow, systematic) thinking.

    System-1 route — similarity retrieval:
      Direct embedding similarity over leaf-level nodes (individual turns or
      event spans). Fast, O(N) over the corpus. Good for factual and
      preference questions where the answer is contained in a specific turn.

    System-2 route — hierarchical traversal:
      Graph has multiple levels: turn → episode → theme → global summary.
      At query time, start from the global summary node, traverse down the
      hierarchy guided by query relevance, collect relevant sub-trees.
      Good for temporal, multi-hop, and knowledge-update questions.

    ROUTING: An LLM classifier decides which route (or both) to use based
    on question type. When both are used, results are merged via RRF.

    GRAPH CONSTRUCTION (write time):
    1. Parse turns into leaf nodes with session metadata.
    2. Cluster consecutive turns into episodes (sliding window + coherence).
    3. Summarise each episode into a theme node.
    4. Build global summary from all theme nodes.
    All edges are bidirectional for upward traversal.

    TO MAKE RUNNABLE:
    Implement HierarchicalMemoryGraph class with add_turns(), retrieve_s1(),
    retrieve_s2(), and merge_routes(). The tricky part is building a
    coherent episode clustering that respects session boundaries.

    STATUS: Not runnable. Requires hierarchical graph construction and
    dual-route retrieval, not yet implemented in this repo.
    """

    name = "mnemis"
    label = "Mnemis"
    paper = PaperRef(
        arxiv_id="2602.15313",
        title="Mnemis: Dual-Route Retrieval on Hierarchical Graphs for Long-Term LLM Memory",
    )
    tags = ["graph-RAG", "hierarchical", "dual-process"]
    runnable = False

    def build_backend(self, conv: Conversation) -> tuple[Backend, dict]:
        raise NotImplementedError(
            "MnemisStrategy is not yet implemented. "
            "See arXiv:2602.15313 (Microsoft, 2026) for the full architecture. "
            "To implement: build hierarchical graph (turn->episode->theme->global) "
            "plus dual System-1/System-2 retrieval. "
            "See the class docstring for details."
        )

    def run(self, question: str, backend: Backend, qa: QA) -> tuple[str, dict]:
        raise NotImplementedError(
            "MnemisStrategy is not yet implemented. "
            "See arXiv:2602.15313 and the class docstring."
        )


# ---------------------------------------------------------------------------
# 8. WorldDBStrategy — stub
# ---------------------------------------------------------------------------

class WorldDBStrategy(MemoryStrategy):
    """
    WorldDB (2026 — arXiv:2604.18478).
    "WorldDB: A Vector Graph-of-Worlds Memory Engine with Ontology-Aware
    Write-Time Reconciliation"
    LongMemEval-S: 96.4% — CURRENT #1 on the LongMemEval leaderboard.

    ARCHITECTURE (content-addressed bitemporal graph):
    The "Graph-of-Worlds" model: each conversation fact exists in a world-node
    that records WHEN the fact was asserted (transaction time) AND what time
    period it describes (valid time). This bitemporal design allows precise
    answers to both "what did the user say on date X" and "what was true at
    time T according to the latest knowledge".

    FOUR-STAGE WRITE-TIME PIPELINE:
    1. EXTRACT: LLM extracts typed entities and relationships from each turn.
       Output: (subject, predicate, object, temporal_qualifiers).
       Temporal qualifiers: valid_from, valid_until, confidence, source_dia_id.

    2. RESOLVE: Entity resolution across sessions. Same entity referenced by
       different names ("my mom", "Sarah", "mom") gets merged. Uses
       embedding similarity + LLM confirmation for ambiguous cases.

    3. RECONCILE: Ontology-aware conflict resolution. If "user lives in NYC"
       and "user moved to LA" both exist, reconcile creates a supersession
       edge. The ontology defines which predicates are "mutable" (location,
       job, relationship_status) vs "immutable" (birthday, name).

    4. COMMIT: Facts are written to HNSW index for vector retrieval and to
       a bitemporal edge store for graph traversal. HNSW ensures O(log N)
       retrieval even with hundreds of thousands of facts.

    RETRIEVAL:
    - Fact query: HNSW similarity over world-nodes → retrieve top-K facts.
    - Temporal query: bitemporal traversal → filter by valid_time window.
    - Ontology query: traverse predicate graph → follow "supersedes" edges
      to get the current value of any mutable attribute.

    TO MAKE RUNNABLE:
    The hardest part is the bitemporal edge store — implement as a dict of
    (subject, predicate) -> list[TemporalFact] sorted by transaction_time.
    HNSW can be approximated with FlatBackend's dense retrieval for small
    corpora. The reconciliation step needs an ontology config file.

    STATUS: Not runnable. Requires content-addressed graph database with
    HNSW index and bitemporal fact store, not yet implemented in this repo.
    """

    name = "worlddb"
    label = "WorldDB"
    paper = PaperRef(
        arxiv_id="2604.18478",
        title=(
            "WorldDB: A Vector Graph-of-Worlds Memory Engine with "
            "Ontology-Aware Write-Time Reconciliation"
        ),
    )
    tags = ["knowledge-graph", "bitemporal", "content-addressing"]
    runnable = False

    def build_backend(self, conv: Conversation) -> tuple[Backend, dict]:
        raise NotImplementedError(
            "WorldDBStrategy is not yet implemented. "
            "See arXiv:2604.18478 (2026) for the full architecture. "
            "To implement: build the four-stage Extract-Resolve-Reconcile-Commit "
            "pipeline with HNSW index and bitemporal edge store. "
            "See the class docstring for the full architecture description."
        )

    def run(self, question: str, backend: Backend, qa: QA) -> tuple[str, dict]:
        raise NotImplementedError(
            "WorldDBStrategy is not yet implemented. "
            "See arXiv:2604.18478 and the class docstring."
        )
