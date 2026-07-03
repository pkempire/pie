"""Canonical LongMemEval comparison matrix.

One row is:
  LongMemEval question × memory/retrieval cell -> answer, score, trace/metrics

This consolidates the repo's disconnected LongMemEval paths:
  - legacy benchmark baselines: full_context, naive_rag_turn/session, pie_fresh
  - mempol reader policies: hybrid_search, rerank_search, expand_search, timeline_synthesis
  - memory/compression baselines: pie_cached_v1, mastra_inspired, hand/gepa flat

Use this as the public LongMemEval harness. Expensive cells are opt-in.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from math import ceil
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from pie.core.world_model import WorldModel

from mempol import config
from mempol.backends.base import Backend, Hit, Unit
from mempol.backends.flat import FlatBackend
from mempol.backends.mastra import MastraBackend
from mempol.backends.pie_kg import PIEBackend
from mempol.data.longmemeval import load as load_lme
from mempol.data.longmemeval import _local_path
from mempol.data.locomo import Conversation, QA
from mempol.eval.judge import judge
from mempol.eval.metrics import Result, summarise
from mempol.eval.runner import conv_to_units
from mempol.policies.base import ReadPolicy, Trace
from mempol.policies.continuity import ContinuityTeacherPolicy
from mempol.policies.rlm_temporal import TemporalRLMPolicy
from mempol.policies.v0_naive import NaivePolicy
from mempol.policies.v1_heuristic import HeuristicPolicy


REPO = Path(__file__).resolve().parents[2]
SCRIPT_DIR = REPO / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


def _compact(text: str, n: int = 1200) -> str:
    text = (text or "").replace("\n", " ").strip()
    return text if len(text) <= n else text[:n] + " ..."


def _token_est(chars: int) -> int:
    """Cheap English token estimate used only for reporting."""
    return int(ceil(chars / 4))


def _raw_conv_metrics(conv: Conversation) -> dict[str, int]:
    raw_chars = sum(len(t.text or "") for t in conv.turns)
    return {
        "raw_turns": len(conv.turns),
        "raw_sessions": len({t.session for t in conv.turns}),
        "raw_chars": raw_chars,
        "raw_tokens_est": _token_est(raw_chars),
    }


def _storage_metrics_for_texts(
    texts: list[str],
    *,
    unit_kind: str,
    embedding_dim: int = 3072,
    has_vectors: bool = True,
    raw_chars: int | None = None,
) -> dict[str, Any]:
    stored_chars = sum(len(t or "") for t in texts)
    vector_bytes = len(texts) * embedding_dim * 4 if has_vectors else 0
    out: dict[str, Any] = {
        "stored_unit_kind": unit_kind,
        "stored_units": len(texts),
        "stored_chars": stored_chars,
        "stored_tokens_est": _token_est(stored_chars),
        "vector_dim": embedding_dim if has_vectors else 0,
        "vector_bytes_est": vector_bytes,
        "vector_mb_est": round(vector_bytes / (1024 * 1024), 3),
    }
    if raw_chars:
        out["storage_compression_ratio"] = stored_chars / raw_chars
    return out


def _world_model_storage_metrics(wm: WorldModel, raw_chars: int | None = None) -> dict[str, Any]:
    texts = []
    for e in wm.entities.values():
        state = e.current_state or {}
        state_str = "; ".join(f"{k}: {v}" for k, v in state.items() if v)
        texts.append(f"{e.name} ({e.type.value}): {state_str}")
    out = _storage_metrics_for_texts(
        texts,
        unit_kind="pie_entity",
        has_vectors=True,
        raw_chars=raw_chars,
    )
    out.update({
        "kg_entities": len(wm.entities),
        "kg_transitions": len(wm.transitions),
        "kg_relationships": len(wm.relationships),
        "kg_procedures": len(wm.procedures),
    })
    return out


def _backend_storage_metrics(backend: Backend, raw_chars: int | None = None) -> dict[str, Any]:
    if hasattr(backend, "units"):
        units = getattr(backend, "units")
        return _storage_metrics_for_texts(
            [u.text for u in units],
            unit_kind="raw_turn",
            has_vectors=True,
            raw_chars=raw_chars,
        )
    if hasattr(backend, "wm"):
        return _world_model_storage_metrics(getattr(backend, "wm"), raw_chars=raw_chars)
    if hasattr(backend, "observations"):
        observations = getattr(backend, "observations")
        reflections = getattr(backend, "reflections", [])
        recent = getattr(backend, "_all_turns", [])
        texts = [o.markdown for o in observations] + [r.markdown for r in reflections]
        out = _storage_metrics_for_texts(
            texts,
            unit_kind="observation_reflection",
            has_vectors=True,
            raw_chars=raw_chars,
        )
        out.update({
            "om_observations": len(observations),
            "om_reflections": len(reflections),
            "om_recent_raw_turns": len(recent),
        })
        return out
    return {}


def _hit_payload(hit: Hit) -> dict:
    md = dict(hit.unit.metadata or {})
    return {
        "uid": hit.unit.uid,
        "source": hit.source,
        "score": hit.score,
        "text": _compact(hit.unit.text, 700),
        "metadata": {
            "dia_id": md.get("dia_id"),
            "session": md.get("session"),
            "session_date": md.get("session_date"),
            "speaker": md.get("speaker"),
            "name": md.get("name"),
            "type": md.get("type"),
            "n_transitions": md.get("n_transitions"),
        },
    }


def _trace_payload(trace: Trace) -> dict:
    return {
        "policy": trace.policy,
        "backend": trace.backend,
        "answer": trace.answer,
        "n_steps": len(trace.steps),
        "n_retrievals": trace.n_retrievals,
        "steps": [asdict(s) for s in trace.steps],
        "retrieved": [_hit_payload(h) for h in trace.final_hits],
    }


class Cell:
    def __init__(
        self,
        name: str,
        run: Callable[[Conversation, QA, dict, argparse.Namespace], dict],
        label: str,
        description: str,
        expensive: bool = False,
    ) -> None:
        self.name = name
        self.run = run
        self.label = label
        self.description = description
        self.expensive = expensive


def _run_policy_cell(
    conv: Conversation,
    qa: QA,
    backend: Backend,
    policy: ReadPolicy,
) -> dict:
    trace = policy.run(qa.question, backend)
    context_chars = sum(len(h.unit.text) for h in trace.final_hits)
    raw_chars = sum(len(t.text or "") for t in conv.turns)
    out = {
        "answer": trace.answer,
        "trace": _trace_payload(trace),
        "context_chars": context_chars,
        "retrieved_tokens_est": _token_est(context_chars),
        "retrieval_count": len(trace.final_hits),
        "n_steps": len(trace.steps),
        "n_retrievals": trace.n_retrievals,
        **_backend_storage_metrics(backend, raw_chars=raw_chars),
    }
    stored_chars = out.get("stored_chars") or 0
    out["retrieval_to_storage_ratio"] = (context_chars / stored_chars) if stored_chars else None
    return out


def _rag_storage_metrics(raw: dict, chunk_by: str) -> dict[str, Any]:
    from benchmarks.longmemeval.baselines import _build_rag_chunks

    chunks = _build_rag_chunks(
        raw["haystack_sessions"],
        raw["haystack_dates"],
        chunk_by=chunk_by,
    )
    raw_chars = sum(
        len(turn.get("content", "") or "")
        for session in raw.get("haystack_sessions", [])
        for turn in session
    )
    return _storage_metrics_for_texts(
        [c.get("text", "") for c in chunks],
        unit_kind=f"{chunk_by}_chunk",
        has_vectors=True,
        raw_chars=raw_chars,
    )


def _flat_backend(conv: Conversation) -> FlatBackend:
    b = FlatBackend()
    b.ingest(conv_to_units(conv))
    return b


def _flat_v0(conv: Conversation, qa: QA, _raw: dict, _args: argparse.Namespace) -> dict:
    return _run_policy_cell(conv, qa, _flat_backend(conv), NaivePolicy(k=10))


def _flat_v1(conv: Conversation, qa: QA, _raw: dict, _args: argparse.Namespace) -> dict:
    policy = HeuristicPolicy(do_reformulate=False, do_route=False, do_expand=True)
    return _run_policy_cell(conv, qa, _flat_backend(conv), policy)


def _flat_v1_expand(conv: Conversation, qa: QA, _raw: dict, _args: argparse.Namespace) -> dict:
    policy = HeuristicPolicy(do_reformulate=False, do_route=True, do_expand=True)
    return _run_policy_cell(conv, qa, _flat_backend(conv), policy)


def _flat_rlm_temporal(conv: Conversation, qa: QA, _raw: dict, args: argparse.Namespace) -> dict:
    policy = TemporalRLMPolicy(
        first_k=args.rlm_first_k,
        final_k=args.rlm_final_k,
        expand_seed_k=args.rlm_expand_seed_k,
        force_timeline=args.rlm_force_timeline,
    )
    return _run_policy_cell(conv, qa, _flat_backend(conv), policy)


def _continuity_teacher(conv: Conversation, qa: QA, raw: dict, args: argparse.Namespace) -> dict:
    policy = ContinuityTeacherPolicy(
        turn_k=args.continuity_turn_k,
        session_k=args.continuity_session_k,
        expand_seed_k=args.continuity_expand_seed_k,
        final_turn_k=args.continuity_final_turn_k,
        max_session_chars=args.continuity_max_session_chars,
    )
    backend = _flat_backend(conv)
    run = policy.run(
        qa.question,
        backend,
        question_date=str(raw.get("question_date", "")),
        question_type=str(raw.get("question_type", qa.category_name)),
    )
    trace = run.trace
    context_chars = sum(len(h.unit.text) for h in trace.final_hits)
    raw_chars = sum(len(t.text or "") for t in conv.turns)
    out = {
        "answer": trace.answer,
        "trace": _trace_payload(trace),
        "context_chars": context_chars,
        "retrieved_tokens_est": _token_est(context_chars),
        "retrieval_count": len(trace.final_hits),
        "n_steps": len(trace.steps),
        "n_retrievals": trace.n_retrievals,
        "continuity_route": run.route,
        "continuity_action": run.action,
        "temporary_states": run.temporary_states,
        "temporary_state_count": len(run.temporary_states),
        "timeline_items": run.timeline,
        "timeline_item_count": len(run.timeline),
        "missing_evidence": run.missing_evidence,
        "session_retrieval_count": len(run.session_hits),
        **_backend_storage_metrics(backend, raw_chars=raw_chars),
    }
    stored_chars = out.get("stored_chars") or 0
    out["retrieval_to_storage_ratio"] = (context_chars / stored_chars) if stored_chars else None
    return out


def _pie_cached(conv: Conversation, qa: QA, _raw: dict, _args: argparse.Namespace) -> dict:
    path = REPO / "benchmarks" / "longmemeval" / "cache" / f"{conv.sample_id}_world_model.json"
    if not path.exists():
        raise FileNotFoundError(f"missing cached LongMemEval PIE world model: {path}")
    backend = PIEBackend(world_model=WorldModel(persist_path=str(path)))
    policy = HeuristicPolicy(do_reformulate=False, do_route=False, do_expand=True)
    return _run_policy_cell(conv, qa, backend, policy)


def _mastra_v0(conv: Conversation, qa: QA, _raw: dict, args: argparse.Namespace) -> dict:
    b = MastraBackend(
        observer_token_threshold=args.mastra_observer_threshold,
        reflector_token_threshold=args.mastra_reflector_threshold,
        keep_recent_n=args.mastra_recent_turns,
    )
    b.ingest(conv_to_units(conv))
    out = _run_policy_cell(conv, qa, b, NaivePolicy(k=10))
    out["mastra_stats"] = b.stats()
    return out


def _mastra_v1(conv: Conversation, qa: QA, _raw: dict, args: argparse.Namespace) -> dict:
    b = MastraBackend(
        observer_token_threshold=args.mastra_observer_threshold,
        reflector_token_threshold=args.mastra_reflector_threshold,
        keep_recent_n=args.mastra_recent_turns,
    )
    b.ingest(conv_to_units(conv))
    policy = HeuristicPolicy(do_reformulate=False, do_route=False, do_expand=True)
    out = _run_policy_cell(conv, qa, b, policy)
    out["mastra_stats"] = b.stats()
    return out


def _consolidated_flat(
    conv: Conversation,
    qa: QA,
    _raw: dict,
    args: argparse.Namespace,
    prompt_path: Path,
) -> dict:
    from compare_pie_vs_gepa import chunk_turns, consolidate_chunk, entry_to_unit

    prompt = prompt_path.read_text()
    chunks = chunk_turns(conv.turns, args.consolidator_chunk_size)
    if args.max_chunks_per_row:
        chunks = chunks[: args.max_chunks_per_row]
    units: list[Unit] = []
    idx = 0
    for ci, ch in enumerate(chunks, start=1):
        print(f"    consolidating {conv.sample_id[:8]} chunk {ci}/{len(chunks)}", flush=True)
        for entry in consolidate_chunk(ch, prompt, args.consolidator_model):
            units.append(entry_to_unit(entry, idx))
            idx += 1
    if not units:
        raise RuntimeError(f"{prompt_path.name} produced zero consolidated units for {conv.sample_id}")
    b = FlatBackend()
    b.ingest(units)
    policy = HeuristicPolicy(do_reformulate=False, do_route=False, do_expand=True)
    out = _run_policy_cell(conv, qa, b, policy)
    out["n_consolidated_units"] = len(units)
    return out


def _hand_flat(conv: Conversation, qa: QA, raw: dict, args: argparse.Namespace) -> dict:
    return _consolidated_flat(
        conv,
        qa,
        raw,
        args,
        REPO / "mempol" / "results" / "gepa_consolidator" / "prompt_original.txt",
    )


def _gepa_flat(conv: Conversation, qa: QA, raw: dict, args: argparse.Namespace) -> dict:
    return _consolidated_flat(
        conv,
        qa,
        raw,
        args,
        REPO / "mempol" / "results" / "gepa_consolidator" / "prompt_optimized.txt",
    )


def _legacy_full_context(_conv: Conversation, _qa: QA, raw: dict, args: argparse.Namespace) -> dict:
    from benchmarks.longmemeval.baselines import full_context

    res = full_context(raw, model=config.ANSWER_MODEL, max_context_chars=args.full_context_chars)
    raw_chars = sum(len(t.text or "") for t in _conv.turns)
    return {
        "answer": res.hypothesis,
        "context_chars": res.context_chars,
        "retrieved_tokens_est": _token_est(res.context_chars),
        "retrieval_count": res.retrieval_count,
        **_storage_metrics_for_texts(
            [t.text for t in _conv.turns],
            unit_kind="full_context_turn",
            has_vectors=False,
            raw_chars=raw_chars,
        ),
        "retrieval_to_storage_ratio": (res.context_chars / raw_chars) if raw_chars else None,
        "latency_ms_inner": res.latency_ms,
        "legacy": res.to_dict(),
    }


def _legacy_naive_turn(_conv: Conversation, _qa: QA, raw: dict, args: argparse.Namespace) -> dict:
    from benchmarks.longmemeval.baselines import naive_rag

    res = naive_rag(raw, model=config.ANSWER_MODEL, top_k=args.legacy_rag_k, chunk_by="turn")
    storage = _rag_storage_metrics(raw, "turn")
    return {
        "answer": res.hypothesis,
        "context_chars": res.context_chars,
        "retrieved_tokens_est": _token_est(res.context_chars),
        "retrieval_count": res.retrieval_count,
        **storage,
        "retrieval_to_storage_ratio": (res.context_chars / storage["stored_chars"]) if storage.get("stored_chars") else None,
        "latency_ms_inner": res.latency_ms,
        "legacy": res.to_dict(),
    }


def _legacy_naive_session(_conv: Conversation, _qa: QA, raw: dict, args: argparse.Namespace) -> dict:
    from benchmarks.longmemeval.baselines import naive_rag

    res = naive_rag(raw, model=config.ANSWER_MODEL, top_k=args.legacy_rag_k, chunk_by="session")
    storage = _rag_storage_metrics(raw, "session")
    return {
        "answer": res.hypothesis,
        "context_chars": res.context_chars,
        "retrieved_tokens_est": _token_est(res.context_chars),
        "retrieval_count": res.retrieval_count,
        **storage,
        "retrieval_to_storage_ratio": (res.context_chars / storage["stored_chars"]) if storage.get("stored_chars") else None,
        "latency_ms_inner": res.latency_ms,
        "legacy": res.to_dict(),
    }


def _legacy_pie_fresh(_conv: Conversation, _qa: QA, raw: dict, args: argparse.Namespace) -> dict:
    from benchmarks.longmemeval.baselines import (
        _ask_llm_temporal,
        _build_world_model_for_question,
        _compile_temporal_context,
        _retrieve_entities_for_question,
    )
    from benchmarks.longmemeval.adapter import parse_question_date
    from pie.core.llm import LLMClient

    llm_client = LLMClient()
    wm = _build_world_model_for_question(
        raw,
        llm=llm_client,
        extraction_model=args.pie_extraction_model,
        max_input_chars=args.pie_extract_max_input_chars,
    )
    retrieved = _retrieve_entities_for_question(
        raw["question"],
        wm,
        llm_client,
        top_k=args.pie_top_k_entities,
    )
    question_ts = parse_question_date(raw["question_date"])
    context = _compile_temporal_context(
        retrieved,
        wm,
        question_ts=question_ts,
        max_chars=args.pie_context_chars,
    )
    answer = _ask_llm_temporal(
        context=context,
        question=raw["question"],
        question_date=raw["question_date"],
        llm=llm_client,
        model=config.ANSWER_MODEL,
    )
    raw_chars = sum(len(t.text or "") for t in _conv.turns)
    storage = _world_model_storage_metrics(wm, raw_chars=raw_chars)
    return {
        "answer": answer,
        "context_chars": len(context),
        "retrieved_tokens_est": _token_est(len(context)),
        "retrieval_count": len(retrieved),
        **storage,
        "retrieval_to_storage_ratio": (len(context) / storage["stored_chars"]) if storage.get("stored_chars") else None,
    }


def _legacy_pie_cached_build(_conv: Conversation, _qa: QA, raw: dict, args: argparse.Namespace) -> dict:
    """Official PIE LongMemEval cached path.

    This is different from `pie_cached_v1`: it calls the original
    PIETemporalCachedBaseline, which builds the world model when cache is
    missing, persists it, then reuses it on later runs.
    """
    from benchmarks.longmemeval.baselines import PIETemporalCachedBaseline

    baseline = PIETemporalCachedBaseline(
        cache_dir=Path(args.pie_cache_dir),
        model=config.ANSWER_MODEL,
        extraction_model=args.pie_extraction_model,
        embed_model=config.EMBED_MODEL,
        top_k_entities=args.pie_top_k_entities,
        max_context_chars=args.pie_context_chars,
        max_input_chars=args.pie_extract_max_input_chars,
    )
    res = baseline.run(raw)
    cache_path = Path(args.pie_cache_dir) / f"{raw['question_id']}_world_model.json"
    raw_chars = sum(len(t.text or "") for t in _conv.turns)
    storage = {}
    if cache_path.exists():
        storage = _world_model_storage_metrics(WorldModel(persist_path=str(cache_path)), raw_chars=raw_chars)
    return {
        "answer": res.hypothesis,
        "context_chars": res.context_chars,
        "retrieved_tokens_est": _token_est(res.context_chars),
        "retrieval_count": res.retrieval_count,
        **storage,
        "retrieval_to_storage_ratio": (res.context_chars / storage["stored_chars"]) if storage.get("stored_chars") else None,
        "latency_ms_inner": res.latency_ms,
        "legacy": res.to_dict(),
    }


CELLS: dict[str, Cell] = {
    "full_context": Cell(
        "full_context",
        _legacy_full_context,
        "Full Context",
        "All sessions stuffed into the answer prompt; optional --full-context-chars truncation.",
        expensive=True,
    ),
    "turn_rag": Cell(
        "turn_rag",
        _legacy_naive_turn,
        "Turn RAG",
        "Dense retrieval over individual conversation turns.",
        expensive=True,
    ),
    "session_rag": Cell(
        "session_rag",
        _legacy_naive_session,
        "Session RAG",
        "Dense retrieval over full conversation sessions.",
        expensive=True,
    ),
    "hybrid_search": Cell(
        "hybrid_search",
        _flat_v0,
        "Hybrid Search",
        "Raw turn chunks; single hybrid BM25+dense retrieve; answer.",
    ),
    "rerank_search": Cell(
        "rerank_search",
        _flat_v1,
        "Rerank Search",
        "Raw turn chunks; hybrid retrieve plus dense rerank; no routed expansion.",
    ),
    "expand_search": Cell(
        "expand_search",
        _flat_v1_expand,
        "Expand Search",
        "Raw turn chunks; routed adjacent-turn expansion when the policy asks for it.",
    ),
    "timeline_synthesis": Cell(
        "timeline_synthesis",
        _flat_rlm_temporal,
        "Timeline Synthesis",
        "Raw turn chunks; broad hybrid retrieve, adjacent-turn expansion, then LLM extracts a dated timeline before answering. Not a true recursive language model.",
        expensive=True,
    ),
    "continuity_teacher": Cell(
        "continuity_teacher",
        _continuity_teacher,
        "Continuity Teacher",
        "Teacher controller: multi-query turn retrieval, session retrieval, temporary state writes, timeline reconstruction, action choice, then answer. Emits traces for later SFT/RL.",
        expensive=True,
    ),
    "cached_pie": Cell(
        "cached_pie",
        _pie_cached,
        "Cached PIE",
        "Load an already-built PIE world model JSON for this LongMemEval row.",
    ),
    "mastra_notes": Cell(
        "mastra_notes",
        _mastra_v0,
        "Mastra Notes",
        "Python Mastra-inspired observational-memory notes plus naive reader; not official Mastra TS.",
        expensive=True,
    ),
    "mastra_rerank": Cell(
        "mastra_rerank",
        _mastra_v1,
        "Mastra Rerank",
        "Python Mastra-inspired observational-memory notes plus rerank reader; not official Mastra TS.",
        expensive=True,
    ),
    "hand_summary": Cell(
        "hand_summary",
        _hand_flat,
        "Hand Summary",
        "Hand-written consolidator prompt; retrieve over compressed summaries.",
        expensive=True,
    ),
    "gepa_summary": Cell(
        "gepa_summary",
        _gepa_flat,
        "GEPA Summary",
        "GEPA-optimized consolidator prompt; retrieve over compressed summaries.",
        expensive=True,
    ),
    "fresh_pie": Cell(
        "fresh_pie",
        _legacy_pie_fresh,
        "Fresh PIE",
        "Build a new PIE world model for the row every time; no cache reuse.",
        expensive=True,
    ),
    "build_pie": Cell(
        "build_pie",
        _legacy_pie_cached_build,
        "Build PIE",
        "Official PIE cached path: build the row world model if missing, then reuse it.",
        expensive=True,
    ),
}


CELL_ALIASES: dict[str, str] = {
    "legacy_naive_rag_turn": "turn_rag",
    "legacy_naive_rag_session": "session_rag",
    "flat_v0": "hybrid_search",
    "flat_v1": "rerank_search",
    "flat_v1_expand": "expand_search",
    "flat_rlm_temporal": "timeline_synthesis",
    "timeline_reader": "timeline_synthesis",
    "temporal_controller": "continuity_teacher",
    "pie_cached_v1": "cached_pie",
    "mastra_inspired_v0": "mastra_notes",
    "mastra_inspired_v1": "mastra_rerank",
    "hand_flat_v1": "hand_summary",
    "gepa_flat_v1": "gepa_summary",
    "pie_fresh": "fresh_pie",
    "pie_temporal_cached": "build_pie",
}


def _canonical_cell_name(name: str) -> str:
    return CELL_ALIASES.get(name, name)


def _cell_label(name: str) -> str:
    canonical = _canonical_cell_name(name)
    cell = CELLS.get(canonical)
    return cell.label if cell else canonical.replace("_", " ").title()


def _dedupe_cells(names: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for name in names:
        if name in seen:
            continue
        out.append(name)
        seen.add(name)
    return out


def _format_cell_list() -> str:
    rows = ["Available strategies:"]
    for name, cell in CELLS.items():
        rows.append(f"  {name:16s} {cell.label:16s} {cell.description}")
    rows.append("\nLegacy aliases:")
    for old, new in CELL_ALIASES.items():
        rows.append(f"  {old:24s} -> {new}")
    return "\n".join(rows)


def _load_raw_rows(variant: str, n_rows: int | None, download: bool) -> list[dict]:
    path = _local_path(variant)
    if not path.exists():
        if not download:
            raise FileNotFoundError(f"{path} not found; rerun without --no-download or set HF_TOKEN")
        load_lme(variant=variant, n_convs=1, download=True)
    rows = []
    with path.open() as f:
        for i, line in enumerate(f):
            if n_rows is not None and i >= n_rows:
                break
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _selected_rows(args: argparse.Namespace) -> list[tuple[Conversation, QA, dict]]:
    n_load = None if args.max_rows == 0 else args.max_rows
    if args.per_category:
        # Need enough rows to draw a balanced prefix from every category.
        n_load = None
    conv_rows = load_lme(variant=args.variant, n_convs=n_load, download=not args.no_download)
    raw_rows = _load_raw_rows(args.variant, n_load, download=not args.no_download)
    out: list[tuple[Conversation, QA, dict]] = []
    for (conv, qas), raw in zip(conv_rows, raw_rows):
        qa = qas[0]
        if args.categories:
            cats = {x.strip() for x in args.categories.split(",") if x.strip()}
            if qa.category_name not in cats:
                continue
        out.append((conv, qa, raw))
    if args.question_ids:
        ids = {x.strip() for x in args.question_ids.split(",") if x.strip()}
        out = [(c, q, r) for c, q, r in out if c.sample_id in ids or q.qid in ids]
    if args.per_category:
        kept: list[tuple[Conversation, QA, dict]] = []
        counts: dict[str, int] = defaultdict(int)
        for item in out:
            cat = item[1].category_name
            if counts[cat] >= args.per_category:
                continue
            kept.append(item)
            counts[cat] += 1
        out = kept
    if args.num_shards > 1:
        if args.shard_index < 0 or args.shard_index >= args.num_shards:
            raise SystemExit("--shard-index must be in [0, --num-shards)")
        out = [item for i, item in enumerate(out) if i % args.num_shards == args.shard_index]
    return out


def _load_done(rows_path: Path, retry_errors: bool = False) -> dict[tuple[str, str], dict]:
    done = {}
    if not rows_path.exists():
        return done
    for line in rows_path.read_text().splitlines():
        try:
            row = json.loads(line)
            if retry_errors and row.get("error"):
                continue
            done[(_canonical_cell_name(row["cell"]), row["question_id"])] = row
        except Exception:
            continue
    return done


def _dedupe_result_rows(rows: list[dict]) -> list[dict]:
    """Last write wins for reruns/retry-errors.

    `rows.jsonl` is append-only so interrupted runs can be resumed safely. When a
    failed row is retried, both the old error row and the new successful row may
    exist in the file. Summaries should count only the latest row for each
    canonical strategy × question.
    """
    latest: dict[tuple[str, str], dict] = {}
    order: list[tuple[str, str]] = []
    for r in rows:
        key = (_canonical_cell_name(r["cell"]), r["question_id"])
        if key not in latest:
            order.append(key)
        latest[key] = r
    return [latest[k] for k in order if k in latest]


def _summaries(rows: list[dict], cells_requested: list[str]) -> dict:
    rows = _dedupe_result_rows(rows)
    by_cell: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_cell[_canonical_cell_name(r["cell"])].append(r)
    out = {
        "variant": rows[0]["variant"] if rows else None,
        "cells_requested": cells_requested,
        "n_rows": len(rows),
        "by_cell": {},
    }
    for cell in cells_requested:
        cr = by_cell.get(cell, [])
        if not cr:
            continue
        results = [
            Result(
                qid=r["question_id"],
                category=0,
                category_name=r["category_name"],
                score=float(r.get("score", 0.0)),
                n_retrievals=int(r.get("n_retrievals") or 0),
                n_steps=int(r.get("n_steps") or 0),
                answer=r.get("answer", ""),
                gold=r.get("gold", ""),
                judge_reason=r.get("judge_reason", ""),
                evidence_recall=None,
            )
            for r in cr
        ]
        s = summarise(results)
        s["label"] = _cell_label(cell)
        s["errors"] = sum(1 for r in cr if r.get("error"))
        numeric_fields = [
            "raw_turns",
            "raw_sessions",
            "raw_chars",
            "raw_tokens_est",
            "stored_units",
            "stored_chars",
            "stored_tokens_est",
            "vector_mb_est",
            "storage_compression_ratio",
            "context_chars",
            "retrieved_tokens_est",
            "retrieval_count",
            "retrieval_to_storage_ratio",
            "kg_entities",
            "kg_transitions",
            "kg_relationships",
            "kg_procedures",
            "om_observations",
            "om_reflections",
            "om_recent_raw_turns",
        ]
        for field in numeric_fields:
            vals = [r.get(field) for r in cr if r.get(field) is not None]
            if vals:
                s[f"avg_{field}"] = sum(float(v) for v in vals) / len(vals)
        out["by_cell"][cell] = s
    return out


def _write_side_by_side(rows: list[dict], out_path: Path, cells_requested: list[str], max_questions: int = 120) -> None:
    rows = _dedupe_result_rows(rows)
    grouped: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in rows:
        grouped[r["question_id"]][_canonical_cell_name(r["cell"])] = r

    parts = ["# LongMemEval Side-By-Side", ""]
    for qi, (qid, by) in enumerate(grouped.items()):
        if qi >= max_questions:
            parts.append(f"\n_Truncated at {max_questions} questions._")
            break
        first = next(iter(by.values()))
        parts.extend([
            f"## {qid} / {first.get('category_name')}",
            f"**Q:** {first.get('question')}",
            f"**Gold:** {first.get('gold')}",
            "",
        ])
        for cell in cells_requested:
            r = by.get(cell)
            if not r:
                parts.append(f"### {_cell_label(cell)} (`{cell}`) — missing\n")
                continue
            parts.extend([
                f"### {_cell_label(cell)} (`{cell}`) — score={r.get('score')}",
                _compact(r.get("answer", ""), 900),
                "",
            ])
            if r.get("error"):
                parts.append(f"Error: `{r['error']}`\n")
            trace = r.get("trace") or {}
            retrieved = trace.get("retrieved") or []
            if retrieved:
                parts.append("Top retrieved:")
                for h in retrieved[:4]:
                    md = h.get("metadata", {})
                    parts.append(
                        f"- `{h.get('uid')}` {md.get('session_date') or ''} "
                        f"{md.get('speaker') or md.get('name') or ''}: {h.get('text')}"
                    )
                parts.append("")
            elif r.get("context_chars") is not None:
                parts.append(f"`context_chars={r.get('context_chars')}`, `retrieval_count={r.get('retrieval_count')}`\n")
        parts.append("---\n")
    out_path.write_text("\n".join(parts))


def run(args: argparse.Namespace) -> dict:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "rows.jsonl"
    summary_path = out_dir / "summary.json"
    side_path = out_dir / "side_by_side.md"

    requested_raw = [c.strip() for c in args.cells.split(",") if c.strip()]
    cells_requested = _dedupe_cells([_canonical_cell_name(c) for c in requested_raw])
    unknown = [c for c in cells_requested if c not in CELLS]
    if unknown:
        raise SystemExit(
            f"unknown cells {unknown}; available={list(CELLS)}; "
            f"legacy aliases={list(CELL_ALIASES)}"
        )

    if args.summarize_only:
        rows = [json.loads(l) for l in rows_path.read_text().splitlines() if l.strip()]
        summary = _summaries(rows, cells_requested)
        summary_path.write_text(json.dumps(summary, indent=2))
        _write_side_by_side(rows, side_path, cells_requested, args.side_by_side_max_questions)
        print(json.dumps(summary, indent=2))
        return summary

    rows_in = _selected_rows(args)
    done = _load_done(rows_path, retry_errors=args.retry_errors)
    print(f"[longmemeval_matrix] variant={args.variant} rows={len(rows_in)}")
    print(f"[longmemeval_matrix] cells={cells_requested}")
    if requested_raw != cells_requested:
        print(f"[longmemeval_matrix] resolved aliases={dict(zip(requested_raw, cells_requested))}")
    print(f"[longmemeval_matrix] out={out_dir}")

    with rows_path.open("a", buffering=1) as f:
        for cell_name in cells_requested:
            cell = CELLS[cell_name]
            print(f"\n[{cell_name}] {cell.description}", flush=True)
            for i, (conv, qa, raw) in enumerate(rows_in, start=1):
                key = (cell_name, conv.sample_id)
                if key in done:
                    continue
                t0 = time.time()
                print(
                    f"  row {i}/{len(rows_in)} {conv.sample_id[:8]} "
                    f"{qa.category_name} turns={len(conv.turns)}",
                    flush=True,
                )
                row: dict[str, Any] = {
                    "variant": args.variant,
                    "cell": cell_name,
                    "strategy": cell.label,
                    "question_id": conv.sample_id,
                    "qid": qa.qid,
                    "category_name": qa.category_name,
                    "question": qa.question,
                    "gold": qa.answer,
                    "n_turns": len(conv.turns),
                    "n_sessions": len({t.session for t in conv.turns}),
                    "description": cell.description,
                    **_raw_conv_metrics(conv),
                }
                try:
                    out = cell.run(conv, qa, raw, args)
                    score, reason = judge(qa.question, qa.answer, out["answer"])
                    row.update(out)
                    row.update({
                        "score": score,
                        "judge_reason": reason,
                        "wall_time_s": round(time.time() - t0, 2),
                    })
                    print(f"    score={score:.1f} answer={_compact(out['answer'], 120)}", flush=True)
                except Exception as e:
                    row.update({
                        "answer": f"ERROR: {e}",
                        "score": 0.0,
                        "judge_reason": "cell_error",
                        "error": str(e),
                        "wall_time_s": round(time.time() - t0, 2),
                    })
                    print(f"    ERROR: {e}", flush=True)
                f.write(json.dumps(row) + "\n")
                f.flush()
                done[key] = row

    rows = [json.loads(l) for l in rows_path.read_text().splitlines() if l.strip()]
    summary = _summaries(rows, cells_requested)
    summary_path.write_text(json.dumps(summary, indent=2))
    _write_side_by_side(rows, side_path, cells_requested, args.side_by_side_max_questions)
    print(f"\nWrote {rows_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {side_path}")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Canonical LongMemEval variant matrix.")
    ap.add_argument("--variant", default="longmemeval_s", choices=["longmemeval_s", "longmemeval_oracle", "longmemeval_m"])
    ap.add_argument("--out-dir", default="mempol/results/longmemeval_matrix")
    ap.add_argument("--max-rows", type=int, default=20, help="0 = all rows")
    ap.add_argument("--per-category", type=int, default=0, help="deterministic balanced prefix: keep N rows per question_type")
    ap.add_argument("--num-shards", type=int, default=1, help="split selected rows across N independent jobs")
    ap.add_argument("--shard-index", type=int, default=0, help="which shard this process runs, 0-indexed")
    ap.add_argument("--categories", default=None, help="comma-separated LongMemEval question_type filter")
    ap.add_argument("--question-ids", default=None, help="comma-separated LongMemEval question IDs")
    ap.add_argument("--cells", default="hybrid_search,rerank_search,timeline_synthesis")
    ap.add_argument("--list-cells", action="store_true")
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("--retry-errors", action="store_true", help="rerun rows that previously wrote an error")
    ap.add_argument("--side-by-side-max-questions", type=int, default=120)

    ap.add_argument("--answer-model", default=None, help="override MEMPOL_ANSWER_MODEL for this run")
    ap.add_argument("--judge-model", default=None, help="override MEMPOL_JUDGE_MODEL for this run")
    ap.add_argument("--reformulate-model", default=None, help="override MEMPOL_REFORMULATE_MODEL for this run")
    ap.add_argument("--embed-model", default=None, help="override MEMPOL_EMBED_MODEL for this run")

    ap.add_argument("--full-context-chars", type=int, default=0, help="0 = no character truncation")
    ap.add_argument("--legacy-rag-k", type=int, default=10)
    ap.add_argument("--pie-extraction-model", default="gpt-4o-mini")
    ap.add_argument("--pie-cache-dir", default=str(REPO / "benchmarks" / "longmemeval" / "cache"))
    ap.add_argument("--pie-top-k-entities", type=int, default=15)
    ap.add_argument("--pie-context-chars", type=int, default=30_000)
    ap.add_argument(
        "--pie-extract-max-input-chars",
        type=int,
        default=0,
        help="0 = do not silently truncate each LongMemEval session before PIE extraction",
    )

    ap.add_argument("--rlm-first-k", type=int, default=32)
    ap.add_argument("--rlm-final-k", type=int, default=14)
    ap.add_argument("--rlm-expand-seed-k", type=int, default=10)
    ap.add_argument("--rlm-force-timeline", action="store_true")

    ap.add_argument("--continuity-turn-k", type=int, default=18)
    ap.add_argument("--continuity-session-k", type=int, default=2)
    ap.add_argument("--continuity-expand-seed-k", type=int, default=8)
    ap.add_argument("--continuity-final-turn-k", type=int, default=10)
    ap.add_argument("--continuity-max-session-chars", type=int, default=4500)

    ap.add_argument("--mastra-observer-threshold", type=int, default=30_000)
    ap.add_argument("--mastra-reflector-threshold", type=int, default=40_000)
    ap.add_argument("--mastra-recent-turns", type=int, default=6)

    ap.add_argument("--consolidator-chunk-size", type=int, default=24)
    ap.add_argument("--max-chunks-per-row", type=int, default=0)
    ap.add_argument("--consolidator-model", default=config.ANSWER_MODEL)
    args = ap.parse_args()
    if args.list_cells:
        print(_format_cell_list())
        return
    if args.answer_model:
        config.ANSWER_MODEL = args.answer_model
        if args.consolidator_model == "gpt-5-mini":
            args.consolidator_model = args.answer_model
    if args.judge_model:
        config.JUDGE_MODEL = args.judge_model
    if args.reformulate_model:
        config.REFORMULATE_MODEL = args.reformulate_model
    if args.embed_model:
        config.EMBED_MODEL = args.embed_model
    run(args)


if __name__ == "__main__":
    main()
