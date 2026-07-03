"""Temporal context compiler.

This is the read-time bridge between raw evidence (`mempol.core`) and temporal
state (`mempol.temporal`). It reconstructs the relevant current state, packs
evidence under a token budget, chooses a simple action, and logs the decision so
offline learners can later improve it.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

from mempol import config
from mempol.core.retrieval import retrieve_budgeted
from mempol.core.schema import TraceEvent
from mempol.core.store import SQLiteMemoryStore, estimate_tokens, now_iso, store_for_run, trace_id

from .schema import ActiveProcess, ContextDecision, StateTransition, TemporalState
from .store import TemporalMemoryStore, temporal_store_for_run, temporal_id


_TOK_RE = re.compile(r"[A-Za-z0-9_./+-]+")


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in _TOK_RE.findall(text or "")}


def _score_text(query: str, text: str) -> float:
    q = _tokens(query)
    d = _tokens(text)
    if not q or not d:
        return 0.0
    overlap = len(q & d)
    if not overlap:
        return 0.0
    return overlap / math.sqrt(len(d) + 8)


def _score_state(task: str, state: TemporalState, now: str) -> float:
    text = f"{state.key}\n{state.content}\n{json.dumps(state.metadata, sort_keys=True)}"
    score = _score_text(task, text)
    score += 0.05 * float(state.confidence)
    if state.valid_until and state.valid_until <= now:
        score -= 0.5
    return score


def _score_process(task: str, process: ActiveProcess, now: str) -> float:
    text = f"{process.kind}\n{process.description}\n{json.dumps(process.metadata, sort_keys=True)}"
    score = _score_text(task, text)
    if process.deadline_at and process.deadline_at <= now:
        score += 1.0
    elif process.expected_at and process.expected_at <= now:
        score += 0.5
    return score


def _state_markdown(states: list[TemporalState], now: str) -> str:
    if not states:
        return "_No selected temporal states._"
    lines = []
    for s in states:
        validity = f"{s.valid_from or '?'} -> {s.valid_until or 'present'}"
        stale = " expired" if s.valid_until and s.valid_until <= now else ""
        supersedes = f" supersedes={','.join(s.supersedes_state_ids)}" if s.supersedes_state_ids else ""
        lines.append(
            f"- `{s.key}` ({s.state_type}, {s.status}{stale}, conf={s.confidence:.2f}, valid={validity}{supersedes})\n"
            f"  {s.content}"
        )
    return "\n".join(lines)


def _process_markdown(processes: list[ActiveProcess], now: str) -> str:
    if not processes:
        return "_No due active processes._"
    lines = []
    for p in processes:
        due_bits = []
        if p.expected_at:
            due_bits.append(f"expected={p.expected_at}")
        if p.deadline_at:
            due_bits.append(f"deadline={p.deadline_at}")
        due = ", ".join(due_bits) or "no due time"
        overdue = " overdue" if (p.deadline_at and p.deadline_at <= now) else ""
        lines.append(f"- `{p.kind}` ({p.status}{overdue}, {due})\n  {p.description}")
    return "\n".join(lines)


def _transition_markdown(transitions: list[StateTransition]) -> str:
    if not transitions:
        return "_No recent transitions._"
    lines = []
    for t in transitions:
        old = ",".join(t.old_state_ids) if t.old_state_ids else "-"
        lines.append(
            f"- `{t.transition_type}` at {t.observed_at or '?'} old=[{old}] new={t.new_state_id or '-'}\n"
            f"  {t.reason}"
        )
    return "\n".join(lines)


def _evidence_markdown(hits: list[dict]) -> str:
    if not hits:
        return "_No retrieved raw evidence._"
    lines = []
    for i, h in enumerate(hits, 1):
        locator = h.get("locator") or ""
        source = h.get("source") or ""
        lines.append(
            f"[{i}] `{h['kind']}` `{h['id']}` score={h.get('score', 0):.3f} source={source} {locator}\n"
            f"{(h.get('text') or '')[:1600]}"
        )
    return "\n\n".join(lines)


def _choose_action(processes: list[ActiveProcess], hits: list[dict], selected_states: list[TemporalState], now: str) -> str:
    if any(p.deadline_at and p.deadline_at <= now for p in processes):
        return "interrupt"
    if any(p.expected_at and p.expected_at <= now for p in processes):
        return "replan"
    if not hits and not selected_states:
        return "refresh"
    return "answer"


def compile_temporal_context_from_stores(
    *,
    core: SQLiteMemoryStore,
    temporal: TemporalMemoryStore,
    run_name: str,
    scope_id: str,
    task: str,
    now: str | None = None,
    k: int = 10,
    token_budget: int = 6000,
    state_limit: int = 12,
    transition_limit: int = 8,
    write_outputs: bool = False,
) -> dict[str, Any]:
    now = now or now_iso()

    current_states = temporal.current_states(scope_id, at=now, include_stale=False, limit=500)
    scored_states = sorted(
        ((s, _score_state(task, s, now)) for s in current_states),
        key=lambda x: x[1],
        reverse=True,
    )
    selected_states = [s for s, score in scored_states if score > 0][:state_limit]
    if not selected_states and current_states:
        selected_states = current_states[: min(4, state_limit)]

    due_processes = temporal.due_processes(scope_id, now=now, limit=100)
    scored_processes = sorted(
        ((p, _score_process(task, p, now)) for p in due_processes),
        key=lambda x: x[1],
        reverse=True,
    )
    selected_processes = [p for p, score in scored_processes if score > 0]
    if due_processes and not selected_processes:
        selected_processes = due_processes[:5]

    transitions = temporal.transitions(scope_id, limit=transition_limit)

    state_text = _state_markdown(selected_states, now)
    process_text = _process_markdown(selected_processes, now)
    transition_text = _transition_markdown(transitions)
    reserved_tokens = estimate_tokens(state_text) + estimate_tokens(process_text) + estimate_tokens(transition_text) + 500
    evidence_budget = max(800, token_budget - reserved_tokens)
    hits, retrieval_metrics = retrieve_budgeted(
        core,
        query=f"{task}\nscope:{scope_id}",
        k=k,
        token_budget=evidence_budget,
        include_spans=True,
    )

    action = _choose_action(selected_processes, hits, selected_states, now)
    source_span_ids = []
    for state in selected_states:
        source_span_ids.extend(state.source_span_ids)
    for process in selected_processes:
        source_span_ids.extend(process.source_span_ids)
    for hit in hits:
        if hit["kind"] == "span":
            source_span_ids.append(hit["id"])
        else:
            source_span_ids.extend(hit.get("source_span_ids") or [])
    source_span_ids = list(dict.fromkeys(source_span_ids))

    evidence_text = _evidence_markdown(hits)
    markdown = "\n".join(
        [
            f"# Temporal Context Pack: {task}",
            "",
            f"- Scope: `{scope_id}`",
            f"- Current time: `{now}`",
            f"- Recommended action: `{action}`",
            f"- Token budget: {token_budget}",
            f"- Estimated tokens: TBD",
            "",
            "## Agent Instruction",
            "Use current temporal state first. Treat expired or superseded information as historical evidence, not current truth. If the recommended action is not `answer`, explain what should be refreshed, waited on, interrupted, or replanned.",
            "",
            "## Current Temporal State",
            "",
            state_text,
            "",
            "## Due Active Processes",
            "",
            process_text,
            "",
            "## Recent Transitions",
            "",
            transition_text,
            "",
            "## Retrieved Evidence",
            "",
            evidence_text,
        ]
    )
    token_estimate = estimate_tokens(markdown)
    markdown = markdown.replace("- Estimated tokens: TBD", f"- Estimated tokens: {token_estimate}")

    decision = ContextDecision(
        id=temporal_id("context_decision", run_name, scope_id, task, now),
        scope_id=scope_id,
        task=task,
        now=now,
        action=action,
        candidate_state_ids=[s.id for s in current_states],
        selected_state_ids=[s.id for s in selected_states],
        selected_span_ids=source_span_ids,
        selected_process_ids=[p.id for p in selected_processes],
        token_budget=token_budget,
        token_estimate=token_estimate,
        rationale=(
            "Due process requires attention." if action in {"interrupt", "replan"}
            else "Answered with selected temporal state and retrieved evidence."
        ),
        metrics={
            **retrieval_metrics,
            "current_state_candidates": len(current_states),
            "selected_states": len(selected_states),
            "due_processes": len(due_processes),
            "selected_processes": len(selected_processes),
            "recent_transitions": len(transitions),
        },
        metadata={"retrieved_ids": [h["id"] for h in hits]},
    )
    temporal.log_context_decision(decision)
    core.log_trace(
        TraceEvent(
            id=trace_id("temporal_compile_context", run_name, scope_id),
            run_name=run_name,
            op="temporal_compile_context",
            input={"scope_id": scope_id, "task": task, "now": now, "k": k, "token_budget": token_budget},
            output={"decision_id": decision.id, "action": action, "retrieved_ids": [h["id"] for h in hits]},
            metrics=decision.metrics,
            created_at=now_iso(),
        )
    )
    temporal.commit()
    core.commit()

    result = {
        "run_name": run_name,
        "scope_id": scope_id,
        "task": task,
        "now": now,
        "action": action,
        "decision": asdict(decision),
        "selected_states": [asdict(s) for s in selected_states],
        "selected_processes": [asdict(p) for p in selected_processes],
        "recent_transitions": [asdict(t) for t in transitions],
        "retrieved": hits,
        "source_span_ids": source_span_ids,
        "metrics": {**decision.metrics, "context_pack_tokens_est": token_estimate},
        "markdown": markdown,
    }

    if write_outputs:
        out_dir = config.RESULTS_DIR / run_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_md = out_dir / "latest_temporal_context_pack.md"
        out_json = out_dir / "latest_temporal_context_pack.json"
        out_md.write_text(markdown, encoding="utf-8")
        out_json.write_text(
            json.dumps({k: v for k, v in result.items() if k != "markdown"} | {"markdown_path": str(out_md)}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        result["markdown_path"] = str(out_md)
        result["json_path"] = str(out_json)

    return result


def compile_temporal_context(
    run_name: str,
    scope_id: str,
    task: str,
    now: str | None = None,
    k: int = 10,
    token_budget: int = 6000,
    state_limit: int = 12,
    transition_limit: int = 8,
) -> dict[str, Any]:
    core = store_for_run(run_name)
    temporal = temporal_store_for_run(run_name)
    try:
        core_stats = core.stats()
        temporal_stats = temporal.stats()
        if core_stats["artifacts"] == 0 and temporal_stats["temporal_states"] == 0:
            out_dir = config.RESULTS_DIR / run_name
            out_dir.mkdir(parents=True, exist_ok=True)
            message = (
                f"No memory data found for run `{run_name}`. "
                "Run repo ingestion first, then bootstrap temporal state:\n\n"
                f"python3 -m mempol.ledger.ingest_repo --root . --run-name {run_name} --max-files 500 --max-commits 80\n"
                f"python3 -m mempol.temporal.bootstrap_repo --run-name {run_name}\n"
            )
            markdown = "\n".join(
                [
                    f"# Temporal Context Pack: {task}",
                    "",
                    f"- Scope: `{scope_id}`",
                    f"- Current time: `{now or now_iso()}`",
                    "- Recommended action: `refresh`",
                    "",
                    "## No Data Loaded",
                    "",
                    message,
                ]
            )
            out_md = out_dir / "latest_temporal_context_pack.md"
            out_json = out_dir / "latest_temporal_context_pack.json"
            out_md.write_text(markdown, encoding="utf-8")
            out_json.write_text(
                json.dumps(
                    {
                        "run_name": run_name,
                        "scope_id": scope_id,
                        "task": task,
                        "action": "refresh",
                        "error": "empty_run",
                        "core_stats": core_stats,
                        "temporal_stats": temporal_stats,
                        "next_commands": [
                            f"python3 -m mempol.ledger.ingest_repo --root . --run-name {run_name} --max-files 500 --max-commits 80",
                            f"python3 -m mempol.temporal.bootstrap_repo --run-name {run_name}",
                        ],
                        "markdown_path": str(out_md),
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            return {
                "markdown": markdown,
                "markdown_path": str(out_md),
                "json_path": str(out_json),
                "action": "refresh",
                "error": "empty_run",
            }
        return compile_temporal_context_from_stores(
            core=core,
            temporal=temporal,
            run_name=run_name,
            scope_id=scope_id,
            task=task,
            now=now,
            k=k,
            token_budget=token_budget,
            state_limit=state_limit,
            transition_limit=transition_limit,
            write_outputs=True,
        )
    finally:
        core.close()
        temporal.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Compile temporal context from core + temporal stores.")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--scope-id", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--now", default="")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--token-budget", type=int, default=6000)
    ap.add_argument("--state-limit", type=int, default=12)
    ap.add_argument("--transition-limit", type=int, default=8)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    result = compile_temporal_context(
        run_name=args.run_name,
        scope_id=args.scope_id,
        task=args.task,
        now=args.now or None,
        k=args.k,
        token_budget=args.token_budget,
        state_limit=args.state_limit,
        transition_limit=args.transition_limit,
    )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(result["markdown"], encoding="utf-8")
    print(result["markdown"])
    print(f"\nWrote {result.get('markdown_path', '')}")


if __name__ == "__main__":
    main()
