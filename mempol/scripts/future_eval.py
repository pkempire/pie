"""Longitudinal future-query eval for personal ChatGPT exports.

This is the practical eval for the "what should memory have remembered?"
question. Instead of generating synthetic QAs from a single conversation, it
uses the user's *real future turns* as evaluation queries:

    past conversations before cutoff T -> build memory M_T
    future user turns after T           -> queries/tasks to answer with M_T

For each cutoff, the script builds one or more memory views from the same past:

  raw      Flat RAG over verbatim past turns (the hard baseline)
  mastra   Observational-memory compression (Observer/Reflector log)
  pie_write Tool-based KG writes via mempol.policies.v1_write

The teacher answer is produced from raw RAG over the full past. If the teacher
says "not in context", the future turn is treated as not memory-dependent and
skipped by default. Each memory view is scored against that teacher answer.

This is not paper-final judging. It is the fast frontier loop we need before
spending time on RL:

  Does compressed/tool memory beat raw retrieval at the same context budget?
  Which future queries actually need memory?
  Where do graph/tool writes help vs. observation text?
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from pie.core.parser import parse_conversations

from mempol import config, llm
from mempol.backends.base import Backend, Hit, Unit
from mempol.backends.flat import FlatBackend
from mempol.backends.mastra import MastraBackend
from mempol.backends.pie_kg import PIEBackend
from mempol.eval.judge import judge
from mempol.policies.v0_naive import answer_with_context
from mempol.policies.v1_write import HeuristicWritePolicy
from mempol.recipes.memory_rl.write_tools import WriteTool


_EVAL_QUERY_SYS = (
    "Decide whether this future user turn is a meaningful memory-evaluation "
    "query/task. It should require or benefit from remembering earlier personal "
    "conversation context, project state, preferences, decisions, or prior work. "
    "Imperative tasks can qualify even if they are not phrased as questions. "
    "Reject tiny acknowledgments, generic chat, and turns that are purely new "
    "unrelated input. Return strict JSON: {\"is_eval_query\": bool, \"reason\": string}."
)


@dataclass
class ChronoTurn:
    idx: int
    conv_id: str
    title: str
    role: str
    text: str
    timestamp: float


@dataclass
class BackendStats:
    name: str
    stored_chars: int
    stored_units: int
    compression_ratio: float
    build_seconds: float
    extra: dict = field(default_factory=dict)


@dataclass
class EvalRow:
    cutoff_idx: int
    query_idx: int
    query: str
    teacher_answer: str
    backend: str
    answer: str
    score: float
    judge_reason: str
    retrieved_chars: int
    retrieved_units: int
    stored_chars: int
    stored_units: int
    compression_ratio: float


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _ts_text(ts: float) -> str:
    if not ts:
        return ""
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _load_turns(conversations_json: Path, year_min: int) -> list[ChronoTurn]:
    convs = parse_conversations(conversations_json, year_min=year_min)
    turns: list[ChronoTurn] = []
    for conv in convs:
        for turn in conv.turns:
            if turn.role not in ("user", "assistant"):
                continue
            if not turn.text.strip():
                continue
            turns.append(ChronoTurn(
                idx=len(turns),
                conv_id=conv.id,
                title=conv.title or "",
                role=turn.role,
                text=turn.text.strip(),
                timestamp=float(turn.timestamp or conv.created_at or 0.0),
            ))
    turns.sort(key=lambda t: (t.timestamp, t.idx))
    for i, t in enumerate(turns):
        t.idx = i
    return turns


def _turns_to_units(turns: Iterable[ChronoTurn]) -> list[Unit]:
    units: list[Unit] = []
    for t in turns:
        speaker = "user" if t.role == "user" else "assistant"
        units.append(Unit(
            uid=f"{t.conv_id}::T{t.idx}",
            text=f"{speaker}: {t.text}",
            metadata={
                "conv_id": t.conv_id,
                "title": t.title,
                "role": t.role,
                "speaker": speaker,
                "dia_id": f"T{t.idx}",
                "timestamp": t.timestamp,
            },
        ))
    return units


def _is_eval_query(t: ChronoTurn, min_chars: int, max_chars: int) -> bool:
    if t.role != "user":
        return False
    text = t.text.strip()
    if len(text) < min_chars or len(text) > max_chars:
        return False
    try:
        raw = llm.chat(
            [
                {"role": "system", "content": _EVAL_QUERY_SYS},
                {"role": "user", "content": text[:2000]},
            ],
            model=config.REFORMULATE_MODEL,
            json_mode=True,
        )
        return bool(json.loads(raw).get("is_eval_query"))
    except Exception:
        return "?" in text


def _retrieved_chars(hits: list[Hit]) -> int:
    return sum(len(h.unit.text or "") for h in hits)


def _answer_backend(question: str, backend: Backend, k: int) -> tuple[str, list[Hit]]:
    hits = backend.retrieve(question, k=k, source="hybrid")
    answer = answer_with_context(question, hits)
    return answer, hits


def _not_in_context(answer: str) -> bool:
    a = (answer or "").strip().lower()
    return (
        not a
        or a.startswith("not in context")
        or "not in the provided context" in a
        or "don't have enough" in a
        or "do not have enough" in a
    )


def _build_raw(past: list[ChronoTurn]) -> tuple[Backend, BackendStats]:
    t0 = time.time()
    b = FlatBackend()
    units = _turns_to_units(past)
    b.ingest(units)
    raw_chars = sum(len(u.text) for u in units)
    return b, BackendStats(
        name="raw",
        stored_chars=raw_chars,
        stored_units=len(units),
        compression_ratio=1.0,
        build_seconds=time.time() - t0,
    )


def _build_mastra(past: list[ChronoTurn]) -> tuple[Backend, BackendStats]:
    t0 = time.time()
    b = MastraBackend()
    units = _turns_to_units(past)
    raw_chars = sum(len(u.text) for u in units)
    b.ingest(units)
    stored = len(b.get_full_context())
    return b, BackendStats(
        name="mastra",
        stored_chars=stored,
        stored_units=len(b.observations) + len(b.reflections) + min(len(b._all_turns), b.keep_recent_n),
        compression_ratio=stored / max(1, raw_chars),
        build_seconds=time.time() - t0,
        extra=b.stats(),
    )


def _build_pie_write(past: list[ChronoTurn], max_turn_chars: int = 1500) -> tuple[Backend, BackendStats]:
    t0 = time.time()
    b = PIEBackend()
    tool = WriteTool(backend=b)
    policy = HeuristicWritePolicy()
    raw_chars = 0
    for i, t in enumerate(past):
        raw_chars += len(t.text)
        tool.current_turn_text = t.text[:max_turn_chars]
        tool.current_dia_id = f"{t.conv_id}::T{t.idx}"
        tool.current_timestamp = t.timestamp
        recent = past[max(0, i - 6):i]
        recent_context = "\n".join(
            f"T{r.idx} {r.role}: {r.text[:400]}" for r in recent
        )
        policy.step(
            turn_text=f"{t.role}: {t.text[:max_turn_chars]}",
            dia_id=tool.current_dia_id,
            timestamp=t.timestamp,
            backend=b,
            write_tool=tool,
            observation_time_text=_ts_text(t.timestamp),
            recent_context_text=recent_context,
        )
    stored = sum(len(h.unit.text or "") for h in b.retrieve("", k=max(1, len(b.wm.entities)), source="bm25"))
    if stored == 0:
        stored = sum(len(json.dumps(e.current_state, ensure_ascii=False)) + len(e.name)
                     for e in b.wm.entities.values())
    return b, BackendStats(
        name="pie_write",
        stored_chars=stored,
        stored_units=len(b.wm.entities),
        compression_ratio=stored / max(1, raw_chars),
        build_seconds=time.time() - t0,
        extra={
            "n_entities": len(b.wm.entities),
            "n_transitions": sum(len(b.wm.get_transitions(uid)) for uid in b.wm.entities),
            "n_relationships": sum(len(b.wm.get_relationships(uid)) for uid in b.wm.entities),
            "write_tool_stats": tool.write_stats(),
        },
    )


_BUILDERS = {
    "raw": _build_raw,
    "mastra": _build_mastra,
    "pie_write": _build_pie_write,
}


def run(
    conversations_json: Path,
    out_dir: Path,
    builders: list[str],
    year_min: int = 2023,
    cutoff_turns: int = 500,
    future_turns: int = 200,
    stride_turns: int = 500,
    max_cutoffs: int = 1,
    max_queries_per_cutoff: int = 25,
    teacher_k: int = 50,
    eval_k: int = 12,
    min_query_chars: int = 12,
    max_query_chars: int = 1200,
    keep_not_in_context: bool = False,
    dry_run: bool = False,
) -> dict:
    turns = _load_turns(conversations_json, year_min=year_min)
    if not turns:
        raise ValueError(f"No turns loaded from {conversations_json}")

    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "rows.jsonl"
    summary_path = out_dir / "summary.json"

    cutoffs = []
    c = cutoff_turns
    while c < len(turns) and len(cutoffs) < max_cutoffs:
        cutoffs.append(c)
        c += stride_turns

    print(f"[future_eval] loaded {len(turns)} turns")
    print(f"[future_eval] cutoffs={cutoffs}")
    print(f"[future_eval] builders={builders}")

    if dry_run:
        preview = []
        for cutoff in cutoffs:
            future = turns[cutoff:cutoff + future_turns]
            queries = [t for t in future if _is_eval_query(t, min_query_chars, max_query_chars)]
            preview.append({
                "cutoff_idx": cutoff,
                "past_turns": cutoff,
                "future_turns": len(future),
                "candidate_queries": len(queries),
                "examples": [
                    {"idx": q.idx, "title": q.title[:80], "text": q.text[:220]}
                    for q in queries[:5]
                ],
            })
        payload = {"dry_run": True, "n_turns": len(turns), "cutoffs": preview}
        summary_path.write_text(json.dumps(payload, indent=2))
        print(json.dumps(payload, indent=2))
        return payload

    all_rows: list[EvalRow] = []
    backend_stats_by_cutoff: dict[str, dict[str, dict]] = {}

    with rows_path.open("w", buffering=1) as f:
        for cutoff in cutoffs:
            past = turns[:cutoff]
            future = turns[cutoff:cutoff + future_turns]
            query_turns = [
                t for t in future
                if _is_eval_query(t, min_query_chars, max_query_chars)
            ][:max_queries_per_cutoff]

            print(f"\n[cutoff {cutoff}] past={len(past)} future={len(future)} queries={len(query_turns)}")

            raw_backend, raw_stats = _build_raw(past)
            built: dict[str, tuple[Backend, BackendStats]] = {"raw": (raw_backend, raw_stats)}
            for name in builders:
                if name == "raw":
                    continue
                print(f"  building {name}...", flush=True)
                built[name] = _BUILDERS[name](past)
                print(
                    f"    {name}: stored_chars={built[name][1].stored_chars} "
                    f"compression={built[name][1].compression_ratio:.3f} "
                    f"build={built[name][1].build_seconds:.1f}s",
                    flush=True,
                )

            backend_stats_by_cutoff[str(cutoff)] = {
                name: asdict(stats) for name, (_backend, stats) in built.items()
                if name in builders
            }

            for qt in query_turns:
                teacher_answer, teacher_hits = _answer_backend(qt.text, raw_backend, k=teacher_k)
                if _not_in_context(teacher_answer) and not keep_not_in_context:
                    continue

                print(f"  q@{qt.idx}: {qt.text[:90].replace(chr(10), ' ')}", flush=True)
                for name in builders:
                    backend, stats = built[name]
                    answer, hits = _answer_backend(qt.text, backend, k=eval_k)
                    score, reason = judge(qt.text, teacher_answer, answer)
                    row = EvalRow(
                        cutoff_idx=cutoff,
                        query_idx=qt.idx,
                        query=qt.text,
                        teacher_answer=teacher_answer,
                        backend=name,
                        answer=answer,
                        score=score,
                        judge_reason=reason,
                        retrieved_chars=_retrieved_chars(hits),
                        retrieved_units=len(hits),
                        stored_chars=stats.stored_chars,
                        stored_units=stats.stored_units,
                        compression_ratio=stats.compression_ratio,
                    )
                    all_rows.append(row)
                    f.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")

    summary = _summarise(all_rows, backend_stats_by_cutoff)
    summary["config"] = {
        "conversations_json": str(conversations_json),
        "builders": builders,
        "year_min": year_min,
        "cutoff_turns": cutoff_turns,
        "future_turns": future_turns,
        "stride_turns": stride_turns,
        "max_cutoffs": max_cutoffs,
        "max_queries_per_cutoff": max_queries_per_cutoff,
        "teacher_k": teacher_k,
        "eval_k": eval_k,
        "keep_not_in_context": keep_not_in_context,
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {rows_path}")
    print(f"Wrote {summary_path}")
    return summary


def _summarise(rows: list[EvalRow], backend_stats_by_cutoff: dict[str, dict[str, dict]]) -> dict:
    by_backend: dict[str, list[EvalRow]] = {}
    for r in rows:
        by_backend.setdefault(r.backend, []).append(r)

    out = {
        "n_rows": len(rows),
        "by_backend": {},
        "backend_stats_by_cutoff": backend_stats_by_cutoff,
    }
    for name, rs in sorted(by_backend.items()):
        n = len(rs)
        score = sum(r.score for r in rs) / max(1, n)
        retrieved_chars = sum(r.retrieved_chars for r in rs) / max(1, n)
        retrieved_tokens = sum(_estimate_tokens("x" * r.retrieved_chars) for r in rs) / max(1, n)
        stored_chars = max((r.stored_chars for r in rs), default=0)
        out["by_backend"][name] = {
            "n": n,
            "avg_score_vs_teacher": score,
            "avg_retrieved_chars": retrieved_chars,
            "avg_retrieved_tokens_est": retrieved_tokens,
            "stored_chars": stored_chars,
            "stored_tokens_est": _estimate_tokens("x" * stored_chars) if stored_chars else 0,
            "avg_compression_ratio": sum(r.compression_ratio for r in rs) / max(1, n),
            "score_per_1k_retrieved_tokens": (
                score / max(1e-9, retrieved_tokens / 1000.0)
            ),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conversations-json", required=True, type=Path)
    ap.add_argument("--out-dir", type=Path, default=config.RESULTS_DIR / "future_eval")
    ap.add_argument("--builders", default="raw,mastra,pie_write",
                    help=f"comma-separated subset of {sorted(_BUILDERS)}")
    ap.add_argument("--year-min", type=int, default=2023)
    ap.add_argument("--cutoff-turns", type=int, default=500)
    ap.add_argument("--future-turns", type=int, default=200)
    ap.add_argument("--stride-turns", type=int, default=500)
    ap.add_argument("--max-cutoffs", type=int, default=1)
    ap.add_argument("--max-queries-per-cutoff", type=int, default=25)
    ap.add_argument("--teacher-k", type=int, default=50)
    ap.add_argument("--eval-k", type=int, default=12)
    ap.add_argument("--min-query-chars", type=int, default=12)
    ap.add_argument("--max-query-chars", type=int, default=1200)
    ap.add_argument("--keep-not-in-context", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    builders = [b.strip() for b in args.builders.split(",") if b.strip()]
    unknown = [b for b in builders if b not in _BUILDERS]
    if unknown:
        raise SystemExit(f"Unknown builders: {unknown}; choices={sorted(_BUILDERS)}")
    if "raw" not in builders:
        builders.insert(0, "raw")

    run(
        conversations_json=args.conversations_json,
        out_dir=args.out_dir,
        builders=builders,
        year_min=args.year_min,
        cutoff_turns=args.cutoff_turns,
        future_turns=args.future_turns,
        stride_turns=args.stride_turns,
        max_cutoffs=args.max_cutoffs,
        max_queries_per_cutoff=args.max_queries_per_cutoff,
        teacher_k=args.teacher_k,
        eval_k=args.eval_k,
        min_query_chars=args.min_query_chars,
        max_query_chars=args.max_query_chars,
        keep_not_in_context=args.keep_not_in_context,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
