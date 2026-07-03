"""Baseline runner for the DSPy consolidator on LoCoMo.

Pipeline:
  1. Load LoCoMo conv-26 (default — overrideable via --conv).
  2. Split its turns into chunks of ~30 turns (1 chunk == 1 working region).
  3. Run the un-optimized Consolidator on each chunk, collect entries.
  4. Index entries in a FlatBackend (hybrid BM25 + dense).
  5. For each question, retrieve top-5 entries and ask the answer LM.
  6. Score against gold using mempol.eval.judge.judge.
  7. Print per-category + overall accuracy and dump a JSONL trace.

This script is intentionally I/O-light and side-effect-clear so it can be
called from a GEPA optimization wrapper in the next step.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import dspy

# Make `python mempol/dspy_consolidator/run_baseline.py` work without install.
_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from mempol import config, llm  # noqa: E402
from mempol.backends.flat import FlatBackend  # noqa: E402
from mempol.backends.base import Unit  # noqa: E402
from mempol.data import locomo  # noqa: E402
from mempol.eval.judge import judge  # noqa: E402

from mempol.dspy_consolidator.consolidator import (  # noqa: E402
    Consolidator,
    ConsolidatedEntry,
    Turn as DspyTurn,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def chunk_turns(turns: list[locomo.Turn], size: int = 30) -> list[list[locomo.Turn]]:
    """Split a flat list of turns into contiguous chunks of `size`."""
    return [turns[i : i + size] for i in range(0, len(turns), size)]


def turns_to_dspy(turns: list[locomo.Turn]) -> list[DspyTurn]:
    return [
        DspyTurn(
            dia_id=t.dia_id,
            speaker=t.speaker,
            text=t.text,
            session_date=t.session_date,
        )
        for t in turns
    ]


def entry_to_unit(entry: ConsolidatedEntry, conv_id: str, idx: int) -> Unit:
    """Flatten one ConsolidatedEntry into a retrievable Unit."""
    if entry.entry_type == "procedural":
        body = "\n".join(f"- {s}" for s in entry.steps) if entry.steps else ""
    else:
        body = entry.details or ""
    text = (
        f"[{entry.entry_type}] {entry.name}\n"
        f"speaker: {entry.speaker}\n"
        f"{entry.summary}\n"
        f"{body}".strip()
    )
    uid = f"{conv_id}::entry_{idx}"
    return Unit(
        uid=uid,
        text=text,
        metadata={
            "entry_type": entry.entry_type,
            "speaker": entry.speaker,
            "source_turn_ids": list(entry.source_turn_ids),
            "name": entry.name,
        },
    )


_ANSWER_SYS = (
    "You answer questions about a long-running conversation between two people "
    "using only the consolidated memory entries provided. "
    "Be concise (one sentence). If the answer is not present, say 'Not in context'."
)


def answer_question(question: str, hits, model: str) -> str:
    """Ask the answer LM using just the retrieved entries."""
    ctx_lines = []
    for i, h in enumerate(hits, 1):
        ctx_lines.append(f"[{i}] {h.unit.text}")
    ctx = "\n\n".join(ctx_lines) if ctx_lines else "(no memory entries retrieved)"
    msgs = [
        {"role": "system", "content": _ANSWER_SYS},
        {
            "role": "user",
            "content": f"Memory entries:\n{ctx}\n\nQuestion: {question}\nAnswer:",
        },
    ]
    try:
        return llm.chat(msgs, model=model).strip()
    except Exception as e:
        return f"error:{e}"


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def run(
    conv_id: str = "conv-26",
    chunk_size: int = 30,
    top_k: int = 5,
    max_chunks: int | None = None,
    max_questions: int | None = None,
    out_path: Path | None = None,
    model: str = "openai/gpt-5-mini",
    answer_model: str = "gpt-5-mini",
) -> dict:
    # 1) Configure DSPy.
    dspy.configure(lm=dspy.LM(model))

    # 2) Load the requested conv.
    all_convs = locomo.load()
    pick = [(c, qs) for c, qs in all_convs if c.sample_id == conv_id]
    if not pick:
        raise SystemExit(f"conv {conv_id!r} not found in LoCoMo. Available: "
                         f"{[c.sample_id for c, _ in all_convs]}")
    conv, qas = pick[0]

    # 3) Chunk turns + run consolidator.
    chunks = chunk_turns(conv.turns, size=chunk_size)
    if max_chunks is not None:
        chunks = chunks[:max_chunks]

    consolidator = Consolidator()
    all_entries: list[ConsolidatedEntry] = []
    chunk_stats: list[dict] = []
    t0 = time.time()
    for ci, chunk in enumerate(chunks):
        t_chunk = time.time()
        try:
            pred = consolidator(working_region=turns_to_dspy(chunk))
            entries = list(pred.consolidated_entries or [])
        except Exception as e:
            print(f"[consolidate] chunk {ci+1}/{len(chunks)} FAILED: {e}", flush=True)
            entries = []
        all_entries.extend(entries)
        dt = time.time() - t_chunk
        chunk_stats.append({"chunk": ci, "n_turns": len(chunk), "n_entries": len(entries), "secs": round(dt, 2)})
        print(
            f"[consolidate] chunk {ci+1}/{len(chunks)} turns={len(chunk)} "
            f"entries={len(entries)} ({dt:.1f}s)",
            flush=True,
        )
    consolidation_secs = time.time() - t0
    print(f"[consolidate] total entries={len(all_entries)} in {consolidation_secs:.1f}s", flush=True)

    # 4) Build memory bank → FlatBackend.
    units = [entry_to_unit(e, conv.sample_id, i) for i, e in enumerate(all_entries)]
    backend = FlatBackend()
    if units:
        backend.ingest(units)
    else:
        print("[warn] no consolidated entries — backend is empty, all answers will be 'Not in context'",
              flush=True)

    # 5) Answer questions + judge.
    questions = qas[:max_questions] if max_questions else qas
    per_cat_total: dict[str, int] = defaultdict(int)
    per_cat_score: dict[str, float] = defaultdict(float)
    rows: list[dict] = []
    t1 = time.time()
    for qi, q in enumerate(questions):
        if not q.answer:
            # Skip questions with no gold (e.g., malformed rows).
            continue
        hits = backend.retrieve(q.question, k=top_k, source="hybrid") if units else []
        pred = answer_question(q.question, hits, model=answer_model)
        score, reason = judge(q.question, q.answer, pred)
        per_cat_total[q.category_name] += 1
        per_cat_score[q.category_name] += score
        rows.append({
            "qid": q.qid,
            "category": q.category_name,
            "question": q.question,
            "gold": q.answer,
            "pred": pred,
            "score": score,
            "reason": reason,
            "retrieved_uids": [h.unit.uid for h in hits],
        })
        if (qi + 1) % 10 == 0 or qi + 1 == len(questions):
            running = sum(per_cat_score.values()) / max(sum(per_cat_total.values()), 1)
            print(f"[qa] {qi+1}/{len(questions)} running_acc={running:.3f}", flush=True)
    qa_secs = time.time() - t1

    # 6) Aggregate.
    overall_total = sum(per_cat_total.values())
    overall_score = sum(per_cat_score.values())
    overall_acc = overall_score / overall_total if overall_total else 0.0
    by_cat = {
        cat: {
            "n": per_cat_total[cat],
            "acc": (per_cat_score[cat] / per_cat_total[cat]) if per_cat_total[cat] else 0.0,
        }
        for cat in sorted(per_cat_total)
    }

    summary = {
        "conv_id": conv.sample_id,
        "n_turns": len(conv.turns),
        "n_chunks": len(chunks),
        "chunk_size": chunk_size,
        "n_entries": len(all_entries),
        "n_questions_scored": overall_total,
        "overall_acc": overall_acc,
        "by_category": by_cat,
        "consolidation_secs": round(consolidation_secs, 1),
        "qa_secs": round(qa_secs, 1),
        "model": model,
        "answer_model": answer_model,
        "chunk_stats": chunk_stats,
    }

    print("\n==== BASELINE RESULTS ====")
    print(json.dumps({k: v for k, v in summary.items() if k not in {"chunk_stats"}}, indent=2))

    if out_path is None:
        out_path = config.RESULTS_DIR / f"dspy_consolidator_baseline_{conv.sample_id}.jsonl"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        f.write(json.dumps({"_summary": summary}) + "\n")
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[out] wrote {len(rows)} rows + summary to {out_path}")
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--conv", default="conv-26")
    p.add_argument("--chunk-size", type=int, default=30)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--max-chunks", type=int, default=None,
                   help="cap number of chunks (smoke test).")
    p.add_argument("--max-questions", type=int, default=None,
                   help="cap number of questions (smoke test).")
    p.add_argument("--model", default="openai/gpt-5-mini",
                   help="DSPy LM identifier for the consolidator.")
    p.add_argument("--answer-model", default="gpt-5-mini",
                   help="OpenAI model name for the answer step (mempol.llm.chat).")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    run(
        conv_id=args.conv,
        chunk_size=args.chunk_size,
        top_k=args.top_k,
        max_chunks=args.max_chunks,
        max_questions=args.max_questions,
        out_path=Path(args.out) if args.out else None,
        model=args.model,
        answer_model=args.answer_model,
    )


if __name__ == "__main__":
    main()
