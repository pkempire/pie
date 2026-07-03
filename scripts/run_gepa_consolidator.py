"""GEPA optimization wrapper for the Auto-Dreamer DSPy consolidator.

Pipeline (per metric call):
  1. Take one working-region example (a chunk of ~30 LoCoMo turns + the
     QA subset that any of those turns appears in evidence for).
  2. Run the candidate `Consolidator` on the working region.
  3. Flatten its entries into Units and index in a FlatBackend.
  4. For each attached question, retrieve top-k, ask the answer LM, judge.
  5. Return mean judge score (+ a textual feedback string for GEPA reflection).

Tiny defaults (CLI without overrides):
  - conv-26
  - 1 working-region example (first chunk)
  - 5 questions
  - max_metric_calls=20   (tiny GEPA budget)
  - judge + answer LM = gpt-5-mini

Run:
  python scripts/run_gepa_consolidator.py --quick

Full (do NOT run blindly — ~$30+):
  python scripts/run_gepa_consolidator.py --auto medium
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import dspy
from dspy.teleprompt import GEPA

# Make the script importable from repo root without install.
_HERE = Path(__file__).resolve()
_REPO = _HERE.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from mempol import config  # noqa: E402
from mempol.backends.flat import FlatBackend  # noqa: E402
from mempol.data import locomo  # noqa: E402
from mempol.eval.judge import judge  # noqa: E402
from mempol.dspy_consolidator.consolidator import Consolidator  # noqa: E402
from mempol.dspy_consolidator.run_baseline import (  # noqa: E402
    chunk_turns,
    turns_to_dspy,
    entry_to_unit,
    answer_question,
)


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------
def build_examples_from_chunks(
    conv,
    qpack: list[dict],
    chunks: list,
    label: str = "trainset",
) -> list[dspy.Example]:
    examples = []
    for ci, chunk in enumerate(chunks):
        wr = turns_to_dspy(chunk)
        ex = dspy.Example(
            working_region=wr,
            questions=qpack,
            conv_id=conv.sample_id,
            chunk_idx=ci,
        ).with_inputs("working_region")
        examples.append(ex)
    print(f"[{label}] {len(examples)} working-region examples, "
          f"{len(qpack)} QA per example", flush=True)
    return examples


def build_train_val_sets(
    conv_id: str = "conv-26",
    chunk_size: int = 30,
    n_chunks_train: int = 1,
    n_chunks_val: int | None = None,
    n_questions: int = 5,
) -> tuple[list[dspy.Example], list[dspy.Example]]:
    """Build (trainset, valset) of dspy.Example objects.

    If `n_chunks_val` is None, the trainset is also returned as the valset
    (current behavior). Otherwise, the first `n_chunks_train` chunks become
    the trainset and the next `n_chunks_val` chunks (no overlap) become the
    valset.
    """
    all_convs = locomo.load()
    pick = [(c, qs) for c, qs in all_convs if c.sample_id == conv_id]
    if not pick:
        raise SystemExit(f"conv {conv_id!r} not found in LoCoMo.")
    conv, qas = pick[0]

    # Use the first `n_questions` questions with a non-empty gold answer.
    qs = [q for q in qas if q.answer][:n_questions]
    qpack = [
        {"qid": q.qid, "question": q.question, "gold": q.answer,
         "category": q.category_name}
        for q in qs
    ]

    all_chunks = chunk_turns(conv.turns, size=chunk_size)

    if n_chunks_val is None:
        train_chunks = all_chunks[:n_chunks_train]
        trainset = build_examples_from_chunks(conv, qpack, train_chunks, label="trainset")
        return trainset, trainset

    train_chunks = all_chunks[:n_chunks_train]
    val_chunks = all_chunks[n_chunks_train: n_chunks_train + n_chunks_val]
    if len(val_chunks) < n_chunks_val:
        print(f"[warn] only {len(val_chunks)} val chunks available "
              f"(requested {n_chunks_val}); conv has {len(all_chunks)} total.",
              flush=True)
    trainset = build_examples_from_chunks(conv, qpack, train_chunks, label="trainset")
    valset = build_examples_from_chunks(conv, qpack, val_chunks, label="valset")
    return trainset, valset


# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------
_CALL_COUNTER = {"metric": 0, "answer": 0, "judge": 0}


def make_metric(top_k: int = 5, answer_model: str = "gpt-5-mini"):
    """Return a GEPA-compatible metric closure."""

    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        _CALL_COUNTER["metric"] += 1
        entries = list(getattr(pred, "consolidated_entries", []) or [])

        # Backend over this prediction's entries only.
        units = [entry_to_unit(e, gold.conv_id, i) for i, e in enumerate(entries)]
        backend = FlatBackend()
        if units:
            backend.ingest(units)

        scores: list[float] = []
        per_q_feedback: list[str] = []
        for q in gold.questions:
            hits = backend.retrieve(q["question"], k=top_k, source="hybrid") if units else []
            _CALL_COUNTER["answer"] += 1
            answer = answer_question(q["question"], hits, model=answer_model)
            _CALL_COUNTER["judge"] += 1
            s, reason = judge(q["question"], q["gold"], answer)
            scores.append(s)
            per_q_feedback.append(
                f"  Q({q['category']}): {q['question'][:80]}\n"
                f"    gold: {q['gold'][:80]}\n"
                f"    pred: {answer[:80]}\n"
                f"    score: {s}  reason: {reason[:100]}"
            )

        mean = sum(scores) / len(scores) if scores else 0.0
        feedback = (
            f"Consolidated {len(entries)} entries from {len(gold.working_region)} turns. "
            f"Mean QA judge score: {mean:.3f} over {len(scores)} questions.\n"
            + "\n".join(per_q_feedback)
        )

        # GEPA understands dspy.Prediction(score=..., feedback=...).
        return dspy.Prediction(score=mean, feedback=feedback)

    return metric


# ---------------------------------------------------------------------------
# Prompt extraction (for diffs)
# ---------------------------------------------------------------------------
def get_prompt(module: dspy.Module) -> str:
    """Concatenate signature instructions for every predictor in the module."""
    parts = []
    for name, predictor in module.named_predictors():
        sig = predictor.signature
        parts.append(f"### predictor: {name}\n{sig.instructions}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conv", default="conv-26")
    ap.add_argument("--chunk-size", type=int, default=30)
    ap.add_argument("--n-chunks", type=int, default=1,
                    help="Used if --n-chunks-train / --n-chunks-val are NOT set. "
                         "trainset == valset behavior.")
    ap.add_argument("--n-chunks-train", type=int, default=None,
                    help="Number of chunks for the trainset. "
                         "If set together with --n-chunks-val, splits chunks "
                         "into disjoint train/val groups.")
    ap.add_argument("--n-chunks-val", type=int, default=None,
                    help="Number of chunks for the valset (no overlap with train).")
    ap.add_argument("--n-questions", type=int, default=5)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--task-model", default="openai/gpt-5-mini",
                    help="Model used by the Consolidator on every metric call.")
    ap.add_argument("--reflection-model", default="openai/gpt-5-mini")
    ap.add_argument("--answer-model", default="gpt-5-mini")
    ap.add_argument("--auto", default=None, choices=[None, "light", "medium", "heavy"])
    ap.add_argument("--max-metric-calls", type=int, default=20,
                    help="Default quick run: 20 rollouts. Ignored if --auto set.")
    ap.add_argument("--num-threads", type=int, default=2,
                    help="Passed to dspy.GEPA(num_threads=...).")
    ap.add_argument("--skip-baseline", action="store_true",
                    help="Skip the un-optimized baseline scoring pass (~25 min).")
    ap.add_argument("--quick", action="store_true",
                    help="Force quick defaults: 1 chunk, 5 Qs, 20 metric calls.")
    ap.add_argument("--smoke", action="store_true",
                    help=argparse.SUPPRESS)
    ap.add_argument("--out-dir", default=str(config.RESULTS_DIR / "gepa_consolidator"))
    args = ap.parse_args()

    if args.quick or args.smoke:
        args.n_chunks = 1
        args.n_chunks_train = None
        args.n_chunks_val = None
        args.n_questions = 5
        args.max_metric_calls = 20
        args.auto = None

    # Resolve chunk counts: explicit train/val pair wins, else fall back to --n-chunks.
    if args.n_chunks_train is not None and args.n_chunks_val is not None:
        n_chunks_train = args.n_chunks_train
        n_chunks_val = args.n_chunks_val
    else:
        n_chunks_train = args.n_chunks_train if args.n_chunks_train is not None else args.n_chunks
        n_chunks_val = None  # signals trainset == valset

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Configure the task LM (used inside Consolidator on every metric call).
    dspy.configure(lm=dspy.LM(args.task_model))
    reflection_lm = dspy.LM(args.reflection_model)

    # Build trainset (+ optional separate valset).
    trainset, valset = build_train_val_sets(
        conv_id=args.conv,
        chunk_size=args.chunk_size,
        n_chunks_train=n_chunks_train,
        n_chunks_val=n_chunks_val,
        n_questions=args.n_questions,
    )

    # ---- Sanity summary ----
    print("\n==== RUN CONFIG ====")
    print(f"  task_model        : {args.task_model}")
    print(f"  reflection_model  : {args.reflection_model}")
    print(f"  answer_model      : {args.answer_model}")
    print(f"  n_chunks_train    : {len(trainset)}")
    print(f"  n_chunks_val      : {len(valset)}"
          f"{'  (== trainset)' if valset is trainset else ''}")
    print(f"  n_questions       : {args.n_questions}")
    print(f"  num_threads       : {args.num_threads}")
    print(f"  auto              : {args.auto}")
    print(f"  max_metric_calls  : "
          f"{args.max_metric_calls if not args.auto else f'(ignored, auto={args.auto})'}")
    print(f"  skip_baseline     : {args.skip_baseline}")
    print("====================\n", flush=True)

    # ---- Baseline ----
    consolidator = Consolidator()
    original_prompt = get_prompt(consolidator)
    metric = make_metric(top_k=args.top_k, answer_model=args.answer_model)
    if args.skip_baseline:
        print("[baseline] skipped", flush=True)
        base_secs = 0.0
        base_mean = float("nan")
    else:
        print("[baseline] scoring un-optimized consolidator...", flush=True)
        t0 = time.time()
        base_scores = []
        for i, ex in enumerate(trainset, start=1):
            print(f"[baseline] example {i}/{len(trainset)} chunk={ex.chunk_idx}", flush=True)
            pred = consolidator(working_region=ex.working_region)
            r = metric(ex, pred)
            base_scores.append(float(r.score))
            print(f"[baseline] example {i}: score={float(r.score):.3f}", flush=True)
        base_secs = time.time() - t0
        base_mean = sum(base_scores) / len(base_scores) if base_scores else 0.0
        print(f"[baseline] mean_score={base_mean:.3f}  ({base_secs:.1f}s, "
              f"{_CALL_COUNTER['metric']} metric calls)", flush=True)

    # Snapshot counters BEFORE GEPA so we can attribute cost.
    pre = dict(_CALL_COUNTER)

    # ---- GEPA optimize ----
    kw = dict(
        metric=metric,
        num_threads=args.num_threads,
        reflection_lm=reflection_lm,
        track_stats=True,
        log_dir=str(out_dir / "gepa_log"),
    )
    if args.auto:
        kw["auto"] = args.auto
    else:
        kw["max_metric_calls"] = args.max_metric_calls

    print(f"\n[gepa] starting compile  kwargs={ {k:v for k,v in kw.items() if k!='metric'} }",
          flush=True)
    gepa = GEPA(**kw)
    t1 = time.time()
    # If valset is a distinct object, pass it; otherwise let GEPA reuse trainset.
    if valset is not trainset:
        optimized = gepa.compile(consolidator, trainset=trainset, valset=valset)
    else:
        optimized = gepa.compile(consolidator, trainset=trainset)
    gepa_secs = time.time() - t1
    optimized_prompt = get_prompt(optimized)

    # ---- Re-score optimized on the valset (== trainset in legacy mode) ----
    print("\n[gepa] scoring optimized consolidator on valset...", flush=True)
    opt_scores = []
    for i, ex in enumerate(valset, start=1):
        print(f"[gepa] val example {i}/{len(valset)} chunk={ex.chunk_idx}", flush=True)
        pred = optimized(working_region=ex.working_region)
        r = metric(ex, pred)
        opt_scores.append(float(r.score))
        print(f"[gepa] val example {i}: score={float(r.score):.3f}", flush=True)
    opt_mean = sum(opt_scores) / len(opt_scores) if opt_scores else 0.0

    # ---- Report ----
    post = dict(_CALL_COUNTER)
    summary = {
        "conv": args.conv,
        "n_chunks_train": len(trainset),
        "n_chunks_val": len(valset),
        "n_questions": args.n_questions,
        "task_model": args.task_model,
        "reflection_model": args.reflection_model,
        "answer_model": args.answer_model,
        "max_metric_calls_setting": args.max_metric_calls if not args.auto else f"auto={args.auto}",
        "baseline_mean": base_mean,
        "gepa_mean": opt_mean,
        "delta": opt_mean - base_mean,
        "baseline_secs": round(base_secs, 1),
        "gepa_secs": round(gepa_secs, 1),
        "metric_calls_during_gepa": post["metric"] - pre["metric"],
        "answer_calls_total": post["answer"],
        "judge_calls_total": post["judge"],
        "metric_calls_total": post["metric"],
        "prompt_changed": original_prompt != optimized_prompt,
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "prompt_original.txt").write_text(original_prompt)
    (out_dir / "prompt_optimized.txt").write_text(optimized_prompt)

    print("\n==== GEPA RUN SUMMARY ====")
    print(json.dumps(summary, indent=2))
    print("\n---- ORIGINAL PROMPT (first 500 chars) ----")
    print(original_prompt[:500])
    print("\n---- OPTIMIZED PROMPT (first 500 chars) ----")
    print(optimized_prompt[:500])
    print(f"\n[out] wrote {out_dir}/{{summary.json,prompt_original.txt,prompt_optimized.txt}}")


if __name__ == "__main__":
    main()
