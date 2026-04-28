"""Run the *real* PIE pipeline on a LongMemEval row.

Calls the existing benchmarks/locomo/baselines.py:pie_temporal() — that's the
canonical PIE flow (full extraction via prompts.PIE_EXTRACTION_PROMPT, 3-tier
entity resolution, world model build, temporal retrieval, context compile,
answer). We convert LongMemEval rows to the LoCoMo-shape `item` dict that
function expects.

Apples-to-apples vs `run_mastra_lme.py` on the same LongMemEval rows.

Usage:
    python -m mempol.scripts.run_pie_lme \\
        --variant longmemeval_oracle --max-qs 3 \\
        --run-name pie_lme_oracle_smoke

    # On the same row Mastra was just run on:
    python -m mempol.scripts.run_pie_lme \\
        --variant longmemeval_s --max-qs 1 \\
        --run-name pie_lme_s_one_row
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

# Force gpt-5-mini for parity with Mastra run (Observer/Reflector/Answer all
# gpt-5-mini in the original 94.87% reproduction). PIE uses extraction_model
# for entity extraction and `model` for the final answer.
import os
os.environ.setdefault("MEMPOL_ANSWER_MODEL", "gpt-5-mini")
os.environ.setdefault("MEMPOL_JUDGE_MODEL", "gpt-4o")

from mempol import config       # noqa: E402
from mempol.data.longmemeval import load as load_lme
from mempol.eval.judge import judge


_LME_DATE_RE = __import__("re").compile(
    r"(\d{4})/(\d{1,2})/(\d{1,2})\s*\(\w+\)\s*(\d{1,2}):(\d{2})"
)
_MONTHS = ["January", "February", "March", "April", "May", "June",
           "July", "August", "September", "October", "November", "December"]


def _normalise_lme_date(s: str) -> str:
    """LongMemEval uses '2023/04/10 (Mon) 17:50'. The LoCoMo date parser only
    accepts '<Day> <MonthName> <Year> at <H>:<M> <am|pm>'. Rewrite to that
    so pie_temporal's adapter doesn't choke."""
    s = (s or "").strip()
    m = _LME_DATE_RE.match(s)
    if not m:
        return s  # unknown format; let downstream raise if it can't handle
    year, month, day, hour, minute = (int(m.group(i)) for i in range(1, 6))
    ampm = "am" if hour < 12 else "pm"
    h12 = hour % 12 or 12
    return f"{day} {_MONTHS[month-1]} {year} at {h12}:{minute:02d} {ampm}"


def lme_row_to_locomo_item(conv, qa) -> dict:
    """Convert a LongMemEval (Conversation, QA) pair into the LoCoMo `item`
    shape that pie_temporal() expects."""
    sess_dict: dict = {}
    for t in conv.turns:
        key = f"session_{t.session}"
        sess_dict.setdefault(key, []).append({
            "speaker": t.speaker,
            "dia_id": t.dia_id,
            "text": t.text,
        })
        date_key = f"session_{t.session}_date_time"
        if date_key not in sess_dict and t.session_date:
            sess_dict[date_key] = _normalise_lme_date(t.session_date)
    sess_dict["speaker_a"] = "user"
    sess_dict["speaker_b"] = "assistant"

    return {
        "question_id": qa.qid,
        "question_type": qa.category_name,
        "question": qa.question,
        "answer": qa.answer,
        "sample_id": conv.sample_id,
        "conversation": sess_dict,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="longmemeval_oracle",
                    choices=["longmemeval_s", "longmemeval_oracle", "longmemeval_m"])
    ap.add_argument("--max-qs", type=int, default=1)
    ap.add_argument("--start-idx", type=int, default=0,
                    help="skip the first N rows (so you can target the same row Mastra ran)")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--extraction-model", default="gpt-4o-mini",
                    help="model for PIE's per-session entity extraction")
    args = ap.parse_args()

    # Lazy imports — pie_temporal pulls in heavy modules
    from benchmarks.locomo.baselines import pie_temporal
    from pie.core.world_model import WorldModel  # noqa: F401  (registered for pickling)

    print(f"[run_pie_lme] models: extraction={args.extraction_model} answer={config.ANSWER_MODEL} judge={config.JUDGE_MODEL}")

    n_load = args.start_idx + args.max_qs
    rows = load_lme(variant=args.variant, n_convs=n_load)
    rows = rows[args.start_idx:args.start_idx + args.max_qs]
    print(f"[{args.variant}] loaded {len(rows)} rows starting at idx {args.start_idx}")

    run_name = args.run_name or f"pie_lme_{args.variant}_{int(time.time())}"
    out_dir = config.RESULTS_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    traces_path = out_dir / "traces.jsonl"
    wm_dir = out_dir / "world_models"
    wm_dir.mkdir(parents=True, exist_ok=True)

    results = []
    t0 = time.time()
    with traces_path.open("a", buffering=1) as f:
        for ci, (conv, qas) in enumerate(rows):
            qa = qas[0]
            item = lme_row_to_locomo_item(conv, qa)
            print(f"\n[{ci+1}/{len(rows)}] {qa.category_name}  conv={conv.sample_id[:8]}  "
                  f"n_turns={len(conv.turns)}  n_sessions={len({t.session for t in conv.turns})}", flush=True)
            print(f"  Q: {qa.question[:100]}", flush=True)
            print(f"  GOLD: {qa.answer}", flush=True)

            t_run = time.time()
            try:
                bres = pie_temporal(
                    item=item,
                    extraction_model=args.extraction_model,
                    model=config.ANSWER_MODEL,
                )
            except Exception as e:
                print(f"  ERROR: {e}", flush=True)
                continue
            run_secs = time.time() - t_run

            score, judge_reason = judge(qa.question, qa.answer, bres.hypothesis)
            print(f"  ANS: {bres.hypothesis[:120]}", flush=True)
            print(f"  judge={score:.1f}  retrieval_count={bres.retrieval_count}  "
                  f"latency={run_secs:.0f}s  ctx_chars={bres.context_chars}", flush=True)

            row_log = {
                "qid": qa.qid,
                "category_name": qa.category_name,
                "n_turns": len(conv.turns),
                "n_sessions": len({t.session for t in conv.turns}),
                "question": qa.question,
                "gold": qa.answer,
                "answer": bres.hypothesis,
                "score": score,
                "judge_reason": judge_reason,
                "retrieval_count": bres.retrieval_count,
                "context_chars": bres.context_chars,
                "tokens_prompt": getattr(bres, "tokens_prompt", None),
                "tokens_completion": getattr(bres, "tokens_completion", None),
                "latency_s": round(run_secs, 1),
            }
            f.write(json.dumps(row_log) + "\n"); f.flush()
            results.append(row_log)

    overall = sum(r["score"] for r in results) / max(1, len(results))
    summary = {
        "variant": args.variant,
        "n": len(results),
        "overall_acc": overall,
        "wall_time_s": round(time.time() - t0, 1),
        "extraction_model": args.extraction_model,
        "answer_model": config.ANSWER_MODEL,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== FINAL ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
