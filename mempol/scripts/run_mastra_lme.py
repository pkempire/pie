"""Reproduce Mastra's LongMemEval result with gpt-5-mini.

Mastra reports 94.87% on LongMemEval with gpt-5-mini (Observer + Reflector +
Answer all gpt-5-mini). This script wraps the standard ingest + eval flow but
forces gpt-5-mini for all three roles. The judge stays at gpt-4o for stronger
grading (matches the LongMemEval paper protocol).

LongMemEval-S has ~115K tokens / 30-40 sessions per question. The Observer
default 30K threshold WILL fire multiple times — that's the regime Mastra is
designed for.

Usage:
    # 1. Get HF token: https://huggingface.co/settings/tokens
    export HF_TOKEN=hf_...

    # 2. One-question smoke
    python -m mempol.scripts.run_mastra_lme \\
        --variant longmemeval_oracle --conv-idx 0 --max-qs 1

    # 3. Full longmemeval_s 500-question run (~$30-50 with gpt-5-mini)
    python -m mempol.scripts.run_mastra_lme \\
        --variant longmemeval_s --max-qs 500 --concurrency 1
"""
from __future__ import annotations
import argparse
import json
import os
import time
from pathlib import Path

# Force gpt-5-mini before mempol.config caches model strings
os.environ.setdefault("MEMPOL_OBSERVER_MODEL",  "gpt-5-mini")
os.environ.setdefault("MEMPOL_REFLECTOR_MODEL", "gpt-5-mini")
os.environ.setdefault("MEMPOL_ANSWER_MODEL",    "gpt-5-mini")
# Judge: keep on gpt-4o (LongMemEval paper protocol). Override via env if needed.
os.environ.setdefault("MEMPOL_JUDGE_MODEL",     "gpt-4o")

from mempol import config       # noqa: E402  imports after env override
from mempol.backends.mastra import MastraBackend
from mempol.data.longmemeval import load as load_lme
from mempol.eval.judge import judge
from mempol.eval.metrics import Result, summarise
from mempol.eval.runner import conv_to_units
from mempol.policies.v0_naive import NaivePolicy
from mempol.policies.v1_heuristic import HeuristicPolicy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="longmemeval_oracle",
                    choices=["longmemeval_s", "longmemeval_oracle", "longmemeval_m"])
    ap.add_argument("--conv-idx", type=int, default=0)
    ap.add_argument("--max-qs", type=int, default=10,
                    help="number of LongMemEval questions to process (0 = all)")
    ap.add_argument("--policy", default="v0_naive", choices=["v0_naive", "v1_heuristic"],
                    help="v0_naive matches Mastra's protocol most closely (no reformulate)")
    ap.add_argument("--observer-threshold", type=int, default=30_000,
                    help="Mastra's actual default for LongMemEval is 30k tokens.")
    ap.add_argument("--reflector-threshold", type=int, default=40_000)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--qids-file", default=None,
                    help="JSON list of question_ids to run (paired comparison with other harnesses)")
    ap.add_argument("--read-protocol", default="full_context",
                    choices=["full_context", "policy"],
                    help="full_context = Mastra's real protocol (whole OM log in context, no "
                         "retrieval). policy = legacy top-k retrieval (INVALID for OM claims).")
    args = ap.parse_args()

    print(f"[run_mastra_lme] models:")
    print(f"  observer  = {config.OBSERVER_MODEL}")
    print(f"  reflector = {config.REFLECTOR_MODEL}")
    print(f"  answer    = {config.ANSWER_MODEL}")
    print(f"  judge     = {config.JUDGE_MODEL}")

    # Each LongMemEval row is (1 conv, 1 qa). We iterate row-by-row.
    if args.qids_file:
        wanted = set(json.load(open(args.qids_file)))
        rows = load_lme(variant=args.variant, n_convs=0 or 1000)
        rows = [(c, q) for (c, q) in rows if c.sample_id in wanted]
        args.max_qs = 0  # run all matched
        print(f"[{args.variant}] qids-file matched {len(rows)}/{len(wanted)} rows")
    else:
        n_load = args.max_qs if args.max_qs else 1000
        rows = load_lme(variant=args.variant, n_convs=n_load)
    print(f"[{args.variant}] loaded {len(rows)} (conv, qa) rows")

    run_name = args.run_name or f"mastra_lme_{args.variant}_{config.ANSWER_MODEL.replace('-','_')}"
    out_dir = config.RESULTS_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    traces_path = out_dir / "traces.jsonl"
    state_dir = out_dir / "states"
    state_dir.mkdir(parents=True, exist_ok=True)

    PolicyCls = NaivePolicy if args.policy == "v0_naive" else HeuristicPolicy
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    results: list[Result] = []
    t0 = time.time()
    with traces_path.open("a", buffering=1) as f:
        for ci, (conv, qas) in enumerate(rows):
            if args.max_qs and ci >= args.max_qs:
                break
            # Per-row backend (Mastra OM is per-thread anyway).
            t_ingest = time.time()
            b = MastraBackend(
                observer_token_threshold=args.observer_threshold,
                reflector_token_threshold=args.reflector_threshold,
            )
            b.ingest(conv_to_units(conv))
            ingest_secs = time.time() - t_ingest

            qa = qas[0]
            t_q = time.time()
            if args.read_protocol == "full_context":
                # Mastra's ACTUAL protocol: the entire compressed observation log goes in
                # context — no retrieval. (Prior runs used top-k snippet retrieval via
                # NaivePolicy, which cripples OM and produced an invalid 24% — harness bug,
                # not a property of their system.)
                from mempol.llm import chat as llm_chat
                full_ctx = b.get_full_context()
                ans = llm_chat(
                    [{"role": "system",
                      "content": "Answer the question using the memory below. Be concise and "
                                 "specific. If the memory lacks the answer, say you don't know."},
                     {"role": "user",
                      "content": f"Memory:\n{full_ctx}\n\nQuestion: {qa.question}"}],
                    model=config.ANSWER_MODEL)
                class _T:  # minimal trace shim
                    answer = ans
                    steps: list = []
                    n_retrievals = 0
                    cost_tokens = 0
                trace = _T()
            else:
                policy = PolicyCls()
                trace = policy.run(qa.question, b)
            score, reason = judge(qa.question, qa.answer, trace.answer)
            q_secs = time.time() - t_q

            # ---- Per-row Mastra memory log dump ----
            log_path = logs_dir / f"{ci:03d}_{conv.sample_id[:8]}_{qa.category_name}.md"
            log_md = [
                f"# Mastra OM log — row {ci} · {qa.category_name}",
                "",
                f"**conv_id**: `{conv.sample_id}`  ·  **n_turns**: {len(conv.turns)}  ·  "
                f"**n_sessions**: {len({t.session for t in conv.turns})}",
                "",
                f"**Question**: {qa.question}",
                f"**Gold**: {qa.answer}",
                f"**Answer**: {trace.answer}",
                f"**Judge score**: {score}  ·  reason: {reason}",
                f"**Compression**: raw_chars={b._stats['n_raw_chars_seen']:,} → "
                f"observation_chars={b._stats['n_observation_chars']:,} = "
                f"{b._stats['n_raw_chars_seen'] / max(1, b._stats['n_observation_chars']):.1f}×",
                "",
                "---",
                "",
                b.memory_log_md(),
            ]
            log_path.write_text("\n".join(log_md))

            results.append(Result(
                qid=qa.qid, category=qa.category, category_name=qa.category_name,
                score=score, n_retrievals=trace.n_retrievals, n_steps=len(trace.steps),
                answer=trace.answer, gold=qa.answer, judge_reason=reason,
                evidence_recall=None,
            ))
            f.write(json.dumps({
                "qid": qa.qid, "category_name": qa.category_name,
                "n_turns": len(conv.turns),
                "n_sessions": len({t.session for t in conv.turns}),
                "question": qa.question, "gold": qa.answer, "answer": trace.answer,
                "score": score, "judge_reason": reason,
                "ingest_secs": round(ingest_secs, 1),
                "q_secs": round(q_secs, 1),
                "stats": b.stats(),
                "compression_ratio": (b._stats["n_raw_chars_seen"] /
                                      max(1, b._stats["n_observation_chars"])),
                "log_path": str(log_path.relative_to(config.RESULTS_DIR.parent)),
            }) + "\n"); f.flush()

            running_acc = sum(r.score for r in results) / len(results)
            print(f"  [{ci+1}/{len(rows)}] {qa.category_name:20s} "
                  f"score={score:.1f}  ingest={ingest_secs:.0f}s  "
                  f"q={q_secs:.0f}s  cum_acc={running_acc:.3f}  "
                  f"log={log_path.name}",
                  flush=True)

    summary = summarise(results)
    summary["wall_time_s"] = round(time.time() - t0, 2)
    summary["variant"] = args.variant
    summary["models"] = {
        "observer": config.OBSERVER_MODEL,
        "reflector": config.REFLECTOR_MODEL,
        "answer": config.ANSWER_MODEL,
        "judge": config.JUDGE_MODEL,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== FINAL ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
