"""Step B: load a pre-ingested Mastra state and run the eval. Cheap to repeat.

Usage:
    python -m mempol.scripts.eval_mastra --run-name mastra_c1 --max-qs 0
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

from mempol import config
from mempol.backends.mastra import MastraBackend
from mempol.data.locomo import load as load_locomo
from mempol.data.longmemeval import load as load_lme
from mempol.eval.judge import judge
from mempol.eval.metrics import Result, summarise
from mempol.policies.v0_naive import NaivePolicy
from mempol.policies.v1_heuristic import HeuristicPolicy


_POLICIES = {"v0_naive": NaivePolicy, "v1_heuristic": HeuristicPolicy}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--dataset", choices=["locomo", "longmemeval_s",
                                           "longmemeval_oracle", "longmemeval_m"],
                    default="locomo")
    ap.add_argument("--conv-idx", type=int, default=0)
    ap.add_argument("--policy", choices=list(_POLICIES), default="v1_heuristic")
    ap.add_argument("--max-qs", type=int, default=0, help="0 = all")
    ap.add_argument("--out-suffix", default="", help="suffix on traces filename to allow chunked runs")
    ap.add_argument("--start-q", type=int, default=0, help="resume from this question index")
    args = ap.parse_args()

    out_dir = config.RESULTS_DIR / args.run_name
    state_path = out_dir / "mastra_state.pkl"
    if not state_path.exists():
        raise SystemExit(f"no state at {state_path}; run ingest_mastra first")
    b = MastraBackend.load(str(state_path))
    print(f"loaded mastra state. stats={b.stats()}")

    if args.dataset == "locomo":
        convs = load_locomo(n_convs=args.conv_idx + 1)
    else:
        convs = load_lme(variant=args.dataset, n_convs=args.conv_idx + 1)
    _, qas = convs[args.conv_idx]
    qas_to_run = qas if args.max_qs == 0 else qas[args.start_q:args.start_q + args.max_qs]
    policy = _POLICIES[args.policy]()

    suffix = f"_{args.out_suffix}" if args.out_suffix else ""
    traces_path = out_dir / f"traces{suffix}.jsonl"
    print(f"evaluating {policy.name} on {len(qas_to_run)} qs (start={args.start_q})…")

    results: list[Result] = []
    t0 = time.time()
    with traces_path.open("a", buffering=1) as f:
        for i, qa in enumerate(qas_to_run):
            trace = policy.run(qa.question, b)
            score, reason = judge(qa.question, qa.answer, trace.answer)
            r = Result(
                qid=qa.qid, category=qa.category, category_name=qa.category_name,
                score=score, n_retrievals=trace.n_retrievals, n_steps=len(trace.steps),
                answer=trace.answer, gold=qa.answer, judge_reason=reason, evidence_recall=None,
            )
            results.append(r)
            f.write(json.dumps({
                "qid": qa.qid, "category": qa.category, "category_name": qa.category_name,
                "question": qa.question, "gold": qa.answer, "answer": trace.answer,
                "score": score, "judge_reason": reason,
                "n_retrievals": trace.n_retrievals,
            }) + "\n")
            f.flush()
            if (i + 1) % 5 == 0:
                acc = sum(x.score for x in results) / len(results)
                print(f"  q {i+1}/{len(qas_to_run)} acc={acc:.3f}", flush=True)

    summary = summarise(results)
    summary["wall_time_s"] = round(time.time() - t0, 2)
    summary["backend"] = b.name
    summary["policy"] = policy.name
    (out_dir / f"summary{suffix}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
