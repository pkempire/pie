"""Run a memory_providers/* provider on a single LoCoMo conv via our shim.

Usage:
    python -m mempol.scripts.run_provider_conv \\
        --provider mem0 --conv-idx 0 --policy v1_heuristic --max-qs 0
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

from mempol import config
from mempol.backends import providers as _p
from mempol.data.locomo import load
from mempol.eval.judge import judge
from mempol.eval.metrics import Result, summarise
from mempol.eval.runner import conv_to_units
from mempol.policies.v0_naive import NaivePolicy
from mempol.policies.v1_heuristic import HeuristicPolicy


_BACKENDS = {
    "mem0":        _p.make_mem0_backend,
    "zep":         _p.make_zep_backend,
    "supermemory": _p.make_supermemory_backend,
    "honcho":      _p.make_honcho_backend,
    "pie_provider": _p.make_pie_provider_backend,
}
_POLICIES = {"v0_naive": NaivePolicy, "v1_heuristic": HeuristicPolicy}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", required=True, choices=list(_BACKENDS))
    ap.add_argument("--conv-idx", type=int, default=0)
    ap.add_argument("--policy", choices=list(_POLICIES), default="v1_heuristic")
    ap.add_argument("--max-qs", type=int, default=0)
    ap.add_argument("--start-q", type=int, default=0)
    ap.add_argument("--run-name", default=None)
    args = ap.parse_args()
    run_name = args.run_name or f"{args.provider}_c{args.conv_idx}"

    convs = load(n_convs=args.conv_idx + 1)
    conv, qas = convs[args.conv_idx]

    out_dir = config.RESULTS_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{conv.sample_id}] ingesting {len(conv.turns)} turns into {args.provider}…")
    t0 = time.time()
    backend = _BACKENDS[args.provider]()
    backend.ingest(conv_to_units(conv))
    print(f"  ingest done in {time.time()-t0:.1f}s")

    qas_to_run = qas if args.max_qs == 0 else qas[args.start_q:args.start_q + args.max_qs]
    policy = _POLICIES[args.policy](do_reformulate=False, do_route=False, do_expand=True)
    print(f"\n  evaluating {policy.name} on {len(qas_to_run)} qs…")

    traces_path = out_dir / f"traces_q{args.start_q}-{args.start_q + len(qas_to_run)}.jsonl"
    results: list[Result] = []
    teval = time.time()
    with traces_path.open("a", buffering=1) as f:
        for i, qa in enumerate(qas_to_run):
            trace = policy.run(qa.question, backend)
            score, reason = judge(qa.question, qa.answer, trace.answer)
            r = Result(
                qid=qa.qid, category=qa.category, category_name=qa.category_name,
                score=score, n_retrievals=trace.n_retrievals, n_steps=len(trace.steps),
                answer=trace.answer, gold=qa.answer, judge_reason=reason, evidence_recall=None,
            )
            results.append(r)
            f.write(json.dumps({
                "qid": qa.qid, "category_name": qa.category_name,
                "question": qa.question, "gold": qa.answer, "answer": trace.answer,
                "score": score, "judge_reason": reason,
                "n_retrievals": trace.n_retrievals,
            }) + "\n")
            f.flush()
            if (i + 1) % 5 == 0:
                acc = sum(x.score for x in results) / len(results)
                print(f"    q {i+1}/{len(qas_to_run)} acc={acc:.3f}", flush=True)

    summary = summarise(results)
    summary["wall_time_s"] = round(time.time() - teval, 2)
    summary["backend"] = backend.name
    summary["policy"] = policy.name
    summary["conv_id"] = conv.sample_id
    suffix = f"_q{args.start_q}-{args.start_q + len(qas_to_run)}"
    (out_dir / f"summary{suffix}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
