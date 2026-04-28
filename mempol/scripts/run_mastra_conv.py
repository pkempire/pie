"""Mastra baseline on a single LoCoMo conversation.

Usage:
    python -m mempol.scripts.run_mastra_conv \\
        --conv-idx 0  --policy v1_heuristic  --max-qs 0  --run-name mastra_conv1

Produces, under mempol/results/<run_name>/:
    memory_log.md     — readable dump of what Mastra stored
    traces.jsonl      — per-question (question, gold, answer, score, judge_reason)
    summary.json      — overall + per-category accuracy, latency, cost stats
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

from mempol import config
from mempol.backends.mastra import MastraBackend
from mempol.data.locomo import load
from mempol.eval.runner import conv_to_units
from mempol.eval.judge import judge
from mempol.eval.metrics import Result, summarise
from mempol.policies.v0_naive import NaivePolicy
from mempol.policies.v1_heuristic import HeuristicPolicy


_POLICIES = {"v0_naive": NaivePolicy, "v1_heuristic": HeuristicPolicy}


def dump_mastra_markdown(backend: MastraBackend, conv_id: str, out_path: Path) -> Path:
    """Write the Mastra Observational Memory log: reflections + observation
    blocks + the trailing recent-raw window."""
    md = backend.memory_log_md()
    md = f"# Mastra OM log — {conv_id}\n\n" + md
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md)
    return out_path


def run(
    conv_idx: int = 0,
    policy_name: str = "v1_heuristic",
    max_qs: int | None = None,
    run_name: str = "mastra_conv1",
) -> dict:
    convs = load(n_convs=conv_idx + 1)
    if conv_idx >= len(convs):
        raise SystemExit(f"only {len(convs)} convs in dataset, need idx {conv_idx}")
    conv, qas = convs[conv_idx]

    # 1. Ingest
    backend = MastraBackend()
    print(f"[{conv.sample_id}] ingesting {len(conv.turns)} turns into Mastra...")
    backend.ingest(conv_to_units(conv))
    print(f"  done. stats={backend.stats()}")

    # 2. Dump memory log
    out_dir = config.RESULTS_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = dump_mastra_markdown(backend, conv.sample_id, out_dir / "memory_log.md")
    print(f"  memory log → {md_path}")

    # 3. Eval
    qas_to_run = qas if max_qs is None else qas[:max_qs]
    policy = _POLICIES[policy_name]()
    traces_path = out_dir / "traces.jsonl"
    print(f"\n  evaluating {policy.name} on {len(qas_to_run)}/{len(qas)} qs…")

    results: list[Result] = []
    t0 = time.time()
    # Line-buffered (1) so partial runs always have valid JSONL on disk.
    with traces_path.open("w", buffering=1) as f:
        for i, qa in enumerate(qas_to_run):
            trace = policy.run(qa.question, backend)
            score, reason = judge(qa.question, qa.answer, trace.answer)
            r = Result(
                qid=qa.qid, category=qa.category, category_name=qa.category_name,
                score=score, n_retrievals=trace.n_retrievals,
                n_steps=len(trace.steps), answer=trace.answer, gold=qa.answer,
                judge_reason=reason, evidence_recall=None,
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
                print(f"    q {i+1}/{len(qas_to_run)}  running_acc={acc:.3f}", flush=True)

    summary = summarise(results)
    summary["wall_time_s"] = round(time.time() - t0, 2)
    summary["backend"] = backend.name
    summary["policy"] = policy.name
    summary["conv_id"] = conv.sample_id
    summary["n_qas"] = len(qas_to_run)
    summary["n_qas_total"] = len(qas)
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nFiles in {out_dir}/")
    for p in (md_path, traces_path, summary_path):
        print(f"  {p.name}")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--conv-idx", type=int, default=0)
    ap.add_argument("--policy", choices=list(_POLICIES), default="v1_heuristic")
    ap.add_argument("--max-qs", type=int, default=0, help="0 = all")
    ap.add_argument("--run-name", default="mastra_conv1")
    args = ap.parse_args()
    run(
        conv_idx=args.conv_idx,
        policy_name=args.policy,
        max_qs=None if args.max_qs == 0 else args.max_qs,
        run_name=args.run_name,
    )
