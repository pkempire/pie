"""Compute summary.json from a (possibly partial) traces.jsonl.

Usage:
    python -m mempol.scripts.summarize_traces  mempol/results/mastra_c1_20q
"""
from __future__ import annotations
import json
import sys
from collections import defaultdict
from pathlib import Path


def summarize(run_dir: Path) -> dict:
    traces = run_dir / "traces.jsonl"
    if not traces.exists():
        raise SystemExit(f"no traces at {traces}")
    rows = [json.loads(l) for l in traces.read_text().splitlines() if l.strip()]
    if not rows:
        raise SystemExit("traces is empty")
    by_cat: dict = defaultdict(list)
    for r in rows:
        by_cat[r.get("category_name", "?")].append(float(r.get("score", 0.0)))
    summary = {
        "n": len(rows),
        "overall_acc": sum(r.get("score", 0.0) for r in rows) / len(rows),
        "avg_n_retrievals": sum(r.get("n_retrievals", 0) for r in rows) / len(rows),
        "by_category": {
            k: {"n": len(v), "acc": sum(v) / len(v)} for k, v in by_cat.items()
        },
    }
    out = run_dir / "summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: summarize_traces.py <run_dir>")
    summarize(Path(sys.argv[1]))
