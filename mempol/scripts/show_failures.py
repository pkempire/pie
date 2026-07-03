"""Inspect failed questions from a baseline-run results JSON.

Reads a `*_results.jsonl` or `*_locomo.json` file produced by
`benchmarks/parallel_runner.py` and surfaces the failures in a way
you can actually reason about. Replaces the eyeball-the-pie_failures
markdown workflow with one CLI you can grep, sort, filter.

Usage:
    # Bottom-20 wrong questions across all categories
    python -m mempol.scripts.show_failures \\
        --results benchmarks/results/20260503_030448/locomo/pie_temporal_results.jsonl \\
        --n 20

    # Only adversarial failures, full text
    python -m mempol.scripts.show_failures \\
        --results <path> --category adversarial --full

    # Group by failure pattern (heuristic clustering)
    python -m mempol.scripts.show_failures \\
        --results <path> --cluster

    # Compare two runs side-by-side on the same questions
    python -m mempol.scripts.show_failures \\
        --results <path_a> --vs <path_b>

The clustering buckets failures into rough patterns that have shown up
in our PIE failure dumps: speaker-confused, no-information, off-by-detail,
list-padding, total-miss. These are heuristic — the cluster label is
based on string patterns in the hypothesis text — but they're useful
for spot-checking which failure mode is dominant.
"""
from __future__ import annotations
import argparse
import collections
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable


# ─── Loaders ────────────────────────────────────────────────────────────────

def _load_results(path: Path) -> list[dict[str, Any]]:
    """Load either a JSONL (one row per line) or a JSON dict with `results`
    key. Both formats are produced by different runs in our benchmarks/."""
    text = path.read_text()
    if not text.strip():
        return []
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    blob = json.loads(text)
    if isinstance(blob, list):
        return blob
    if isinstance(blob, dict) and "results" in blob:
        return blob["results"]
    raise ValueError(f"Don't know how to parse {path}")


def _normalize(row: dict[str, Any]) -> dict[str, Any]:
    """Coerce a row from any of our results formats into a common schema.

    Common fields:
      question, gold, hypothesis, score, category, latency_ms,
      reasoning (judge reasoning, optional)
    """
    return {
        "question": row.get("question") or row.get("q") or "",
        "gold": row.get("gold") or row.get("gold_answer") or row.get("answer") or "",
        "hypothesis": row.get("hypothesis") or row.get("predicted") or
                       row.get("prediction") or row.get("answer_text") or
                       # locomo_matrix rows use `answer` for prediction and
                       # `gold` for reference.
                       (row.get("answer") if row.get("gold") else "") or "",
        "score": float(row.get("score", row.get("reward", 0.0))),
        "category": row.get("category") or row.get("type") or "uncategorized",
        "latency_ms": row.get("latency_ms"),
        "reasoning": row.get("reasoning") or row.get("judge_reasoning") or "",
        # passthrough for advanced viewing
        "raw": row,
    }


# ─── Failure clustering ─────────────────────────────────────────────────────

_NO_INFO_PATTERNS = [
    re.compile(r"\bno information\b", re.I),
    re.compile(r"\bnot mentioned\b", re.I),
    re.compile(r"\bno specific\b", re.I),
    re.compile(r"\bnot in the (knowledge base|context|conversation)\b", re.I),
]
_OTHER_SPEAKER_PATTERNS = [
    re.compile(r"\b(caroline|melanie|john|james|joanna|nate|andrew|audrey)\b "
               r"(did|got|said|attended|made|created)", re.I),
]


def _cluster_failure(row: dict[str, Any]) -> str:
    """Heuristic bucket label for a failed question. Strictly approximate,
    but useful for "which kind of failure dominates" spot checks."""
    h = row["hypothesis"] or ""
    g = row["gold"] or ""
    score = row["score"]

    if not h.strip():
        return "empty"
    if any(p.search(h) for p in _NO_INFO_PATTERNS):
        return "no-information"
    if any(p.search(h) for p in _OTHER_SPEAKER_PATTERNS):
        # check if hypothesis names a different person than the question
        if "name" in (row.get("reasoning") or "").lower() or \
           "wrong person" in (row.get("reasoning") or "").lower() or \
           "instead of" in (row.get("reasoning") or "").lower():
            return "speaker-confused"
    # list questions where hypothesis is much longer than gold = padding
    if "," in h and len(h.split(",")) > 2 and len(g.split(",")) <= 2:
        return "list-padding"
    if 0 < score < 1:
        return "off-by-detail"
    if score == 0 and h.strip() and g.strip():
        # last-resort: if hypothesis has any token overlap with gold, "near"
        h_toks = set(h.lower().split())
        g_toks = set(g.lower().split())
        if h_toks & g_toks:
            return "near-miss"
        return "total-miss"
    return "other"


# ─── Display ────────────────────────────────────────────────────────────────

def _print_row(row: dict[str, Any], idx: int, full: bool = False) -> None:
    print(f"\n── #{idx} score={row['score']:.2f} cat={row['category']} ──")
    print(f"  Q: {row['question']}")
    print(f"  G: {row['gold']}")
    h = row["hypothesis"]
    if not full and len(h) > 200:
        h = h[:200] + "..."
    print(f"  P: {h}")
    if row["reasoning"]:
        r = row["reasoning"]
        if not full and len(r) > 200:
            r = r[:200] + "..."
        print(f"  J: {r}")


def _summary(rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    by_cat = collections.defaultdict(lambda: {"full": 0, "partial": 0, "wrong": 0})
    for r in rows:
        s = r["score"]
        bucket = "full" if s >= 1.0 else "partial" if s > 0 else "wrong"
        by_cat[r["category"]][bucket] += 1
    print("=== Per-category result counts ===")
    print(f"  {'category':15s} {'full':>5s} {'partial':>8s} {'wrong':>5s} "
          f"{'acc':>6s}")
    for cat in sorted(by_cat):
        c = by_cat[cat]
        n = c["full"] + c["partial"] + c["wrong"]
        acc = (c["full"] + 0.5 * c["partial"]) / max(n, 1) * 100
        print(f"  {cat:15s} {c['full']:>5d} {c['partial']:>8d} {c['wrong']:>5d} "
              f"{acc:>5.1f}%")


# ─── Compare two runs ───────────────────────────────────────────────────────

def _diff_runs(rows_a: list[dict], rows_b: list[dict],
               n: int = 30) -> None:
    """Find questions where A and B disagree — A wrong, B right OR vice versa.
    Print a side-by-side."""
    a_by_q = {r["question"]: r for r in rows_a}
    b_by_q = {r["question"]: r for r in rows_b}
    common = set(a_by_q) & set(b_by_q)

    flips_a_to_b = []   # A wrong, B right
    flips_b_to_a = []   # B wrong, A right
    for q in common:
        ra, rb = a_by_q[q], b_by_q[q]
        if ra["score"] < 1.0 and rb["score"] >= 1.0:
            flips_a_to_b.append((ra, rb))
        elif rb["score"] < 1.0 and ra["score"] >= 1.0:
            flips_b_to_a.append((ra, rb))

    print(f"=== A→B GAINS (A wrong, B right): {len(flips_a_to_b)} ===")
    for ra, rb in flips_a_to_b[:n]:
        print(f"\nQ: {ra['question']}")
        print(f"  GOLD:  {ra['gold']}")
        print(f"  A({ra['score']:.1f}): {(ra['hypothesis'] or '')[:160]}")
        print(f"  B({rb['score']:.1f}): {(rb['hypothesis'] or '')[:160]}")

    print(f"\n=== A→B REGRESSIONS (A right, B wrong): {len(flips_b_to_a)} ===")
    for ra, rb in flips_b_to_a[:n]:
        print(f"\nQ: {ra['question']}")
        print(f"  GOLD:  {ra['gold']}")
        print(f"  A({ra['score']:.1f}): {(ra['hypothesis'] or '')[:160]}")
        print(f"  B({rb['score']:.1f}): {(rb['hypothesis'] or '')[:160]}")

    print(f"\n=== Net A→B: +{len(flips_a_to_b)} -{len(flips_b_to_a)} = "
          f"{len(flips_a_to_b) - len(flips_b_to_a):+d} questions ===")


# ─── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", required=True, type=Path,
                    help="path to a results json/jsonl file")
    p.add_argument("--vs", type=Path, default=None,
                    help="optional second results file for side-by-side diff")
    p.add_argument("--n", type=int, default=20, help="how many failures to show")
    p.add_argument("--category", default=None,
                    help="filter to a single category "
                         "(adversarial/multi_hop/single_hop/temporal/open_domain)")
    p.add_argument("--full", action="store_true",
                    help="show full hypothesis + reasoning text (no truncation)")
    p.add_argument("--cluster", action="store_true",
                    help="group failures by heuristic failure-mode bucket")
    p.add_argument("--summary", action="store_true",
                    help="print per-category result counts and exit")
    p.add_argument("--worst-clusters", action="store_true",
                    help="show top failure-mode clusters with example "
                         "questions per cluster")
    args = p.parse_args()

    rows = [_normalize(r) for r in _load_results(args.results)]

    if args.category:
        rows = [r for r in rows if r["category"] == args.category]

    if args.vs:
        rows_b = [_normalize(r) for r in _load_results(args.vs)]
        if args.category:
            rows_b = [r for r in rows_b if r["category"] == args.category]
        _diff_runs(rows, rows_b, n=args.n)
        return

    if args.summary:
        _summary(rows)
        return

    if args.cluster or args.worst_clusters:
        wrong = [r for r in rows if r["score"] < 1.0]
        clusters = collections.defaultdict(list)
        for r in wrong:
            clusters[_cluster_failure(r)].append(r)
        print("=== Failure clusters (count, fraction-of-wrong) ===")
        total_wrong = max(len(wrong), 1)
        for k, v in sorted(clusters.items(), key=lambda kv: -len(kv[1])):
            print(f"  {k:20s} {len(v):4d} ({100*len(v)/total_wrong:.1f}%)")
        if args.worst_clusters:
            print("\n=== Examples per cluster (3 each) ===")
            for k, v in sorted(clusters.items(), key=lambda kv: -len(kv[1])):
                print(f"\n--- cluster: {k} ---")
                for i, row in enumerate(v[:3]):
                    _print_row(row, i, full=args.full)
        return

    # Default: bottom-N failures sorted by score ascending
    wrong = [r for r in rows if r["score"] < 1.0]
    wrong.sort(key=lambda r: r["score"])
    _summary(rows)
    print(f"\n=== Bottom-{args.n} failures (score asc) ===")
    for i, row in enumerate(wrong[:args.n]):
        _print_row(row, i, full=args.full)


if __name__ == "__main__":
    main()
