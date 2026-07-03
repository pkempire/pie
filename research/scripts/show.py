"""Query the research/ corpus from the CLI.

Usage:
  # Every paper that uses LongMemEval
  python -m research.scripts.show --benchmark LongMemEval

  # Every paper in the RL-for-memory class, sorted by year
  python -m research.scripts.show --approach RL-for-memory

  # Every high-relevance paper
  python -m research.scripts.show --relevance high

  # Free-text grep against title + problem + approach + tags
  python -m research.scripts.show --grep "counterfactual"

  # Full per-paper dump for one ID
  python -m research.scripts.show --id 2508.19828

  # Comparison table: approach class × benchmark coverage
  python -m research.scripts.show --matrix
"""
from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from research.scripts.aggregate import Paper, load_papers


def _print_paper(p: Paper, full: bool = False) -> None:
    print(f"\n── {p.title}  ({p.year or '?'})")
    if p.arxiv_id:
        print(f"   arxiv:{p.arxiv_id}")
    print(f"   class: {p.approach_class}    relevance: {p.relevance}")
    if p.fm.get("problem"):
        print(f"   problem: {p.fm['problem']}")
    if p.fm.get("approach"):
        print(f"   approach: {p.fm['approach']}")
    if p.benchmarks:
        print(f"   benchmarks: {', '.join(p.benchmarks)}")
    res = p.fm.get("results") or []
    if res:
        print("   results:")
        for r in res:
            print(f"     - {r}")
    if full:
        lims = p.fm.get("limitations") or []
        if lims:
            print("   limitations:")
            for l in lims:
                print(f"     - {l}")
        if p.fm.get("relevance_reason"):
            print(f"   relevance_reason: {p.fm['relevance_reason']}")


def cmd_filter(papers: list[Paper], args) -> None:
    out = papers
    if args.benchmark:
        out = [p for p in out if any(
            args.benchmark.lower() in b.lower() for b in p.benchmarks)]
    if args.approach:
        out = [p for p in out if args.approach.lower() in p.approach_class.lower()]
    if args.relevance:
        out = [p for p in out if p.relevance == args.relevance]
    if args.grep:
        needle = args.grep.lower()
        def hit(p: Paper) -> bool:
            haystack = " ".join([
                p.title,
                str(p.fm.get("problem") or ""),
                str(p.fm.get("approach") or ""),
                " ".join(p.fm.get("tags") or []),
            ]).lower()
            return needle in haystack
        out = [p for p in out if hit(p)]
    if args.id:
        out = [p for p in out if p.arxiv_id == args.id]
    out.sort(key=lambda p: -(p.year or 0))
    for p in out:
        _print_paper(p, full=bool(args.id))
    print(f"\n{len(out)} paper(s) matched.")


def cmd_matrix(papers: list[Paper]) -> None:
    """Approach-class × benchmark matrix (cells = paper count)."""
    classes = sorted({p.approach_class for p in papers})
    benchmarks = sorted({b for p in papers for b in p.benchmarks})
    if not benchmarks:
        print("(no benchmarks listed in any paper)")
        return
    grid: dict[tuple[str, str], int] = collections.defaultdict(int)
    for p in papers:
        for b in p.benchmarks:
            grid[(p.approach_class, b)] += 1

    # Width
    col_w = max(8, max(len(b) for b in benchmarks))
    row_w = max(12, max(len(c) for c in classes))

    # Header
    print(" " * row_w + " | " + " | ".join(b.ljust(col_w) for b in benchmarks))
    print("-" * (row_w + 3 + (col_w + 3) * len(benchmarks)))
    for c in classes:
        row = c.ljust(row_w) + " | "
        row += " | ".join(str(grid.get((c, b), "")).ljust(col_w) for b in benchmarks)
        print(row)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark")
    p.add_argument("--approach")
    p.add_argument("--relevance", choices=["high", "medium", "low", "?"])
    p.add_argument("--grep")
    p.add_argument("--id", help="arxiv id exact match (also prints full detail)")
    p.add_argument("--matrix", action="store_true",
                    help="print approach × benchmark coverage matrix")
    args = p.parse_args()

    papers = load_papers()
    if not papers:
        print("No papers ingested yet. Try:")
        print("  python -m research.scripts.ingest 2508.19828")
        return

    if args.matrix:
        cmd_matrix(papers)
        return

    cmd_filter(papers, args)


if __name__ == "__main__":
    main()
