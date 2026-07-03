"""Strategy registry.

REGISTRY maps snake_case names to live strategy instances.
Import this module to get the full strategy catalogue.

Usage:
    from mempol.strategies.registry import REGISTRY, describe_all
    strategy = REGISTRY["chronos"]
    describe_all()
"""
from __future__ import annotations

from .base import MemoryStrategy
from .implementations import (
    ContinuityTeacher,
    ChronosStrategy,
    HindsightStrategy,
    HybridSearchBaseline,
    MnemisStrategy,
    TimelineSynthesis,
    TurnRAG,
    WorldDBStrategy,
)

REGISTRY: dict[str, MemoryStrategy] = {
    "turn_rag": TurnRAG(),
    "timeline_synthesis": TimelineSynthesis(),
    "continuity_teacher": ContinuityTeacher(),
    "hybrid_search": HybridSearchBaseline(),
    "chronos": ChronosStrategy(),
    "hindsight": HindsightStrategy(),   # stub — runnable=False
    "mnemis": MnemisStrategy(),         # stub — runnable=False
    "worlddb": WorldDBStrategy(),       # stub — runnable=False
}

_TAG_WIDTH = 56
_COL_WIDTHS = (20, 22, 12, 12, _TAG_WIDTH, 12)
_HEADERS = ("Name", "Label", "arXiv", "Runnable", "Tags", "LME-S*")

_PERF = {
    "worlddb": "96.4%",
    "chronos": "95.6%",
    "hindsight": "91.4%",
    "mnemis": "91.6%",
    "continuity_teacher": "—",
    "timeline_synthesis": "—",
    "hybrid_search": "—",
    "turn_rag": "—",
}


def describe_all(*, only_runnable: bool = False) -> None:
    """Print a formatted table of all registered strategies."""
    strategies = [
        (name, s)
        for name, s in REGISTRY.items()
        if not only_runnable or s.runnable
    ]

    sep = "  ".join("-" * w for w in _COL_WIDTHS)
    header = "  ".join(h.ljust(w) for h, w in zip(_HEADERS, _COL_WIDTHS))

    print("\nMemory Strategy Registry")
    print("=" * len(sep))
    print(header)
    print(sep)
    for name, s in strategies:
        tags_str = ", ".join(s.tags)
        if len(tags_str) > _TAG_WIDTH:
            tags_str = tags_str[: _TAG_WIDTH - 3] + "..."
        runnable_str = "yes" if s.runnable else "stub"
        lme_s = _PERF.get(name, "—")
        row = "  ".join(
            v.ljust(w)
            for v, w in zip(
                [name, s.label, s.paper.arxiv_id, runnable_str, tags_str, lme_s],
                _COL_WIDTHS,
            )
        )
        print(row)
    print(sep)
    print(
        f"\n* LME-S = LongMemEval-S accuracy from paper (where known). "
        f"'—' = our implementation, paper number not directly comparable.\n"
    )
    print("Paper refs:")
    seen_arxiv: set[str] = set()
    for _, s in strategies:
        if s.paper.arxiv_id not in seen_arxiv:
            print(f"  arXiv:{s.paper.arxiv_id}  {s.paper.title}")
            seen_arxiv.add(s.paper.arxiv_id)
    print()
