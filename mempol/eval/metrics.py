"""Aggregate scores by category + cost summary."""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class Result:
    qid: str
    category: int
    category_name: str
    score: float
    n_retrievals: int
    n_steps: int
    answer: str
    gold: str
    judge_reason: str
    evidence_recall: float | None = None  # fraction of gold dia_ids appearing in retrieved hits


def summarise(results: list[Result]) -> dict:
    by_cat: dict[str, list[float]] = defaultdict(list)
    for r in results:
        by_cat[r.category_name].append(r.score)
    out: dict = {
        "n": len(results),
        "overall_acc": sum(r.score for r in results) / max(1, len(results)),
        "avg_steps": sum(r.n_steps for r in results) / max(1, len(results)),
        "avg_retrievals": sum(r.n_retrievals for r in results) / max(1, len(results)),
        "by_category": {
            k: {"n": len(v), "acc": sum(v) / len(v)} for k, v in by_cat.items()
        },
    }
    ev = [r.evidence_recall for r in results if r.evidence_recall is not None]
    if ev:
        out["avg_evidence_recall"] = sum(ev) / len(ev)
    return out
