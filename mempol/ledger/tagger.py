"""Deterministic project/thread assignment for local artifacts.

This is intentionally boring.  The learned router should replace it later, but
the first ingestion pass needs stable partitions so humans can inspect what the
system thinks happened.
"""
from __future__ import annotations

from dataclasses import dataclass

from .schema import Project, Thread


DEFAULT_PROJECT = Project(
    id="memory_context_systems",
    title="Memory / Context Systems",
    objective="Build a useful, publishable, long-running memory and context system for AI agents.",
)


@dataclass(frozen=True)
class ThreadRule:
    thread_id: str
    title: str
    summary: str
    prefixes: tuple[str, ...]
    contains: tuple[str, ...] = ()


THREAD_RULES: tuple[ThreadRule, ...] = (
    ThreadRule(
        "mempol_memory_policy",
        "mempol memory policy",
        "RL/GEPA/read-write memory-policy experiments, backends, and evals.",
        ("mempol/", "paper/", "tests/test_universal_memory"),
        ("gepa", "longmemeval", "locomo", "counterfactual", "universal_memory"),
    ),
    ThreadRule(
        "pie_temporal_world_model",
        "PIE temporal world model",
        "Legacy personal world-model, temporal retrieval, and MCP integration.",
        ("pie/", "mcp_server.py", "output/world_model"),
        ("temporal_memory", "world_model", "pie_"),
    ),
    ThreadRule(
        "research_wiki_content",
        "research wiki and content",
        "Literature review, public content, scripts, paper notes, and research synthesis.",
        ("research/", "docs/", "BLOG", "VIDEO", "PAPER", "MEMORY-", "TEMPORAL-"),
        ("lit-review", "working-memory", "temporal-awareness"),
    ),
    ThreadRule(
        "benchmark_harnesses",
        "benchmark harnesses",
        "LoCoMo, LongMemEval, MSC, Test-of-Time, providers, cached result matrices.",
        ("benchmarks/", "benchmark_results/", "memory_providers/", "logs/"),
        ("summary.json", "rows.jsonl", "side_by_side"),
    ),
    ThreadRule(
        "architect_planner",
        "architect planner",
        "AI-for-system-design component index, architecture miner, and planner.",
        ("architect/",),
        ("component", "planner", "architecture"),
    ),
    ThreadRule(
        "footnote_video_product",
        "Footnote video product",
        "AI video annotation pipeline, Remotion render assets, and video content artifacts.",
        ("scripts/footnote/", "remotion/footnote/"),
        ("footnote", "remotion"),
    ),
    ThreadRule(
        "visual_memory_multimodal",
        "visual-memory multimodal",
        "Video/frame memory experiments and multimodal temporal logs.",
        ("visual-memory/",),
        ("frame_", "big-brother", "video"),
    ),
    ThreadRule(
        "sales_lucid_side_projects",
        "sales and Lucid side projects",
        "Sales/product side projects that reuse the context infrastructure.",
        ("sales/",),
        ("lucid", "sponsorfind", "revenue"),
    ),
)


def seed_threads(project_id: str = DEFAULT_PROJECT.id) -> list[Thread]:
    return [
        Thread(
            id=rule.thread_id,
            project_id=project_id,
            title=rule.title,
            summary=rule.summary,
            metadata={"rule_prefixes": list(rule.prefixes), "rule_contains": list(rule.contains)},
        )
        for rule in THREAD_RULES
    ]


def assign_thread(rel_path: str) -> tuple[str, str, float]:
    path = rel_path.replace("\\", "/")
    path_l = path.lower()
    for rule in THREAD_RULES:
        if any(path.startswith(p) for p in rule.prefixes):
            return rule.thread_id, f"path prefix matched {rule.thread_id}", 0.95
    for rule in THREAD_RULES:
        if any(s.lower() in path_l for s in rule.contains):
            return rule.thread_id, f"path content matched {rule.thread_id}", 0.75
    return "research_wiki_content", "default catch-all for repo knowledge artifacts", 0.35
