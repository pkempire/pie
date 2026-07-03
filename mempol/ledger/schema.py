"""Research-ledger primitives.

The ledger is the project/process layer above the universal memory core.  The
core stores raw artifacts and spans; this layer says which project/thread they
belong to and what research work they support.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Project:
    id: str
    title: str
    objective: str = ""
    status: str = "active"
    created_at: str = ""
    updated_at: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Thread:
    id: str
    project_id: str
    title: str
    summary: str = ""
    status: str = "active"
    created_at: str = ""
    updated_at: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Membership:
    id: str
    target_type: str
    target_id: str
    project_id: str
    thread_id: str = ""
    confidence: float = 1.0
    assigned_by: str = "rule"
    rationale: str = ""
    valid_from: str = ""
    valid_until: str = ""
    created_at: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ResearchObject:
    id: str
    project_id: str
    thread_id: str
    role: str
    content: str
    source_span_ids: list[str] = field(default_factory=list)
    parent_ids: list[str] = field(default_factory=list)
    status: str = "open"
    confidence: float = 1.0
    novelty_score: float | None = None
    utility_score: float | None = None
    created_at: str = ""
    updated_at: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunRecord:
    id: str
    project_id: str
    thread_id: str
    title: str
    started_at: str = ""
    ended_at: str = ""
    actor: str = ""
    command: str = ""
    status: str = "unknown"
    metrics: dict[str, Any] = field(default_factory=dict)
    artifact_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ContextPack:
    id: str
    project_id: str
    thread_id: str
    task: str
    markdown: str
    source_span_ids: list[str] = field(default_factory=list)
    research_object_ids: list[str] = field(default_factory=list)
    token_budget: int = 0
    token_estimate: int = 0
    created_at: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
