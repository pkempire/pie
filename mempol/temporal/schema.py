"""Temporal state primitives for agent memory.

These are intentionally domain-light. A "state" can describe a user preference,
project belief, account stage, web-page cache, experiment result, or simulated
world fact. Domain-specific objects are views over this layer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TemporalState:
    """A valid-time state backed by evidence.

    `scope_id` partitions memory by user, project, account, simulation world, or
    benchmark episode. `key` is freeform but should be stable within a scope
    (for example: "diet.preference", "account.stage", "benchmark.best_baseline").
    """

    id: str
    scope_id: str
    key: str
    content: str
    state_type: str = "state"
    valid_from: str = ""
    valid_until: str = ""
    observed_at: str = ""
    status: str = "active"  # active | stale | superseded | archived
    confidence: float = 1.0
    volatility_seconds: float | None = None
    source_span_ids: list[str] = field(default_factory=list)
    supersedes_state_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StateTransition:
    """A change in temporal state."""

    id: str
    scope_id: str
    transition_type: str  # create | update | supersede | archive | refresh
    old_state_ids: list[str] = field(default_factory=list)
    new_state_id: str = ""
    reason: str = ""
    observed_at: str = ""
    source_span_ids: list[str] = field(default_factory=list)
    trace_event_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ActiveProcess:
    """A process whose status changes with elapsed time."""

    id: str
    scope_id: str
    kind: str
    description: str
    status: str = "active"  # active | waiting | blocked | done | stale | cancelled
    started_at: str = ""
    expected_at: str = ""
    deadline_at: str = ""
    last_checked_at: str = ""
    source_span_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ContextDecision:
    """A logged read-time context/action decision."""

    id: str
    scope_id: str
    task: str
    now: str
    action: str = "answer"  # answer | refresh | wait | interrupt | replan
    candidate_state_ids: list[str] = field(default_factory=list)
    selected_state_ids: list[str] = field(default_factory=list)
    selected_span_ids: list[str] = field(default_factory=list)
    selected_process_ids: list[str] = field(default_factory=list)
    token_budget: int = 0
    token_estimate: int = 0
    rationale: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class OutcomeEvent:
    """Downstream feedback for a context/action decision."""

    id: str
    decision_id: str
    scope_id: str
    score: float = 0.0
    outcome_type: str = "unknown"  # judge | human | benchmark | tool | product
    feedback: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
