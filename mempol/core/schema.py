"""Domain-light primitives for the universal memory core."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Artifact:
    """Immutable raw source object: chat turn, PDF page, repo file, CRM event."""

    id: str
    source: str
    kind: str
    title: str
    content: str
    uri: str = ""
    created_at: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Span:
    """Addressable evidence slice inside an artifact."""

    id: str
    artifact_id: str
    text: str
    locator: str = ""
    start: int | None = None
    end: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryState:
    """Freeform compressed state with provenance.

    No domain ontology is required. Domain labels can live in metadata/views,
    but the substrate only requires content and source spans.
    """

    id: str
    content: str
    source_span_ids: list[str] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""
    archived: bool = False
    embedding_ref: str = ""
    utility: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TraceEvent:
    """A logged memory/tool decision for later training and debugging."""

    id: str
    run_name: str
    op: str
    input: dict[str, Any] = field(default_factory=dict)
    output: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
