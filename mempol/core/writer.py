"""Generic writer interface for universal memory.

The first implementation is intentionally conservative and deterministic:
it creates one memory state per high-signal span. LLM/learned writers can
replace this module without changing storage, retrieval, or dashboards.
"""
from __future__ import annotations

from .schema import MemoryState, Span
from .store import now_iso, stable_id


def simple_state_from_span(span: Span, adapter: str = "generic") -> MemoryState:
    return MemoryState(
        id=stable_id("state", adapter, span.id),
        content=span.text,
        source_span_ids=[span.id],
        created_at=now_iso(),
        updated_at=now_iso(),
        metadata={"adapter": adapter, "writer": "simple_state_from_span"},
    )
