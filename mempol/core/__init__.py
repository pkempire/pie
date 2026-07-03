"""Universal memory substrate.

This package is intentionally domain-light. Apps like PIE, architect, sales,
research, and benchmarks adapt their raw data into the same four primitives:
Artifact, Span, MemoryState, and TraceEvent.
"""

from .schema import Artifact, MemoryState, Span, TraceEvent
from .store import SQLiteMemoryStore, store_for_run

__all__ = [
    "Artifact",
    "Span",
    "MemoryState",
    "TraceEvent",
    "SQLiteMemoryStore",
    "store_for_run",
]
