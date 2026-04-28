"""Backend interface — both Tree (FS) and Graph (KG) backends conform to this.

A *unit* is the atom of memory. For LoCoMo, a unit = one turn (or a small group).
The backend is responsible for ingestion, lexical/dense retrieval, neighbour
expansion (graph follow / folder walk), and time filtering.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class Unit:
    uid: str                 # stable id
    text: str
    metadata: dict[str, Any]   # session, speaker, dia_id, timestamp, etc.


@dataclass
class Hit:
    unit: Unit
    score: float
    source: str              # "dense" | "bm25" | "expand" | ...


class Backend(ABC):
    name: str = "base"

    @abstractmethod
    def ingest(self, units: list[Unit]) -> None: ...

    @abstractmethod
    def retrieve(self, query: str, k: int = 10, source: str = "hybrid") -> list[Hit]:
        """source ∈ {dense, bm25, hybrid}."""

    def expand(self, seed_uids: list[str], k_per: int = 3) -> list[Hit]:
        """Default: no expansion. Subclasses override."""
        return []

    def filter_by_time(self, hits: list[Hit], window: tuple[float | None, float | None]) -> list[Hit]:
        lo, hi = window
        out = []
        for h in hits:
            ts = h.unit.metadata.get("timestamp")
            if ts is None:
                continue
            if lo is not None and ts < lo:
                continue
            if hi is not None and ts > hi:
                continue
            out.append(h)
        return out
