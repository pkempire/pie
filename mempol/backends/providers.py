"""Shim: wrap any existing `memory_providers.MemoryProvider` as a `Backend`.

The repo already has Mem0 / Zep / Supermemory / Honcho / PIE adapters under
`memory_providers/`. They use a different interface (sessions list of dicts;
single `answer()` method). This shim adapts them to our `Backend` ABC so the
same `mempol.eval` runner + same `v0_naive` / `v1_heuristic` policies can
evaluate them apples-to-apples.

Notes on semantics:
  - Their `search(query, top_k) -> SearchResult` maps cleanly to `retrieve()`.
  - Their `ingest(sessions, dates)` takes a higher-level shape; we synthesise
    sessions from our Unit list by grouping by `metadata['session']` if present.
  - Their `answer(question)` is a higher-level alternative to v0/v1 policies.
    Some providers (Mem0/Zep) implement strong end-to-end answer pipelines; we
    expose those via an `EndToEndBackendAnswerer` shim too so we can compare
    "their full pipeline" vs "our policy on top of their retrieve()".
"""
from __future__ import annotations
from collections import defaultdict
from typing import Any

from memory_providers.interface import (
    MemoryProvider, MemoryProviderConfig, SearchResult,
)
from .base import Backend, Hit, Unit


def _units_to_sessions(units: list[Unit]) -> tuple[list[list[dict]], list[str]]:
    """Group units by metadata['session'] (or all-in-one if absent).
    Each session becomes [{"role": speaker, "content": text}, ...].
    Returns (sessions, dates_per_session)."""
    by_sess: dict = defaultdict(list)
    sess_date: dict = {}
    for u in units:
        sess = u.metadata.get("session", 0)
        by_sess[sess].append({
            "role": u.metadata.get("speaker", "user"),
            "content": u.text,
            "dia_id": u.metadata.get("dia_id", ""),
        })
        d = u.metadata.get("session_date")
        if d and sess not in sess_date:
            sess_date[sess] = d
    keys = sorted(by_sess.keys(), key=lambda x: (isinstance(x, str), x))
    return [by_sess[k] for k in keys], [sess_date.get(k, "") for k in keys]


class ProviderBackend(Backend):
    """Generic Backend that delegates ingest/search to a MemoryProvider."""

    def __init__(self, provider: MemoryProvider, name: str | None = None):
        self.provider = provider
        self.name = name or f"provider:{provider.name}"
        # Keep a copy of units so we can answer expand() and provide a default
        # text-bundle fallback if the provider's search returns empty.
        self._units: list[Unit] = []
        self._uid_index: dict[str, int] = {}

    def ingest(self, units: list[Unit]) -> None:
        for u in units:
            self._uid_index[u.uid] = len(self._units)
            self._units.append(u)
        sessions, dates = _units_to_sessions(units)
        self.provider.ingest(sessions, dates)

    def retrieve(self, query: str, k: int = 10, source: str = "hybrid") -> list[Hit]:
        results = self.provider.search(query, top_k=k)
        out: list[Hit] = []
        for i, r in enumerate(results):
            uid = r.metadata.get("uid") or r.metadata.get("dia_id") or f"{self.name}::r{i}"
            out.append(Hit(
                unit=Unit(uid=uid, text=r.content, metadata=dict(r.metadata)),
                score=float(r.score), source=self.name,
            ))
        return out

    def expand(self, seed_uids: list[str], k_per: int = 2) -> list[Hit]:
        """Approximate via adjacent-turn fallback over the cached units."""
        out, seen = [], set(seed_uids)
        for uid in seed_uids:
            i = self._uid_index.get(uid)
            if i is None:
                continue
            for j in (i - 1, i + 1):
                if 0 <= j < len(self._units):
                    nb = self._units[j]
                    if nb.uid in seen:
                        continue
                    seen.add(nb.uid)
                    out.append(Hit(
                        unit=Unit(uid=nb.uid, text=nb.text, metadata=nb.metadata),
                        score=0.4, source="adjacent",
                    ))
        return out[: k_per * len(seed_uids)]


def make_mem0_backend() -> ProviderBackend:
    from memory_providers.mem0_provider import Mem0Provider
    return ProviderBackend(Mem0Provider(), name="mem0")


def make_zep_backend() -> ProviderBackend:
    from memory_providers.zep_provider import ZepProvider
    return ProviderBackend(ZepProvider(), name="zep")


def make_supermemory_backend() -> ProviderBackend:
    from memory_providers.supermemory_provider import SupermemoryProvider
    return ProviderBackend(SupermemoryProvider(), name="supermemory")


def make_honcho_backend() -> ProviderBackend:
    from memory_providers.honcho_provider import HonchoProvider
    return ProviderBackend(HonchoProvider(), name="honcho")


def make_pie_provider_backend() -> ProviderBackend:
    """The existing benchmarks/locomo PIE pipeline wrapped as a Backend."""
    from memory_providers.pie_provider import PIEProvider
    return ProviderBackend(PIEProvider(), name="pie_provider")
