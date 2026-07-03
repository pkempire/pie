"""Tinker tools for RL over the universal memory substrate.

This is intentionally not a domain-specific KG toolset. The policy sees raw
spans, writes freeform memory states with provenance, retrieves from those
states, and answers under a budget.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from mempol.core.schema import MemoryState
from mempol.core.store import SQLiteMemoryStore, estimate_tokens, now_iso, stable_id
from mempol.recipes.memory_rl.tinker_compat import ToolResult, simple_tool_result, tool


def _obs(payload: dict) -> ToolResult:
    return simple_tool_result(json.dumps(payload, ensure_ascii=False))


@dataclass
class UniversalMemoryTool:
    """Per-env tools over one mutable universal memory store."""

    store: SQLiteMemoryStore
    raw_searches: int = 0
    memory_searches: int = 0
    writes: int = 0
    token_cost: int = 0
    written_state_ids: list[str] = field(default_factory=list)
    max_raw_searches: int = 8
    max_memory_searches: int = 8
    max_writes: int = 24
    raw_enabled: bool = True

    def search_raw_spans_impl(self, query: str, k: int = 8) -> ToolResult:
        if not self.raw_enabled:
            return _obs({"error": "raw_span_search_disabled_after_memory_build", "hits": []})
        if self.raw_searches >= self.max_raw_searches:
            return _obs({"error": "max_raw_searches_reached", "hits": []})
        k = max(1, min(int(k), 20))
        hits = [h for h in self.store.retrieve(query, k=k * 2, include_spans=True) if h["kind"] == "span"][:k]
        self.raw_searches += 1
        self.token_cost += sum(int(h.get("token_estimate") or 0) for h in hits)
        return _obs({
            "hits": [
                {
                    "span_id": h["id"],
                    "artifact_id": h.get("artifact_id"),
                    "source": h.get("source"),
                    "score": round(float(h.get("score") or 0), 3),
                    "text": h.get("text", "")[:900],
                    "locator": h.get("locator", ""),
                }
                for h in hits
            ],
            "n_hits": len(hits),
        })

    @tool
    def search_raw_spans(self, query: str, k: int = 8) -> ToolResult:
        """Search immutable raw/evidence spans. Use during memory-building.

        Args:
            query: natural-language search query
            k: max raw spans to return
        """
        return self.search_raw_spans_impl(query=query, k=k)

    def write_memory_state_impl(self, content: str, source_span_ids: list[str]) -> ToolResult:
        if self.writes >= self.max_writes:
            return _obs({"error": "max_writes_reached"})
        source_span_ids = [str(s) for s in source_span_ids][:8]
        missing = [sid for sid in source_span_ids if self.store.get_span(sid) is None]
        if missing:
            return _obs({"error": "unknown_source_span_ids", "missing": missing})
        content = str(content).strip()
        if not content:
            return _obs({"error": "empty_content"})
        sid = stable_id("rl_state", content, source_span_ids)
        state = MemoryState(
            id=sid,
            content=content,
            source_span_ids=source_span_ids,
            created_at=now_iso(),
            updated_at=now_iso(),
            metadata={"adapter": "rl_policy", "writer": "universal_rl"},
        )
        self.store.upsert_memory_state(state)
        self.store.commit()
        self.writes += 1
        self.written_state_ids.append(sid)
        self.token_cost += estimate_tokens(content)
        return _obs({"written_state_id": sid, "tokens_est": estimate_tokens(content)})

    @tool
    def write_memory_state(self, content: str, source_span_ids: list[str]) -> ToolResult:
        """Write a freeform compressed memory state backed by raw spans.

        Args:
            content: compact memory text useful for future answers/tasks
            source_span_ids: raw span ids supporting this memory
        """
        return self.write_memory_state_impl(content=content, source_span_ids=source_span_ids)

    def retrieve_memory_states_impl(self, query: str, k: int = 8) -> ToolResult:
        if self.memory_searches >= self.max_memory_searches:
            return _obs({"error": "max_memory_searches_reached", "hits": []})
        k = max(1, min(int(k), 20))
        hits = [h for h in self.store.retrieve(query, k=k * 3, include_spans=False) if h["kind"] == "memory_state"][:k]
        self.memory_searches += 1
        self.token_cost += sum(int(h.get("token_estimate") or 0) for h in hits)
        return _obs({
            "hits": [
                {
                    "memory_state_id": h["id"],
                    "source": h.get("source"),
                    "score": round(float(h.get("score") or 0), 3),
                    "content": h.get("text", "")[:1200],
                    "source_span_ids": h.get("source_span_ids", [])[:8],
                }
                for h in hits
            ],
            "n_hits": len(hits),
        })

    @tool
    def retrieve_memory_states(self, query: str, k: int = 8) -> ToolResult:
        """Search only compressed memory states. Use before answering."""
        return self.retrieve_memory_states_impl(query=query, k=k)

    def freeze_raw_access_impl(self, reason: str = "") -> ToolResult:
        self.raw_enabled = False
        return _obs({"raw_enabled": False, "reason": reason})

    @tool
    def freeze_raw_access(self, reason: str = "") -> ToolResult:
        """Disable raw-span search for the rest of the episode.

        This forces the policy to answer from compressed memory, making memory
        writes causally useful instead of letting the policy do raw RAG forever.
        """
        return self.freeze_raw_access_impl(reason=reason)

    def stats(self) -> dict[str, Any]:
        return {
            "raw_searches": self.raw_searches,
            "memory_searches": self.memory_searches,
            "writes": self.writes,
            "token_cost": self.token_cost,
            "written_state_ids": list(self.written_state_ids),
            "raw_enabled": self.raw_enabled,
        }
