"""Memory ops exposed as Tinker tools (analog of search_tool/tools.py:ChromaTool).

The policy emits tool calls like:
  <tool_call>{"name": "memory_search", "arguments": {"query": "...", "k": 10, "source": "hybrid"}}</tool_call>
The env runs the tool against a Backend instance and returns observations.

We expose 4 tools (smaller than the full 8-op vocab — we'll grow):
  memory_search    — the workhorse retrieval, picks bm25/dense/hybrid + k
  memory_expand    — 1-hop expansion from seed uids
  memory_filter    — temporal / speaker filter on existing hits
  memory_rerank    — re-rank with a different strategy

`stop_and_answer` is implicit: when the model emits an "Answer:" line, the env
detects the terminal and runs the judge. Same convention as the Search-R1 recipe.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import Any

from mempol.backends.base import Backend, Hit, Unit
from mempol.recipes.memory_rl.tinker_compat import tool, simple_tool_result, ToolResult


# ── Helpers to format hits for observation tokens ──
def _format_hit(h: Hit, max_chars: int = 200) -> dict:
    m = h.unit.metadata
    return {
        "uid": h.unit.uid,
        "score": round(h.score, 3),
        "source": h.source,
        "speaker": m.get("speaker"),
        "session": m.get("session"),
        "session_date": m.get("session_date"),
        "text": h.unit.text[:max_chars],
    }


def _format_observation(hits: list[Hit], note: str = "") -> str:
    """Compact JSON observation the model gets after a tool call."""
    return json.dumps({
        "note": note,
        "hits": [_format_hit(h) for h in hits[:10]],  # cap at 10 to control context
        "n_hits": len(hits),
    }, ensure_ascii=False)


# ── Memory tool class — one per-environment instance, holds the backend. ──
@dataclass
class MemoryTool:
    """Per-env wrapper. Holds a Backend and exposes memory ops as tools.

    The Tinker `@tool` decorator pattern is used at registration time below.
    """
    backend: Backend
    last_hits: list[Hit] = field(default_factory=list)
    n_searches: int = 0
    max_searches: int = 8

    # ----- tool 1: search -----
    @tool
    def memory_search(self, query: str, k: int = 10, source: str = "hybrid") -> ToolResult:
        """Search the memory backend.

        Args:
            query: natural-language search query
            k: top-k results to return (max 20)
            source: 'bm25' (lexical), 'dense' (semantic), or 'hybrid'
        """
        if self.n_searches >= self.max_searches:
            return simple_tool_result(json.dumps({"error": "max_searches reached", "hits": []}))
        k = max(1, min(int(k), 20))
        if source not in ("bm25", "dense", "hybrid"):
            source = "hybrid"
        hits = self.backend.retrieve(query=query, k=k, source=source)
        self.last_hits = hits
        self.n_searches += 1
        return simple_tool_result(_format_observation(hits, note=f"search query={query!r} k={k} source={source}"))

    # ----- tool 2: expand -----
    @tool
    def memory_expand(self, seed_uids: list[str], k_per: int = 2) -> ToolResult:
        """Expand from previously-retrieved seed uids (1-hop)."""
        seed_uids = list(seed_uids)[:5]
        new_hits = self.backend.expand(seed_uids, k_per=k_per)
        merged = list(self.last_hits)
        seen = {h.unit.uid for h in merged}
        for h in new_hits:
            if h.unit.uid not in seen:
                merged.append(h)
                seen.add(h.unit.uid)
        self.last_hits = merged
        return simple_tool_result(_format_observation(merged, note=f"expanded {len(seed_uids)} seeds → +{len(new_hits)}"))

    # ----- tool 3: filter -----
    @tool
    def memory_filter(self, predicate: str, value: Any | None = None) -> ToolResult:
        """Filter the last hit set.

        Args:
            predicate: 'session_lt' | 'session_gt' | 'session_eq' | 'speaker_eq'
            value: int for session_*, str for speaker_eq
        """
        if not self.last_hits:
            return simple_tool_result(json.dumps({"note": "nothing to filter", "hits": []}))
        kept: list[Hit] = []
        for h in self.last_hits:
            m = h.unit.metadata
            keep = True
            if predicate == "session_lt":
                keep = (m.get("session", 0) < int(value))
            elif predicate == "session_gt":
                keep = (m.get("session", 0) > int(value))
            elif predicate == "session_eq":
                keep = (m.get("session", 0) == int(value))
            elif predicate == "speaker_eq":
                keep = (str(m.get("speaker", "")).lower() == str(value).lower())
            if keep:
                kept.append(h)
        self.last_hits = kept
        return simple_tool_result(_format_observation(kept, note=f"filter {predicate}={value} kept {len(kept)}"))

    # ----- tool 4: rerank -----
    @tool
    def memory_rerank(self, strategy: str = "dense", query: str | None = None) -> ToolResult:
        """Rerank the current hit set."""
        if not self.last_hits:
            return simple_tool_result(json.dumps({"note": "nothing to rerank", "hits": []}))
        if strategy == "dense" and query:
            fresh = self.backend.retrieve(query=query, k=len(self.last_hits) * 2, source="dense")
            order = {h.unit.uid: i for i, h in enumerate(fresh)}
            self.last_hits.sort(key=lambda h: order.get(h.unit.uid, 9999))
        elif strategy == "session_desc":
            self.last_hits.sort(key=lambda h: h.unit.metadata.get("session", 0), reverse=True)
        elif strategy == "session_asc":
            self.last_hits.sort(key=lambda h: h.unit.metadata.get("session", 0))
        return simple_tool_result(_format_observation(self.last_hits, note=f"reranked by {strategy}"))


# ── Tinker registration shim ──
# The actual `@tool` decorator + `.to_spec()` lives in tinker_cookbook.tool_use.
# When this file is dropped into a tinker-cookbook clone, the registration looks
# something like:
#
# from tinker_cookbook.tool_use import tool
#
# class MemoryTool:
#     @tool
#     def memory_search(self, query: str, k: int = 10, source: str = "hybrid") -> str:
#         ...
#
# Here we keep the methods undecorated so the file is testable standalone.
# In the cookbook clone, add `@tool` to each of the four methods above.


def smoke():
    """Verify the tools work against a FlatBackend."""
    from mempol.backends.flat import FlatBackend
    from mempol.data.locomo import load
    from mempol.eval.runner import conv_to_units

    convs = load(n_convs=1)
    conv, qas = convs[0]
    b = FlatBackend(); b.ingest(conv_to_units(conv))
    mt = MemoryTool(backend=b)

    print("--- memory_search ---")
    obs = mt.memory_search(query="LGBTQ support group", k=5)
    print(obs[:500])

    print("\n--- memory_expand ---")
    seed = json.loads(obs)["hits"][0]["uid"]
    obs2 = mt.memory_expand(seed_uids=[seed], k_per=2)
    print(obs2[:500])

    print("\n--- memory_filter ---")
    obs3 = mt.memory_filter(predicate="session_eq", value=1)
    print(obs3[:500])

    print("\n--- memory_rerank ---")
    obs4 = mt.memory_rerank(strategy="dense", query="when did Caroline go to support group")
    print(obs4[:500])


if __name__ == "__main__":
    smoke()
