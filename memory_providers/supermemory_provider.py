"""Supermemory Memory Provider — fast routing + disambiguation.

From supermemory.ai/research: state of the art on MemoryBench, focus on
disambiguation to handle similar but different entities.

Requires SUPERMEMORY_API_KEY environment variable. No local simulation fallback.
"""
from __future__ import annotations
import logging, os, uuid
from typing import Any

import requests

from .interface import MemoryProvider, MemoryProviderConfig, SearchResult, MemoryStats

logger = logging.getLogger("supermemory.provider")

class SupermemoryProvider(MemoryProvider):
    name = "supermemory"

    def __init__(self, config: MemoryProviderConfig | None = None):
        super().__init__(config)
        self.api_key = (config.api_key if config else None) or os.environ.get("SUPERMEMORY_API_KEY")
        if not self.api_key:
            raise ValueError("SUPERMEMORY_API_KEY not set — Supermemory requires an API key. "
                             "Get one at https://supermemory.ai")
        self._base_url = "https://api.supermemory.ai"
        self._headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        self._container_tag = "benchmark"

    def ingest(self, sessions: list[list[dict]], dates: list[str] | None = None) -> None:
        for session in sessions:
            messages = [
                {"role": t.get("role", "user"), "content": t.get("content", "")}
                for t in session if t.get("content", "").strip()
            ]
            if not messages:
                continue
            try:
                resp = requests.post(
                    f"{self._base_url}/conversations",
                    headers=self._headers,
                    json={
                        "conversationId": f"bench_{uuid.uuid4().hex[:8]}",
                        "containerTag": self._container_tag,
                        "messages": messages,
                    },
                    timeout=30,
                )
                if resp.status_code not in (200, 201):
                    raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
            except Exception as e:
                raise RuntimeError(f"Supermemory ingest failed: {e}") from e
        logger.info(f"Supermemory: ingested {len(sessions)} sessions")

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        try:
            resp = requests.post(
                f"{self._base_url}/search/memories",
                headers=self._headers,
                json={"query": query, "containerTags": [self._container_tag], "limit": top_k},
                timeout=30,
            )
            if resp.status_code != 200:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
            data = resp.json()
            results = data.get("results", data.get("memories", []))
            return [
                SearchResult(
                    content=r.get("content", r.get("memory", str(r))),
                    score=r.get("score", r.get("similarity", 0.5)),
                    metadata=r.get("metadata", {}),
                )
                for r in results
            ]
        except Exception as e:
            raise RuntimeError(f"Supermemory search failed: {e}") from e

    def answer(self, question: str, question_date: str | None = None) -> str:
        results = self.search(question, top_k=10)
        if not results:
            return "not in context"
        return "\n".join([r.content[:500] for r in results])

    def stats(self) -> MemoryStats:
        return MemoryStats(num_memories=0)

    def clear(self) -> None:
        pass
