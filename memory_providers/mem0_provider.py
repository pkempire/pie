"""Mem0 Memory Provider — flat fact store with embedding retrieval.

Paper: "Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory" (arXiv:2504.19413)

Architecture: LLM extracts salient facts → stored with embeddings → cosine retrieval.
Requires MEM0_API_KEY environment variable. No local simulation fallback.
"""
from __future__ import annotations
import logging, os, requests
from typing import Any

from .interface import MemoryProvider, MemoryProviderConfig, SearchResult, MemoryStats

logger = logging.getLogger("mem0.provider")

class Mem0Provider(MemoryProvider):
    name = "mem0"

    def __init__(self, config: MemoryProviderConfig | None = None):
        super().__init__(config)
        self.api_key = (config.api_key if config else None) or os.environ.get("MEM0_API_KEY")
        if not self.api_key:
            raise ValueError("MEM0_API_KEY not set — Mem0 requires an API key. "
                             "Get one at https://mem0.ai")
        self._base_url = "https://api.mem0.ai/v1"
        self._headers = {"Authorization": f"Token {self.api_key}", "Content-Type": "application/json"}
        self._user_id = "benchmark_user"
        self._n_ingested = 0

    def ingest(self, sessions: list[list[dict]], dates: list[str] | None = None) -> None:
        ingested = 0
        for session in sessions:
            messages = [
                {"role": t.get("role", "user"), "content": t.get("content", "")}
                for t in session if t.get("content", "").strip()
            ]
            if not messages:
                continue
            try:
                resp = requests.post(
                    f"{self._base_url}/memories/",
                    headers=self._headers,
                    json={"messages": messages, "user_id": self._user_id},
                    timeout=30,
                )
                resp.raise_for_status()
                ingested += 1
            except Exception as e:
                raise RuntimeError(f"Mem0 API ingest failed: {e}") from e
        self._n_ingested += ingested
        logger.info(f"Mem0: ingested {ingested} sessions (total {self._n_ingested})")

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        try:
            resp = requests.post(
                f"{self._base_url}/memories/search/",
                headers=self._headers,
                json={"query": query, "user_id": self._user_id, "limit": top_k},
                timeout=30,
            )
            resp.raise_for_status()
            results = resp.json()
            return [
                SearchResult(
                    content=r.get("memory", str(r)),
                    score=r.get("score", 0.5),
                    metadata=r.get("metadata", {}),
                )
                for r in results
            ]
        except Exception as e:
            raise RuntimeError(f"Mem0 search failed: {e}") from e

    def answer(self, question: str, question_date: str | None = None) -> str:
        results = self.search(question, top_k=10)
        if not results:
            return "not in context"
        memories_str = "\n".join([f"- {r.content}" for r in results])
        return memories_str  # caller (HeuristicPolicy) handles the LLM answer

    def stats(self) -> MemoryStats:
        return MemoryStats(num_memories=self._n_ingested)

    def clear(self) -> None:
        self._n_ingested = 0
