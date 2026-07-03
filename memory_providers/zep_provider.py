"""Zep/Graphiti Memory Provider — temporal knowledge graph.

Paper: "Zep: A Temporal Knowledge Graph Architecture for Agent Memory" (arXiv:2501.13956)

Requires ZEP_API_KEY environment variable. No local simulation fallback.
"""
from __future__ import annotations
import logging, os, uuid
from typing import Any

from .interface import MemoryProvider, MemoryProviderConfig, SearchResult, MemoryStats

logger = logging.getLogger("zep.provider")

class ZepProvider(MemoryProvider):
    name = "zep"

    def __init__(self, config: MemoryProviderConfig | None = None):
        super().__init__(config)
        self.api_key = (config.api_key if config else None) or os.environ.get("ZEP_API_KEY")
        if not self.api_key:
            raise ValueError("ZEP_API_KEY not set — Zep requires an API key. "
                             "Get one at https://www.getzep.com")
        try:
            from zep_cloud.client import Zep
            self._client = Zep(api_key=self.api_key)
        except ImportError:
            raise ImportError("zep-cloud package not installed. Run: pip install zep-cloud")
        self._user_id = "benchmark_user"
        self._session_id: str | None = None

    def ingest(self, sessions: list[list[dict]], dates: list[str] | None = None) -> None:
        try:
            self._client.user.add(user_id=self._user_id)
        except Exception:
            pass  # user may already exist
        self._session_id = f"bench_{uuid.uuid4().hex[:8]}"
        self._client.memory.add_session(session_id=self._session_id, user_id=self._user_id)
        for session in sessions:
            for turn in session:
                if not turn.get("content", "").strip():
                    continue
                self._client.memory.add(
                    session_id=self._session_id,
                    messages=[{
                        "role_type": turn.get("role", "user"),
                        "content": turn.get("content", ""),
                    }],
                )
        logger.info(f"Zep: ingested {len(sessions)} sessions")

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        try:
            results = self._client.memory.search(
                session_id=self._session_id, text=query, limit=top_k,
            )
            return [
                SearchResult(
                    content=r.message.content if hasattr(r, "message") else str(r),
                    score=r.score if hasattr(r, "score") else 0.5,
                    metadata={"source": "zep"},
                )
                for r in results
            ]
        except Exception as e:
            raise RuntimeError(f"Zep search failed: {e}") from e

    def answer(self, question: str, question_date: str | None = None) -> str:
        results = self.search(question, top_k=10)
        if not results:
            return "not in context"
        return "\n".join([r.content[:500] for r in results])

    def stats(self) -> MemoryStats:
        return MemoryStats(num_memories=1)

    def clear(self) -> None:
        self._session_id = None
