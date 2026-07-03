"""Honcho Memory Provider — dialectical user modeling (psychology-based).

From plastic-labs/honcho: "Honcho uses an entity-centric model where both users
and agents are represented as 'peers'." Builds a psychological profile — answers
"what kind of person is this?" not "what facts do I know?"

Requires HONCHO_API_KEY environment variable. No local simulation fallback.
"""
from __future__ import annotations
import logging, os
from typing import Any

from .interface import MemoryProvider, MemoryProviderConfig, SearchResult, MemoryStats

logger = logging.getLogger("honcho.provider")

class HonchoProvider(MemoryProvider):
    name = "honcho"

    def __init__(self, config: MemoryProviderConfig | None = None):
        super().__init__(config)
        self.api_key = (config.api_key if config else None) or os.environ.get("HONCHO_API_KEY")
        if not self.api_key:
            raise ValueError("HONCHO_API_KEY not set — Honcho requires an API key.")
        try:
            from honcho import Honcho
            self._client = Honcho(workspace_id="pie_benchmark")
        except ImportError:
            raise ImportError("honcho-ai not installed. Run: pip install honcho-ai")
        self._n_ingested = 0

    def ingest(self, sessions: list[list[dict]], dates: list[str] | None = None) -> None:
        user = self._client.peer("user")
        assistant = self._client.peer("assistant")
        for i, session in enumerate(sessions):
            sess = self._client.session(f"session_{i}")
            messages = []
            for turn in session:
                peer = user if turn.get("role") == "user" else assistant
                content = turn.get("content", "")
                if content.strip():
                    messages.append(peer.message(content))
            if messages:
                sess.add_messages(messages)
            self._n_ingested += 1
        logger.info(f"Honcho: ingested {len(sessions)} sessions")

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        try:
            user = self._client.peer("user")
            results = user.search(query)
            return [
                SearchResult(content=str(r), score=0.5, metadata={})
                for r in results[:top_k]
            ]
        except Exception as e:
            raise RuntimeError(f"Honcho search failed: {e}") from e

    def answer(self, question: str, question_date: str | None = None) -> str:
        results = self.search(question, top_k=10)
        if not results:
            return "not in context"
        return "\n".join([r.content[:500] for r in results])

    def stats(self) -> MemoryStats:
        return MemoryStats(num_memories=self._n_ingested)

    def clear(self) -> None:
        self._n_ingested = 0
