"""
Mem0 Memory Provider

Flat fact store with embedding retrieval.

Paper: "Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory" (arXiv:2504.19413)

Architecture:
    1. Memory Extraction: LLM extracts salient facts from conversations
    2. Storage: Key-value pairs with vector embeddings
    3. Retrieval: Embedding similarity search
    4. Optional Graph (Mem0^g): Adds entity relationships

Key characteristics:
    - Simple and fast
    - No explicit temporal model (just timestamps)
    - LLM-based conflict resolution (overwrites or keeps both)
    - Strong baseline on LOCOMO (+26% over OpenAI Memory)

Differences from PIE:
    - No state transition chains
    - No typed change detection
    - No procedural pattern extraction
    - Simpler = faster, but less temporal reasoning depth

Claims from paper:
    - +26% accuracy over OpenAI Memory on LOCOMO
    - 91% faster than full-context
    - 90% fewer tokens
"""

from __future__ import annotations
import logging
import os
from typing import Any, Optional, List, Dict

from .interface import MemoryProvider, MemoryProviderConfig, SearchResult, MemoryStats

logger = logging.getLogger("mem0.provider")


class Mem0Provider(MemoryProvider):
    """
    Mem0 memory provider.
    
    Uses either:
    1. Mem0 Cloud API (mem0.ai) - managed service
    2. mem0 Python package - self-hosted
    3. Local simulation - for benchmarks without API
    """
    
    name = "mem0"
    
    def __init__(self, config: MemoryProviderConfig | None = None):
        super().__init__(config)
        
        self.api_key = config.api_key if config else os.environ.get("MEM0_API_KEY")
        self._use_package = False
        self._memory = None
        self._user_id = "benchmark_user"
        
        # Local fallback state
        self._memories: list[dict] = []
        self._embeddings: list = []
        
        self._init_client()
    
    def _init_client(self):
        """Initialize Mem0 client."""
        if self.api_key:
            # Use direct HTTP API (more reliable than package)
            self._use_http_api = True
            self._base_url = "https://api.mem0.ai/v1"
            self._headers = {
                "Authorization": f"Token {self.api_key}",
                "Content-Type": "application/json"
            }
            logger.info("Initialized Mem0 HTTP API client")
            return
        
        # Try local package
        try:
            from mem0 import Memory
            self._memory = Memory()
            self._use_package = True
            logger.info("Initialized local Mem0 package")
        except ImportError:
            logger.info("mem0 package not installed, using local simulation")
    
    @property
    def _use_http_api(self):
        return getattr(self, '_http_api_enabled', False)
    
    @_use_http_api.setter
    def _use_http_api(self, value):
        self._http_api_enabled = value
    
    def ingest(self, sessions, dates = None) -> None:
        """
        Ingest sessions into Mem0.
        
        Mem0's approach:
        1. Send conversation to memory.add()
        2. LLM extracts salient facts
        3. Facts stored with embeddings
        """
        if self._use_http_api or (self._use_package and self._memory):
            self._ingest_package(sessions, dates)
        else:
            self._ingest_local(sessions, dates)
    
    def _ingest_package(self, sessions, dates):
        """Ingest using mem0 package or HTTP API. Falls back to local on failure."""
        import requests

        if self._use_http_api:
            failures = 0
            for i, session in enumerate(sessions):
                messages = [
                    {"role": turn.get("role", "user"), "content": turn.get("content", "")}
                    for turn in session
                    if turn.get("content", "").strip()
                ]
                if messages:
                    try:
                        resp = requests.post(
                            f"{self._base_url}/memories/",
                            headers=self._headers,
                            json={"messages": messages, "user_id": self._user_id},
                            timeout=30
                        )
                        resp.raise_for_status()
                    except Exception as e:
                        failures += 1
                        logger.warning(f"Mem0 HTTP add failed: {e}")
                        if failures >= 2:
                            logger.warning("Mem0 HTTP API unreliable, falling back to local simulation")
                            self._use_http_api = False
                            self._ingest_local(sessions, dates)
                            return

            if failures == 0:
                logger.info(f"Ingested {len(sessions)} sessions via Mem0 HTTP API")
            else:
                logger.warning(f"Mem0 HTTP: {failures} failures during ingest, falling back to local")
                self._use_http_api = False
                self._ingest_local(sessions, dates)
        else:
            for i, session in enumerate(sessions):
                messages = [
                    {"role": turn.get("role", "user"), "content": turn.get("content", "")}
                    for turn in session
                    if turn.get("content", "").strip()
                ]
                if messages:
                    try:
                        self._memory.add(messages, user_id=self._user_id)
                    except Exception as e:
                        logger.warning(f"Mem0 add failed: {e}")
            logger.info(f"Ingested {len(sessions)} sessions via Mem0 package")
    
    def _ingest_local(self, sessions: list[list[dict]], dates: list[str] | None):
        """
        Local simulation of Mem0's extraction.
        
        Mem0 extracts facts like:
        - "User prefers dark mode"
        - "User lives in NYC"
        - "User is working on Project X"
        """
        from openai import OpenAI
        client = OpenAI()
        
        for i, session in enumerate(sessions):
            date = dates[i] if dates and i < len(dates) else f"2025-01-{i+1:02d}"
            
            text = "\n".join([
                f"{t.get('role', 'user')}: {t.get('content', '')}"
                for t in session
            ])
            
            if len(text) < 50:
                continue
            
            # Extract facts (Mem0's core operation)
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{
                        "role": "user",
                        "content": f"""Extract key facts about the user from this conversation.
Return as JSON: {{"facts": ["fact 1", "fact 2", ...]}}

Conversation:
{text[:4000]}"""
                    }],
                    response_format={"type": "json_object"},
                    max_tokens=500,
                )
                
                import json
                result = json.loads(response.choices[0].message.content)
                
                for fact in result.get("facts", []):
                    self._memories.append({
                        "memory": fact,
                        "date": date,
                        "session_idx": i,
                    })
                    
            except Exception as e:
                logger.warning(f"Fact extraction failed: {e}")
        
        # Compute embeddings for all memories
        if self._memories:
            try:
                texts = [m["memory"] for m in self._memories]
                response = client.embeddings.create(
                    model="text-embedding-3-small",
                    input=texts[:500],  # Limit
                )
                self._embeddings = [d.embedding for d in response.data]
            except Exception as e:
                logger.warning(f"Embedding failed: {e}")
        
        logger.info(f"Extracted {len(self._memories)} facts locally")
    
    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """Search memories by embedding similarity."""
        if self._use_http_api or (self._use_package and self._memory):
            return self._search_package(query, top_k)
        else:
            return self._search_local(query, top_k)
    
    def _search_package(self, query: str, top_k: int) -> List[SearchResult]:
        """Search via Mem0 package or HTTP API. Falls back to local on failure."""
        import requests

        if self._use_http_api:
            try:
                resp = requests.post(
                    f"{self._base_url}/memories/search/",
                    headers=self._headers,
                    json={"query": query, "user_id": self._user_id, "limit": top_k},
                    timeout=30
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
                logger.error(f"Mem0 HTTP search failed: {e}, falling back to local")
                return self._search_local(query, top_k)
        else:
            try:
                results = self._memory.search(query=query, user_id=self._user_id, limit=top_k)
                
                result_list = results.get("results", results) if isinstance(results, dict) else results
                return [
                    SearchResult(
                        content=r.get("memory", str(r)),
                        score=r.get("score", 0.5),
                        metadata=r.get("metadata", {}),
                    )
                    for r in result_list
                ]
            except Exception as e:
                logger.error(f"Mem0 search failed: {e}")
                return []
    
    def _search_local(self, query: str, top_k: int) -> list[SearchResult]:
        """Local embedding search."""
        if not self._memories or not self._embeddings:
            return []
        
        from openai import OpenAI
        import numpy as np
        
        client = OpenAI()
        
        # Get query embedding
        query_resp = client.embeddings.create(
            model="text-embedding-3-small",
            input=query,
        )
        query_emb = np.array(query_resp.data[0].embedding)
        
        # Score memories
        scored = []
        for i, mem in enumerate(self._memories):
            if i < len(self._embeddings):
                emb = np.array(self._embeddings[i])
                score = np.dot(query_emb, emb) / (np.linalg.norm(query_emb) * np.linalg.norm(emb) + 1e-8)
                scored.append((mem, float(score)))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        
        return [
            SearchResult(
                content=mem["memory"],
                score=score,
                metadata={"date": mem.get("date")},
            )
            for mem, score in scored[:top_k]
        ]
    
    def answer(self, question: str, question_date: str | None = None) -> str:
        """Answer using retrieved memories."""
        results = self.search(question, top_k=10)
        
        if not results:
            return "I don't have enough information."
        
        # Mem0's approach: inject memories into system prompt
        memories_str = "\n".join([f"- {r.content}" for r in results])
        
        from openai import OpenAI
        client = OpenAI()
        
        response = client.chat.completions.create(
            model=self.config.model,
            messages=[
                {
                    "role": "system", 
                    "content": f"You are a helpful AI. Answer based on these user memories:\n{memories_str}"
                },
                {"role": "user", "content": question},
            ],
            max_tokens=300,
        )
        
        return response.choices[0].message.content.strip()
    
    def stats(self) -> MemoryStats:
        if self._use_package and self._memory:
            try:
                all_mems = self._memory.get_all(user_id=self._user_id)
                count = len(all_mems.get("results", all_mems) if isinstance(all_mems, dict) else all_mems)
                return MemoryStats(num_memories=count)
            except:
                pass
        
        return MemoryStats(num_memories=len(self._memories))
    
    def clear(self) -> None:
        if self._use_package and self._memory:
            try:
                self._memory.delete_all(user_id=self._user_id)
            except:
                pass
        
        self._memories = []
        self._embeddings = []
