"""
Supermemory Memory Provider

Fast routing + disambiguation architecture.

From supermemory.ai/research:
    - State of the art on MemoryBench (LongMemEval + LoCoMo)
    - Focus on disambiguation to handle similar but different entities
    - Fast routing to relevant context

Architecture (inferred from docs/research):
    1. Memory Ingestion: Add from URLs, PDFs, text
    2. Disambiguation: Handle "Apple (company)" vs "apple (fruit)"
    3. Fast Routing: Route queries to relevant memory chunks
    4. Retrieval: Semantic search with disambiguation-aware ranking

Key claims:
    - 71.4% on LongMemEval (vs 86% Emergence, 71.2% Zep)
    - Sub-200ms latency
    - Disambiguation as key differentiator

API: https://api.supermemory.ai
"""

from __future__ import annotations
import logging
import os
from typing import Any, Optional, List, Dict

from .interface import MemoryProvider, MemoryProviderConfig, SearchResult, MemoryStats

logger = logging.getLogger("supermemory.provider")


class SupermemoryProvider(MemoryProvider):
    """
    Supermemory memory provider.
    
    Uses Supermemory Cloud API when available,
    otherwise simulates the disambiguation approach locally.
    """
    
    name = "supermemory"
    
    def __init__(self, config: MemoryProviderConfig | None = None):
        super().__init__(config)
        
        self.api_key = config.api_key if config else os.environ.get("SUPERMEMORY_API_KEY")
        self._use_cloud = bool(self.api_key)
        self._base_url = "https://api.supermemory.ai"  # No version prefix
        self._container_tag = "benchmark"  # Group memories by container
        
        # Local state
        self._memories: list[dict] = []
        self._entities: dict[str, dict] = {}  # For disambiguation
        self._embeddings: list = []
    
    def ingest(self, sessions: list[list[dict]], dates: list[str] | None = None) -> None:
        """
        Ingest sessions into Supermemory.
        
        Supermemory's approach emphasizes:
        1. Entity extraction with disambiguation
        2. Fast indexing for routing
        """
        if self._use_cloud:
            self._ingest_cloud(sessions, dates)
        else:
            self._ingest_local(sessions, dates)
    
    def _ingest_cloud(self, sessions, dates):
        """Ingest via Supermemory API (conversations endpoint)."""
        import requests
        import uuid
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        # Use conversations API for chat data
        for i, session in enumerate(sessions):
            messages = []
            for turn in session:
                messages.append({
                    "role": turn.get("role", "user"),
                    "content": turn.get("content", ""),
                })
            
            if not messages:
                continue
            
            try:
                # Ingest as conversation
                response = requests.post(
                    f"{self._base_url}/conversations",
                    headers=headers,
                    json={
                        "conversationId": f"bench_{uuid.uuid4().hex[:8]}",
                        "containerTag": self._container_tag,
                        "messages": messages,
                    },
                    timeout=30,
                )
                if response.status_code not in [200, 201]:
                    logger.warning(f"Supermemory response: {response.status_code} {response.text[:200]}")
            except Exception as e:
                logger.warning(f"Supermemory API failed: {e}")
                # Fall back to local
                self._use_cloud = False
                self._ingest_local(sessions, dates)
                return
        
        logger.info(f"Ingested {len(sessions)} sessions to Supermemory Cloud")
    
    def _ingest_local(self, sessions: list[list[dict]], dates: list[str] | None):
        """
        Local simulation with disambiguation.
        
        Key insight from Supermemory: disambiguation matters.
        "Apple" in tech context ≠ "apple" in food context.
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
            
            # Store memory
            self._memories.append({
                "content": text,
                "date": date,
                "session_idx": i,
            })
            
            # Extract entities with disambiguation context
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{
                        "role": "user",
                        "content": f"""Extract entities with disambiguation from this conversation.
For each entity, provide context to distinguish it from similar entities.

Return JSON: {{"entities": [{{"name": "...", "type": "...", "disambiguation": "context to identify this specific entity"}}]}}

Conversation:
{text[:3000]}"""
                    }],
                    response_format={"type": "json_object"},
                    max_tokens=500,
                )
                
                import json
                result = json.loads(response.choices[0].message.content)
                
                for ent in result.get("entities", []):
                    key = f"{ent.get('name', '').lower()}:{ent.get('disambiguation', '')[:50]}"
                    if key not in self._entities:
                        self._entities[key] = {
                            "name": ent.get("name"),
                            "type": ent.get("type"),
                            "disambiguation": ent.get("disambiguation"),
                            "memory_indices": [i],
                        }
                    else:
                        self._entities[key]["memory_indices"].append(i)
                        
            except Exception as e:
                logger.warning(f"Entity extraction failed: {e}")
        
        # Compute embeddings
        if self._memories:
            try:
                texts = [m["content"][:2000] for m in self._memories]
                response = client.embeddings.create(
                    model="text-embedding-3-small",
                    input=texts[:500],
                )
                self._embeddings = [d.embedding for d in response.data]
            except Exception as e:
                logger.warning(f"Embedding failed: {e}")
        
        logger.info(f"Ingested {len(self._memories)} memories, {len(self._entities)} disambiguated entities")
    
    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        """
        Search with disambiguation-aware routing.
        
        Supermemory's key innovation: route to the right context
        by understanding which "Apple" you mean.
        """
        if self._use_cloud:
            return self._search_cloud(query, top_k)
        else:
            return self._search_local(query, top_k)
    
    def _search_cloud(self, query: str, top_k: int) -> List[SearchResult]:
        """Search via Supermemory API (memory search endpoint)."""
        import requests
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        try:
            # Search memory entries
            response = requests.post(
                f"{self._base_url}/search/memories",
                headers=headers,
                json={
                    "query": query,
                    "containerTags": [self._container_tag],
                    "limit": top_k,
                },
                timeout=30,
            )
            
            if response.status_code != 200:
                logger.warning(f"Supermemory search: {response.status_code} {response.text[:200]}")
                return self._search_local(query, top_k)
            
            data = response.json()
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
            logger.error(f"Supermemory search failed: {e}")
            return self._search_local(query, top_k)
    
    def _search_local(self, query: str, top_k: int) -> list[SearchResult]:
        """Local search with disambiguation boost."""
        if not self._memories:
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
                base_score = np.dot(query_emb, emb) / (np.linalg.norm(query_emb) * np.linalg.norm(emb) + 1e-8)
                
                # Disambiguation boost: if query mentions a disambiguated entity, boost relevant memories
                boost = 0.0
                query_lower = query.lower()
                for key, ent in self._entities.items():
                    if ent["name"].lower() in query_lower and i in ent.get("memory_indices", []):
                        boost += 0.05  # Small boost for entity match
                
                scored.append((mem, float(base_score) + boost))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        
        return [
            SearchResult(
                content=mem["content"][:1500],
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
        
        context = "\n\n".join([f"[{r.metadata.get('date', '?')}]\n{r.content}" for r in results])
        
        from openai import OpenAI
        client = OpenAI()
        
        response = client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": "Answer based on the context. Be concise."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"},
            ],
            max_tokens=300,
        )
        
        return response.choices[0].message.content.strip()
    
    def stats(self) -> MemoryStats:
        return MemoryStats(
            num_memories=len(self._memories),
            num_entities=len(self._entities),
        )
    
    def clear(self) -> None:
        self._memories = []
        self._entities = {}
        self._embeddings = []
