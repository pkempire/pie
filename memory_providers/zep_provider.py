"""
Zep/Graphiti Memory Provider

Zep's temporal knowledge graph architecture.

Paper: "Zep: A Temporal Knowledge Graph Architecture for Agent Memory" (arXiv:2501.13956)

Architecture (from paper):
    1. Episodic Memory: Raw conversation episodes with timestamps
    2. Semantic Entity Subgraph: Extracted entities and relationships
    3. Community Subgraph: Clustered entity communities
    
Key features:
    - Bi-temporal model: event_time (when it happened) + ingestion_time (when recorded)
    - Edge invalidation for contradictions (not explicit transition types like PIE)
    - Community detection for entity clustering
    - Hybrid retrieval: semantic + keyword + graph traversal

Differences from PIE:
    - No typed state transitions (creation/update/contradiction/resolution)
    - No procedural pattern extraction
    - Uses community detection vs PIE's explicit relationships
    - Faster (~200ms) but less temporal reasoning depth
"""

from __future__ import annotations
import logging
import os
from typing import Any

from .interface import MemoryProvider, MemoryProviderConfig, SearchResult, MemoryStats

logger = logging.getLogger("zep.provider")


class ZepProvider(MemoryProvider):
    """
    Zep memory provider using Graphiti.
    
    Can use either:
    1. Zep Cloud API (api.getzep.com) - managed service
    2. Self-hosted Graphiti - open source
    
    For benchmarks, we use the Zep Cloud API when available,
    otherwise fall back to a simplified local implementation.
    """
    
    name = "zep"
    
    def __init__(self, config: MemoryProviderConfig | None = None):
        super().__init__(config)
        
        self.api_key = config.api_key if config else os.environ.get("ZEP_API_KEY")
        self._use_cloud = bool(self.api_key)
        self._client = None
        self._user_id = "benchmark_user"
        self._session_id = None
        
        # Local fallback state
        self._episodes = []
        self._entities = {}
        self._episode_embeddings = []  # Cached during ingest

        if self._use_cloud:
            self._init_cloud_client()
        else:
            logger.info("No ZEP_API_KEY - using local Graphiti simulation")
    
    def _init_cloud_client(self):
        """Initialize Zep Cloud client."""
        try:
            from zep_cloud.client import Zep
            self._client = Zep(api_key=self.api_key)
            logger.info("Initialized Zep Cloud client")
        except ImportError:
            logger.warning("zep-cloud not installed, using local simulation")
            self._use_cloud = False
    
    def ingest(self, sessions: list[list[dict]], dates: list[str] | None = None) -> None:
        """
        Ingest sessions as Zep episodes.
        
        In Zep/Graphiti:
        - Each session becomes an "episode"
        - Episodes are processed to extract entities and edges
        - Entities are deduplicated via community detection
        """
        if self._use_cloud and self._client:
            self._ingest_cloud(sessions, dates)
        else:
            self._ingest_local(sessions, dates)
    
    def _ingest_cloud(self, sessions: list[list[dict]], dates: list[str] | None):
        """Ingest via Zep Cloud API."""
        try:
            # Create or get user
            try:
                self._client.user.add(user_id=self._user_id)
            except:
                pass  # User may already exist
            
            # Create session
            import uuid
            self._session_id = f"bench_{uuid.uuid4().hex[:8]}"
            self._client.memory.add_session(
                session_id=self._session_id,
                user_id=self._user_id,
            )
            
            # Add messages
            for i, session in enumerate(sessions):
                for turn in session:
                    self._client.memory.add(
                        session_id=self._session_id,
                        messages=[{
                            "role_type": turn.get("role", "user"),
                            "content": turn.get("content", ""),
                        }],
                    )
            
            logger.info(f"Ingested {len(sessions)} sessions to Zep Cloud")
            
        except Exception as e:
            logger.error(f"Zep Cloud ingestion failed: {e}")
            # Fall back to local
            self._use_cloud = False
            self._ingest_local(sessions, dates)
    
    def _ingest_local(self, sessions: list[list[dict]], dates: list[str] | None):
        """
        Local Graphiti-style ingestion.
        
        Simulates Graphiti's approach:
        1. Store episodes with timestamps
        2. Extract entities using LLM
        3. Build entity index with embeddings
        """
        from openai import OpenAI
        client = OpenAI()
        
        for i, session in enumerate(sessions):
            date = dates[i] if dates and i < len(dates) else f"2025-01-{i+1:02d}"
            
            # Store episode
            text = "\n".join([
                f"{t.get('role', 'user')}: {t.get('content', '')}"
                for t in session
            ])
            
            self._episodes.append({
                "id": f"ep_{i}",
                "text": text,
                "date": date,
                "event_time": i,  # Bi-temporal: when it happened
                "ingestion_time": i,  # Bi-temporal: when recorded
            })
            
            # Extract entities (simplified - Graphiti uses more sophisticated extraction)
            if len(text) > 100:
                try:
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{
                            "role": "user",
                            "content": f"Extract key entities (people, places, things, events) from:\n\n{text[:4000]}\n\nList as JSON: {{\"entities\": [{{\"name\": ..., \"type\": ...}}]}}"
                        }],
                        response_format={"type": "json_object"},
                        max_tokens=500,
                    )
                    
                    import json
                    result = json.loads(response.choices[0].message.content)
                    for ent in result.get("entities", []):
                        name = ent.get("name", "").lower()
                        if name and name not in self._entities:
                            self._entities[name] = {
                                "name": ent.get("name"),
                                "type": ent.get("type", "unknown"),
                                "episodes": [f"ep_{i}"],
                                "first_seen": date,
                            }
                        elif name:
                            self._entities[name]["episodes"].append(f"ep_{i}")
                            
                except Exception as e:
                    logger.warning(f"Entity extraction failed: {e}")
        
        # Cache episode embeddings for fast search
        if self._episodes:
            try:
                texts = [ep["text"][:2000] for ep in self._episodes]
                emb_resp = client.embeddings.create(
                    model="text-embedding-3-small",
                    input=texts[:500],
                )
                self._episode_embeddings = [d.embedding for d in emb_resp.data]
            except Exception as e:
                logger.warning(f"Episode embedding failed: {e}")

        logger.info(f"Ingested {len(sessions)} sessions locally → {len(self._entities)} entities")

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        """
        Search using Zep's hybrid approach.
        
        Graphiti combines:
        1. Semantic (embedding) search
        2. Keyword (BM25) search  
        3. Graph traversal
        """
        if self._use_cloud and self._client and self._session_id:
            return self._search_cloud(query, top_k)
        else:
            return self._search_local(query, top_k)
    
    def _search_cloud(self, query: str, top_k: int) -> list[SearchResult]:
        """Search via Zep Cloud."""
        try:
            results = self._client.memory.search(
                session_id=self._session_id,
                text=query,
                limit=top_k,
            )
            
            return [
                SearchResult(
                    content=r.message.content if hasattr(r, 'message') else str(r),
                    score=r.score if hasattr(r, 'score') else 0.5,
                    metadata={"source": "zep_cloud"},
                )
                for r in results
            ]
        except Exception as e:
            logger.error(f"Zep search failed: {e}")
            return []
    
    def _search_local(self, query: str, top_k: int) -> list[SearchResult]:
        """Local embedding search using cached episode embeddings."""
        import numpy as np

        if not self._episodes or not self._episode_embeddings:
            return []

        from openai import OpenAI
        client = OpenAI()

        # Get query embedding
        query_resp = client.embeddings.create(
            model="text-embedding-3-small",
            input=query,
        )
        query_emb = np.array(query_resp.data[0].embedding)

        # Score against cached embeddings
        scored = []
        for i, ep in enumerate(self._episodes):
            if i < len(self._episode_embeddings):
                emb = np.array(self._episode_embeddings[i])
                score = np.dot(query_emb, emb) / (np.linalg.norm(query_emb) * np.linalg.norm(emb) + 1e-8)
                scored.append((ep, float(score)))

        scored.sort(key=lambda x: x[1], reverse=True)

        return [
            SearchResult(
                content=ep["text"][:1000],
                score=score,
                metadata={"episode_id": ep["id"], "date": ep["date"]},
            )
            for ep, score in scored[:top_k]
        ]
    
    def answer(self, question: str, question_date: str | None = None) -> str:
        """Answer using Zep's context assembly."""
        results = self.search(question, top_k=10)
        
        if not results:
            return "I don't have enough information."
        
        # Compile context (Zep sorts by relevance + recency)
        context = "\n\n".join([
            f"[{r.metadata.get('date', 'Unknown')}]\n{r.content}"
            for r in results
        ])
        
        from openai import OpenAI
        client = OpenAI()
        
        response = client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": "Answer based on the provided context. Be concise."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"},
            ],
            max_tokens=300,
        )
        
        return response.choices[0].message.content.strip()
    
    def stats(self) -> MemoryStats:
        return MemoryStats(
            num_memories=len(self._episodes),
            num_entities=len(self._entities),
            num_relationships=0,  # Would need graph DB for this
            extra={"use_cloud": self._use_cloud},
        )
    
    def clear(self) -> None:
        self._episodes = []
        self._entities = {}
        self._episode_embeddings = []
        self._session_id = None
