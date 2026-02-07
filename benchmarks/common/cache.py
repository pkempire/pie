"""
CachedWorldModel — wrapper for PIE world model with embedding caching.

This eliminates the need to rebuild the world model for each question
in benchmark runs. Load once, query many times.

Usage:
    # Load or build cached world model
    cached_wm = CachedWorldModel.load_or_build(
        cache_path="cache/question_123.json",
        build_fn=lambda: _build_world_model_for_question(item, llm, model),
        llm=llm,
    )
    
    # Retrieve relevant entities (embeddings cached)
    results = cached_wm.retrieve("What did I do at MoMA?", top_k=15)
    
    # Each result: (entity_id, entity, similarity_score)
"""

from __future__ import annotations

import json
import logging
import hashlib
from pathlib import Path
from typing import Any, Callable

from pie.core.world_model import WorldModel, cosine_similarity
from pie.core.llm import LLMClient

logger = logging.getLogger("pie.bench.cache")


class CachedWorldModel:
    """
    Wrapper around WorldModel that caches embeddings for efficient retrieval.
    
    Key optimizations:
    1. World model loaded from JSON once (entities, transitions, relationships)
    2. Entity embeddings computed lazily and cached in memory
    3. Query embeddings cached to avoid re-embedding identical queries
    4. Optional disk persistence for embeddings
    """
    
    def __init__(
        self,
        world_model: WorldModel,
        llm: LLMClient | None = None,
        embed_model: str = "text-embedding-3-large",
        cache_embeddings_path: Path | None = None,
    ):
        """
        Initialize cached world model.
        
        Args:
            world_model: The underlying PIE WorldModel
            llm: LLM client for computing embeddings (if not already cached)
            embed_model: Embedding model to use
            cache_embeddings_path: Optional path to persist embeddings to disk
        """
        self.wm = world_model
        self._llm = llm  # Lazy init - only create if needed
        self.embed_model = embed_model
        self.cache_embeddings_path = Path(cache_embeddings_path) if cache_embeddings_path else None
        
        # In-memory caches
        self._entity_embeddings: dict[str, list[float]] = {}  # entity_id -> embedding
        self._query_cache: dict[str, list[float]] = {}  # query_hash -> embedding
        self._entity_texts: dict[str, str] = {}  # entity_id -> text used for embedding
        
        # Stats for debugging
        self.stats = {
            "entity_embed_hits": 0,
            "entity_embed_misses": 0,
            "query_embed_hits": 0,
            "query_embed_misses": 0,
            "total_queries": 0,
        }
        
        # Load cached embeddings if available
        if self.cache_embeddings_path and self.cache_embeddings_path.exists():
            self._load_embeddings()
    
    @property
    def llm(self) -> LLMClient:
        """Lazy init LLM client."""
        if self._llm is None:
            self._llm = LLMClient()
        return self._llm
    
    @classmethod
    def load_or_build(
        cls,
        cache_path: Path | str,
        build_fn: Callable[[], WorldModel],
        llm: LLMClient | None = None,
        embed_model: str = "text-embedding-3-large",
        force_rebuild: bool = False,
    ) -> "CachedWorldModel":
        """
        Load cached world model from disk, or build and cache it.
        
        Args:
            cache_path: Path to world model JSON cache
            build_fn: Function to build world model if not cached
            llm: LLM client
            embed_model: Embedding model
            force_rebuild: Force rebuilding even if cache exists
        
        Returns:
            CachedWorldModel ready for queries
        """
        cache_path = Path(cache_path)
        embeddings_path = cache_path.with_suffix(".embeddings.json")
        
        llm = llm or LLMClient()
        
        if not force_rebuild and cache_path.exists():
            # Load from cache
            logger.info(f"Loading cached world model from {cache_path}")
            wm = WorldModel(persist_path=cache_path)
            logger.info(f"  Loaded {len(wm.entities)} entities, {len(wm.transitions)} transitions")
        else:
            # Build fresh
            logger.info(f"Building world model (cache not found at {cache_path})")
            wm = build_fn()
            
            # Persist for next time
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            wm.persist_path = cache_path
            wm.save()
            logger.info(f"  Cached to {cache_path}")
        
        return cls(
            world_model=wm,
            llm=llm,
            embed_model=embed_model,
            cache_embeddings_path=embeddings_path,
        )
    
    def _get_entity_text(self, entity_id: str) -> str:
        """Get text representation of entity for embedding."""
        if entity_id in self._entity_texts:
            return self._entity_texts[entity_id]
        
        entity = self.wm.entities.get(entity_id)
        if not entity:
            return ""
        
        state = entity.current_state or {}
        desc = state.get("description", "")
        if not desc and isinstance(state, dict):
            # Build description from state dict
            desc = "; ".join(
                f"{k}: {v}" for k, v in state.items()
                if k not in ("description", "_is_event") and v
            )
        
        text = f"{entity.name} ({entity.type.value}): {desc}"[:500]
        self._entity_texts[entity_id] = text
        return text
    
    def _get_entity_embedding(self, entity_id: str) -> list[float] | None:
        """Get or compute embedding for an entity."""
        # Check memory cache
        if entity_id in self._entity_embeddings:
            self.stats["entity_embed_hits"] += 1
            return self._entity_embeddings[entity_id]
        
        # Check if entity has embedding stored
        entity = self.wm.entities.get(entity_id)
        if entity and entity.embedding:
            self._entity_embeddings[entity_id] = entity.embedding
            self.stats["entity_embed_hits"] += 1
            return entity.embedding
        
        self.stats["entity_embed_misses"] += 1
        return None  # Will be batch-computed later
    
    def _ensure_entity_embeddings(self):
        """Ensure all entities have embeddings (batch compute if needed)."""
        needs_embedding = []
        needs_embedding_ids = []
        
        for eid in self.wm.entities:
            if eid not in self._entity_embeddings:
                entity = self.wm.entities[eid]
                if entity.embedding:
                    self._entity_embeddings[eid] = entity.embedding
                else:
                    text = self._get_entity_text(eid)
                    if text:
                        needs_embedding.append(text)
                        needs_embedding_ids.append(eid)
        
        if needs_embedding:
            logger.info(f"Computing embeddings for {len(needs_embedding)} entities...")
            embeddings = self._batch_embed(needs_embedding)
            for eid, emb in zip(needs_embedding_ids, embeddings):
                self._entity_embeddings[eid] = emb
                # Also store on entity for future
                self.wm.entities[eid].embedding = emb
            logger.info(f"  Computed {len(embeddings)} embeddings")
            
            # Persist embeddings
            if self.cache_embeddings_path:
                self._save_embeddings()
    
    def _batch_embed(
        self,
        texts: list[str],
        batch_size: int = 512,
    ) -> list[list[float]]:
        """Embed texts in batches."""
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch = [t[:8000] for t in batch]  # Truncate for API limits
            embeddings = self.llm.embed(batch, model=self.embed_model)
            all_embeddings.extend(embeddings)
        return all_embeddings
    
    def _hash_query(self, query: str) -> str:
        """Hash query for caching."""
        return hashlib.md5(query.encode()).hexdigest()[:16]
    
    def _get_query_embedding(self, query: str) -> list[float]:
        """Get or compute embedding for a query."""
        query_hash = self._hash_query(query)
        
        if query_hash in self._query_cache:
            self.stats["query_embed_hits"] += 1
            return self._query_cache[query_hash]
        
        self.stats["query_embed_misses"] += 1
        emb = self.llm.embed_single(query, model=self.embed_model)
        self._query_cache[query_hash] = emb
        return emb
    
    def retrieve(
        self,
        query: str,
        top_k: int = 15,
        entity_type: str | None = None,
    ) -> list[tuple[str, Any, float]]:
        """
        Retrieve relevant entities for a query.
        
        Args:
            query: Search query (e.g., question text)
            top_k: Number of results to return
            entity_type: Optional filter by entity type
        
        Returns:
            List of (entity_id, entity, similarity_score) tuples
        """
        self.stats["total_queries"] += 1
        
        # Ensure all entities have embeddings
        self._ensure_entity_embeddings()
        
        # Get query embedding
        query_emb = self._get_query_embedding(query)
        
        # Score all entities
        scored = []
        for eid, entity in self.wm.entities.items():
            if entity_type and entity.type.value != entity_type:
                continue
            
            emb = self._entity_embeddings.get(eid)
            if emb:
                sim = cosine_similarity(query_emb, emb)
                scored.append((eid, entity, sim))
        
        # Sort by similarity
        scored.sort(key=lambda x: x[2], reverse=True)
        return scored[:top_k]
    
    def get_entity(self, entity_id: str) -> Any:
        """Get entity by ID."""
        return self.wm.get_entity(entity_id)
    
    def get_transitions(self, entity_id: str, ordered: bool = True) -> list:
        """Get transitions for an entity."""
        return self.wm.get_transitions(entity_id, ordered=ordered)
    
    def get_relationships(self, entity_id: str) -> list:
        """Get relationships for an entity."""
        return self.wm.get_relationships(entity_id)
    
    @property
    def entities(self) -> dict:
        """Access underlying entities dict."""
        return self.wm.entities
    
    @property
    def transitions(self) -> dict:
        """Access underlying transitions dict."""
        return self.wm.transitions
    
    def _save_embeddings(self):
        """Save embeddings to disk."""
        if not self.cache_embeddings_path:
            return
        
        data = {
            "entity_embeddings": self._entity_embeddings,
            "embed_model": self.embed_model,
        }
        
        self.cache_embeddings_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_embeddings_path, "w") as f:
            json.dump(data, f)
        
        logger.debug(f"Saved {len(self._entity_embeddings)} embeddings to {self.cache_embeddings_path}")
    
    def _load_embeddings(self):
        """Load embeddings from disk."""
        if not self.cache_embeddings_path or not self.cache_embeddings_path.exists():
            return
        
        try:
            with open(self.cache_embeddings_path) as f:
                data = json.load(f)
            
            # Only load if same embed model
            if data.get("embed_model") == self.embed_model:
                self._entity_embeddings = data.get("entity_embeddings", {})
                logger.info(f"Loaded {len(self._entity_embeddings)} cached embeddings")
            else:
                logger.info(f"Embed model changed, recomputing embeddings")
        except Exception as e:
            logger.warning(f"Failed to load embeddings cache: {e}")
    
    def print_stats(self):
        """Print cache statistics."""
        print(f"CachedWorldModel Stats:")
        print(f"  Entities: {len(self.wm.entities)}")
        print(f"  Entity embeddings cached: {len(self._entity_embeddings)}")
        print(f"  Query embeddings cached: {len(self._query_cache)}")
        print(f"  Entity embed hits/misses: {self.stats['entity_embed_hits']}/{self.stats['entity_embed_misses']}")
        print(f"  Query embed hits/misses: {self.stats['query_embed_hits']}/{self.stats['query_embed_misses']}")
        print(f"  Total queries: {self.stats['total_queries']}")
