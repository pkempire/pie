"""
PIE Memory Provider

Our temporal knowledge graph approach.

Architecture:
    1. Ingestion: Parse conversations → Extract entities/relationships/state changes
    2. Resolution: 3-tier entity resolution (string → embedding → LLM)
    3. World Model: Temporal KG with typed state transitions
    4. Query: Semantic + temporal context compilation

Key differentiator: Explicit state transition chains, not just timestamps.
"""

from __future__ import annotations
import logging
from typing import Any

from .interface import MemoryProvider, MemoryProviderConfig, SearchResult, MemoryStats

logger = logging.getLogger("pie.provider")


class PIEProvider(MemoryProvider):
    """
    PIE (Personal Intelligence Engine) memory provider.
    
    Uses a temporal knowledge graph with:
    - Typed entities (person, project, tool, belief, event, etc.)
    - State transition chains (creation → update → contradiction → resolution)
    - Semantic time anchors ("during freshman year", not just timestamps)
    - Procedural patterns extracted from entity lifecycles
    """
    
    name = "pie"
    
    def __init__(self, config: MemoryProviderConfig | None = None):
        super().__init__(config)

        from pie.core.llm import LLMClient
        from pie.core.world_model import WorldModel

        self.llm = LLMClient()
        self.world_model = WorldModel()
        self._ingested = False
        self._retriever = None  # built lazily after ingestion
    
    def ingest(self, sessions: list[list[dict]], dates: list[str] | None = None) -> None:
        """
        Ingest sessions using PIE's extraction pipeline.
        
        This performs:
        1. Entity extraction with state tracking
        2. Relationship extraction
        3. Entity resolution (deduplication)
        4. State transition recording
        """
        from pie.core.models import Conversation, Turn
        from pie.ingestion.pipeline import IngestionPipeline
        from pie.config import PIEConfig
        import tempfile
        from pathlib import Path
        
        # Convert to PIE conversation format
        conversations = []
        for i, session in enumerate(sessions):
            date = dates[i] if dates and i < len(dates) else f"2025-01-{i+1:02d}"
            turns = []
            for j, turn in enumerate(session):
                turns.append(Turn(
                    role=turn.get("role", "user"),
                    text=turn.get("content", ""),
                    timestamp=j,
                ))
            conversations.append(Conversation(
                id=f"conv_{i}",
                title=f"Session {i+1}",
                created_at=0,
                updated_at=None,
                model=None,
                turns=turns,
            ))
        
        # Create temporary config
        with tempfile.TemporaryDirectory() as tmpdir:
            config = PIEConfig(output_dir=Path(tmpdir))
            config.use_web_grounding = False  # Skip for benchmarks
            
            # Note: For full ingestion, we'd use the pipeline
            # For benchmarks, we do a simplified extraction
            self._simple_extraction(conversations, dates)
        
        self._ingested = True
        self._build_retriever()
        logger.info(f"Ingested {len(sessions)} sessions → {self.world_model.stats}")
    
    def _simple_extraction(self, conversations, dates):
        """Simplified extraction for benchmark speed."""
        from pie.ingestion.prompts import EXTRACTION_SYSTEM_PROMPT, build_extraction_user_message
        from pie.core.llm import parse_extraction_result

        for i, conv in enumerate(conversations):
            date = dates[i] if dates and i < len(dates) else f"2025-01-{i+1:02d}"

            # Format conversation text
            text = "\n".join([
                f"{t.role.capitalize()}: {t.text}"
                for t in conv.turns
                if t.text and t.text.strip()
            ])

            if len(text) < 100:  # Skip very short conversations
                continue

            try:
                # Extract using LLM
                result = self.llm.chat(
                    messages=[
                        {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                        {"role": "user", "content": f"Date: {date}\n\n{text[:8000]}"},
                    ],
                    model=self.config.extra.get("extraction_model", "gpt-4o-mini"),
                    json_mode=True,
                )

                extraction = parse_extraction_result(
                    raw=result["content"],
                    conversation_ids=[conv.id],
                    tokens=result.get("tokens", {}),
                )

                # Add entities to world model with name-based dedup
                for entity in extraction.entities:
                    state = entity.state if isinstance(entity.state, dict) else {"description": str(entity.state)}
                    existing = self.world_model.find_by_name(entity.name)
                    if existing:
                        self.world_model.update_entity_state(
                            entity_id=existing.id,
                            new_state=state,
                            source_conversation_id=conv.id,
                            timestamp=0,
                        )
                    else:
                        self.world_model.create_entity(
                            name=entity.name,
                            type=entity.type,
                            state=state,
                            source_conversation_id=conv.id,
                            timestamp=0,
                        )

            except Exception as e:
                logger.warning(f"Extraction failed for conv {i}: {e}")

        # ── Compute embeddings for all entities (batch) ──
        # Without embeddings, search() returns nothing.
        entities_needing_embeddings = [
            (eid, entity) for eid, entity in self.world_model.entities.items()
            if not entity.embedding
        ]
        if entities_needing_embeddings:
            try:
                texts = [
                    f"{e.name} ({e.type}): {e.current_state.get('description', str(e.current_state)[:200])}"
                    for _, e in entities_needing_embeddings
                ]
                embeddings = self.llm.embed(texts)
                for (eid, entity), emb in zip(entities_needing_embeddings, embeddings):
                    entity.embedding = emb
                logger.info(f"Computed embeddings for {len(embeddings)} entities")
            except Exception as e:
                logger.warning(f"Batch embedding failed: {e}")

        self.world_model.rebuild_embedding_matrix()
    
    def _build_retriever(self):
        """Build the hybrid retriever after ingestion."""
        from pie.retrieval.hybrid_retriever import HybridRetriever
        from pie.config import PIEConfig
        self._retriever = HybridRetriever(self.world_model, self.llm, PIEConfig())

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        """Search world model using hybrid retrieval (BM25 + dense + RRF)."""
        if not self.world_model.entities:
            return []

        if self._retriever is None:
            self._build_retriever()

        entity_ids = self._retriever.retrieve(query, top_k=top_k)
        results = []
        for eid in entity_ids:
            entity = self.world_model.entities.get(eid)
            if entity:
                state_desc = ""
                if isinstance(entity.current_state, dict):
                    state_desc = entity.current_state.get("description", str(entity.current_state)[:300])
                else:
                    state_desc = str(entity.current_state)[:300]
                results.append(SearchResult(
                    content=f"{entity.name} ({entity.type}): {state_desc}",
                    score=1.0 / (1 + entity_ids.index(eid)),  # rank-based score
                    metadata={"entity_id": entity.id, "type": entity.type},
                ))
        return results

    def answer(self, question: str, question_date: str | None = None) -> str:
        """Answer using PIE's hybrid retrieval + full temporal context compilation."""
        if not self.world_model.entities:
            return "I don't have enough information to answer this question."

        if self._retriever is None:
            self._build_retriever()

        now_dt = None
        if question_date:
            try:
                from datetime import datetime
                now_dt = datetime.fromisoformat(question_date)
            except (ValueError, TypeError):
                pass

        entity_ids = self._retriever.retrieve(question, top_k=15, now=now_dt)
        if not entity_ids:
            return "I don't have enough information to answer this question."

        context = self._retriever.compile_context(entity_ids, query=question, now=now_dt)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a personal knowledge assistant. Answer the question using "
                    "ONLY the provided context. Use temporal information (history, "
                    "contradictions, change dates) where relevant. Be concise."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Context:\n\n{context}\n\n---\n\nQuestion: {question}"
                    + (f"\n(Question date: {question_date})" if question_date else "")
                ),
            },
        ]

        response = self.llm.chat(
            messages=messages,
            model=self.config.model,
            max_tokens=400,
        )
        return (response.get("content") or "").strip()
    
    def stats(self) -> MemoryStats:
        """Get PIE world model statistics."""
        return MemoryStats(
            num_memories=len(self.world_model.entities),
            num_entities=len(self.world_model.entities),
            num_relationships=len(self.world_model.relationships),
            extra=self.world_model.stats,
        )
    
    def clear(self) -> None:
        """Clear world model."""
        self.world_model = type(self.world_model)()
        self._ingested = False
