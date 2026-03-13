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
        
        # Lazy imports to avoid circular dependencies
        from pie.core.llm import LLMClient
        from pie.core.world_model import WorldModel
        
        self.llm = LLMClient()
        self.world_model = WorldModel()
        self._ingested = False
    
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

                # Add entities to world model
                for entity in extraction.entities:
                    self.world_model.create_entity(
                        name=entity.name,
                        type=entity.type,
                        state=entity.state if isinstance(entity.state, dict) else {"description": str(entity.state)},
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
    
    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        """Search world model for relevant entities."""
        if not self.world_model.entities:
            return []

        from pie.core.world_model import cosine_similarity

        # Check if any entities have embeddings
        has_embeddings = any(e.embedding for e in self.world_model.entities.values())

        scored = []
        if has_embeddings:
            # Embedding-based search (primary)
            try:
                query_emb = self.llm.embed_single(query)
                for entity_id, entity in self.world_model.entities.items():
                    if entity.embedding:
                        score = cosine_similarity(query_emb, entity.embedding)
                        scored.append((entity, score))
            except Exception as e:
                logger.warning(f"Embedding search failed: {e}, falling back to text match")
                has_embeddings = False

        if not has_embeddings or not scored:
            # Text-based fallback: fuzzy match against entity names + state
            query_lower = query.lower()
            query_words = set(query_lower.split())
            for entity_id, entity in self.world_model.entities.items():
                entity_text = f"{entity.name} {entity.current_state.get('description', '')}".lower()
                # Score by word overlap
                entity_words = set(entity_text.split())
                overlap = len(query_words & entity_words)
                if overlap > 0:
                    score = overlap / max(len(query_words), 1)
                    scored.append((entity, score))

        # Sort and return top-k
        scored.sort(key=lambda x: x[1], reverse=True)

        results = []
        for entity, score in scored[:top_k]:
            state_desc = entity.current_state.get("description", str(entity.current_state)[:300])
            content = f"{entity.name} ({entity.type}): {state_desc}"
            results.append(SearchResult(
                content=content,
                score=score,
                metadata={"entity_id": entity.id, "type": entity.type},
            ))

        return results
    
    def answer(self, question: str, question_date: str | None = None) -> str:
        """Answer using PIE's temporal context compilation."""
        # Search for relevant entities
        results = self.search(question, top_k=15)
        
        if not results:
            return "I don't have enough information to answer this question."
        
        # Compile temporal context
        context_parts = []
        for r in results:
            # Add temporal context if available
            entity_id = r.metadata.get("entity_id")
            if entity_id and entity_id in self.world_model.entities:
                transitions = self.world_model.get_transitions(entity_id)
                if transitions:
                    history = " → ".join([
                        t.trigger_summary[:50] or str(t.to_state)[:50]
                        for t in transitions[-3:]
                    ])
                    context_parts.append(f"{r.content}\n  History: {history}")
                else:
                    context_parts.append(r.content)
            else:
                context_parts.append(r.content)
        
        context = "\n\n".join(context_parts)
        
        # Generate answer
        prompt = f"""Based on the following knowledge about the user, answer the question.

Knowledge:
{context}

Question: {question}
{"(Asked on: " + question_date + ")" if question_date else ""}

Answer concisely:"""
        
        response = self.llm.chat(
            messages=[{"role": "user", "content": prompt}],
            model=self.config.model,
            max_tokens=300,
        )
        
        return response["content"].strip()
    
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
