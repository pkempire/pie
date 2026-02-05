"""
Ingestion Pipeline — the main orchestrator.

Groups conversations into daily batches, builds activity context,
extracts entities/relationships/state changes, resolves entities,
and updates the world model.
"""

from __future__ import annotations
import time
import logging
import datetime
from pathlib import Path

from pie.config import PIEConfig
from pie.core.models import (
    Entity, EntityType, RelationshipType, ExtractionResult,
    DailyBatch, Conversation,
)
from pie.core.parser import parse_conversations, group_into_daily_batches, get_stats
from pie.core.world_model import WorldModel
from pie.core.llm import LLMClient, parse_extraction_result
from pie.ingestion.prompts import (
    EXTRACTION_SYSTEM_PROMPT,
    build_extraction_user_message,
    format_conversations_for_extraction,
)
from pie.resolution.resolver import EntityResolver, ResolvedEntity
from pie.resolution.web_grounder import WebGrounder

logger = logging.getLogger("pie")


class IngestionPipeline:
    """
    Main ingestion pipeline. Processes conversations chronologically
    with sliding window context and hybrid entity resolution.
    """
    
    def __init__(self, config: PIEConfig, web_search_fn=None):
        self.config = config
        
        # Core components
        self.llm = LLMClient()
        self.world_model = WorldModel(
            persist_path=config.output_dir / "world_model.json"
        )
        self.resolver = EntityResolver(
            world_model=self.world_model,
            llm=self.llm,
            config=config.resolution,
        )
        self.web_grounder = WebGrounder(
            search_fn=web_search_fn if config.use_web_grounding else None
        )
        
        # Progress tracking
        self._batches_processed = 0
        self._conversations_processed = 0
        self._errors: list[dict] = []
    
    def run(
        self,
        conversations_path: str | Path | None = None,
        year_min: int | None = None,
        limit_batches: int | None = None,
        limit_conversations: int | None = None,
        save_every: int = 10,
        skip_batches: int = 0,
    ):
        """
        Run the full ingestion pipeline.
        
        Args:
            conversations_path: Path to conversations.json (defaults to config)
            year_min: Override minimum year filter
            limit_batches: Stop after N batches (for testing)
            limit_conversations: Only parse first N conversations
            save_every: Save world model every N batches
            skip_batches: Skip the first N batches (for resuming)
        """
        path = conversations_path or self.config.conversations_path
        year = year_min or self.config.ingestion.year_min
        
        # Phase 1: Parse
        logger.info(f"Parsing conversations from {path}...")
        conversations = parse_conversations(path, year_min=year)
        
        if limit_conversations:
            conversations = conversations[:limit_conversations]
        
        stats = get_stats(conversations)
        logger.info(f"Parsed {stats['count']} conversations: {stats['date_range']}")
        logger.info(f"  Avg {stats['avg_turns']:.0f} turns, {stats['avg_chars']:.0f} chars per conversation")
        
        # Phase 2: Group into daily batches
        batches = group_into_daily_batches(conversations)
        logger.info(f"Grouped into {len(batches)} daily batches")
        
        if skip_batches:
            batches = batches[skip_batches:]
            logger.info(f"  (skipping first {skip_batches} batches, resuming from batch {skip_batches + 1})")
        
        if limit_batches:
            batches = batches[:limit_batches]
            logger.info(f"  (limited to {limit_batches} batches for testing)")
        
        # Phase 3: Process each batch chronologically
        total = len(batches)
        t0 = time.time()
        
        for i, batch in enumerate(batches):
            t_batch = time.time()
            
            logger.info(
                f"[{i+1}/{total}] {batch.date} — "
                f"{len(batch.conversations)} conversations, "
                f"{batch.total_turns} turns, {batch.total_chars:,} chars"
            )
            
            try:
                self._process_batch(batch)
                self._batches_processed += 1
                self._conversations_processed += len(batch.conversations)
                
                elapsed = time.time() - t_batch
                logger.info(
                    f"  → Processed in {elapsed:.1f}s | "
                    f"World model: {self.world_model.stats['entities']} entities, "
                    f"{self.world_model.stats['transitions']} transitions, "
                    f"{self.world_model.stats['relationships']} relationships"
                )
                
            except Exception as e:
                logger.error(f"  → ERROR processing batch {batch.date}: {e}")
                self._errors.append({
                    "batch_date": batch.date,
                    "error": str(e),
                    "conversations": [c.id for c in batch.conversations],
                })
                continue
            
            # Periodic save
            if (i + 1) % save_every == 0:
                logger.info(f"  Saving checkpoint...")
                self.world_model.save()
        
        # Final save
        self.world_model.save()
        
        total_time = time.time() - t0
        self._print_summary(total_time)
    
    def _process_batch(self, batch: DailyBatch):
        """Process a single daily batch through extraction and resolution."""
        
        # Step 1: Build context preamble from current world model state
        batch_timestamp = batch.conversations[0].created_at
        context_preamble = ""
        if self.config.use_sliding_window and len(self.world_model.entities) > 0:
            context_preamble = self.world_model.build_context_preamble(batch_timestamp)
        
        # Step 2: Format conversations for extraction
        conversations_text = format_conversations_for_extraction(
            batch.conversations,
            max_chars_per_turn=self.config.ingestion.max_chars_per_turn,
            max_turns_per_conversation=self.config.ingestion.max_turns_per_conversation,
        )
        
        # Step 3: Run extraction
        user_message = build_extraction_user_message(
            batch_date=batch.date,
            conversations_text=conversations_text,
            context_preamble=context_preamble,
            num_conversations=len(batch.conversations),
        )
        
        result = self.llm.chat(
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            model=self.config.llm.extraction_model,
            json_mode=True,
        )
        
        extraction = parse_extraction_result(
            raw=result["content"],
            conversation_ids=[c.id for c in batch.conversations],
            tokens=result["tokens"],
        )
        
        logger.info(
            f"  Extracted: {len(extraction.entities)} entities, "
            f"{len(extraction.state_changes)} state changes, "
            f"{len(extraction.relationships)} relationships, "
            f"significance={extraction.significance:.2f} "
            f"({result['tokens']['total']} tokens)"
        )
        
        # Step 4: Resolve entities
        resolved = self.resolver.resolve(extraction.entities)
        
        creates = sum(1 for r in resolved if r.action == "create")
        updates = sum(1 for r in resolved if r.action == "update")
        deduped = sum(1 for r in resolved if r.action == "intra_batch_dedup")
        if deduped:
            logger.info(f"  Resolved: {creates} new, {updates} matched existing, {deduped} intra-batch dedup")
        else:
            logger.info(f"  Resolved: {creates} new, {updates} matched existing")
        
        # Step 5: Web ground new entities
        if self.config.use_web_grounding:
            for r in resolved:
                if r.action == "create" and r.extracted.type in self.config.resolution.web_ground_entity_types:
                    grounding = self.web_grounder.ground(r.extracted)
                    if grounding.verified:
                        r.web_grounding = {
                            "canonical_name": grounding.canonical_name,
                            "description": grounding.description,
                            "url": grounding.url,
                        }
        
        # Step 6: Update world model
        self._apply_to_world_model(batch, extraction, resolved)
    
    def _apply_to_world_model(
        self,
        batch: DailyBatch,
        extraction: ExtractionResult,
        resolved: list[ResolvedEntity],
    ):
        """Apply extraction results to the world model."""
        batch_timestamp = batch.conversations[0].created_at
        convo_ids = [c.id for c in batch.conversations]
        
        # Track entity name → ID mapping for relationship creation
        name_to_id: dict[str, str] = {}
        
        # Batch-compute embeddings for new entities
        new_entities_for_embed = []
        new_embed_texts = []
        for r in resolved:
            if r.action == "create":
                state = r.extracted.state if isinstance(r.extracted.state, dict) else {"description": str(r.extracted.state)}
                new_entities_for_embed.append(r)
                new_embed_texts.append(f"{r.extracted.name} ({r.extracted.type}): {state.get('description', str(state))}")
        
        new_embeddings = {}
        if new_embed_texts:
            try:
                embeddings = self.llm.embed(new_embed_texts)
                for r, emb in zip(new_entities_for_embed, embeddings):
                    new_embeddings[r.extracted.name] = emb
            except Exception as e:
                logger.warning(f"Batch embedding for new entities failed: {e}")
        
        # Create/update entities
        for r in resolved:
            entity_type = _parse_entity_type(r.extracted.type)
            state = r.extracted.state if isinstance(r.extracted.state, dict) else {"description": str(r.extracted.state)}
            
            if r.action == "create":
                embedding = new_embeddings.get(r.extracted.name)
                
                # Apply web grounding if available
                aliases = []
                if r.web_grounding and r.web_grounding.get("canonical_name"):
                    canonical = r.web_grounding["canonical_name"]
                    if canonical.lower() != r.extracted.name.lower():
                        aliases = [r.extracted.name]
                        name = canonical
                    else:
                        name = r.extracted.name
                else:
                    name = r.extracted.name
                
                entity = self.world_model.create_entity(
                    name=name,
                    type=entity_type,
                    state=state,
                    source_conversation_id=convo_ids[0] if convo_ids else None,
                    timestamp=batch_timestamp,
                    aliases=aliases,
                    embedding=embedding,
                )
                
                # Apply web grounding metadata
                if r.web_grounding:
                    entity.web_canonical_name = r.web_grounding.get("canonical_name")
                    entity.web_description = r.web_grounding.get("description")
                    entity.web_verified = True
                
                name_to_id[r.extracted.name.lower()] = entity.id
                
            elif r.action == "intra_batch_dedup":
                # This entity was deduped against another new entity in the same batch
                target_idx = getattr(r, '_dedup_target_idx', None)
                if target_idx is not None and target_idx < len(resolved):
                    target_name = resolved[target_idx].extracted.name.lower()
                    target_id = name_to_id.get(target_name)
                    if target_id:
                        existing = self.world_model.get_entity(target_id)
                        if existing:
                            self.world_model.add_alias(target_id, r.extracted.name)
                            self.world_model.update_entity_state(
                                entity_id=target_id,
                                new_state=state,
                                source_conversation_id=convo_ids[0] if convo_ids else "",
                                timestamp=batch_timestamp,
                                trigger_summary=f"Merged from intra-batch dedup: {r.extracted.name}",
                            )
                            name_to_id[r.extracted.name.lower()] = target_id

            elif r.action == "update" and r.matched_entity_id:
                existing = self.world_model.get_entity(r.matched_entity_id)
                if existing:
                    self.world_model.update_entity_state(
                        entity_id=r.matched_entity_id,
                        new_state=state,
                        source_conversation_id=convo_ids[0] if convo_ids else "",
                        timestamp=batch_timestamp,
                        trigger_summary=f"Updated from {batch.date} batch",
                    )
                    name_to_id[r.extracted.name.lower()] = r.matched_entity_id
                    if existing.name:
                        name_to_id[existing.name.lower()] = r.matched_entity_id
        
        # Apply state changes
        for sc in extraction.state_changes:
            entity = self.world_model.find_by_name(sc.entity_name)
            if not entity and sc.entity_name.lower() in name_to_id:
                entity = self.world_model.get_entity(name_to_id[sc.entity_name.lower()])
            
            if entity:
                self.world_model.update_entity_state(
                    entity_id=entity.id,
                    new_state={"description": sc.new_state} if isinstance(sc.new_state, str) else {},
                    source_conversation_id=convo_ids[0] if convo_ids else "",
                    timestamp=batch_timestamp,
                    trigger_summary=sc.what_changed,
                    is_contradiction=sc.is_contradiction,
                )
        
        # Apply relationships
        for rel in extraction.relationships:
            source_id = self._find_entity_id(rel.source, name_to_id)
            target_id = self._find_entity_id(rel.target, name_to_id)
            
            if source_id and target_id:
                rel_type = _parse_relationship_type(rel.type)
                self.world_model.add_relationship(
                    source_id=source_id,
                    target_id=target_id,
                    rel_type=rel_type,
                    description=rel.description,
                    source_conversation_id=convo_ids[0] if convo_ids else None,
                    timestamp=batch_timestamp,
                )
    
    def _find_entity_id(self, name: str, name_to_id: dict[str, str]) -> str | None:
        """Find entity ID by name, checking local mapping then world model."""
        if name.lower() in name_to_id:
            return name_to_id[name.lower()]
        entity = self.world_model.find_by_name(name)
        return entity.id if entity else None
    
    def _print_summary(self, total_time: float):
        """Print final summary after ingestion."""
        logger.info("\n" + "=" * 60)
        logger.info("INGESTION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Time: {total_time:.0f}s ({total_time/60:.1f}m)")
        logger.info(f"Batches: {self._batches_processed}")
        logger.info(f"Conversations: {self._conversations_processed}")
        logger.info(f"Errors: {len(self._errors)}")
        logger.info(f"\nWorld Model: {self.world_model.stats}")
        logger.info(f"LLM Stats: {self.llm.stats}")
        logger.info(f"Resolution Stats: {self.resolver.stats}")
        if self.config.use_web_grounding:
            logger.info(f"Web Grounding Stats: {self.web_grounder.stats}")
        logger.info("=" * 60)


def _parse_entity_type(type_str: str) -> EntityType:
    """Safely parse entity type string to enum."""
    try:
        return EntityType(type_str.lower())
    except ValueError:
        return EntityType.CONCEPT


def _parse_relationship_type(type_str: str) -> RelationshipType:
    """Safely parse relationship type string to enum."""
    try:
        return RelationshipType(type_str.lower())
    except ValueError:
        return RelationshipType.RELATED_TO
