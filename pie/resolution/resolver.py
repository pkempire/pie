"""
Hybrid Entity Resolution — the three-tier matching system.

Tier 1: Exact/fuzzy string match (free, fast)
Tier 2: Embedding similarity (cheap, catches semantic matches)
Tier 3: LLM verification (expensive, only for ambiguous cases)

Plus optional web grounding for new tool/org/concept entities.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass

from pie.core.models import (
    Entity, EntityType, ExtractedEntity,
)
from pie.core.world_model import WorldModel
from pie.core.llm import LLMClient
from pie.config import ResolutionConfig

logger = logging.getLogger("pie.resolution")


@dataclass
class ResolvedEntity:
    """Result of entity resolution."""
    extracted: ExtractedEntity
    matched_entity_id: str | None = None  # existing entity ID if matched
    action: str = "create"                # "create" or "update"
    match_method: str = ""                # "string", "embedding", "llm", "extraction_hint"
    match_score: float = 0.0
    web_grounding: dict | None = None     # web verification data


class EntityResolver:
    """
    Resolves extracted entities against the world model.
    Three-tier: string match → embedding → LLM verify.
    """
    
    def __init__(
        self,
        world_model: WorldModel,
        llm: LLMClient,
        config: ResolutionConfig,
    ):
        self.world_model = world_model
        self.llm = llm
        self.config = config
        self._resolution_stats = {
            "string_matches": 0,
            "embedding_matches": 0,
            "llm_matches": 0,
            "new_entities": 0,
            "extraction_hint_matches": 0,
        }
    
    @property
    def stats(self) -> dict:
        return self._resolution_stats.copy()
    
    def resolve(self, entities: list[ExtractedEntity]) -> list[ResolvedEntity]:
        """Resolve a batch of extracted entities against the world model."""
        # Pre-compute embeddings in batch for efficiency
        self._precompute_embeddings(entities)
        
        resolved = []
        # Track new entities created within this batch for intra-batch dedup
        batch_new_names: dict[str, int] = {}  # normalized_name -> index in resolved list
        
        for entity in entities:
            result = self._resolve_single(entity)
            
            # Intra-batch dedup: if this resolved as "new", check against
            # other new entities already resolved in this batch
            if result.action == "create":
                from pie.core.world_model import _normalize, _fuzzy_ratio
                norm_name = _normalize(entity.name)
                
                merged = False
                for known_name, idx in batch_new_names.items():
                    score = _fuzzy_ratio(entity.name, resolved[idx].extracted.name)
                    # Containment check with safety: min 4 chars, word-like
                    norm_known = _normalize(resolved[idx].extracted.name)
                    shorter_len = min(len(norm_name), len(norm_known))
                    longer_len = max(len(norm_name), len(norm_known))
                    if shorter_len >= 4 and shorter_len / longer_len >= 0.3:
                        if norm_name in norm_known or norm_known in norm_name:
                            score = max(score, 0.87)
                    
                    if score >= 0.85 and entity.type == resolved[idx].extracted.type:
                        logger.info(f"  Intra-batch dedup: '{entity.name}' -> '{resolved[idx].extracted.name}' (score={score:.2f})")
                        # Convert this to an update of the first entity
                        result = ResolvedEntity(
                            extracted=entity,
                            matched_entity_id=None,  # will be set when first entity is added
                            action="intra_batch_dedup",
                            match_method="intra_batch_string",
                            match_score=score,
                        )
                        result._dedup_target_idx = idx
                        merged = True
                        self._resolution_stats["string_matches"] += 1
                        break
                
                if not merged:
                    batch_new_names[norm_name] = len(resolved)
            
            resolved.append(result)
        return resolved
    
    def _precompute_embeddings(self, entities: list[ExtractedEntity]):
        """Batch-compute embeddings for all entities at once."""
        texts = []
        for e in entities:
            state_desc = e.state.get("description", str(e.state)) if isinstance(e.state, dict) else str(e.state)
            texts.append(f"{e.name} ({e.type}): {state_desc}")
        
        if not texts:
            return
        
        try:
            embeddings = self.llm.embed(texts)
            self._embedding_cache = {}
            for e, emb in zip(entities, embeddings):
                self._embedding_cache[e.name] = emb
        except Exception as ex:
            logger.warning(f"Batch embedding failed: {ex}")
            self._embedding_cache = {}
    
    def _resolve_single(self, extracted: ExtractedEntity) -> ResolvedEntity:
        """Resolve a single extracted entity."""

        # If the extraction LLM already identified a match, try that first
        if extracted.matches_existing:
            existing = self.world_model.find_by_name(extracted.matches_existing)
            if existing:
                # TYPE GUARD: extraction hints can be wrong, especially with
                # accumulated aliases pointing to the wrong entity.
                types_match = extracted.type == existing.type.value
                if types_match:
                    self._resolution_stats["extraction_hint_matches"] += 1
                    return ResolvedEntity(
                        extracted=extracted,
                        matched_entity_id=existing.id,
                        action="update",
                        match_method="extraction_hint",
                        match_score=1.0,
                    )
                else:
                    # Cross-type hint — don't blindly trust.
                    # Verify via LLM before accepting.
                    logger.info(
                        f"  Extraction hint cross-type: '{extracted.name}' ({extracted.type}) "
                        f"hinted at '{existing.name}' ({existing.type.value}) — LLM verify"
                    )
                    is_match = self._llm_verify_match(extracted, existing)
                    if is_match:
                        self._resolution_stats["extraction_hint_matches"] += 1
                        return ResolvedEntity(
                            extracted=extracted,
                            matched_entity_id=existing.id,
                            action="update",
                            match_method="extraction_hint_verified",
                            match_score=0.9,
                        )
                    # Fall through to normal resolution if LLM says no

        # Tier 1: String match — ALWAYS filter by type (no exceptions for concepts)
        string_matches = self.world_model.find_by_string_match(
            name=extracted.name,
            threshold=self.config.string_match_threshold,
            entity_type=extracted.type,
        )

        if string_matches:
            best_entity, best_score = string_matches[0]
            types_match = extracted.type == best_entity.type.value

            if best_score >= 0.95:
                # Near-exact name match — accept regardless of type
                self._resolution_stats["string_matches"] += 1
                if extracted.name.lower() != best_entity.name.lower():
                    self.world_model.add_alias(best_entity.id, extracted.name)
                return ResolvedEntity(
                    extracted=extracted,
                    matched_entity_id=best_entity.id,
                    action="update",
                    match_method="string",
                    match_score=best_score,
                )
            elif best_score >= 0.90 and types_match:
                # High confidence + same type — accept
                self._resolution_stats["string_matches"] += 1
                if extracted.name.lower() != best_entity.name.lower():
                    self.world_model.add_alias(best_entity.id, extracted.name)
                return ResolvedEntity(
                    extracted=extracted,
                    matched_entity_id=best_entity.id,
                    action="update",
                    match_method="string",
                    match_score=best_score,
                )
            elif best_score >= 0.90 and not types_match:
                # High string match but cross-type — escalate to LLM
                logger.info(
                    f"  String match cross-type: '{extracted.name}' ({extracted.type}) "
                    f"vs '{best_entity.name}' ({best_entity.type.value}) score={best_score:.2f} — LLM verify"
                )
                is_match = self._llm_verify_match(extracted, best_entity)
                if is_match:
                    self._resolution_stats["llm_matches"] += 1
                    if extracted.name.lower() != best_entity.name.lower():
                        self.world_model.add_alias(best_entity.id, extracted.name)
                    return ResolvedEntity(
                        extracted=extracted,
                        matched_entity_id=best_entity.id,
                        action="update",
                        match_method="string+llm",
                        match_score=best_score,
                    )
            elif best_score >= self.config.string_match_threshold:
                # Good string match but not perfect — could be a different entity with similar name
                # Fall through to embedding check for confirmation
                pass
        
        # Tier 2: Embedding similarity
        # Use pre-computed embedding from batch, or compute on the fly
        embedding = getattr(self, '_embedding_cache', {}).get(extracted.name)
        if embedding is None:
            state_desc = extracted.state.get("description", str(extracted.state)) if isinstance(extracted.state, dict) else str(extracted.state)
            embed_text = f"{extracted.name} ({extracted.type}): {state_desc}"
            try:
                embedding = self.llm.embed_single(embed_text)
            except Exception as e:
                logger.warning(f"Embedding failed for {extracted.name}: {e}")
                embedding = None
        
        if embedding:
            embedding_matches = self.world_model.find_by_embedding(
                embedding=embedding,
                top_k=3,
                entity_type=extracted.type,
            )
            
            if embedding_matches:
                best_entity, best_sim = embedding_matches[0]
                
                if best_sim >= self.config.embedding_accept_threshold:
                    # High similarity — but verify if types differ (common source of false merges)
                    types_match = extracted.type == best_entity.type.value
                    if types_match or not getattr(self.config, 'require_llm_for_cross_type', True):
                        self._resolution_stats["embedding_matches"] += 1
                        if extracted.name.lower() != best_entity.name.lower():
                            self.world_model.add_alias(best_entity.id, extracted.name)
                        return ResolvedEntity(
                            extracted=extracted,
                            matched_entity_id=best_entity.id,
                            action="update",
                            match_method="embedding",
                            match_score=best_sim,
                        )
                    else:
                        # Cross-type match above threshold — escalate to LLM anyway
                        logger.info(f"  Cross-type high-sim match: {extracted.name} ({extracted.type}) vs {best_entity.name} ({best_entity.type.value}) — LLM verify")
                        is_match = self._llm_verify_match(extracted, best_entity)
                        if is_match:
                            self._resolution_stats["llm_matches"] += 1
                            if extracted.name.lower() != best_entity.name.lower():
                                self.world_model.add_alias(best_entity.id, extracted.name)
                            return ResolvedEntity(
                                extracted=extracted,
                                matched_entity_id=best_entity.id,
                                action="update",
                                match_method="llm_cross_type",
                                match_score=best_sim,
                            )
                
                elif best_sim >= self.config.embedding_reject_threshold:
                    # Ambiguous zone — need LLM verification (Tier 3)
                    # Combine with string match evidence if available
                    string_score = string_matches[0][1] if string_matches else 0
                    
                    if string_score >= self.config.string_match_threshold:
                        # String match + embedding both suggest a match — accept
                        self._resolution_stats["embedding_matches"] += 1
                        if extracted.name.lower() != best_entity.name.lower():
                            self.world_model.add_alias(best_entity.id, extracted.name)
                        return ResolvedEntity(
                            extracted=extracted,
                            matched_entity_id=best_entity.id,
                            action="update",
                            match_method="embedding+string",
                            match_score=best_sim,
                        )
                    
                    # Tier 3: LLM verification
                    is_match = self._llm_verify_match(extracted, best_entity)
                    if is_match:
                        self._resolution_stats["llm_matches"] += 1
                        if extracted.name.lower() != best_entity.name.lower():
                            self.world_model.add_alias(best_entity.id, extracted.name)
                        return ResolvedEntity(
                            extracted=extracted,
                            matched_entity_id=best_entity.id,
                            action="update",
                            match_method="llm",
                            match_score=best_sim,
                        )
        
        # No match found — this is a new entity
        self._resolution_stats["new_entities"] += 1
        return ResolvedEntity(
            extracted=extracted,
            matched_entity_id=None,
            action="create",
            match_method="",
            match_score=0.0,
        )
    
    def _llm_verify_match(self, extracted: ExtractedEntity, candidate: Entity) -> bool:
        """Ask the LLM to verify if an extracted entity matches an existing one.

        Uses a strict prompt that defaults to "no" for ambiguous cases.
        """
        state_desc_new = extracted.state.get("description", str(extracted.state)) if isinstance(extracted.state, dict) else str(extracted.state)
        state_desc_existing = candidate.current_state.get("description", str(candidate.current_state))

        # Limit alias display to prevent prompt bloat from corrupted entities
        aliases_display = ', '.join(candidate.aliases[:5]) if candidate.aliases else 'none'
        if len(candidate.aliases) > 5:
            aliases_display += f" (+{len(candidate.aliases) - 5} more)"

        prompt = f"""Are Entity A and Entity B referring to the SAME real-world thing?

IMPORTANT: Only answer "yes" if they are clearly the same entity — same name, same concept, same project, same person. Two things that are merely RELATED (e.g., "React" and "React Native", "Python" and "PyTorch") are NOT the same entity. If the entity types differ (e.g., one is a "tool" and one is a "concept"), that's strong evidence they are DIFFERENT entities.

When in doubt, answer "no". It is much better to create a duplicate than to merge unrelated things.

Entity A (newly extracted):
- Name: {extracted.name}
- Type: {extracted.type}
- Description: {state_desc_new[:500]}

Entity B (existing in knowledge graph):
- Name: {candidate.name}
- Type: {candidate.type.value}
- First 5 aliases: {aliases_display}
- Description: {state_desc_existing[:500]}

Answer with ONLY "yes" or "no":"""

        try:
            result = self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                model="gpt-4o-mini",
                temperature=0.0,
            )
            answer = result["content"].strip().lower()
            is_match = answer.startswith("yes")
            if is_match:
                logger.info(f"  LLM verified match: '{extracted.name}' <-> '{candidate.name}'")
            else:
                logger.info(f"  LLM rejected match: '{extracted.name}' <-> '{candidate.name}'")
            return is_match
        except Exception as e:
            logger.warning(f"LLM verify failed: {e}")
            return False  # Default to "no" on failure
