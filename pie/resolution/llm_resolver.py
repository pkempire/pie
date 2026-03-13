"""
LLM-Native Entity Resolution — No hardcoded thresholds.

Instead of: 
    if similarity > 0.85: match
    elif similarity > 0.70: maybe
    else: no match
    
We do:
    LLM decides if it's a match, given both entities.
    
This is more expensive but eliminates arbitrary thresholds entirely.
The LLM sees the full context and makes a judgment call.

For efficiency, we still use embeddings for CANDIDATE RETRIEVAL,
but the DECISION is always made by the LLM.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from pie.core.models import Entity, EntityType, ExtractedEntity
from pie.core.llm import LLMClient

logger = logging.getLogger("pie.resolution.llm")


@dataclass
class ResolutionResult:
    """Result of LLM-based entity resolution."""
    extracted: ExtractedEntity
    matched_entity: Optional[Entity]
    action: str  # "create" | "update" | "merge"
    confidence: float  # LLM's stated confidence
    reasoning: str  # LLM's explanation


RESOLUTION_PROMPT = """You are resolving whether a newly extracted entity matches an existing entity in a knowledge graph.

## Extracted Entity (from current conversation)
Name: {new_name}
Type: {new_type}
State: {new_state}

## Candidate Entity (from knowledge graph)
Name: {existing_name}
Type: {existing_type}
Aliases: {existing_aliases}
State: {existing_state}
First seen: {existing_first_seen}
Last seen: {existing_last_seen}

## Task
Determine if these refer to the SAME real-world entity.

Consider:
- Could be same entity with different names (nicknames, abbreviations, typos)
- Could be same entity with evolved state (project pivoted, person changed roles)
- Could be DIFFERENT entities with similar names (React the library vs React the concept)

Respond with JSON:
{{
    "same_entity": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "one sentence explanation"
}}"""


class LLMEntityResolver:
    """
    Pure LLM-based entity resolution.
    
    Flow:
    1. Embed extracted entity
    2. Find top-k candidates by embedding similarity (cheap retrieval)
    3. For each candidate, ask LLM: same entity? (expensive decision)
    4. Take highest-confidence match, or create new if none confident
    
    No thresholds except:
    - How many candidates to consider (top_k)
    - Minimum LLM confidence to accept match
    
    Both of these are naturally interpretable.
    """
    
    def __init__(
        self,
        world_model,  # WorldModel
        llm: LLMClient,
        candidate_top_k: int = 5,
        min_match_confidence: float = 0.7,  # LLM's stated confidence
    ):
        self.world_model = world_model
        self.llm = llm
        self.candidate_top_k = candidate_top_k
        self.min_match_confidence = min_match_confidence
    
    def resolve(self, extracted: ExtractedEntity) -> ResolutionResult:
        """Resolve a single extracted entity."""
        
        # 1. Get embedding for extracted entity
        state_desc = self._format_state(extracted.state)
        embed_text = f"{extracted.name} ({extracted.type}): {state_desc}"
        
        try:
            embedding = self.llm.embed_single(embed_text)
        except Exception as e:
            logger.warning(f"Embedding failed: {e}")
            # Can't find candidates without embedding — create new
            return ResolutionResult(
                extracted=extracted,
                matched_entity=None,
                action="create",
                confidence=1.0,
                reasoning="Could not compute embedding for candidate search",
            )
        
        # 2. Find candidate entities by embedding similarity
        candidates = self.world_model.find_by_embedding(
            embedding=embedding,
            top_k=self.candidate_top_k,
            entity_type=None,  # Don't filter by type — LLM decides
        )
        
        if not candidates:
            return ResolutionResult(
                extracted=extracted,
                matched_entity=None,
                action="create",
                confidence=1.0,
                reasoning="No similar entities found in knowledge graph",
            )
        
        # 3. Ask LLM about each candidate
        best_match: Optional[tuple[Entity, float, str]] = None
        
        for entity, sim_score in candidates:
            result = self._llm_compare(extracted, entity)
            
            if result["same_entity"] and result["confidence"] >= self.min_match_confidence:
                if best_match is None or result["confidence"] > best_match[1]:
                    best_match = (entity, result["confidence"], result["reasoning"])
        
        # 4. Return result
        if best_match:
            entity, confidence, reasoning = best_match
            return ResolutionResult(
                extracted=extracted,
                matched_entity=entity,
                action="update",
                confidence=confidence,
                reasoning=reasoning,
            )
        else:
            return ResolutionResult(
                extracted=extracted,
                matched_entity=None,
                action="create",
                confidence=1.0,
                reasoning="No candidates matched with sufficient confidence",
            )
    
    def _llm_compare(self, extracted: ExtractedEntity, existing: Entity) -> dict:
        """Ask LLM if two entities are the same."""
        from datetime import datetime
        
        prompt = RESOLUTION_PROMPT.format(
            new_name=extracted.name,
            new_type=extracted.type,
            new_state=self._format_state(extracted.state),
            existing_name=existing.name,
            existing_type=existing.type.value,
            existing_aliases=", ".join(existing.aliases) if existing.aliases else "none",
            existing_state=self._format_state(existing.current_state),
            existing_first_seen=datetime.fromtimestamp(existing.first_seen).strftime("%Y-%m-%d"),
            existing_last_seen=datetime.fromtimestamp(existing.last_seen).strftime("%Y-%m-%d"),
        )
        
        try:
            response = self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            
            import json
            result = json.loads(response["content"])
            return {
                "same_entity": result.get("same_entity", False),
                "confidence": float(result.get("confidence", 0.0)),
                "reasoning": result.get("reasoning", ""),
            }
        except Exception as e:
            logger.warning(f"LLM compare failed: {e}")
            return {"same_entity": False, "confidence": 0.0, "reasoning": str(e)}
    
    def _format_state(self, state) -> str:
        """Format state dict as readable string."""
        if isinstance(state, dict):
            parts = [f"{k}: {v}" for k, v in state.items() if k != "embedding"]
            return "; ".join(parts) if parts else "none"
        return str(state) if state else "none"


# =============================================================================
# Batch resolver with caching
# =============================================================================

class BatchLLMResolver:
    """
    Batch resolution with LLM decision caching.
    
    Caches LLM decisions to avoid re-asking about the same pairs.
    Also deduplicates within a batch before resolution.
    """
    
    def __init__(self, resolver: LLMEntityResolver):
        self.resolver = resolver
        self._decision_cache: dict[tuple[str, str], dict] = {}  # (new_name, existing_id) -> result
    
    def resolve_batch(self, entities: list[ExtractedEntity]) -> list[ResolutionResult]:
        """Resolve a batch of entities."""
        results = []
        
        # Track new entities created in this batch for intra-batch dedup
        batch_new: dict[str, int] = {}  # normalized_name -> index in results
        
        for entity in entities:
            # Check intra-batch duplicate first
            norm_name = entity.name.lower().strip()
            if norm_name in batch_new:
                # Duplicate within batch — merge with first occurrence
                first_idx = batch_new[norm_name]
                results.append(ResolutionResult(
                    extracted=entity,
                    matched_entity=None,
                    action="intra_batch_merge",
                    confidence=1.0,
                    reasoning=f"Duplicate of {results[first_idx].extracted.name} in same batch",
                ))
                continue
            
            # Resolve against world model
            result = self.resolver.resolve(entity)
            
            if result.action == "create":
                batch_new[norm_name] = len(results)
            
            results.append(result)
        
        return results
