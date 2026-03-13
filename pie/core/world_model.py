"""
World Model Store — the in-memory graph with JSON persistence.

Starts as JSON-backed for fast iteration. FalkorDB adapter comes later.
The interface stays the same regardless of backend.
"""

from __future__ import annotations
import json
import math
import logging
from pathlib import Path
from collections import defaultdict
from difflib import SequenceMatcher

from .models import (
    Entity, EntityType, StateTransition, TransitionType,
    Relationship, RelationshipType, Procedure,
)

logger = logging.getLogger("pie.world_model")


def _normalize(name: str) -> str:
    """Normalize entity name for matching."""
    return name.lower().strip().replace("-", " ").replace("_", " ")


def _fuzzy_ratio(a: str, b: str) -> float:
    """Fuzzy string similarity ratio."""
    return SequenceMatcher(None, _normalize(a), _normalize(b)).ratio()


def _normalize_value(v) -> str:
    """Normalize a state value for comparison."""
    if v is None:
        return ""
    s = str(v).strip().lower()
    # Collapse whitespace
    return " ".join(s.split())


def _states_meaningfully_differ(old: dict, new: dict) -> bool:
    """Check if two state dicts have meaningful differences.

    Returns False when the new state is essentially a re-extraction of the
    same information (same keys, same values after normalization).
    """
    all_keys = set(old.keys()) | set(new.keys())
    if not all_keys:
        return False

    # New keys that weren't in old → real change
    added_keys = set(new.keys()) - set(old.keys())
    # Only count added keys that have non-empty values
    real_adds = [k for k in added_keys if _normalize_value(new.get(k))]
    if real_adds:
        return True

    # Check existing keys for value changes
    for key in old:
        if key not in new:
            continue  # key wasn't re-extracted, not a removal
        old_norm = _normalize_value(old[key])
        new_norm = _normalize_value(new[key])
        if old_norm != new_norm:
            # For description fields, also check if it's just minor rewording
            # (less than 20% different by length) — still counts as change
            return True

    return False


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class WorldModel:
    """
    In-memory world model graph with JSON persistence.
    
    Stores entities, transitions, relationships, and procedures.
    Provides entity resolution (string match + embedding similarity).
    """
    
    def __init__(self, persist_path: str | Path | None = None):
        self.entities: dict[str, Entity] = {}               # id -> Entity
        self.transitions: dict[str, StateTransition] = {}    # id -> StateTransition
        self.relationships: dict[str, Relationship] = {}     # id -> Relationship
        self.procedures: dict[str, Procedure] = {}           # id -> Procedure
        
        # Indexes for fast lookup
        self._name_index: dict[str, str] = {}                # normalized_name -> entity_id
        self._alias_index: dict[str, str] = {}               # normalized_alias -> entity_id
        self._type_index: dict[str, set[str]] = defaultdict(set)  # type -> set of entity_ids
        self._entity_transitions: dict[str, list[str]] = defaultdict(list)  # entity_id -> transition_ids
        self._entity_relationships: dict[str, list[str]] = defaultdict(list)  # entity_id -> relationship_ids
        
        self.persist_path = Path(persist_path) if persist_path else None
        if self.persist_path and self.persist_path.exists():
            self._load()
    
    # --- Entity CRUD ---
    
    def create_entity(
        self,
        name: str,
        type: EntityType,
        state: dict,
        source_conversation_id: str | None = None,
        timestamp: float = 0.0,
        aliases: list[str] | None = None,
        embedding: list[float] | None = None,
    ) -> Entity:
        """Create a new entity in the world model."""
        entity = Entity(
            type=type,
            name=name,
            aliases=aliases or [],
            current_state=state,
            created_from=source_conversation_id,
            first_seen=timestamp,
            last_seen=timestamp,
            embedding=embedding,
        )
        
        self.entities[entity.id] = entity
        self._index_entity(entity)
        
        # Record creation transition
        self.record_transition(
            entity_id=entity.id,
            from_state=None,
            to_state=state,
            transition_type=TransitionType.CREATION,
            trigger_conversation_id=source_conversation_id or "",
            trigger_summary=f"First appearance of {name}",
            timestamp=timestamp,
        )
        
        return entity
    
    def get_entity(self, entity_id: str) -> Entity | None:
        return self.entities.get(entity_id)
    
    def update_entity_state(
        self,
        entity_id: str,
        new_state: dict,
        source_conversation_id: str,
        timestamp: float,
        trigger_summary: str = "",
        is_contradiction: bool = False,
    ) -> StateTransition | None:
        """Update an entity's state and record the transition.

        Only records a transition if the state actually changed meaningfully.
        Prevents noise transitions from re-extraction of unchanged entities.
        """
        entity = self.entities.get(entity_id)
        if not entity:
            logger.warning(f"Entity {entity_id} not found for state update")
            return None

        old_state = entity.current_state.copy()

        # Merge new state into current (don't overwrite everything)
        entity.current_state.update(new_state)
        entity.last_seen = max(entity.last_seen, timestamp)

        # Check if anything actually changed meaningfully
        if not _states_meaningfully_differ(old_state, entity.current_state):
            return None  # No-op: skip recording redundant transition

        transition = self.record_transition(
            entity_id=entity_id,
            from_state=old_state,
            to_state=entity.current_state,
            transition_type=TransitionType.CONTRADICTION if is_contradiction else TransitionType.UPDATE,
            trigger_conversation_id=source_conversation_id,
            trigger_summary=trigger_summary,
            timestamp=timestamp,
        )

        return transition
    
    def add_alias(self, entity_id: str, alias: str):
        """Add an alias to an entity with validation.

        Safety measures to prevent alias accumulation cascade:
        1. Cap at MAX_ALIASES per entity
        2. Alias must share at least one word with entity name or existing aliases
        3. Skip obviously wrong aliases (empty, same as name)
        """
        MAX_ALIASES = 10

        entity = self.entities.get(entity_id)
        if not entity:
            return
        if alias in entity.aliases or _normalize(alias) == _normalize(entity.name):
            return

        # Cap alias count — if an entity keeps gaining aliases, something is wrong
        if len(entity.aliases) >= MAX_ALIASES:
            logger.warning(
                f"Alias cap ({MAX_ALIASES}) reached for '{entity.name}' — "
                f"skipping '{alias}'"
            )
            return

        # Basic semantic check: alias should share at least one word with
        # the entity name or an existing alias. This prevents cascade merges
        # where unrelated entities get aliased together.
        alias_words = set(_normalize(alias).split())
        name_words = set(_normalize(entity.name).split())

        # Collect words from all existing aliases too
        all_known_words = name_words.copy()
        for existing in entity.aliases:
            all_known_words.update(_normalize(existing).split())

        # Remove very common words that don't carry semantic meaning
        stop_words = {"the", "a", "an", "of", "in", "on", "for", "to", "and", "or", "is", "it", "with"}
        meaningful_alias = alias_words - stop_words
        meaningful_known = all_known_words - stop_words

        # If the alias has 3+ meaningful words and shares NONE with entity,
        # it's almost certainly a wrong merge
        if len(meaningful_alias) >= 3 and not (meaningful_alias & meaningful_known):
            logger.warning(
                f"Alias '{alias}' shares no words with '{entity.name}' "
                f"(aliases: {entity.aliases[:3]}) — skipping"
            )
            return

        entity.aliases.append(alias)
        self._alias_index[_normalize(alias)] = entity_id

    def remove_alias(self, entity_id: str, alias: str) -> bool:
        """Remove an alias from an entity. Returns True if removed."""
        entity = self.entities.get(entity_id)
        if not entity:
            return False
        if alias in entity.aliases:
            entity.aliases.remove(alias)
            norm = _normalize(alias)
            if self._alias_index.get(norm) == entity_id:
                del self._alias_index[norm]
            return True
        return False

    # --- Transitions ---
    
    def record_transition(
        self,
        entity_id: str,
        from_state: dict | None,
        to_state: dict,
        transition_type: TransitionType,
        trigger_conversation_id: str,
        trigger_summary: str,
        timestamp: float,
        confidence: float = 1.0,
    ) -> StateTransition:
        """Record a state transition for an entity."""
        transition = StateTransition(
            entity_id=entity_id,
            from_state=from_state,
            to_state=to_state,
            transition_type=transition_type,
            trigger_conversation_id=trigger_conversation_id,
            trigger_summary=trigger_summary,
            timestamp=timestamp,
            confidence=confidence,
        )
        self.transitions[transition.id] = transition
        self._entity_transitions[entity_id].append(transition.id)
        return transition
    
    def get_transitions(self, entity_id: str, ordered: bool = True) -> list[StateTransition]:
        """Get all transitions for an entity."""
        tids = self._entity_transitions.get(entity_id, [])
        transitions = [self.transitions[tid] for tid in tids if tid in self.transitions]
        if ordered:
            transitions.sort(key=lambda t: t.timestamp)
        return transitions
    
    # --- Relationships ---
    
    def add_relationship(
        self,
        source_id: str,
        target_id: str,
        rel_type: RelationshipType,
        description: str = "",
        source_conversation_id: str | None = None,
        timestamp: float = 0.0,
    ) -> Relationship | None:
        """Add a relationship between two entities. Deduplicates by (source, target, type)."""
        # Check for existing relationship between same entities of same type
        for rid in self._entity_relationships.get(source_id, []):
            existing = self.relationships.get(rid)
            if (existing and existing.source_id == source_id
                    and existing.target_id == target_id
                    and existing.type == rel_type):
                # Already exists — update timestamp if newer
                if timestamp > existing.timestamp:
                    existing.timestamp = timestamp
                return None

        rel = Relationship(
            source_id=source_id,
            target_id=target_id,
            type=rel_type,
            description=description,
            source_conversation_id=source_conversation_id,
            timestamp=timestamp,
        )
        self.relationships[rel.id] = rel
        self._entity_relationships[source_id].append(rel.id)
        self._entity_relationships[target_id].append(rel.id)
        return rel
    
    def get_relationships(self, entity_id: str) -> list[Relationship]:
        """Get all relationships involving an entity."""
        rids = self._entity_relationships.get(entity_id, [])
        return [self.relationships[rid] for rid in rids if rid in self.relationships]
    
    def get_neighbors(self, entity_id: str) -> list[str]:
        """Get IDs of entities connected to this one."""
        neighbors = set()
        for rel in self.get_relationships(entity_id):
            if rel.source_id == entity_id:
                neighbors.add(rel.target_id)
            else:
                neighbors.add(rel.source_id)
        return list(neighbors)
    
    # --- Entity Resolution ---
    
    def find_by_name(
        self,
        name: str,
        entity_type: str | None = None,
    ) -> Entity | None:
        """Exact name lookup (case-insensitive), checks primary name then aliases.

        Args:
            name: Entity name to look up.
            entity_type: If provided, only return the entity if its type matches.
                         This prevents alias collisions across types.
        """
        normalized = _normalize(name)
        eid = self._name_index.get(normalized) or self._alias_index.get(normalized)
        if eid:
            entity = self.entities.get(eid)
            if entity and entity_type and entity.type.value != entity_type:
                return None  # Type mismatch — don't resolve through wrong-type alias
            return entity
        return None
    
    def find_by_string_match(
        self,
        name: str,
        threshold: float = 0.85,
        entity_type: str | None = None,
    ) -> list[tuple[Entity, float]]:
        """
        Find entities by fuzzy string matching.
        Returns list of (entity, score) pairs sorted by score descending.
        """
        normalized = _normalize(name)
        matches = []
        
        # Check candidates (optionally filtered by type)
        if entity_type:
            candidate_ids = self._type_index.get(entity_type, set())
        else:
            candidate_ids = set(self.entities.keys())
        
        for eid in candidate_ids:
            entity = self.entities[eid]
            best_score = 0.0
            
            # Check against primary name first, then aliases
            for known_name in entity.all_names:
                score = _fuzzy_ratio(name, known_name)
                best_score = max(best_score, score)

                # Containment check — much more conservative than before.
                # Requirements: shorter string ≥ 4 chars, represents ≥ 30% of
                # the longer string, and boost is capped at 0.87 (below auto-
                # accept threshold of 0.90 so it goes through extra verification).
                norm_known = _normalize(known_name)
                shorter_len = min(len(normalized), len(norm_known))
                longer_len = max(len(normalized), len(norm_known))
                if shorter_len >= 4 and longer_len > 0 and shorter_len / longer_len >= 0.3:
                    if normalized in norm_known or norm_known in normalized:
                        best_score = max(best_score, 0.87)
            
            if best_score >= threshold:
                matches.append((entity, best_score))
        
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches
    
    def find_by_embedding(
        self,
        embedding: list[float],
        top_k: int = 5,
        entity_type: str | None = None,
        exclude_ids: set[str] | None = None,
    ) -> list[tuple[Entity, float]]:
        """
        Find entities by embedding similarity.
        Returns list of (entity, cosine_similarity) pairs.
        """
        if entity_type:
            candidate_ids = self._type_index.get(entity_type, set())
        else:
            candidate_ids = set(self.entities.keys())
        
        if exclude_ids:
            candidate_ids -= exclude_ids
        
        scores = []
        for eid in candidate_ids:
            entity = self.entities[eid]
            if entity.embedding:
                sim = cosine_similarity(embedding, entity.embedding)
                scores.append((entity, sim))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
    
    # --- Embedding Management ---

    def get_entities_without_embeddings(self) -> list[Entity]:
        """Get entities that don't have embeddings computed."""
        return [e for e in self.entities.values() if e.embedding is None]

    def set_entity_embedding(self, entity_id: str, embedding: list[float]):
        """Set the embedding for an entity."""
        entity = self.entities.get(entity_id)
        if entity:
            entity.embedding = embedding

    # --- Context Building (for sliding window) ---
    
    def get_recently_active_entities(
        self,
        before_timestamp: float,
        lookback_seconds: float = 3 * 86400,  # 3 days
        max_entities: int = 50,
    ) -> list[Entity]:
        """
        Get entities that had state transitions recently (relative to a timestamp).
        Used for building the activity-based context preamble.
        """
        cutoff = before_timestamp - lookback_seconds
        
        active_entities = {}
        for tid, transition in self.transitions.items():
            if cutoff <= transition.timestamp <= before_timestamp:
                eid = transition.entity_id
                if eid in self.entities:
                    entity = self.entities[eid]
                    if eid not in active_entities or entity.importance > active_entities[eid].importance:
                        active_entities[eid] = entity
        
        # Sort by importance, take top N
        sorted_entities = sorted(active_entities.values(), key=lambda e: e.importance, reverse=True)
        return sorted_entities[:max_entities]
    
    def get_recent_transitions(
        self,
        before_timestamp: float,
        lookback_seconds: float = 3 * 86400,
        max_transitions: int = 10,
    ) -> list[StateTransition]:
        """Get recent state transitions for context preamble."""
        cutoff = before_timestamp - lookback_seconds
        recent = [
            t for t in self.transitions.values()
            if cutoff <= t.timestamp <= before_timestamp
        ]
        recent.sort(key=lambda t: t.timestamp, reverse=True)
        return recent[:max_transitions]
    
    def get_active_projects(self, min_importance: float = 0.1) -> list[Entity]:
        """Get all project entities above importance threshold."""
        project_ids = self._type_index.get("project", set())
        projects = [self.entities[eid] for eid in project_ids if eid in self.entities]
        return [p for p in projects if p.importance >= min_importance]
    
    def build_context_preamble(
        self,
        batch_timestamp: float,
        max_chars: int = 100_000,
    ) -> str:
        """
        Build the context preamble for extraction.

        Includes FULL structured state for each entity (not truncated).
        Greedily fills the token budget, prioritizing:
          1. Recently active entities (most likely to reappear)
          2. High-importance entities (core knowledge)

        The LLM needs to see the COMPLETE state to:
          - Correctly match re-extracted entities against existing ones
          - Know what has already been captured (avoid re-invention)
          - Detect actual state changes vs rewording
        """
        import json as _json

        # Collect candidate entities, scored by relevance to this batch
        candidates: list[tuple[float, Entity]] = []
        lookback = 7 * 86400  # 7 days

        for eid, entity in self.entities.items():
            # Score: recency + importance
            time_since = batch_timestamp - entity.last_seen
            recency_score = max(0, 1.0 - time_since / (30 * 86400))  # 0 at 30 days
            # Boost if recently active
            if time_since < lookback:
                recency_score += 1.0  # big boost for last-7-days
            score = recency_score + (entity.importance or 0)
            candidates.append((score, entity))

        # Sort by score descending
        candidates.sort(key=lambda x: -x[0])

        # Build preamble, greedily filling budget
        lines = []
        chars_used = 0
        entities_included = 0
        seen_ids = set()

        for score, entity in candidates:
            if chars_used >= max_chars:
                break

            # Serialize full state
            state_json = _json.dumps(entity.current_state, indent=None, default=str)

            # NOTE: We intentionally do NOT include aliases in the preamble.
            # Showing aliases to the extraction LLM caused it to write
            # matches_existing pointing to aliases, which resolved through
            # the alias index and created wrong merges. The entity resolver
            # handles alias matching internally; the LLM only needs the
            # canonical name.

            # Get last transition info
            transitions = self.get_transitions(entity.id)
            last_change = ""
            if transitions:
                days_ago = (batch_timestamp - transitions[-1].timestamp) / 86400
                if days_ago < 1:
                    last_change = " [last changed: today]"
                elif days_ago < 2:
                    last_change = " [last changed: yesterday]"
                elif days_ago < 30:
                    last_change = f" [last changed: {int(days_ago)} days ago]"

            entry = (
                f"### {entity.name} ({entity.type.value}){last_change}\n"
                f"{state_json}\n"
            )

            if chars_used + len(entry) > max_chars:
                # Try a compact version: just name + description
                desc = entity.current_state.get("description", "")
                compact = f"- {entity.name} ({entity.type.value}): {desc}\n"
                if chars_used + len(compact) > max_chars:
                    break
                lines.append(compact)
                chars_used += len(compact)
            else:
                lines.append(entry)
                chars_used += len(entry)

            entities_included += 1
            seen_ids.add(entity.id)

        # Add recent state changes at the end (important for detecting contradictions)
        recent_transitions = self.get_recent_transitions(batch_timestamp, max_transitions=15)
        if recent_transitions:
            header = "\n## RECENT STATE CHANGES\n"
            if chars_used + len(header) < max_chars:
                lines.append(header)
                chars_used += len(header)
                for t in recent_transitions:
                    entity = self.entities.get(t.entity_id)
                    entity_name = entity.name if entity else "unknown"
                    days_ago = (batch_timestamp - t.timestamp) / 86400
                    time_str = "today" if days_ago < 1 else f"{int(days_ago)}d ago"
                    marker = " ⚠ CONTRADICTION" if t.transition_type == TransitionType.CONTRADICTION else ""
                    entry = f"- {entity_name}: {t.trigger_summary} ({time_str}){marker}\n"
                    if chars_used + len(entry) > max_chars:
                        break
                    lines.append(entry)
                    chars_used += len(entry)

        logger.debug(
            f"Context preamble: {entities_included} entities, "
            f"{chars_used:,} chars (~{chars_used//4:,} tokens)"
        )

        return "".join(lines) if lines else ""
    
    # --- Stats ---
    
    @property
    def stats(self) -> dict:
        type_counts = defaultdict(int)
        for e in self.entities.values():
            type_counts[e.type.value] += 1
        
        return {
            "entities": len(self.entities),
            "transitions": len(self.transitions),
            "relationships": len(self.relationships),
            "procedures": len(self.procedures),
            "entity_types": dict(type_counts),
        }
    
    # --- Indexing ---
    
    def _index_entity(self, entity: Entity):
        """Add entity to lookup indexes."""
        self._name_index[_normalize(entity.name)] = entity.id
        for alias in entity.aliases:
            self._alias_index[_normalize(alias)] = entity.id
        self._type_index[entity.type.value].add(entity.id)
    
    # --- Persistence ---
    
    def save(self):
        """Save world model to JSON."""
        if not self.persist_path:
            return
        
        data = {
            "entities": {
                eid: {
                    "id": e.id, "type": e.type.value, "name": e.name,
                    "aliases": e.aliases, "current_state": e.current_state,
                    "created_from": e.created_from,
                    "first_seen": e.first_seen, "last_seen": e.last_seen,
                    "importance": e.importance,
                    "web_canonical_name": e.web_canonical_name,
                    "web_description": e.web_description,
                    "web_verified": e.web_verified,
                    # Skip embeddings in JSON — too large. Recompute on load.
                }
                for eid, e in self.entities.items()
            },
            "transitions": {
                tid: {
                    "id": t.id, "entity_id": t.entity_id,
                    "from_state": t.from_state, "to_state": t.to_state,
                    "transition_type": t.transition_type.value,
                    "trigger_conversation_id": t.trigger_conversation_id,
                    "trigger_summary": t.trigger_summary,
                    "timestamp": t.timestamp, "confidence": t.confidence,
                }
                for tid, t in self.transitions.items()
            },
            "relationships": {
                rid: {
                    "id": r.id, "source_id": r.source_id, "target_id": r.target_id,
                    "type": r.type.value, "description": r.description,
                    "source_conversation_id": r.source_conversation_id,
                    "timestamp": r.timestamp,
                }
                for rid, r in self.relationships.items()
            },
        }
        
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.persist_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Saved world model: {self.stats}")
    
    def _load(self):
        """Load world model from JSON."""
        with open(self.persist_path) as f:
            data = json.load(f)
        
        for eid, edata in data.get("entities", {}).items():
            entity = Entity(
                id=edata["id"],
                type=EntityType(edata["type"]),
                name=edata["name"],
                aliases=edata.get("aliases", []),
                current_state=edata.get("current_state", {}),
                created_from=edata.get("created_from"),
                first_seen=edata.get("first_seen", 0),
                last_seen=edata.get("last_seen", 0),
                importance=edata.get("importance", 0),
                web_canonical_name=edata.get("web_canonical_name"),
                web_description=edata.get("web_description"),
                web_verified=edata.get("web_verified", False),
            )
            self.entities[entity.id] = entity
            self._index_entity(entity)
        
        for tid, tdata in data.get("transitions", {}).items():
            t = StateTransition(
                id=tdata["id"],
                entity_id=tdata["entity_id"],
                from_state=tdata.get("from_state"),
                to_state=tdata.get("to_state", {}),
                transition_type=TransitionType(tdata["transition_type"]),
                trigger_conversation_id=tdata.get("trigger_conversation_id", ""),
                trigger_summary=tdata.get("trigger_summary", ""),
                timestamp=tdata.get("timestamp", 0),
                confidence=tdata.get("confidence", 1.0),
            )
            self.transitions[t.id] = t
            self._entity_transitions[t.entity_id].append(t.id)
        
        for rid, rdata in data.get("relationships", {}).items():
            r = Relationship(
                id=rdata["id"],
                source_id=rdata["source_id"],
                target_id=rdata["target_id"],
                type=RelationshipType(rdata["type"]),
                description=rdata.get("description", ""),
                source_conversation_id=rdata.get("source_conversation_id"),
                timestamp=rdata.get("timestamp", 0),
            )
            self.relationships[r.id] = r
            self._entity_relationships[r.source_id].append(r.id)
            self._entity_relationships[r.target_id].append(r.id)
        
        logger.info(f"Loaded world model: {self.stats}")
