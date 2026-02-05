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
        """Update an entity's state and record the transition."""
        entity = self.entities.get(entity_id)
        if not entity:
            logger.warning(f"Entity {entity_id} not found for state update")
            return None
        
        old_state = entity.current_state.copy()
        
        # Merge new state into current (don't overwrite everything)
        entity.current_state.update(new_state)
        entity.last_seen = max(entity.last_seen, timestamp)
        
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
        """Add an alias to an entity."""
        entity = self.entities.get(entity_id)
        if entity and alias not in entity.aliases and alias != entity.name:
            entity.aliases.append(alias)
            self._alias_index[_normalize(alias)] = entity_id
    
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
    ) -> Relationship:
        """Add a relationship between two entities."""
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
    
    def find_by_name(self, name: str) -> Entity | None:
        """Exact name lookup (case-insensitive)."""
        normalized = _normalize(name)
        eid = self._name_index.get(normalized) or self._alias_index.get(normalized)
        if eid:
            return self.entities.get(eid)
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
            
            # Check against all names
            for known_name in entity.all_names:
                score = _fuzzy_ratio(name, known_name)
                best_score = max(best_score, score)
                
                # Also check if one contains the other
                norm_known = _normalize(known_name)
                if normalized in norm_known or norm_known in normalized:
                    best_score = max(best_score, 0.9)
            
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
    
    def build_context_preamble(self, batch_timestamp: float) -> str:
        """
        Build the activity-based context preamble for extraction.
        This is what makes the sliding window work.
        """
        lines = []
        
        # Active projects
        projects = self.get_active_projects(min_importance=0.1)
        if projects:
            lines.append("ACTIVE PROJECTS:")
            for p in sorted(projects, key=lambda p: p.importance, reverse=True)[:15]:
                state_summary = p.current_state.get("description", str(p.current_state)[:200])
                transitions = self.get_transitions(p.id)
                last_change = ""
                if transitions:
                    last_t = transitions[-1]
                    days_ago = (batch_timestamp - last_t.timestamp) / 86400
                    if days_ago < 1:
                        last_change = "today"
                    elif days_ago < 2:
                        last_change = "yesterday"
                    else:
                        last_change = f"{int(days_ago)} days ago"
                    last_change = f" (last changed: {last_change})"
                
                aliases = f" (aka: {', '.join(p.aliases)})" if p.aliases else ""
                lines.append(f"- {p.name}{aliases}: {state_summary}{last_change}")
            lines.append("")
        
        # Recently active entities (non-project)
        active = self.get_recently_active_entities(batch_timestamp)
        non_project_active = [e for e in active if e.type != EntityType.PROJECT]
        if non_project_active:
            lines.append("RECENTLY ACTIVE ENTITIES:")
            for e in non_project_active[:20]:
                state_summary = e.current_state.get("description", str(e.current_state)[:150])
                lines.append(f"- {e.name} ({e.type.value}): {state_summary}")
            lines.append("")
        
        # Recent state changes
        recent_transitions = self.get_recent_transitions(batch_timestamp)
        if recent_transitions:
            lines.append("RECENT STATE CHANGES:")
            for t in recent_transitions[:10]:
                entity = self.entities.get(t.entity_id)
                entity_name = entity.name if entity else "unknown"
                days_ago = (batch_timestamp - t.timestamp) / 86400
                time_str = "today" if days_ago < 1 else f"{int(days_ago)} days ago"
                marker = " ⚠ CONTRADICTION" if t.transition_type == TransitionType.CONTRADICTION else ""
                lines.append(f"- {entity_name}: {t.trigger_summary} ({time_str}){marker}")
            lines.append("")
        
        return "\n".join(lines) if lines else ""
    
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
