"""Core data models for PIE world model."""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
import uuid
import time


class EntityType(str, Enum):
    PERSON = "person"
    PROJECT = "project"
    TOOL = "tool"
    ORGANIZATION = "organization"
    BELIEF = "belief"
    DECISION = "decision"
    CONCEPT = "concept"
    PERIOD = "period"
    EVENT = "event"  # User activities with specific dates (visits, meetings, purchases, etc.)


class TransitionType(str, Enum):
    CREATION = "creation"
    UPDATE = "update"
    CONTRADICTION = "contradiction"
    RESOLUTION = "resolution"
    ARCHIVAL = "archival"


class RelationshipType(str, Enum):
    USES = "uses"
    WORKS_ON = "works_on"
    COLLABORATES_WITH = "collaborates_with"
    RELATED_TO = "related_to"
    PART_OF = "part_of"
    CAUSED_BY = "caused_by"
    DURING = "during"
    REPLACES = "replaces"
    INTEGRATES_WITH = "integrates_with"


# --- Source Data ---

@dataclass
class Turn:
    """A single message in a conversation."""
    role: str           # "user" or "assistant"
    text: str
    timestamp: float | None = None


@dataclass
class Conversation:
    """A parsed conversation from the export."""
    id: str
    title: str
    created_at: float
    updated_at: float | None
    model: str | None
    turns: list[Turn]
    
    @property
    def turn_count(self) -> int:
        return len(self.turns)
    
    @property
    def total_chars(self) -> int:
        return sum(len(t.text) for t in self.turns)
    
    @property
    def user_turns(self) -> list[Turn]:
        return [t for t in self.turns if t.role == "user"]
    
    @property
    def assistant_turns(self) -> list[Turn]:
        return [t for t in self.turns if t.role == "assistant"]


# --- World Model ---

@dataclass
class Entity:
    """An entity in the world model graph."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: EntityType = EntityType.CONCEPT
    name: str = ""
    aliases: list[str] = field(default_factory=list)
    current_state: dict[str, Any] = field(default_factory=dict)
    created_from: str | None = None      # conversation_id
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    importance: float = 0.0
    embedding: list[float] | None = None
    
    # Web grounding data (if verified)
    web_canonical_name: str | None = None
    web_description: str | None = None
    web_verified: bool = False
    
    @property
    def all_names(self) -> list[str]:
        """All known names/aliases for this entity."""
        names = [self.name] + self.aliases
        if self.web_canonical_name and self.web_canonical_name not in names:
            names.append(self.web_canonical_name)
        return names


@dataclass
class StateTransition:
    """A state change for an entity."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    entity_id: str = ""
    from_state: dict[str, Any] | None = None
    to_state: dict[str, Any] = field(default_factory=dict)
    transition_type: TransitionType = TransitionType.UPDATE
    trigger_conversation_id: str = ""
    trigger_summary: str = ""
    timestamp: float = field(default_factory=time.time)
    confidence: float = 1.0


@dataclass
class Relationship:
    """A relationship between two entities."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_id: str = ""
    target_id: str = ""
    type: RelationshipType = RelationshipType.RELATED_TO
    description: str = ""
    source_conversation_id: str | None = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class Procedure:
    """An extracted procedural pattern."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""
    pattern: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)  # conversation_ids
    times_observed: int = 0
    first_observed: float = 0.0
    last_observed: float = 0.0
    domain: str | None = None
    confidence: float = 0.0


# --- Extraction Results (LLM output) ---

@dataclass
class ExtractedEntity:
    """An entity extracted from a conversation by the LLM."""
    name: str
    type: str                       # string from LLM, validated to EntityType
    state: dict[str, Any]
    is_new: bool = True
    matches_existing: str | None = None  # name of existing entity if match detected
    confidence: float = 1.0


@dataclass
class ExtractedStateChange:
    """A state change extracted from a conversation."""
    entity_name: str
    what_changed: str
    old_state: str | None = None
    new_state: str = ""
    is_contradiction: bool = False
    confidence: float = 1.0


@dataclass 
class ExtractedRelationship:
    """A relationship extracted from a conversation."""
    source: str                     # entity name
    target: str                     # entity name
    type: str                       # string, validated to RelationshipType
    description: str = ""


@dataclass
class ExtractionResult:
    """Full extraction output from one batch/conversation."""
    entities: list[ExtractedEntity] = field(default_factory=list)
    state_changes: list[ExtractedStateChange] = field(default_factory=list)
    relationships: list[ExtractedRelationship] = field(default_factory=list)
    period_context: str = ""
    summary: str = ""
    significance: float = 0.0
    user_state: str | None = None
    
    # Metadata
    conversation_ids: list[str] = field(default_factory=list)
    tokens_used: dict[str, int] = field(default_factory=dict)


@dataclass
class DailyBatch:
    """A day's worth of conversations grouped for batch extraction."""
    date: str                       # YYYY-MM-DD
    conversations: list[Conversation]
    
    @property
    def total_chars(self) -> int:
        return sum(c.total_chars for c in self.conversations)
    
    @property
    def total_turns(self) -> int:
        return sum(c.turn_count for c in self.conversations)
