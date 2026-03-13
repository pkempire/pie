"""
PIE Configuration — All constants with derivations.

Every threshold/constant must have one of:
1. A derivation (from data or principle)
2. A TODO noting it needs to be learned
3. A citation to external work

No magic numbers without justification.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


# =============================================================================
# Derivation Notes (referenced by constants below)
# =============================================================================
#
# [D1] String match threshold 0.85:
#      Standard in fuzzy matching literature. Levenshtein ratio > 0.85
#      catches typos/variants while avoiding false matches.
#      TODO: Validate on labeled entity pairs from our data.
#
# [D2] Embedding similarity thresholds:
#      - reject < 0.70: Based on typical cosine similarity distributions
#        where unrelated text pairs cluster below 0.7
#      - accept > 0.85: High confidence match threshold
#      TODO: Derive from actual similarity distributions of match/non-match pairs.
#
# [D3] Context window limits:
#      - GPT-4o: 128K tokens (~400K chars)
#      - We use ~25% for safety margin and response
#      - 50 entities at ~500 chars each = ~25K chars = safe
#      TODO: Dynamic sizing based on actual model context window.
#
# [D4] Temporal halflife 30 days:
#      Arbitrary starting point for "default" memory decay.
#      TODO: Learn per-entity-type halflife from access patterns.
#
# [D5] Retrieval weights (0.6/0.25/0.15):
#      Initial guess: semantic most important, then temporal, then recency.
#      TODO: Learn from query-relevance labels via logistic regression.
#
# =============================================================================


@dataclass
class LLMConfig:
    """LLM model configuration."""
    
    extraction_model: str = "gpt-4o-mini"
    """Model for entity/relationship extraction. Needs good instruction following."""
    
    resolution_model: str = "gpt-4o-mini"
    """Model for entity resolution verification (Tier 3). Binary classification."""
    
    embedding_model: str = "text-embedding-3-large"
    """Embedding model. Dimensions fixed by model choice."""
    
    embedding_dimensions: int = 3072
    """text-embedding-3-large produces 3072-dim vectors."""
    
    extraction_temperature: float = 0.0
    """Zero temperature for deterministic extraction."""
    
    resolution_temperature: float = 0.0
    """Zero temperature for deterministic resolution."""


@dataclass
class ResolutionConfig:
    """Entity resolution thresholds. See derivation notes above."""
    
    string_match_threshold: float = 0.85
    """[D1] Levenshtein ratio threshold for string match."""
    
    embedding_reject_threshold: float = 0.70
    """[D2] Below this = definitely different entities."""
    
    embedding_accept_threshold: float = 0.85
    """[D2] Above this = auto-accept as same entity."""
    
    # Note: values in [reject, accept) go to LLM verification (Tier 3)
    
    require_llm_for_cross_type: bool = True
    """Always verify via LLM if entity types differ."""
    
    web_ground_types: tuple[str, ...] = ("tool", "organization")
    """Entity types to verify against web search."""


@dataclass
class IngestionConfig:
    """Ingestion pipeline settings."""
    
    batch_mode: Literal["daily", "window"] = "daily"
    """How to group conversations. Daily is simpler and matches natural time units."""
    
    max_context_entities: int = 50
    """[D3] Max entities in extraction context preamble."""
    
    max_context_chars: int = 60_000
    """[D3] Max chars for conversation content in extraction."""
    
    max_turns_per_conversation: int = 200
    """Truncate very long conversations. 200 turns ≈ reasonable upper bound."""
    
    max_chars_per_turn: int = 4000
    """Truncate very long messages. 4K chars ≈ 1K tokens."""
    
    year_min: int = 2023
    """Earliest year to process. Configurable per dataset."""
    
    activity_lookback_days: int = 7
    """How far back to look for "recently active" entities in context."""


@dataclass
class RetrievalConfig:
    """Retrieval scoring weights. See derivation notes above."""
    
    semantic_weight: float = 0.6
    """[D5] Weight for embedding similarity. TODO: Learn from data."""
    
    temporal_weight: float = 0.25
    """[D5] Weight for temporal constraint match. TODO: Learn from data."""
    
    recency_weight: float = 0.15
    """[D5] Weight for recency decay. TODO: Learn from data."""
    
    recency_halflife_days: float = 30.0
    """[D4] Days until recency score halves. TODO: Learn per entity type."""
    
    @property
    def recency_halflife_seconds(self) -> float:
        return self.recency_halflife_days * 86400


@dataclass
class TemporalConfig:
    """Temporal reference parsing settings."""
    
    implicit_recent_days: int = 7
    """What "recently" means when no explicit time given. Arbitrary."""
    
    implicit_confidence: float = 0.6
    """Confidence for implicit temporal references like "recently"."""
    
    explicit_confidence: float = 0.95
    """Confidence for explicit dates like "March 15, 2025"."""
    
    relative_confidence: float = 0.90
    """Confidence for relative dates like "last week"."""


@dataclass
class GraphConfig:
    """Graph database configuration."""
    
    backend: Literal["json", "falkordb"] = "json"
    """Storage backend. JSON for dev, FalkorDB for production."""
    
    persist_path: str = "output/world_model.json"
    """Path for JSON persistence."""
    
    # FalkorDB settings (when backend == "falkordb")
    falkor_host: str = "localhost"
    falkor_port: int = 6379
    falkor_graph_name: str = "pie_world"


@dataclass
class PIEConfig:
    """Top-level configuration."""
    
    conversations_path: Path = field(
        default_factory=lambda: Path("~/Downloads/conversations.json").expanduser()
    )
    output_dir: Path = field(default_factory=lambda: Path("./output"))
    
    llm: LLMConfig = field(default_factory=LLMConfig)
    resolution: ResolutionConfig = field(default_factory=ResolutionConfig)
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    
    # Feature flags
    use_web_grounding: bool = False  # Disabled until we have reliable API
    use_sliding_window: bool = True  # Build context preamble from world model state
    verbose: bool = True
    
    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)


# Default config instance
default_config = PIEConfig()
