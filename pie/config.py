"""PIE configuration."""

from dataclasses import dataclass, field
from pathlib import Path
import os


@dataclass
class LLMConfig:
    """LLM model configuration."""
    extraction_model: str = "gpt-5-mini"       # primary extraction
    resolution_model: str = "gpt-5-mini"       # entity resolution verification
    cheap_model: str = "gpt-5-nano"            # quick classifications
    embedding_model: str = "text-embedding-3-large"
    embedding_dimensions: int = 3072
    temperature: float = 1.0  # gpt-5-mini/nano only support default (1.0)


@dataclass
class IngestionConfig:
    """Ingestion pipeline settings."""
    batch_mode: str = "daily"                   # "daily" or "window"
    window_size: int = 10                       # conversations per window (if window mode)
    window_overlap: int = 5                     # overlap between windows
    activity_lookback_days: int = 3             # how far back for "recently active" entities
    max_active_entities_in_preamble: int = 50   # cap on context preamble size
    max_turns_per_conversation: int = 200       # truncate very long convos
    max_chars_per_turn: int = 4000              # truncate very long messages
    year_min: int = 2023                        # earliest year to process


@dataclass
class ResolutionConfig:
    """Entity resolution settings."""
    string_match_threshold: float = 0.85        # fuzzy string match threshold
    embedding_similarity_threshold: float = 0.70 # below this = definitely different
    embedding_ambiguous_threshold: float = 0.78  # above this = auto-accept (was 0.85, lowered to reduce false merges)
    require_llm_for_cross_type: bool = True     # always LLM verify if types differ
    web_ground_entity_types: list[str] = field(
        default_factory=lambda: ["tool", "organization", "concept"]
    )
    web_ground_new_only: bool = True            # only web-ground new entities


@dataclass
class GraphConfig:
    """Graph database configuration."""
    host: str = "localhost"
    port: int = 6379
    graph_name: str = "pie_world"


@dataclass
class PIEConfig:
    """Top-level configuration."""
    conversations_path: Path = Path("~/Downloads/conversations.json").expanduser()
    output_dir: Path = Path("./output")
    llm: LLMConfig = field(default_factory=LLMConfig)
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    resolution: ResolutionConfig = field(default_factory=ResolutionConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    
    # Feature flags
    use_web_grounding: bool = True
    use_sliding_window: bool = True
    verbose: bool = True
    
    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
