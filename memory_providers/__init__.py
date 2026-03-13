"""
Memory Provider Implementations
===============================

Each provider implements the same interface for fair comparison.

Providers:
    - PIE: Our temporal knowledge graph approach
    - Graphiti/Zep: Temporal KG with bi-temporal model (Zep paper)
    - Mem0: Flat fact store with embeddings
    - Supermemory: Fast routing + disambiguation
    - Honcho: Dialectical user modeling (psychology-based)

Architecture comparison:

| Provider    | Memory Type        | Temporal      | Entity Resolution | Key Insight                    |
|-------------|-------------------|---------------|-------------------|--------------------------------|
| PIE         | Temporal KG       | State chains  | 3-tier hybrid     | Explicit state transitions     |
| Graphiti    | Temporal KG       | Bi-temporal   | Community-based   | Event+ingestion time tracking  |
| Mem0        | Fact store        | Timestamps    | LLM-based         | Simple key-value, fast         |
| Supermemory | Hybrid routing    | Basic         | Disambiguation    | Fast routing to right context  |
| Honcho      | Dialectical       | Continuous    | Psychology-based  | User modeling, not facts       |
"""

from .interface import MemoryProvider, MemoryProviderConfig
from .registry import PROVIDERS, get_provider, list_providers

__all__ = [
    "MemoryProvider",
    "MemoryProviderConfig", 
    "PROVIDERS",
    "get_provider",
    "list_providers",
]
