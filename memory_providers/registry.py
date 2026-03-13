"""
Memory Provider Registry

Central registry for all memory providers.
"""

from __future__ import annotations
from typing import Type

from .interface import MemoryProvider, MemoryProviderConfig


def _lazy_import(module: str, cls: str) -> Type[MemoryProvider]:
    """Lazy import to avoid loading all providers at startup."""
    def loader():
        import importlib
        mod = importlib.import_module(module, package="memory_providers")
        return getattr(mod, cls)
    return loader


# Registry of available providers
PROVIDERS: dict[str, callable] = {
    "pie": _lazy_import(".pie_provider", "PIEProvider"),
    "zep": _lazy_import(".zep_provider", "ZepProvider"),
    "graphiti": _lazy_import(".zep_provider", "ZepProvider"),  # Alias
    "mem0": _lazy_import(".mem0_provider", "Mem0Provider"),
    "supermemory": _lazy_import(".supermemory_provider", "SupermemoryProvider"),
    "honcho": _lazy_import(".honcho_provider", "HonchoProvider"),
}


def get_provider(name: str, config: MemoryProviderConfig | None = None) -> MemoryProvider:
    """
    Get a memory provider by name.
    
    Args:
        name: Provider name (pie, zep, graphiti, mem0, supermemory, honcho)
        config: Optional configuration
        
    Returns:
        Initialized MemoryProvider instance
        
    Raises:
        ValueError: If provider name is unknown
    """
    name = name.lower()
    
    if name not in PROVIDERS:
        available = ", ".join(PROVIDERS.keys())
        raise ValueError(f"Unknown provider: {name}. Available: {available}")
    
    # Load the provider class
    loader = PROVIDERS[name]
    if callable(loader) and not isinstance(loader, type):
        provider_cls = loader()
    else:
        provider_cls = loader
    
    return provider_cls(config)


def list_providers() -> list[str]:
    """List all available provider names."""
    return list(PROVIDERS.keys())


# Provider comparison summary
PROVIDER_COMPARISON = """
╔════════════════╦══════════════════════╦═══════════════════╦═══════════════════════════════════════╗
║ Provider       ║ Memory Architecture  ║ Temporal Model    ║ Best For                              ║
╠════════════════╬══════════════════════╬═══════════════════╬═══════════════════════════════════════╣
║ PIE            ║ Temporal KG          ║ State transitions ║ "How did X evolve?" questions         ║
║ Zep/Graphiti   ║ Temporal KG          ║ Bi-temporal       ║ Enterprise agents, real-time updates  ║
║ Mem0           ║ Flat fact store      ║ Timestamps only   ║ Simple personalization, fast          ║
║ Supermemory    ║ Hybrid routing       ║ Basic             ║ Fast recall, disambiguation           ║
║ Honcho         ║ Dialectical          ║ Continuous        ║ User psychology, personalization      ║
╚════════════════╩══════════════════════╩═══════════════════╩═══════════════════════════════════════╝
"""
