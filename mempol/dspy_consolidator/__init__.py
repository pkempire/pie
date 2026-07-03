"""DSPy-based Auto-Dreamer-style consolidator for LoCoMo.

This subpackage holds a single DSPy `Module` that turns a working region of
raw conversation turns into a list of consolidated semantic / procedural
memory entries. It is intended to be optimized by GEPA in a subsequent step.

Public API:
    from mempol.dspy_consolidator import (
        Turn,
        ConsolidatedEntry,
        Consolidator,
    )
"""
from .consolidator import (
    Turn,
    ConsolidatedEntry,
    Consolidator,
    ConsolidateSignature,
)

__all__ = [
    "Turn",
    "ConsolidatedEntry",
    "Consolidator",
    "ConsolidateSignature",
]
