"""DSPy consolidator module — Auto-Dreamer-style.

Given a *working region* of raw conversation turns (typically 20-40 turns from
a single LoCoMo session), produce a small set of `ConsolidatedEntry`s capturing
the semantic and procedural memories that should be persisted to long-term
storage.

This file deliberately keeps the module tiny: one `dspy.ChainOfThought` call.
The prompt that GEPA optimizes lives in `ConsolidateSignature.__doc__` and
the field descriptions — DSPy/GEPA rewrites those at optimization time.
"""
from __future__ import annotations

from typing import Literal

import dspy
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# I/O types
# ---------------------------------------------------------------------------
class Turn(BaseModel):
    """One raw conversation turn from LoCoMo."""
    dia_id: str = Field(..., description="Turn id, e.g. 'D1:3' (session:index).")
    speaker: str = Field(..., description="Speaker name, e.g. 'Caroline' or 'Melanie'.")
    text: str = Field(..., description="Utterance text.")
    session_date: str = Field("", description="Human-readable session timestamp.")


class ConsolidatedEntry(BaseModel):
    """One consolidated long-term memory entry."""
    entry_type: Literal["semantic", "procedural"] = Field(
        ..., description="'semantic' for facts/preferences/events; 'procedural' for how-to steps."
    )
    name: str = Field(..., description="Short title (<= 80 chars) identifying the entry.")
    summary: str = Field(..., description="One-sentence summary of the entry's content.")
    details: str = Field(
        "",
        description="Free-form details (only for semantic entries). Empty string for procedural.",
    )
    steps: list[str] = Field(
        default_factory=list,
        description="Ordered list of steps (only for procedural entries). Empty list for semantic.",
    )
    speaker: str = Field(
        ...,
        description=(
            "Whose memory this belongs to — the speaker(s) whose life/knowledge it describes. "
            "For LoCoMo this is typically 'Caroline', 'Melanie', or 'Caroline & Melanie' "
            "for shared facts. Critical for correct attribution."
        ),
    )
    source_turn_ids: list[str] = Field(
        default_factory=list,
        description="dia_ids of the source turns that justify this entry (provenance).",
    )


# ---------------------------------------------------------------------------
# DSPy signature + module
# ---------------------------------------------------------------------------
class ConsolidateSignature(dspy.Signature):
    """Consolidate a working region of raw conversation turns into a small
    set of long-term memory entries, in the spirit of Auto-Dreamer.

    Read the turns carefully. Identify:
      - Semantic memories: durable facts, preferences, opinions, events,
        relationships, possessions, plans, named entities.
      - Procedural memories: how-to knowledge — sequences of steps the
        speaker described or learned.

    Attribute every entry to the speaker whose life/knowledge it describes
    (NOT necessarily the speaker of the source turn — e.g., if Melanie says
    'Caroline runs marathons', that entry belongs to 'Caroline').

    Always include the dia_ids of the source turns that justify each entry.
    Prefer a small number of high-quality, non-redundant entries over many
    fragmentary ones. Skip greetings, filler, and turns with no durable
    content.
    """

    working_region: list[Turn] = dspy.InputField(
        desc="Raw conversation turns (typically 20-40) from one LoCoMo session."
    )
    consolidated_entries: list[ConsolidatedEntry] = dspy.OutputField(
        desc="Consolidated long-term memory entries extracted from the working region."
    )


class Consolidator(dspy.Module):
    """One-shot ChainOfThought consolidator. GEPA optimizes the underlying
    signature's prompt + field descriptions."""

    def __init__(self):
        super().__init__()
        self.consolidate = dspy.ChainOfThought(ConsolidateSignature)

    def forward(self, working_region: list[Turn]) -> dspy.Prediction:
        return self.consolidate(working_region=working_region)
