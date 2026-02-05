"""
Sales Entity Extraction — adapt PIE's extraction for sales-specific entity types.

Entity types: Prospect, PainPoint, Objection, BuyingSignal, Stakeholder,
CompetitiveContext, NextStep
"""

from __future__ import annotations
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pie.core.llm import LLMClient

logger = logging.getLogger("sales.extraction")

SALES_EXTRACTION_PROMPT = """You are a sales intelligence analyst. Extract structured entities from this sales call transcript.

For each entity, provide confidence (0.0-1.0) based on how explicitly it was stated vs inferred.

Return a JSON object with these arrays:

1. "prospect": Single object with:
   - name, title, company, email (if mentioned)
   - personality_traits: array of inferred traits (e.g., "detail-oriented", "risk-averse", "data-driven")
   - communication_style: how they communicate (e.g., "direct", "asks clarifying questions", "wants proof")
   - decision_style: how they make decisions (e.g., "consensus-builder", "needs stakeholder buy-in", "ROI-focused")

2. "pain_points": array of objects with:
   - description: what the problem is
   - severity: "critical" | "high" | "medium" | "low"
   - category: "reliability" | "compliance" | "security" | "cost" | "efficiency" | "governance" | "other"
   - verbatim_quote: closest quote from transcript (or null)
   - meeting_number: which meeting this was mentioned in
   - evolution: how it changed from previous mention (null if first mention)

3. "objections": array of objects with:
   - description: the concern or blocker
   - type: "technical" | "process" | "budget" | "timeline" | "political" | "risk"
   - severity: "hard_blocker" | "concern" | "question"
   - resolution_status: "unresolved" | "partially_resolved" | "resolved"
   - what_would_resolve: what needs to happen to resolve this

4. "buying_signals": array of objects with:
   - signal: what the positive indicator is
   - strength: "strong" | "moderate" | "weak"
   - type: "timeline" | "budget" | "need" | "champion" | "process" | "urgency"
   - verbatim_quote: closest quote (or null)

5. "stakeholders": array of objects with:
   - name: person's name (or role if name not given)
   - role: their role/title
   - influence: "decision_maker" | "influencer" | "evaluator" | "user" | "blocker" | "unknown"
   - sentiment: "positive" | "neutral" | "negative" | "unknown"
   - mentioned_context: why/how they were mentioned

6. "competitive_context": array of objects with:
   - competitor_or_alternative: name of competitor/current solution/alternative
   - type: "current_solution" | "competitor" | "alternative_approach"
   - sentiment: "positive" | "negative" | "neutral" — how prospect feels about it
   - details: additional context

7. "next_steps": array of objects with:
   - action: what needs to happen
   - owner: who owns this action ("rep" | "prospect" | "both" | specific name)
   - timeline: when it should happen (or null)
   - commitment_level: "firm" | "tentative" | "suggested"

8. "meeting_metadata": object with:
   - meeting_count: how many meetings are in this transcript
   - meetings: array of objects with {number, topic, tone}
   - overall_deal_stage: "discovery" | "evaluation" | "proposal" | "negotiation" | "closed"
   - momentum: "accelerating" | "steady" | "stalling" | "unknown"

Return ONLY valid JSON. Be thorough — extract everything, even subtle signals."""


def extract_sales_entities(
    transcript: str,
    llm: LLMClient,
    model: str = "gpt-4o-mini",
) -> dict[str, Any]:
    """
    Extract sales-specific entities from a transcript using LLM.
    
    Returns structured dict with all entity types.
    """
    messages = [
        {"role": "system", "content": SALES_EXTRACTION_PROMPT},
        {"role": "user", "content": f"TRANSCRIPT:\n\n{transcript}"},
    ]
    
    result = llm.chat(
        messages=messages,
        model=model,
        json_mode=True,
        max_tokens=16000,  # reasoning models need headroom for thinking + output
    )
    
    extracted = result["content"]
    tokens = result["tokens"]
    
    logger.info(
        f"Extraction complete: {tokens['total']} tokens, "
        f"{len(extracted.get('pain_points', []))} pain points, "
        f"{len(extracted.get('buying_signals', []))} buying signals"
    )
    
    return {
        "extraction": extracted,
        "tokens": tokens,
    }


@dataclass
class SalesEntity:
    """A sales-domain entity for the prospect model."""
    id: str = ""
    type: str = ""  # prospect, pain_point, objection, buying_signal, stakeholder, competitive, next_step
    name: str = ""
    data: dict = field(default_factory=dict)
    meeting_number: int | None = None
    confidence: float = 1.0
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "name": self.name,
            "data": self.data,
            "meeting_number": self.meeting_number,
            "confidence": self.confidence,
        }
