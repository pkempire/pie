"""
Proactive Uncertainty Detection — surface questions instead of hallucinated answers.

Detects:
- Contradictions between meetings
- Low confidence on important signals
- Missing information that would change the simulation
- Ambiguous statements with multiple interpretations
"""

from __future__ import annotations
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pie.core.llm import LLMClient

logger = logging.getLogger("sales.uncertainty")


UNCERTAINTY_PROMPT = """You are a sales intelligence analyst focused on identifying WHAT WE DON'T KNOW.

Given a prospect's extracted intelligence data, identify uncertainties, contradictions, 
and gaps that could affect sales strategy.

For each uncertainty, categorize it:

1. **contradiction**: Two pieces of information that conflict
   - Example: Meeting 1 says "budget approved" but Meeting 2 says "need stakeholder buy-in"

2. **missing_critical**: Information we don't have that would significantly change our approach
   - Example: We don't know their evaluation timeline
   - Example: We don't know who the final decision maker is

3. **ambiguous**: Statements that could be interpreted multiple ways
   - Example: "I'll loop in stakeholders" — are they supportive or hedging?

4. **low_confidence**: Signals where our confidence should be low
   - Example: We inferred budget availability from indirect signals only

5. **stale**: Information that might be outdated
   - Example: Pain point mentioned in Meeting 1 but not revisited — still relevant?

Return JSON with:
{
  "uncertainties": [
    {
      "type": "contradiction" | "missing_critical" | "ambiguous" | "low_confidence" | "stale",
      "entity_type": "pain_point" | "objection" | "buying_signal" | "stakeholder" | "competitive" | "next_step" | "prospect",
      "entity_reference": "brief description of the entity this relates to",
      "question": "The specific question we should ask or investigate",
      "context": "Why this matters / what we observed that triggered this flag",
      "impact": "high" | "medium" | "low" — how much this affects our strategy,
      "suggested_action": "What the rep should do to resolve this uncertainty"
    }
  ],
  "overall_confidence": 0.0-1.0 — how confident are we in our overall read of this prospect,
  "top_3_questions": ["The 3 most important questions we need answered to move this deal forward"]
}

Be thorough. It's better to flag too many uncertainties than to miss a critical one.
The goal is to prevent the sales team from acting on assumptions."""


def detect_uncertainties(
    prospect_data: dict,
    llm: LLMClient,
    model: str = "gpt-4o-mini",
) -> dict[str, Any]:
    """
    Analyze a prospect's data for uncertainties and contradictions.
    
    Args:
        prospect_data: The full prospect model dict
        llm: LLM client for analysis
        model: Model to use
        
    Returns:
        Dict with uncertainties, overall_confidence, top_3_questions
    """
    # Build a focused view of the data for the LLM
    focused_data = {
        "prospect": {
            "name": prospect_data.get("name"),
            "title": prospect_data.get("title"),
            "company": prospect_data.get("company"),
            "personality_traits": prospect_data.get("personality_traits"),
            "decision_style": prospect_data.get("decision_style"),
        },
        "pain_points": prospect_data.get("pain_points", []),
        "objections": prospect_data.get("objections", []),
        "buying_signals": prospect_data.get("buying_signals", []),
        "stakeholders": prospect_data.get("stakeholders", []),
        "competitive_context": prospect_data.get("competitive_context", []),
        "next_steps": prospect_data.get("next_steps", []),
        "meeting_metadata": prospect_data.get("meeting_metadata", {}),
        "temporal_evolution": [
            e for e in prospect_data.get("evolutions", [])
        ],
    }
    
    messages = [
        {"role": "system", "content": UNCERTAINTY_PROMPT},
        {
            "role": "user",
            "content": f"Analyze this prospect intelligence for uncertainties:\n\n{json.dumps(focused_data, indent=2, default=str)}",
        },
    ]
    
    result = llm.chat(
        messages=messages,
        model=model,
        json_mode=True,
        max_tokens=16000,  # reasoning models need headroom for thinking + output
    )
    
    uncertainties = result["content"]
    tokens = result["tokens"]
    
    logger.info(
        f"Uncertainty detection: {len(uncertainties.get('uncertainties', []))} flags, "
        f"confidence: {uncertainties.get('overall_confidence', 'N/A')}, "
        f"{tokens['total']} tokens"
    )
    
    return uncertainties


def add_rule_based_uncertainties(
    prospect_data: dict,
    llm_uncertainties: dict,
) -> dict:
    """
    Add rule-based uncertainty checks on top of LLM analysis.
    
    These are deterministic checks that don't need an LLM.
    """
    uncertainties = llm_uncertainties.get("uncertainties", [])
    
    # Check: No clear decision maker identified
    stakeholders = prospect_data.get("stakeholders", [])
    decision_makers = [s for s in stakeholders if s.get("influence") == "decision_maker"]
    if not decision_makers:
        uncertainties.append({
            "type": "missing_critical",
            "entity_type": "stakeholder",
            "entity_reference": "Decision maker",
            "question": "Who is the final decision maker for this deal?",
            "context": f"We identified {len(stakeholders)} stakeholders but none confirmed as the decision maker.",
            "impact": "high",
            "suggested_action": "Ask the prospect directly who makes the final call and what their approval process looks like.",
        })
    
    # Check: No budget/timeline signals
    buying_signals = prospect_data.get("buying_signals", [])
    budget_signals = [s for s in buying_signals if s.get("type") == "budget"]
    timeline_signals = [s for s in buying_signals if s.get("type") == "timeline"]
    
    if not budget_signals:
        uncertainties.append({
            "type": "missing_critical",
            "entity_type": "buying_signal",
            "entity_reference": "Budget",
            "question": "Is there allocated budget for this initiative?",
            "context": "No budget-related signals detected in the transcripts.",
            "impact": "high",
            "suggested_action": "In the next meeting, explore budget availability and fiscal year timing.",
        })
    
    if not timeline_signals:
        uncertainties.append({
            "type": "missing_critical",
            "entity_type": "buying_signal",
            "entity_reference": "Timeline",
            "question": "What is the prospect's evaluation/implementation timeline?",
            "context": "No timeline signals detected. We don't know when they want to move.",
            "impact": "high",
            "suggested_action": "Ask about their target go-live date and any deadlines driving urgency.",
        })
    
    # Check: All objections unresolved
    objections = prospect_data.get("objections", [])
    unresolved = [o for o in objections if o.get("resolution_status") == "unresolved"]
    hard_blockers = [o for o in objections if o.get("severity") == "hard_blocker"]
    
    if hard_blockers:
        for blocker in hard_blockers:
            uncertainties.append({
                "type": "low_confidence",
                "entity_type": "objection",
                "entity_reference": blocker.get("description", "Unknown blocker"),
                "question": f"Can we actually resolve: {blocker.get('description', '')}?",
                "context": "This was flagged as a hard blocker. If we can't resolve it, the deal is at risk.",
                "impact": "high",
                "suggested_action": blocker.get("what_would_resolve", "Investigate resolution path."),
            })
    
    # Check: Mentioned stakeholders not yet engaged
    mentioned_stakeholders = [
        s for s in stakeholders 
        if s.get("influence") in ("decision_maker", "blocker")
        and s.get("sentiment") == "unknown"
    ]
    for s in mentioned_stakeholders:
        uncertainties.append({
            "type": "missing_critical",
            "entity_type": "stakeholder",
            "entity_reference": s.get("name", "Unknown"),
            "question": f"What is {s.get('name', 'this stakeholder')}'s position on this initiative?",
            "context": f"{s.get('name', 'A stakeholder')} was mentioned as {s.get('influence', 'important')} but we don't know their sentiment.",
            "impact": "medium",
            "suggested_action": f"Get a meeting with {s.get('name', 'this stakeholder')} or ask the champion about their likely position.",
        })
    
    llm_uncertainties["uncertainties"] = uncertainties
    return llm_uncertainties
