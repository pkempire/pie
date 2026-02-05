"""
Prospect World Model — builds temporal entity graphs per prospect.

Tracks state changes across meetings, links to company-level intelligence.
Uses PIE's WorldModel patterns but with sales-specific entity types.
"""

from __future__ import annotations
import json
import logging
import sys
import uuid
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("sales.prospect_model")


@dataclass
class TemporalState:
    """A state snapshot at a point in time."""
    meeting_number: int
    state: dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> dict:
        return {
            "meeting_number": self.meeting_number,
            "state": self.state,
            "timestamp": self.timestamp,
        }


@dataclass
class StateEvolution:
    """Tracks how an entity evolved across meetings."""
    entity_type: str
    entity_key: str  # unique key for this entity within its type
    states: list[TemporalState] = field(default_factory=list)
    
    @property
    def changed(self) -> bool:
        """Did this entity change across meetings?"""
        if len(self.states) < 2:
            return False
        return self.states[0].state != self.states[-1].state
    
    @property
    def evolution_summary(self) -> str:
        """Human-readable summary of how this evolved."""
        if len(self.states) == 1:
            return f"First mentioned in Meeting {self.states[0].meeting_number}"
        
        changes = []
        for i in range(1, len(self.states)):
            prev = self.states[i-1]
            curr = self.states[i]
            # Find what changed
            for key in set(list(prev.state.keys()) + list(curr.state.keys())):
                old_val = prev.state.get(key)
                new_val = curr.state.get(key)
                if old_val != new_val and key not in ("verbatim_quote", "mentioned_context"):
                    changes.append(
                        f"Meeting {prev.meeting_number}→{curr.meeting_number}: "
                        f"{key}: {old_val} → {new_val}"
                    )
        
        return "; ".join(changes) if changes else "No significant changes"
    
    def to_dict(self) -> dict:
        return {
            "entity_type": self.entity_type,
            "entity_key": self.entity_key,
            "states": [s.to_dict() for s in self.states],
            "changed": self.changed,
            "evolution_summary": self.evolution_summary,
        }


@dataclass
class ProspectModel:
    """Complete intelligence model for a single prospect."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Core prospect info
    name: str = ""
    title: str = ""
    company: str = ""
    email: str = ""
    
    # Personality model
    personality_traits: list[str] = field(default_factory=list)
    communication_style: str = ""
    decision_style: str = ""
    
    # Entity collections (temporal)
    pain_points: list[dict] = field(default_factory=list)
    objections: list[dict] = field(default_factory=list)
    buying_signals: list[dict] = field(default_factory=list)
    stakeholders: list[dict] = field(default_factory=list)
    competitive_context: list[dict] = field(default_factory=list)
    next_steps: list[dict] = field(default_factory=list)
    
    # Meeting metadata
    meeting_metadata: dict = field(default_factory=dict)
    
    # Temporal evolution tracking
    evolutions: list[StateEvolution] = field(default_factory=list)
    
    # Web enrichment (filled later)
    web_enrichment: dict = field(default_factory=dict)
    
    # Uncertainty flags (filled later)
    uncertainties: list[dict] = field(default_factory=list)
    
    # Raw extraction data
    raw_extractions: list[dict] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "title": self.title,
            "company": self.company,
            "email": self.email,
            "personality_traits": self.personality_traits,
            "communication_style": self.communication_style,
            "decision_style": self.decision_style,
            "pain_points": self.pain_points,
            "objections": self.objections,
            "buying_signals": self.buying_signals,
            "stakeholders": self.stakeholders,
            "competitive_context": self.competitive_context,
            "next_steps": self.next_steps,
            "meeting_metadata": self.meeting_metadata,
            "evolutions": [e.to_dict() for e in self.evolutions],
            "web_enrichment": self.web_enrichment,
            "uncertainties": self.uncertainties,
        }


class ProspectWorldModel:
    """
    Builds and maintains prospect models from sales transcripts.
    
    Ingests extraction results, tracks temporal evolution,
    and links to company-level intelligence.
    """
    
    def __init__(self, output_dir: str | Path = "sales/output"):
        self.prospects: dict[str, ProspectModel] = {}  # name -> model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def ingest_extraction(self, extraction: dict) -> ProspectModel:
        """
        Ingest a sales extraction result and build/update a prospect model.
        
        Args:
            extraction: The raw extraction dict from extract_sales_entities
            
        Returns:
            The updated ProspectModel
        """
        data = extraction.get("extraction", extraction)
        
        # Get or create prospect
        prospect_data = data.get("prospect", {})
        name = prospect_data.get("name", "Unknown")
        
        if name in self.prospects:
            model = self.prospects[name]
        else:
            model = ProspectModel(name=name)
            self.prospects[name] = model
        
        # Update core info
        model.title = prospect_data.get("title", model.title) or model.title
        model.company = prospect_data.get("company", model.company) or model.company
        model.email = prospect_data.get("email", model.email) or model.email
        model.personality_traits = prospect_data.get("personality_traits", model.personality_traits)
        model.communication_style = prospect_data.get("communication_style", model.communication_style) or model.communication_style
        model.decision_style = prospect_data.get("decision_style", model.decision_style) or model.decision_style
        
        # Store entity collections
        model.pain_points = data.get("pain_points", [])
        model.objections = data.get("objections", [])
        model.buying_signals = data.get("buying_signals", [])
        model.stakeholders = data.get("stakeholders", [])
        model.competitive_context = data.get("competitive_context", [])
        model.next_steps = data.get("next_steps", [])
        model.meeting_metadata = data.get("meeting_metadata", {})
        
        # Store raw
        model.raw_extractions.append(data)
        
        # Build temporal evolution
        model.evolutions = self._build_evolutions(data)
        
        logger.info(
            f"Ingested prospect: {name} ({model.company}) — "
            f"{len(model.pain_points)} pain points, "
            f"{len(model.buying_signals)} signals, "
            f"{len(model.objections)} objections"
        )
        
        return model
    
    def _build_evolutions(self, data: dict) -> list[StateEvolution]:
        """Build temporal evolution tracking from extraction data."""
        evolutions = []
        
        # Track pain point evolution
        pain_by_desc = {}
        for pp in data.get("pain_points", []):
            key = pp.get("description", "")[:50]
            meeting = pp.get("meeting_number", 1)
            if key not in pain_by_desc:
                pain_by_desc[key] = StateEvolution(
                    entity_type="pain_point",
                    entity_key=key,
                )
            pain_by_desc[key].states.append(TemporalState(
                meeting_number=meeting,
                state={
                    "severity": pp.get("severity"),
                    "category": pp.get("category"),
                    "evolution": pp.get("evolution"),
                },
            ))
        evolutions.extend(pain_by_desc.values())
        
        # Track objection evolution
        obj_by_desc = {}
        for obj in data.get("objections", []):
            key = obj.get("description", "")[:50]
            if key not in obj_by_desc:
                obj_by_desc[key] = StateEvolution(
                    entity_type="objection",
                    entity_key=key,
                )
            obj_by_desc[key].states.append(TemporalState(
                meeting_number=1,  # objections from extraction
                state={
                    "severity": obj.get("severity"),
                    "resolution_status": obj.get("resolution_status"),
                    "type": obj.get("type"),
                },
            ))
        evolutions.extend(obj_by_desc.values())
        
        # Track buying signal evolution
        sig_by_desc = {}
        for sig in data.get("buying_signals", []):
            key = sig.get("signal", "")[:50]
            if key not in sig_by_desc:
                sig_by_desc[key] = StateEvolution(
                    entity_type="buying_signal",
                    entity_key=key,
                )
            sig_by_desc[key].states.append(TemporalState(
                meeting_number=1,
                state={
                    "strength": sig.get("strength"),
                    "type": sig.get("type"),
                },
            ))
        evolutions.extend(sig_by_desc.values())
        
        return evolutions
    
    def get_prospect(self, name: str) -> ProspectModel | None:
        """Get a prospect model by name."""
        return self.prospects.get(name)
    
    def save(self):
        """Save all prospect models to JSON."""
        all_data = {}
        for name, model in self.prospects.items():
            all_data[name] = model.to_dict()
        
        output_path = self.output_dir / "prospects.json"
        with open(output_path, "w") as f:
            json.dump(all_data, f, indent=2, default=str)
        
        logger.info(f"Saved {len(self.prospects)} prospects to {output_path}")
        return output_path
    
    def save_individual(self, name: str) -> Path:
        """Save a single prospect model to its own JSON file."""
        model = self.prospects.get(name)
        if not model:
            raise ValueError(f"Prospect {name} not found")
        
        safe_name = name.lower().replace(" ", "_").replace(",", "")
        output_path = self.output_dir / f"prospect_{safe_name}.json"
        
        with open(output_path, "w") as f:
            json.dump(model.to_dict(), f, indent=2, default=str)
        
        return output_path
    
    def build_simulation_prompt(self, name: str, pitch: str = "") -> str:
        """
        Build a simulation prompt for a prospect.
        
        This creates a detailed persona prompt that captures everything we know
        about the prospect, so an LLM can simulate their likely response.
        """
        model = self.prospects.get(name)
        if not model:
            return f"No data available for {name}"
        
        # Handle both string list and dict list formats for personality_traits
        traits = []
        for t in model.personality_traits:
            if isinstance(t, dict):
                traits.append(t.get("trait", t.get("name", str(t))))
            else:
                traits.append(str(t))
        
        # Extract communication and decision style from dict or string
        comm_style = model.communication_style
        if isinstance(comm_style, dict):
            comm_style = comm_style.get("style", str(comm_style))
        
        dec_style = model.decision_style
        if isinstance(dec_style, dict):
            dec_style = dec_style.get("style", str(dec_style))
        
        prompt_parts = [
            f"You are {model.name}, {model.title} at {model.company}.",
            "",
            "YOUR PERSONALITY AND STYLE:",
            f"- Traits: {', '.join(traits)}",
            f"- Communication style: {comm_style}",
            f"- Decision style: {dec_style}",
            "",
            "YOUR PAIN POINTS:",
        ]
        
        for pp in model.pain_points:
            prompt_parts.append(
                f"- [{pp.get('severity', 'medium')}] {pp.get('description', '')}"
            )
        
        prompt_parts.append("")
        prompt_parts.append("YOUR CONCERNS/OBJECTIONS:")
        for obj in model.objections:
            prompt_parts.append(
                f"- [{obj.get('type', 'unknown')}] {obj.get('description', '')} "
                f"(Status: {obj.get('resolution_status', 'unknown')})"
            )
        
        if model.competitive_context:
            prompt_parts.append("")
            prompt_parts.append("YOUR CURRENT SOLUTIONS:")
            for cc in model.competitive_context:
                prompt_parts.append(
                    f"- {cc.get('competitor_or_alternative', '')}: {cc.get('details', '')}"
                )
        
        if model.web_enrichment:
            company_info = model.web_enrichment.get("company", {})
            if company_info:
                prompt_parts.append("")
                prompt_parts.append("YOUR COMPANY CONTEXT:")
                if company_info.get("description"):
                    prompt_parts.append(f"- About: {company_info['description']}")
                if company_info.get("recent_news"):
                    for news in company_info["recent_news"][:3]:
                        prompt_parts.append(f"- Recent: {news}")
        
        prompt_parts.append("")
        prompt_parts.append("WHERE YOU ARE IN THE PROCESS:")
        prompt_parts.append(f"- Stage: {model.meeting_metadata.get('overall_deal_stage', 'unknown')}")
        prompt_parts.append(f"- Momentum: {model.meeting_metadata.get('momentum', 'unknown')}")
        
        if model.next_steps:
            prompt_parts.append("")
            prompt_parts.append("AGREED NEXT STEPS:")
            for ns in model.next_steps:
                prompt_parts.append(f"- {ns.get('action', '')} (owner: {ns.get('owner', 'unknown')})")
        
        prompt_parts.append("")
        prompt_parts.append(
            "Respond AS this person would. Stay in character. "
            "Be realistic about your concerns and enthusiasm level. "
            "Don't be overly positive or negative — be authentic to the personality above."
        )
        
        if pitch:
            prompt_parts.append("")
            prompt_parts.append(f"THE REP SAYS TO YOU:\n{pitch}")
        
        return "\n".join(prompt_parts)
