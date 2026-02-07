"""
Box Demo Data Loader

Loads realistic Box sales data for the Prism demo.
"""

import json
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import random

DATA_DIR = Path(__file__).parent / "data"


class Stage(Enum):
    LEAD = "lead"
    DISCOVERY = "discovery"
    EVALUATION = "evaluation"
    PROPOSAL = "proposal"
    NEGOTIATION = "negotiation"
    CLOSED_WON = "closed_won"
    CLOSED_LOST = "closed_lost"


@dataclass
class Stakeholder:
    name: str
    role: str  # Champion, Economic Buyer, Technical Evaluator, Blocker
    title: str
    sentiment: str = "neutral"


@dataclass
class Prospect:
    id: str
    name: str
    title: str
    company: str
    vertical: str
    stage: Stage
    deal_value: float
    pain_points: list[str]
    objections: list[str]
    stakeholders: list[Stakeholder]
    next_step: str
    created_at: datetime
    probability: float = 0.0
    days_in_stage: int = 0
    
    def __post_init__(self):
        # Calculate probability based on stage
        stage_probs = {
            Stage.LEAD: 0.10,
            Stage.DISCOVERY: 0.20,
            Stage.EVALUATION: 0.40,
            Stage.PROPOSAL: 0.60,
            Stage.NEGOTIATION: 0.80,
            Stage.CLOSED_WON: 1.0,
            Stage.CLOSED_LOST: 0.0,
        }
        self.probability = stage_probs.get(self.stage, 0.0)


@dataclass  
class Activity:
    timestamp: datetime
    type: str  # email, call, meeting, note
    subject: str
    outcome: str
    next_action: Optional[str] = None


@dataclass
class ProspectTimeline:
    prospect_id: str
    prospect_name: str
    company: str
    current_stage: Stage
    current_stage_display: str
    deal_value: float
    probability: float
    activities: list[Activity]
    days_in_pipeline: int


def load_box_prospects() -> list[Prospect]:
    """Load prospects from Box demo data."""
    prospects_file = DATA_DIR / "box_prospects.json"
    
    if not prospects_file.exists():
        print(f"Warning: {prospects_file} not found, using defaults")
        return _get_default_prospects()
    
    with open(prospects_file) as f:
        data = json.load(f)
    
    prospects = []
    for p in data:
        stakeholders = [
            Stakeholder(
                name=s["name"],
                role=s["role"],
                title=s["title"],
                sentiment=s.get("sentiment", "neutral")
            )
            for s in p.get("stakeholders", [])
        ]
        
        created_at = datetime.fromisoformat(p["created_at"].replace("Z", "+00:00"))
        days_in_stage = (datetime.now(created_at.tzinfo) - created_at).days if created_at.tzinfo else (datetime.now() - created_at).days
        
        prospect = Prospect(
            id=p["id"],
            name=p["name"],
            title=p["title"],
            company=p["company"],
            vertical=p["vertical"],
            stage=Stage(p["stage"]),
            deal_value=p["deal_value"],
            pain_points=p["pain_points"],
            objections=p["objections"],
            stakeholders=stakeholders,
            next_step=p["next_step"],
            created_at=created_at,
            days_in_stage=days_in_stage,
        )
        prospects.append(prospect)
    
    return prospects


def load_prospect_timelines() -> list[ProspectTimeline]:
    """Generate timelines for Box prospects."""
    prospects = load_box_prospects()
    timelines = []
    
    stage_display = {
        Stage.LEAD: "Lead",
        Stage.DISCOVERY: "Discovery", 
        Stage.EVALUATION: "Evaluation",
        Stage.PROPOSAL: "Proposal",
        Stage.NEGOTIATION: "Negotiation",
        Stage.CLOSED_WON: "Closed Won",
        Stage.CLOSED_LOST: "Closed Lost",
    }
    
    for p in prospects:
        # Generate realistic activities
        activities = _generate_activities(p)
        
        timeline = ProspectTimeline(
            prospect_id=p.id,
            prospect_name=p.name,
            company=p.company,
            current_stage=p.stage,
            current_stage_display=stage_display[p.stage],
            deal_value=p.deal_value,
            probability=p.probability,
            activities=activities,
            days_in_pipeline=p.days_in_stage,
        )
        timelines.append(timeline)
    
    return timelines


def _generate_activities(prospect: Prospect) -> list[Activity]:
    """Generate realistic activity history for a prospect."""
    activities = []
    
    activity_templates = {
        Stage.LEAD: [
            ("email", "Initial outreach", "Opened email, no reply yet", "Follow up in 3 days"),
        ],
        Stage.DISCOVERY: [
            ("email", "Initial outreach", "Positive response, interested", "Schedule discovery call"),
            ("call", "Discovery call", "Identified 3 key pain points", "Send case study"),
            ("email", "Case study follow-up", "Shared with team", "Schedule demo"),
        ],
        Stage.EVALUATION: [
            ("email", "Initial outreach", "Positive response", "Schedule call"),
            ("call", "Discovery call", "Mapped stakeholders", "Schedule demo"),
            ("meeting", "Product demo", "Strong interest from team", "Send proposal"),
            ("email", "Technical questions", "Answered security concerns", "Schedule architecture review"),
        ],
        Stage.PROPOSAL: [
            ("email", "Initial outreach", "Positive response", "Schedule call"),
            ("call", "Discovery call", "Identified budget and timeline", "Schedule demo"),
            ("meeting", "Product demo", "Positive feedback", "Send proposal"),
            ("email", "Proposal sent", "Under review", "Follow up on questions"),
            ("call", "Proposal review", "Negotiating terms", "Revise pricing"),
        ],
        Stage.NEGOTIATION: [
            ("email", "Initial outreach", "Positive response", "Schedule call"),
            ("call", "Discovery call", "Strong fit identified", "Schedule demo"),
            ("meeting", "Product demo", "Very positive", "Send proposal"),
            ("email", "Proposal sent", "Moving to legal", "Address legal questions"),
            ("call", "Legal review", "Minor redlines", "Finalize contract"),
            ("meeting", "Contract negotiation", "Terms agreed", "Awaiting signature"),
        ],
    }
    
    templates = activity_templates.get(prospect.stage, [])
    base_date = prospect.created_at
    
    for i, (atype, subject, outcome, next_action) in enumerate(templates):
        activity = Activity(
            timestamp=base_date + timedelta(days=i*3 + random.randint(0, 2)),
            type=atype,
            subject=subject,
            outcome=outcome,
            next_action=next_action if i == len(templates) - 1 else None,
        )
        activities.append(activity)
    
    return activities


def _get_default_prospects() -> list[Prospect]:
    """Fallback prospects if Box data not available."""
    return [
        Prospect(
            id="default-001",
            name="Demo User",
            title="IT Director",
            company="Acme Corp",
            vertical="Enterprise",
            stage=Stage.DISCOVERY,
            deal_value=100000,
            pain_points=["Legacy systems"],
            objections=[],
            stakeholders=[],
            next_step="Schedule demo",
            created_at=datetime.now() - timedelta(days=7),
        )
    ]


def get_funnel_data() -> list[dict]:
    """Get funnel statistics."""
    prospects = load_box_prospects()
    
    stages = [Stage.LEAD, Stage.DISCOVERY, Stage.EVALUATION, Stage.PROPOSAL, Stage.NEGOTIATION, Stage.CLOSED_WON, Stage.CLOSED_LOST]
    stage_display = {
        Stage.LEAD: "Lead",
        Stage.DISCOVERY: "Discovery",
        Stage.EVALUATION: "Evaluation", 
        Stage.PROPOSAL: "Proposal",
        Stage.NEGOTIATION: "Negotiation",
        Stage.CLOSED_WON: "Closed Won",
        Stage.CLOSED_LOST: "Closed Lost",
    }
    
    total_leads = len(prospects)
    funnel = []
    
    for stage in stages:
        count = len([p for p in prospects if p.stage == stage])
        value = sum(p.deal_value for p in prospects if p.stage == stage)
        
        # Calculate conversion from lead
        conversion = (count / total_leads * 100) if total_leads > 0 else 0
        
        funnel.append({
            "stage": stage.value,
            "display_name": stage_display[stage],
            "count": count,
            "value": value,
            "conversion_from_lead": round(conversion, 1),
        })
    
    return funnel


def get_total_pipeline_value() -> float:
    """Get total pipeline value (excluding closed)."""
    prospects = load_box_prospects()
    return sum(
        p.deal_value for p in prospects 
        if p.stage not in [Stage.CLOSED_WON, Stage.CLOSED_LOST]
    )


def get_weighted_pipeline_value() -> float:
    """Get probability-weighted pipeline value."""
    prospects = load_box_prospects()
    return sum(
        p.deal_value * p.probability for p in prospects
        if p.stage not in [Stage.CLOSED_WON, Stage.CLOSED_LOST]
    )


def get_demo_prospects() -> list[dict]:
    """Get demo prospects as dicts for Flask templates."""
    prospects = load_box_prospects()
    return [
        {
            "id": p.id,
            "name": p.name,
            "title": p.title,
            "company": p.company,
            "vertical": p.vertical,
            "stage": p.stage.value,
            "deal_value": p.deal_value,
            "probability": p.probability,
            "pain_points": p.pain_points,
            "objections": p.objections,
            "stakeholders": [
                {"name": s.name, "role": s.role, "title": s.title, "sentiment": s.sentiment}
                for s in p.stakeholders
            ],
            "next_step": p.next_step,
            "days_in_stage": p.days_in_stage,
        }
        for p in prospects
    ]


def get_pipeline_summary() -> dict:
    """Get pipeline summary stats."""
    prospects = load_box_prospects()
    active = [p for p in prospects if p.stage not in [Stage.CLOSED_WON, Stage.CLOSED_LOST]]
    
    total_value = sum(p.deal_value for p in active)
    weighted_value = sum(p.deal_value * p.probability for p in active)
    
    return {
        # Keys expected by app.py
        "total_deals": len(active),
        "total_pipeline_value": total_value,
        "weighted_pipeline_value": weighted_value,
        # Additional keys
        "total_value": total_value,
        "weighted_value": weighted_value,
        "active_deals": len(active),
        "avg_deal_size": total_value / len(active) if active else 0,
        "by_stage": {
            stage.value: len([p for p in prospects if p.stage == stage])
            for stage in Stage
        },
    }
