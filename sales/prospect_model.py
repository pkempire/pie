"""
Prospect World Model - PIE-Powered

Each prospect is a mini world model:
- Entities extracted from transcripts/notes via PIE
- Web-grounded with real data (Brave API)
- Dynamic state tracking over time
- No hardcoded data - everything extracted or enriched live
"""

from __future__ import annotations
import json
import os
import sys
import uuid
import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Any
import logging

# Add parent for PIE imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pie.core.llm import LLMClient
from pie.resolution.web_grounder import WebGrounder

logger = logging.getLogger(__name__)

# Storage
DATA_DIR = Path(__file__).parent / "prospects"
DATA_DIR.mkdir(exist_ok=True)


@dataclass
class Stakeholder:
    name: str
    title: str
    role: str  # champion, economic_buyer, technical_buyer, blocker, influencer
    email: Optional[str] = None
    linkedin: Optional[str] = None
    sentiment: str = "neutral"  # positive, neutral, negative
    notes: list[str] = field(default_factory=list)
    
    # Web-grounded data
    enriched: bool = False
    company_tenure: Optional[str] = None
    previous_companies: list[str] = field(default_factory=list)
    education: Optional[str] = None


@dataclass
class PainPoint:
    description: str
    severity: str  # critical, high, medium, low
    category: str  # budget, timeline, technical, political, competition
    verbatim_quote: Optional[str] = None
    source: str = "transcript"  # transcript, web, inferred
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class Objection:
    description: str
    status: str = "open"  # open, addressed, resolved
    response: Optional[str] = None
    source: str = "transcript"


@dataclass
class Activity:
    timestamp: str
    type: str  # email, call, meeting, note
    subject: str
    summary: str
    outcome: Optional[str] = None
    next_steps: list[str] = field(default_factory=list)
    raw_content: Optional[str] = None  # Original transcript/note


@dataclass
class ProspectModel:
    """Simplified prospect model for simulation (matches InteractionSimulator interface)."""
    id: str
    name: str
    title: str = ""
    company: str = ""
    
    # For simulation
    pain_points: list[dict] = field(default_factory=list)
    objections: list[dict] = field(default_factory=list)
    buying_signals: list[dict] = field(default_factory=list)
    competitive_context: list[dict] = field(default_factory=list)
    
    # Personality (optional)
    communication_style: Optional[str] = None
    decision_style: Optional[str] = None
    personality_traits: list[str] = field(default_factory=list)
    
    @classmethod
    def from_world_model(cls, wm: "ProspectWorldModel") -> "ProspectModel":
        """Convert from full world model."""
        return cls(
            id=wm.id,
            name=wm.name,
            title=wm.title,
            company=wm.company,
            pain_points=[{"description": p.description, "severity": p.severity} for p in wm.pain_points],
            objections=[{"text": o.description, "strength": 0.7 if o.status == "open" else 0.3} for o in wm.objections],
            buying_signals=[{"signal": s} for s in wm.buying_signals],
            competitive_context=[{"competitor": c, "relationship": "mentioned"} for c in wm.competitors_mentioned],
        )


@dataclass 
class ProspectWorldModel:
    """A prospect as a PIE-style world model."""
    
    id: str
    name: str
    title: str
    company: str
    email: Optional[str] = None
    
    # Deal info
    stage: str = "lead"  # lead, discovery, evaluation, proposal, negotiation, closed_won, closed_lost
    deal_value: float = 0.0
    probability: float = 0.0
    close_date: Optional[str] = None
    
    # Extracted entities
    stakeholders: list[Stakeholder] = field(default_factory=list)
    pain_points: list[PainPoint] = field(default_factory=list)
    objections: list[Objection] = field(default_factory=list)
    buying_signals: list[str] = field(default_factory=list)
    
    # Company context (web-grounded)
    company_description: Optional[str] = None
    company_industry: Optional[str] = None
    company_size: Optional[str] = None
    company_recent_news: list[str] = field(default_factory=list)
    competitors_mentioned: list[str] = field(default_factory=list)
    
    # Activity timeline
    activities: list[Activity] = field(default_factory=list)
    
    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    enriched_at: Optional[str] = None
    
    def save(self):
        """Save to disk."""
        self.updated_at = datetime.now().isoformat()
        path = DATA_DIR / f"{self.id}.json"
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2, default=str)
        logger.info(f"Saved prospect {self.id}: {self.name}")
    
    @classmethod
    def load(cls, prospect_id: str) -> Optional["ProspectWorldModel"]:
        """Load from disk."""
        path = DATA_DIR / f"{prospect_id}.json"
        if not path.exists():
            return None
        with open(path) as f:
            data = json.load(f)
        # Reconstruct nested dataclasses
        data['stakeholders'] = [Stakeholder(**s) for s in data.get('stakeholders', [])]
        data['pain_points'] = [PainPoint(**p) for p in data.get('pain_points', [])]
        data['objections'] = [Objection(**o) for o in data.get('objections', [])]
        data['activities'] = [Activity(**a) for a in data.get('activities', [])]
        return cls(**data)
    
    @classmethod
    def list_all(cls) -> list["ProspectWorldModel"]:
        """List all prospects."""
        prospects = []
        for path in DATA_DIR.glob("*.json"):
            try:
                prospect = cls.load(path.stem)
                if prospect:
                    prospects.append(prospect)
            except Exception as e:
                logger.error(f"Failed to load {path}: {e}")
        return sorted(prospects, key=lambda p: p.updated_at, reverse=True)
    
    def calculate_probability(self):
        """Calculate win probability based on stage and signals."""
        stage_probs = {
            "lead": 0.05,
            "discovery": 0.15,
            "evaluation": 0.35,
            "proposal": 0.55,
            "negotiation": 0.75,
            "closed_won": 1.0,
            "closed_lost": 0.0,
        }
        base = stage_probs.get(self.stage, 0.1)
        
        # Adjust for signals
        if any("champion" in s.role.lower() for s in self.stakeholders):
            base += 0.10
        if len(self.pain_points) >= 3:
            base += 0.05
        if any(s.sentiment == "positive" for s in self.stakeholders):
            base += 0.05
        if any(o.status == "open" for o in self.objections):
            base -= 0.10
            
        self.probability = min(max(base, 0.0), 1.0)
        return self.probability


class ProspectExtractor:
    """Extract prospect data from transcripts using PIE/LLM."""
    
    EXTRACTION_PROMPT = """Analyze this sales conversation and extract structured data.

TRANSCRIPT:
{transcript}

Extract the following as JSON:
{{
    "prospect_name": "Full name of the main prospect",
    "prospect_title": "Their job title",
    "company": "Company name",
    "stakeholders": [
        {{"name": "...", "title": "...", "role": "champion|economic_buyer|technical_buyer|blocker|influencer"}}
    ],
    "pain_points": [
        {{"description": "...", "severity": "critical|high|medium|low", "category": "budget|timeline|technical|political|competition", "verbatim_quote": "exact quote if available"}}
    ],
    "objections": [
        {{"description": "...", "status": "open|addressed"}}
    ],
    "buying_signals": ["signal 1", "signal 2"],
    "competitors_mentioned": ["competitor 1"],
    "next_steps": ["action 1", "action 2"],
    "deal_stage": "lead|discovery|evaluation|proposal|negotiation",
    "estimated_deal_value": 0
}}

Be specific and extract actual quotes where possible. Only include what's explicitly mentioned."""

    def __init__(self):
        self.llm = LLMClient()
    
    def extract_from_transcript(self, transcript: str, existing_prospect: Optional[ProspectWorldModel] = None) -> ProspectWorldModel:
        """Extract or update prospect from a transcript."""
        
        result = self.llm.chat(
            messages=[{"role": "user", "content": self.EXTRACTION_PROMPT.format(transcript=transcript)}],
            model="gpt-4o-mini",
            json_mode=True
        )
        try:
            data = result["content"] if isinstance(result["content"], dict) else json.loads(result["content"])
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse extraction response: {e}")
            raise ValueError("Failed to extract structured data from transcript")
        
        if existing_prospect:
            # Update existing
            prospect = existing_prospect
            prospect.activities.append(Activity(
                timestamp=datetime.now().isoformat(),
                type="transcript",
                subject="New conversation",
                summary=f"Extracted {len(data.get('pain_points', []))} pain points, {len(data.get('stakeholders', []))} stakeholders",
                raw_content=transcript[:2000],
            ))
        else:
            # Create new
            prospect = ProspectWorldModel(
                id=str(uuid.uuid4())[:8],
                name=data.get('prospect_name', 'Unknown'),
                title=data.get('prospect_title', ''),
                company=data.get('company', 'Unknown'),
            )
        
        # Merge extracted data
        for s in data.get('stakeholders', []):
            if not any(existing.name == s['name'] for existing in prospect.stakeholders):
                prospect.stakeholders.append(Stakeholder(**s))
        
        for p in data.get('pain_points', []):
            prospect.pain_points.append(PainPoint(**p))
        
        for o in data.get('objections', []):
            prospect.objections.append(Objection(**o))
        
        prospect.buying_signals.extend(data.get('buying_signals', []))
        prospect.competitors_mentioned.extend(data.get('competitors_mentioned', []))
        
        if data.get('deal_stage'):
            prospect.stage = data['deal_stage']
        if data.get('estimated_deal_value'):
            prospect.deal_value = float(data['estimated_deal_value'])
        
        prospect.calculate_probability()
        prospect.save()
        
        return prospect


class ProspectEnricher:
    """Enrich prospect data with web grounding."""
    
    def __init__(self):
        self.grounder = WebGrounder()
        self.llm = LLMClient()
    
    def enrich_company(self, prospect: ProspectWorldModel) -> ProspectWorldModel:
        """Enrich company data from web."""
        if not prospect.company:
            return prospect
        
        # Search for company info
        query = f"{prospect.company} company overview industry size"
        results = self.grounder.search(query, max_results=3)
        
        if results:
            # Summarize with LLM
            context = "\n".join([f"- {r.get('title', '')}: {r.get('snippet', '')}" for r in results])
            
            result = self.llm.chat(
                messages=[{"role": "user", "content": f"""Based on these search results about {prospect.company}, extract:
                
{context}

Return JSON:
{{
    "description": "1-2 sentence company description",
    "industry": "primary industry",
    "size": "employee count range if mentioned",
    "recent_news": ["news item 1", "news item 2"]
}}"""}],
                model="gpt-4o-mini",
                json_mode=True
            )
            data = result["content"] if isinstance(result["content"], dict) else json.loads(result["content"])
            
            try:
                prospect.company_description = data.get('description')
                prospect.company_industry = data.get('industry')
                prospect.company_size = data.get('size')
                prospect.company_recent_news = data.get('recent_news', [])
            except:
                pass
        
        prospect.enriched_at = datetime.now().isoformat()
        prospect.save()
        return prospect
    
    def enrich_stakeholder(self, prospect: ProspectWorldModel, stakeholder_name: str) -> ProspectWorldModel:
        """Enrich stakeholder data from LinkedIn/web."""
        stakeholder = next((s for s in prospect.stakeholders if s.name == stakeholder_name), None)
        if not stakeholder:
            return prospect
        
        query = f"{stakeholder.name} {prospect.company} LinkedIn"
        results = self.grounder.search(query, max_results=2)
        
        if results:
            context = "\n".join([f"- {r.get('title', '')}: {r.get('snippet', '')}" for r in results])
            
            result = self.llm.chat(
                messages=[{"role": "user", "content": f"""Based on search results for {stakeholder.name} at {prospect.company}:

{context}

Return JSON:
{{
    "linkedin_url": "LinkedIn URL if found",
    "tenure": "how long at company if mentioned",
    "previous_companies": ["company 1"],
    "education": "education if mentioned"
}}"""}],
                model="gpt-4o-mini",
                json_mode=True
            )
            data = result["content"] if isinstance(result["content"], dict) else json.loads(result["content"])
            
            try:
                stakeholder.linkedin = data.get('linkedin_url')
                stakeholder.company_tenure = data.get('tenure')
                stakeholder.previous_companies = data.get('previous_companies', [])
                stakeholder.education = data.get('education')
                stakeholder.enriched = True
            except:
                pass
        
        prospect.save()
        return prospect
    
    def enrich_stakeholders(self, prospect: ProspectWorldModel) -> ProspectWorldModel:
        """Enrich all stakeholders (limited to avoid rate limits)."""
        for s in prospect.stakeholders[:3]:
            prospect = self.enrich_stakeholder(prospect, s.name)
            import time
            time.sleep(1)  # Rate limit
        return prospect


# Convenience functions for Flask app
def get_all_prospects() -> list[dict]:
    """Get all prospects as dicts."""
    return [asdict(p) for p in ProspectWorldModel.list_all()]


def get_prospect(prospect_id: str) -> Optional[dict]:
    """Get single prospect as dict."""
    p = ProspectWorldModel.load(prospect_id)
    return asdict(p) if p else None


def create_prospect_from_transcript(transcript: str) -> dict:
    """Create new prospect from transcript."""
    extractor = ProspectExtractor()
    prospect = extractor.extract_from_transcript(transcript)
    return asdict(prospect)


def enrich_prospect(prospect_id: str) -> dict:
    """Enrich prospect with web data."""
    prospect = ProspectWorldModel.load(prospect_id)
    if not prospect:
        raise ValueError(f"Prospect {prospect_id} not found")
    
    enricher = ProspectEnricher()
    prospect = enricher.enrich_company(prospect)
    
    for s in prospect.stakeholders[:3]:  # Limit to avoid rate limits
        prospect = enricher.enrich_stakeholder(prospect, s.name)
    
    return asdict(prospect)


def get_pipeline_stats() -> dict:
    """Get pipeline statistics."""
    prospects = ProspectWorldModel.list_all()
    
    by_stage = {}
    for p in prospects:
        by_stage[p.stage] = by_stage.get(p.stage, 0) + 1
    
    active = [p for p in prospects if p.stage not in ['closed_won', 'closed_lost']]
    
    return {
        "total_prospects": len(prospects),
        "total_deals": len(active),
        "total_pipeline_value": sum(p.deal_value for p in active),
        "weighted_pipeline_value": sum(p.deal_value * p.probability for p in active),
        "by_stage": by_stage,
        "avg_deal_size": sum(p.deal_value for p in active) / len(active) if active else 0,
    }
