"""
Sales Process Mining — Extract stages, build transition graphs, find bottlenecks.

Implements:
- Stage extraction from unstructured transcripts
- Markov chain for stage transitions
- Bottleneck detection (stalled deals)
- Conversion probability computation
"""

from __future__ import annotations
import json
import logging
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import random

logger = logging.getLogger("sales.process_mining")

# Sales process stages (ordered)
SALES_STAGES = [
    "lead",           # Initial contact / inbound
    "discovery",      # Understanding needs
    "qualification",  # BANT/MEDDIC qualification
    "demo",           # Product demonstration
    "evaluation",     # Technical evaluation / pilot
    "proposal",       # Proposal / pricing sent
    "negotiation",    # Contract negotiation
    "closed_won",     # Deal won
    "closed_lost",    # Deal lost
]

STAGE_DISPLAY = {
    "lead": "Lead",
    "discovery": "Discovery",
    "qualification": "Qualification",
    "demo": "Demo",
    "evaluation": "Evaluation",
    "proposal": "Proposal",
    "negotiation": "Negotiation",
    "closed_won": "Closed Won",
    "closed_lost": "Closed Lost",
}

# Stage indicators for extraction
STAGE_INDICATORS = {
    "lead": [
        "inbound", "inquiry", "reached out", "first contact", "new prospect",
        "marketing qualified", "MQL", "form submission"
    ],
    "discovery": [
        "discovery call", "initial meeting", "understanding needs", "pain points",
        "challenges", "current solution", "requirements gathering", "intro call",
        "learning about", "tell me about"
    ],
    "qualification": [
        "budget", "authority", "need", "timeline", "BANT", "MEDDIC",
        "decision maker", "stakeholders", "approval process", "fiscal year",
        "evaluation criteria", "qualified"
    ],
    "demo": [
        "demo", "demonstration", "product walkthrough", "show you", "platform tour",
        "capabilities", "features", "use case demo", "live demo"
    ],
    "evaluation": [
        "pilot", "proof of concept", "POC", "trial", "evaluation period",
        "testing", "technical review", "security review", "compliance review",
        "technical validation", "hands-on"
    ],
    "proposal": [
        "proposal", "pricing", "quote", "contract", "commercial terms",
        "pricing discussion", "package", "tier", "licensing", "MSA"
    ],
    "negotiation": [
        "negotiate", "terms", "redline", "legal review", "procurement",
        "discount", "concession", "final terms", "sign off", "approval"
    ],
    "closed_won": [
        "signed", "closed", "won", "deal closed", "contract signed",
        "purchase order", "kick off", "implementation", "onboarding"
    ],
    "closed_lost": [
        "lost", "no decision", "went dark", "competitor", "budget cut",
        "project cancelled", "postponed", "not moving forward", "declined"
    ],
}


@dataclass
class StageTransition:
    """A transition between sales stages."""
    from_stage: str
    to_stage: str
    count: int = 0
    avg_days: float = 0.0
    prospects: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "from_stage": self.from_stage,
            "to_stage": self.to_stage,
            "count": self.count,
            "avg_days": round(self.avg_days, 1),
            "prospects": self.prospects,
        }


@dataclass
class StageMetrics:
    """Metrics for a single stage."""
    stage: str
    count: int = 0
    avg_time_days: float = 0.0
    conversion_rate: float = 0.0
    stall_rate: float = 0.0
    prospects: List[str] = field(default_factory=list)
    stalled_prospects: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "stage": self.stage,
            "display_name": STAGE_DISPLAY.get(self.stage, self.stage),
            "count": self.count,
            "avg_time_days": round(self.avg_time_days, 1),
            "conversion_rate": round(self.conversion_rate * 100, 1),
            "stall_rate": round(self.stall_rate * 100, 1),
            "prospects": self.prospects,
            "stalled_prospects": self.stalled_prospects,
        }


@dataclass
class DealTimeline:
    """Timeline of a deal's journey through stages."""
    prospect_id: str
    prospect_name: str
    company: str
    stages: List[Dict[str, Any]] = field(default_factory=list)  # [{stage, entered_at, exited_at, days_in_stage}]
    current_stage: str = "lead"
    is_stalled: bool = False
    stall_days: int = 0
    outcome: Optional[str] = None  # closed_won, closed_lost, or None (active)
    
    def to_dict(self) -> dict:
        return {
            "prospect_id": self.prospect_id,
            "prospect_name": self.prospect_name,
            "company": self.company,
            "stages": self.stages,
            "current_stage": self.current_stage,
            "current_stage_display": STAGE_DISPLAY.get(self.current_stage, self.current_stage),
            "is_stalled": self.is_stalled,
            "stall_days": self.stall_days,
            "outcome": self.outcome,
            "total_days": sum(s.get("days_in_stage", 0) for s in self.stages),
        }


class SalesProcessMiner:
    """
    Mine sales process patterns from prospect data and transcripts.
    
    Builds:
    - Stage transition Markov chain
    - Bottleneck identification
    - Conversion funnel metrics
    - Deal timelines
    """
    
    def __init__(self, stall_threshold_days: int = 14):
        self.stall_threshold_days = stall_threshold_days
        self.transitions: Dict[Tuple[str, str], StageTransition] = {}
        self.stage_metrics: Dict[str, StageMetrics] = {}
        self.deal_timelines: List[DealTimeline] = []
        self.stage_counts: Dict[str, int] = defaultdict(int)
        
    def extract_stage_from_metadata(self, prospect_data: dict) -> str:
        """
        Extract current stage from prospect meeting metadata.
        """
        metadata = prospect_data.get("meeting_metadata", {})
        stage = metadata.get("overall_deal_stage", "discovery")
        
        # Normalize stage names
        stage = stage.lower().replace(" ", "_").replace("-", "_")
        
        # Map common variations
        stage_map = {
            "discovery": "discovery",
            "evaluation": "evaluation",
            "proposal": "proposal",
            "negotiation": "negotiation",
            "closed": "closed_won",
            "won": "closed_won",
            "lost": "closed_lost",
        }
        
        return stage_map.get(stage, stage)
    
    def extract_stages_from_transcript(self, transcript: str) -> List[str]:
        """
        Extract likely stages mentioned in a transcript.
        
        Uses keyword matching with confidence weighting.
        Returns list of detected stages in order of confidence.
        """
        transcript_lower = transcript.lower()
        stage_scores = defaultdict(int)
        
        for stage, indicators in STAGE_INDICATORS.items():
            for indicator in indicators:
                # Count occurrences
                count = transcript_lower.count(indicator.lower())
                if count > 0:
                    stage_scores[stage] += count
        
        # Sort by score and return
        sorted_stages = sorted(
            stage_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [s[0] for s in sorted_stages if s[1] > 0]
    
    def build_timeline_from_prospect(
        self,
        prospect_data: dict,
        base_date: Optional[datetime] = None,
    ) -> DealTimeline:
        """
        Build a deal timeline from prospect data.
        
        Infers stage progression from meeting metadata and signals.
        """
        if base_date is None:
            base_date = datetime.now() - timedelta(days=30)
        
        prospect_id = prospect_data.get("id", "unknown")
        prospect_name = prospect_data.get("name", "Unknown")
        company = prospect_data.get("company", "Unknown")
        
        timeline = DealTimeline(
            prospect_id=prospect_id,
            prospect_name=prospect_name,
            company=company,
        )
        
        # Get current stage
        current_stage = self.extract_stage_from_metadata(prospect_data)
        timeline.current_stage = current_stage
        
        # Build stage history
        # Infer from meeting metadata
        meetings = prospect_data.get("meeting_metadata", {}).get("meetings", [])
        meeting_count = len(meetings)
        
        # Standard progression based on meeting count and metadata
        stage_order = ["lead", "discovery", "qualification", "demo", "evaluation", "proposal", "negotiation"]
        current_idx = stage_order.index(current_stage) if current_stage in stage_order else 0
        
        # Build history up to current stage
        days_per_stage = max(3, 30 // (current_idx + 1))
        stage_date = base_date
        
        for i, stage in enumerate(stage_order[:current_idx + 1]):
            # Add some variance
            days = days_per_stage + random.randint(-2, 5)
            if days < 1:
                days = 1
                
            timeline.stages.append({
                "stage": stage,
                "display_name": STAGE_DISPLAY.get(stage, stage),
                "entered_at": stage_date.isoformat(),
                "exited_at": (stage_date + timedelta(days=days)).isoformat() if i < current_idx else None,
                "days_in_stage": days if i < current_idx else (datetime.now() - stage_date).days,
            })
            stage_date += timedelta(days=days)
        
        # Check for stall
        momentum = prospect_data.get("meeting_metadata", {}).get("momentum", "steady")
        if momentum == "stalling":
            timeline.is_stalled = True
            timeline.stall_days = random.randint(7, 21)
        elif timeline.stages:
            last_stage = timeline.stages[-1]
            days_in_current = last_stage.get("days_in_stage", 0)
            if days_in_current > self.stall_threshold_days:
                timeline.is_stalled = True
                timeline.stall_days = days_in_current
        
        return timeline
    
    def compute_transition_matrix(
        self,
        timelines: List[DealTimeline]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute Markov chain transition probabilities.
        
        Returns: {from_stage: {to_stage: probability}}
        """
        # Count transitions
        transition_counts = defaultdict(lambda: defaultdict(int))
        
        for timeline in timelines:
            stages = timeline.stages
            for i in range(len(stages) - 1):
                from_stage = stages[i]["stage"]
                to_stage = stages[i + 1]["stage"]
                transition_counts[from_stage][to_stage] += 1
            
            # Add current stage -> outcome if closed
            if timeline.outcome:
                if timeline.stages:
                    last_stage = timeline.stages[-1]["stage"]
                    transition_counts[last_stage][timeline.outcome] += 1
        
        # Convert to probabilities
        transition_probs = {}
        for from_stage, to_stages in transition_counts.items():
            total = sum(to_stages.values())
            transition_probs[from_stage] = {
                to_stage: count / total
                for to_stage, count in to_stages.items()
            }
        
        return transition_probs
    
    def identify_bottlenecks(
        self,
        timelines: List[DealTimeline]
    ) -> List[Dict[str, Any]]:
        """
        Identify stages where deals commonly stall or get lost.
        
        Returns list of bottleneck stages with metrics.
        """
        stage_stats = defaultdict(lambda: {
            "count": 0,
            "total_days": 0,
            "stalled": 0,
            "lost": 0,
            "prospects": [],
        })
        
        for timeline in timelines:
            for stage_data in timeline.stages:
                stage = stage_data["stage"]
                days = stage_data.get("days_in_stage", 0)
                
                stage_stats[stage]["count"] += 1
                stage_stats[stage]["total_days"] += days
                stage_stats[stage]["prospects"].append(timeline.prospect_name)
                
                if days > self.stall_threshold_days:
                    stage_stats[stage]["stalled"] += 1
            
            # Track where deals are lost
            if timeline.outcome == "closed_lost" and timeline.stages:
                last_stage = timeline.stages[-1]["stage"]
                stage_stats[last_stage]["lost"] += 1
        
        # Compute bottleneck scores
        bottlenecks = []
        for stage, stats in stage_stats.items():
            if stats["count"] == 0:
                continue
                
            avg_days = stats["total_days"] / stats["count"]
            stall_rate = stats["stalled"] / stats["count"]
            loss_rate = stats["lost"] / stats["count"]
            
            # Bottleneck score: weighted combination
            score = (avg_days * 0.3) + (stall_rate * 50) + (loss_rate * 100)
            
            bottlenecks.append({
                "stage": stage,
                "display_name": STAGE_DISPLAY.get(stage, stage),
                "count": stats["count"],
                "avg_days": round(avg_days, 1),
                "stall_rate": round(stall_rate * 100, 1),
                "loss_rate": round(loss_rate * 100, 1),
                "bottleneck_score": round(score, 1),
                "is_bottleneck": score > 30,  # Threshold for "bottleneck" classification
            })
        
        # Sort by bottleneck score
        bottlenecks.sort(key=lambda x: x["bottleneck_score"], reverse=True)
        
        return bottlenecks
    
    def compute_funnel_metrics(
        self,
        timelines: List[DealTimeline]
    ) -> List[Dict[str, Any]]:
        """
        Compute conversion funnel metrics.
        
        Shows how many deals progress through each stage.
        """
        stage_order = ["lead", "discovery", "qualification", "demo", "evaluation", "proposal", "negotiation", "closed_won"]
        
        # Count deals reaching each stage
        stage_reached = defaultdict(set)
        
        for timeline in timelines:
            for stage_data in timeline.stages:
                stage = stage_data["stage"]
                stage_reached[stage].add(timeline.prospect_id)
            
            if timeline.outcome == "closed_won":
                stage_reached["closed_won"].add(timeline.prospect_id)
        
        # Build funnel
        total_leads = len(timelines)
        funnel = []
        
        for stage in stage_order:
            count = len(stage_reached[stage])
            conversion = (count / total_leads * 100) if total_leads > 0 else 0
            
            funnel.append({
                "stage": stage,
                "display_name": STAGE_DISPLAY.get(stage, stage),
                "count": count,
                "conversion_from_lead": round(conversion, 1),
                "width_percent": max(20, conversion),  # For visualization
            })
        
        return funnel
    
    def simulate_scenario(
        self,
        timeline: DealTimeline,
        changes: Dict[str, Any],
        transition_matrix: Dict[str, Dict[str, float]]
    ) -> Dict[str, Any]:
        """
        Simulate "what if" scenarios.
        
        Changes can include:
        - rep_change: True (different rep could change conversion)
        - timing_speedup: 0.5 (50% faster through stages)
        - remove_objection: True (main objection resolved)
        - add_champion: True (internal champion added)
        """
        current_stage = timeline.current_stage
        
        # Base win probability from transition matrix
        base_prob = 0.0
        if current_stage in transition_matrix:
            # Calculate path to close
            remaining_stages = []
            found = False
            for stage in SALES_STAGES:
                if stage == current_stage:
                    found = True
                elif found and stage not in ("closed_won", "closed_lost"):
                    remaining_stages.append(stage)
            
            # Multiply probabilities along the path
            prob = 1.0
            current = current_stage
            for next_stage in remaining_stages + ["closed_won"]:
                if current in transition_matrix and next_stage in transition_matrix[current]:
                    prob *= transition_matrix[current][next_stage]
                else:
                    prob *= 0.5  # Default 50% if no data
                current = next_stage
            
            base_prob = prob
        else:
            base_prob = 0.3  # Default
        
        # Apply scenario modifiers
        modified_prob = base_prob
        changes_applied = []
        
        if changes.get("rep_change"):
            modified_prob *= 1.15  # 15% boost from fresh approach
            changes_applied.append("New rep assignment: +15%")
        
        if changes.get("timing_speedup"):
            speedup = changes["timing_speedup"]
            modifier = 1 + (1 - speedup) * 0.2  # Faster = better
            modified_prob *= modifier
            changes_applied.append(f"Timing speedup ({speedup}x): +{round((modifier-1)*100)}%")
        
        if changes.get("remove_objection"):
            modified_prob *= 1.25  # Removing blockers is huge
            changes_applied.append("Main objection resolved: +25%")
        
        if changes.get("add_champion"):
            modified_prob *= 1.35  # Champions are critical
            changes_applied.append("Internal champion added: +35%")
        
        # Cap at 95%
        modified_prob = min(0.95, modified_prob)
        
        # Estimate time to close
        base_days = sum(s.get("days_in_stage", 7) for s in timeline.stages)
        estimated_days = base_days
        
        if changes.get("timing_speedup"):
            estimated_days *= changes["timing_speedup"]
        
        if changes.get("add_champion"):
            estimated_days *= 0.8  # Champions speed things up
        
        return {
            "prospect_name": timeline.prospect_name,
            "current_stage": current_stage,
            "base_win_probability": round(base_prob * 100, 1),
            "modified_win_probability": round(modified_prob * 100, 1),
            "probability_change": round((modified_prob - base_prob) * 100, 1),
            "changes_applied": changes_applied,
            "estimated_days_to_close": round(estimated_days),
            "recommendation": self._get_recommendation(modified_prob, timeline),
        }
    
    def _get_recommendation(self, prob: float, timeline: DealTimeline) -> str:
        """Generate recommendation based on probability and deal state."""
        if prob > 0.7:
            return "High probability deal. Focus on closing actions and removing final blockers."
        elif prob > 0.4:
            return "Moderate probability. Identify and engage decision maker, resolve open objections."
        elif prob > 0.2:
            return "Low probability. Consider qualification review or strategic pivot."
        else:
            return "Very low probability. Evaluate if deal is worth continued investment."
    
    def get_full_analysis(
        self,
        prospects: List[dict]
    ) -> Dict[str, Any]:
        """
        Run full process mining analysis on prospect data.
        
        Returns complete metrics package.
        """
        # Build timelines
        timelines = []
        for prospect in prospects:
            timeline = self.build_timeline_from_prospect(prospect)
            timelines.append(timeline)
        
        self.deal_timelines = timelines
        
        # Compute metrics
        transition_matrix = self.compute_transition_matrix(timelines)
        bottlenecks = self.identify_bottlenecks(timelines)
        funnel = self.compute_funnel_metrics(timelines)
        
        # Summary stats
        total_deals = len(timelines)
        active_deals = sum(1 for t in timelines if t.outcome is None)
        stalled_deals = sum(1 for t in timelines if t.is_stalled)
        
        # Stage distribution
        stage_distribution = defaultdict(int)
        for timeline in timelines:
            stage_distribution[timeline.current_stage] += 1
        
        return {
            "summary": {
                "total_deals": total_deals,
                "active_deals": active_deals,
                "stalled_deals": stalled_deals,
                "stall_rate": round(stalled_deals / total_deals * 100, 1) if total_deals > 0 else 0,
            },
            "timelines": [t.to_dict() for t in timelines],
            "transition_matrix": transition_matrix,
            "bottlenecks": bottlenecks,
            "funnel": funnel,
            "stage_distribution": dict(stage_distribution),
        }


def generate_demo_pipeline() -> List[dict]:
    """
    Generate realistic demo pipeline data for visualization.
    
    Creates a mix of deals at various stages with realistic patterns.
    """
    companies = [
        ("TechCorp Industries", "technology"),
        ("HealthFirst Systems", "healthcare"),
        ("FinServe Global", "finance"),
        ("RetailMax Group", "retail"),
        ("ManuPro Solutions", "manufacturing"),
        ("EduTech Learning", "education"),
        ("GreenEnergy Co", "energy"),
        ("LogiFlow Transport", "logistics"),
    ]
    
    first_names = ["Sarah", "Michael", "Jennifer", "David", "Emily", "James", "Amanda", "Robert"]
    last_names = ["Chen", "Williams", "Garcia", "Johnson", "Brown", "Martinez", "Lee", "Wilson"]
    titles = ["VP Engineering", "CTO", "Director of IT", "Head of Data", "VP Operations", "Chief Data Officer"]
    
    pipeline = []
    stages_weighted = [
        ("lead", 0.1),
        ("discovery", 0.15),
        ("qualification", 0.15),
        ("demo", 0.2),
        ("evaluation", 0.2),
        ("proposal", 0.1),
        ("negotiation", 0.05),
        ("closed_won", 0.03),
        ("closed_lost", 0.02),
    ]
    
    import uuid
    
    for i in range(15):
        company, industry = random.choice(companies)
        first = random.choice(first_names)
        last = random.choice(last_names)
        
        # Weighted random stage selection
        r = random.random()
        cumulative = 0
        stage = "discovery"
        for s, weight in stages_weighted:
            cumulative += weight
            if r <= cumulative:
                stage = s
                break
        
        momentum = random.choice(["accelerating", "steady", "steady", "stalling"])
        
        # Determine meeting count based on stage
        stage_idx = SALES_STAGES.index(stage) if stage in SALES_STAGES else 1
        meeting_count = max(1, stage_idx)
        
        prospect = {
            "id": str(uuid.uuid4()),
            "name": f"{first} {last}",
            "title": random.choice(titles),
            "company": company,
            "email": f"{first.lower()}.{last.lower()}@{company.lower().replace(' ', '')}.com",
            "personality_traits": random.sample(
                ["detail-oriented", "data-driven", "risk-averse", "innovative", "collaborative", "decisive"],
                k=random.randint(2, 3)
            ),
            "communication_style": random.choice(["direct", "analytical", "collaborative", "formal"]),
            "decision_style": random.choice(["ROI-focused", "consensus-builder", "fast-mover", "methodical"]),
            "pain_points": [
                {
                    "description": random.choice([
                        "Legacy system integration challenges",
                        "Data quality and reliability issues",
                        "Compliance and audit requirements",
                        "Scaling infrastructure concerns",
                        "Team productivity bottlenecks",
                    ]),
                    "severity": random.choice(["critical", "high", "medium"]),
                    "category": random.choice(["reliability", "compliance", "efficiency", "governance"]),
                    "meeting_number": 1,
                }
            ],
            "objections": [] if random.random() > 0.4 else [
                {
                    "description": random.choice([
                        "Concerns about implementation timeline",
                        "Need to validate security requirements",
                        "Budget approval process unclear",
                        "Integration with existing tools",
                    ]),
                    "type": random.choice(["technical", "process", "budget", "timeline"]),
                    "severity": random.choice(["hard_blocker", "concern", "question"]),
                    "resolution_status": random.choice(["unresolved", "partially_resolved", "resolved"]),
                }
            ],
            "buying_signals": [
                {
                    "signal": random.choice([
                        "Expressed urgency to solve the problem",
                        "Asked about pricing and implementation",
                        "Mentioned upcoming budget cycle",
                        "Requested technical deep-dive",
                    ]),
                    "strength": random.choice(["strong", "moderate", "weak"]),
                    "type": random.choice(["timeline", "budget", "need", "champion"]),
                }
            ],
            "stakeholders": [
                {
                    "name": f"{random.choice(first_names)} {random.choice(last_names)}",
                    "role": random.choice(["CFO", "CEO", "VP", "Director"]),
                    "influence": random.choice(["decision_maker", "influencer", "evaluator"]),
                    "sentiment": random.choice(["positive", "neutral", "unknown"]),
                }
            ],
            "competitive_context": [],
            "next_steps": [
                {
                    "action": random.choice([
                        "Schedule follow-up call",
                        "Send proposal document",
                        "Technical architecture review",
                        "Executive sponsor meeting",
                    ]),
                    "owner": random.choice(["rep", "prospect", "both"]),
                    "timeline": None,
                    "commitment_level": random.choice(["firm", "tentative"]),
                }
            ],
            "meeting_metadata": {
                "meeting_count": meeting_count,
                "meetings": [{"number": j+1, "topic": f"Meeting {j+1}", "tone": "collaborative"} for j in range(meeting_count)],
                "overall_deal_stage": stage,
                "momentum": momentum,
            },
            "evolutions": [],
            "web_enrichment": {},
            "uncertainties": [],
        }
        
        pipeline.append(prospect)
    
    return pipeline


# Standalone usage
if __name__ == "__main__":
    import json
    
    # Generate demo data
    pipeline = generate_demo_pipeline()
    
    # Run analysis
    miner = SalesProcessMiner()
    analysis = miner.get_full_analysis(pipeline)
    
    print("=== Sales Process Mining Results ===\n")
    print(f"Summary: {json.dumps(analysis['summary'], indent=2)}")
    print(f"\nFunnel:")
    for stage in analysis["funnel"]:
        print(f"  {stage['display_name']}: {stage['count']} ({stage['conversion_from_lead']}%)")
    
    print(f"\nTop Bottlenecks:")
    for bn in analysis["bottlenecks"][:3]:
        print(f"  {bn['display_name']}: score={bn['bottleneck_score']}, stall_rate={bn['stall_rate']}%")
