"""
Interaction Simulator — Simulates how sales content interacts with prospect world models.

Core MVP feature: "What happens if I send this email sequence to Nina?"

Predicts:
- Objection strength changes
- Pain point resonance
- New signals / concerns surfacing
- Engagement probability
- Optimal timing
"""

from __future__ import annotations
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from openai import OpenAI

from sales.content_model import SalesContent, ContentSequence, ContentTone
from sales.prospect_model import ProspectModel

logger = logging.getLogger("sales.interaction_simulator")

# Initialize OpenAI client  
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


@dataclass
class ObjectionChange:
    """Change in an objection's strength after content interaction."""
    objection: str
    before: float  # 0-1
    after: float  # 0-1
    reason: str
    
    @property
    def delta(self) -> float:
        return self.after - self.before
    
    def to_dict(self) -> dict:
        return {
            "objection": self.objection,
            "before": self.before,
            "after": self.after,
            "delta": round(self.delta, 2),
            "reason": self.reason,
        }


@dataclass
class PainResonance:
    """How well content resonates with a prospect's pain point."""
    pain: str
    resonance_score: float  # 0-1
    addressed_by: list[str]  # Which claims/proof points address this
    
    def to_dict(self) -> dict:
        return {
            "pain": self.pain,
            "resonance_score": round(self.resonance_score, 2),
            "addressed_by": self.addressed_by,
        }


@dataclass
class InteractionResult:
    """Result of simulating content ↔ prospect interaction."""
    
    # Identifiers
    prospect_name: str
    content_name: str
    content_id: str
    
    # State changes
    objection_changes: list[ObjectionChange] = field(default_factory=list)
    pain_resonance: list[PainResonance] = field(default_factory=list)
    new_signals: list[str] = field(default_factory=list)  # New buying signals
    new_concerns: list[str] = field(default_factory=list)  # New objections that might surface
    
    # Engagement prediction
    open_probability: float = 0.5
    response_probability: float = 0.3
    predicted_response_type: str = "neutral"  # positive, objection, clarification, ghost
    
    # Timing recommendations
    optimal_send_time: str = "morning"
    optimal_day_of_week: str = "tuesday"
    
    # Next action
    recommended_followup: str = ""
    risk_level: str = "medium"  # low, medium, high
    risk_factors: list[str] = field(default_factory=list)
    
    # Simulation metadata
    model_used: str = ""
    simulated_at: float = field(default_factory=time.time)
    
    @property
    def net_objection_change(self) -> float:
        """Net change in objection strength (negative = good)."""
        if not self.objection_changes:
            return 0.0
        return sum(c.delta for c in self.objection_changes)
    
    @property
    def avg_resonance(self) -> float:
        """Average pain point resonance."""
        if not self.pain_resonance:
            return 0.0
        return sum(p.resonance_score for p in self.pain_resonance) / len(self.pain_resonance)
    
    def to_dict(self) -> dict:
        return {
            "prospect_name": self.prospect_name,
            "content_name": self.content_name,
            "content_id": self.content_id,
            "objection_changes": [o.to_dict() for o in self.objection_changes],
            "pain_resonance": [p.to_dict() for p in self.pain_resonance],
            "new_signals": self.new_signals,
            "new_concerns": self.new_concerns,
            "open_probability": round(self.open_probability, 2),
            "response_probability": round(self.response_probability, 2),
            "predicted_response_type": self.predicted_response_type,
            "optimal_send_time": self.optimal_send_time,
            "optimal_day_of_week": self.optimal_day_of_week,
            "recommended_followup": self.recommended_followup,
            "risk_level": self.risk_level,
            "risk_factors": self.risk_factors,
            "net_objection_change": round(self.net_objection_change, 2),
            "avg_resonance": round(self.avg_resonance, 2),
            "model_used": self.model_used,
            "simulated_at": self.simulated_at,
        }


@dataclass
class SequenceSimulation:
    """Full simulation of a content sequence against a prospect."""
    
    # Identifiers
    prospect_name: str
    sequence_name: str
    sequence_id: str
    
    # Per-step results
    step_results: list[InteractionResult] = field(default_factory=list)
    
    # Trajectory
    trajectory: list[dict] = field(default_factory=list)  # {step, win_prob, state_summary}
    
    # Aggregate metrics
    cumulative_objection_reduction: float = 0.0
    pains_addressed_count: int = 0
    new_risks_surfaced: int = 0
    
    # Overall assessment
    sequence_rating: str = "moderate"  # strong, moderate, weak, risky
    improvement_suggestions: list[str] = field(default_factory=list)
    
    # Key moments
    inflection_points: list[dict] = field(default_factory=list)
    ghost_risk_step: Optional[int] = None  # Step where ghost risk is highest
    
    # Metadata
    simulated_at: float = field(default_factory=time.time)
    total_duration_days: int = 0
    
    def to_dict(self) -> dict:
        return {
            "prospect_name": self.prospect_name,
            "sequence_name": self.sequence_name,
            "sequence_id": self.sequence_id,
            "step_results": [r.to_dict() for r in self.step_results],
            "trajectory": self.trajectory,
            "cumulative_objection_reduction": round(self.cumulative_objection_reduction, 2),
            "pains_addressed_count": self.pains_addressed_count,
            "new_risks_surfaced": self.new_risks_surfaced,
            "sequence_rating": self.sequence_rating,
            "improvement_suggestions": self.improvement_suggestions,
            "inflection_points": self.inflection_points,
            "ghost_risk_step": self.ghost_risk_step,
            "simulated_at": self.simulated_at,
            "total_duration_days": self.total_duration_days,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Simulation Engine
# ─────────────────────────────────────────────────────────────────────────────

INTERACTION_SIMULATION_PROMPT = """You are a sales psychology expert simulating how a prospect will react to sales content.

## Prospect Profile
Name: {prospect_name}
Title: {prospect_title}
Company: {prospect_company}
Communication Style: {communication_style}
Decision Style: {decision_style}
Personality Traits: {personality_traits}

### Current Pain Points
{pain_points}

### Current Objections
{objections}

### Buying Signals Observed
{buying_signals}

### Competitive Context
{competitive_context}

## Content Being Sent
Type: {content_type}
Subject: {subject_line}
---
{content}
---

### Content Claims
{claims}

### Content Addresses These Pains
{pains_addressed}

### Content Handles These Objections
{objections_handled}

## Simulation Task
Predict how this specific prospect will react to this specific content, given their profile.

Consider:
1. Does the content tone match their communication style?
2. Do the claims resonate with their expressed pain points?
3. Does the content address their specific objections?
4. Given their personality, how likely are they to engage?
5. What new concerns might this content surface?

Return valid JSON:
```json
{{
    "objection_changes": [
        {{"objection": "the objection text", "before_strength": 0.8, "after_strength": 0.6, "reason": "why it changed"}}
    ],
    "pain_resonance": [
        {{"pain": "the pain point", "resonance_score": 0.8, "addressed_by": ["claim 1", "proof point 2"]}}
    ],
    "new_signals": ["potential new buying signals that might emerge"],
    "new_concerns": ["new objections or concerns this might surface"],
    "open_probability": 0.7,
    "response_probability": 0.4,
    "predicted_response_type": "positive|objection|clarification|ghost",
    "optimal_send_time": "morning|afternoon|evening",
    "optimal_day_of_week": "monday|tuesday|wednesday|thursday|friday",
    "recommended_followup": "what to do next based on predicted response",
    "risk_level": "low|medium|high",
    "risk_factors": ["specific risks with this content for this prospect"]
}}
```

Be realistic and specific to this prospect's profile. Don't be overly optimistic."""


class InteractionSimulator:
    """
    Simulates how sales content interacts with prospect world models.
    
    Core capability: Predict state evolution when content meets prospect.
    """
    
    def __init__(self, model: str = "gpt-4o"):
        self.model = model
    
    def simulate_interaction(
        self,
        prospect: ProspectModel,
        content: SalesContent,
    ) -> InteractionResult:
        """
        Simulate how a prospect will react to a piece of content.
        
        Returns predicted state changes and engagement metrics.
        """
        logger.info(f"Simulating: {content.name} → {prospect.name}")
        
        # Format prospect data for prompt
        pain_points_str = "\n".join([
            f"- {pp.get('description', pp)} (severity: {pp.get('severity', 'unknown')})"
            for pp in prospect.pain_points[:5]
        ]) or "None identified"
        
        objections_str = "\n".join([
            f"- {obj.get('text', obj)} (strength: {obj.get('strength', 'unknown')})"
            for obj in prospect.objections[:5]
        ]) or "None identified"
        
        buying_signals_str = "\n".join([
            f"- {sig.get('signal', sig)}"
            for sig in prospect.buying_signals[:5]
        ]) or "None observed"
        
        competitive_str = "\n".join([
            f"- {comp.get('competitor', comp)}: {comp.get('relationship', 'unknown')}"
            for comp in prospect.competitive_context[:3]
        ]) or "No competitive context"
        
        prompt = INTERACTION_SIMULATION_PROMPT.format(
            prospect_name=prospect.name,
            prospect_title=prospect.title,
            prospect_company=prospect.company,
            communication_style=prospect.communication_style or "unknown",
            decision_style=prospect.decision_style or "unknown",
            personality_traits=", ".join(prospect.personality_traits) or "unknown",
            pain_points=pain_points_str,
            objections=objections_str,
            buying_signals=buying_signals_str,
            competitive_context=competitive_str,
            content_type=content.type.value,
            subject_line=content.subject_line or "(no subject)",
            content=content.raw_content,
            claims="\n".join([f"- {c}" for c in content.claims]) or "None extracted",
            pains_addressed="\n".join([f"- {p}" for p in content.pains_addressed]) or "None",
            objections_handled="\n".join([f"- {o}" for o in content.objections_handled]) or "None",
        )
        
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a sales psychology expert. Return only valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.4,
                response_format={"type": "json_object"},
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Parse into structured result
            interaction = InteractionResult(
                prospect_name=prospect.name,
                content_name=content.name,
                content_id=content.id,
                model_used=self.model,
            )
            
            # Parse objection changes
            for oc in result.get("objection_changes", []):
                interaction.objection_changes.append(ObjectionChange(
                    objection=oc.get("objection", ""),
                    before=oc.get("before_strength", 0.5),
                    after=oc.get("after_strength", 0.5),
                    reason=oc.get("reason", ""),
                ))
            
            # Parse pain resonance
            for pr in result.get("pain_resonance", []):
                interaction.pain_resonance.append(PainResonance(
                    pain=pr.get("pain", ""),
                    resonance_score=pr.get("resonance_score", 0.5),
                    addressed_by=pr.get("addressed_by", []),
                ))
            
            # Simple fields
            interaction.new_signals = result.get("new_signals", [])
            interaction.new_concerns = result.get("new_concerns", [])
            interaction.open_probability = result.get("open_probability", 0.5)
            interaction.response_probability = result.get("response_probability", 0.3)
            interaction.predicted_response_type = result.get("predicted_response_type", "neutral")
            interaction.optimal_send_time = result.get("optimal_send_time", "morning")
            interaction.optimal_day_of_week = result.get("optimal_day_of_week", "tuesday")
            interaction.recommended_followup = result.get("recommended_followup", "")
            interaction.risk_level = result.get("risk_level", "medium")
            interaction.risk_factors = result.get("risk_factors", [])
            
            logger.info(
                f"Simulation complete: "
                f"open={interaction.open_probability:.0%}, "
                f"response={interaction.response_probability:.0%}, "
                f"type={interaction.predicted_response_type}"
            )
            
            return interaction
            
        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            # Return a default result
            return InteractionResult(
                prospect_name=prospect.name,
                content_name=content.name,
                content_id=content.id,
                risk_level="high",
                risk_factors=[f"Simulation error: {str(e)}"],
            )
    
    def simulate_sequence(
        self,
        prospect: ProspectModel,
        sequence: ContentSequence,
    ) -> SequenceSimulation:
        """
        Simulate an entire content sequence against a prospect.
        
        Tracks cumulative state evolution across all steps.
        """
        logger.info(f"Simulating sequence: {sequence.name} → {prospect.name}")
        
        simulation = SequenceSimulation(
            prospect_name=prospect.name,
            sequence_name=sequence.name,
            sequence_id=sequence.id,
            total_duration_days=sequence.total_duration_days,
        )
        
        # Track cumulative state
        cumulative_objection_reduction = 0.0
        addressed_pains = set()
        all_new_concerns = []
        ghost_risk_max = 0.0
        ghost_risk_step = None
        
        # Run simulation for each step
        for i, content in enumerate(sequence.items):
            result = self.simulate_interaction(prospect, content)
            simulation.step_results.append(result)
            
            # Accumulate metrics
            cumulative_objection_reduction += result.net_objection_change
            
            for pr in result.pain_resonance:
                if pr.resonance_score > 0.6:
                    addressed_pains.add(pr.pain)
            
            all_new_concerns.extend(result.new_concerns)
            
            # Track ghost risk
            ghost_risk = 1 - result.response_probability
            if ghost_risk > ghost_risk_max:
                ghost_risk_max = ghost_risk
                ghost_risk_step = i
            
            # Add to trajectory
            # Estimate win probability change based on simulation
            base_prob = 0.3  # Assume baseline
            prob_delta = (
                -result.net_objection_change * 0.2 +  # Objection reduction helps
                result.avg_resonance * 0.1 +  # Pain resonance helps
                (0.1 if result.predicted_response_type == "positive" else 0)
            )
            current_prob = min(0.95, max(0.05, base_prob + prob_delta * (i + 1)))
            
            simulation.trajectory.append({
                "step": i + 1,
                "content_name": content.name,
                "win_prob": round(current_prob, 2),
                "response_prob": round(result.response_probability, 2),
                "state_summary": f"{result.predicted_response_type}: {len(result.objection_changes)} objection changes",
            })
            
            # Check for inflection points
            if result.net_objection_change < -0.2:
                simulation.inflection_points.append({
                    "step": i + 1,
                    "type": "breakthrough",
                    "description": f"Significant objection reduction after {content.name}",
                })
            elif len(result.new_concerns) > 1:
                simulation.inflection_points.append({
                    "step": i + 1,
                    "type": "risk",
                    "description": f"Multiple new concerns surfaced: {', '.join(result.new_concerns[:2])}",
                })
            
            # Rate limit
            if i < len(sequence.items) - 1:
                time.sleep(0.5)
        
        # Aggregate metrics
        simulation.cumulative_objection_reduction = cumulative_objection_reduction
        simulation.pains_addressed_count = len(addressed_pains)
        simulation.new_risks_surfaced = len(set(all_new_concerns))
        simulation.ghost_risk_step = ghost_risk_step
        
        # Rate the sequence
        simulation.sequence_rating = self._rate_sequence(simulation)
        simulation.improvement_suggestions = self._generate_suggestions(simulation)
        
        logger.info(
            f"Sequence simulation complete: "
            f"rating={simulation.sequence_rating}, "
            f"objection_delta={simulation.cumulative_objection_reduction:.2f}, "
            f"pains_addressed={simulation.pains_addressed_count}"
        )
        
        return simulation
    
    def _rate_sequence(self, sim: SequenceSimulation) -> str:
        """Rate the overall sequence effectiveness."""
        score = 0
        
        # Objection reduction is good
        if sim.cumulative_objection_reduction < -0.3:
            score += 2
        elif sim.cumulative_objection_reduction < -0.1:
            score += 1
        elif sim.cumulative_objection_reduction > 0.1:
            score -= 1
        
        # Pain addressing is good
        if sim.pains_addressed_count >= 3:
            score += 2
        elif sim.pains_addressed_count >= 1:
            score += 1
        
        # New risks are bad
        if sim.new_risks_surfaced > 2:
            score -= 2
        elif sim.new_risks_surfaced > 0:
            score -= 1
        
        # Response trajectory
        if sim.step_results:
            avg_response = sum(r.response_probability for r in sim.step_results) / len(sim.step_results)
            if avg_response > 0.5:
                score += 1
            elif avg_response < 0.2:
                score -= 1
        
        if score >= 3:
            return "strong"
        elif score >= 1:
            return "moderate"
        elif score >= -1:
            return "weak"
        else:
            return "risky"
    
    def _generate_suggestions(self, sim: SequenceSimulation) -> list[str]:
        """Generate improvement suggestions based on simulation results."""
        suggestions = []
        
        # Check for ghost risk
        if sim.ghost_risk_step is not None:
            step = sim.step_results[sim.ghost_risk_step]
            if step.response_probability < 0.2:
                suggestions.append(
                    f"High ghost risk after step {sim.ghost_risk_step + 1} ({step.content_name}). "
                    f"Consider adding a softer touch or extending timing before this step."
                )
        
        # Check for new concerns
        if sim.new_risks_surfaced > 1:
            concerns = []
            for r in sim.step_results:
                concerns.extend(r.new_concerns)
            suggestions.append(
                f"This sequence surfaces new concerns: {', '.join(set(concerns)[:3])}. "
                f"Consider adding content to preempt these objections."
            )
        
        # Check pain coverage
        if sim.pains_addressed_count < 2:
            suggestions.append(
                "Low pain point coverage. Consider adding content that speaks directly to the prospect's specific challenges."
            )
        
        # Check timing
        total_days = sim.total_duration_days
        if total_days > 14 and len(sim.step_results) <= 3:
            suggestions.append(
                f"Sequence spans {total_days} days with only {len(sim.step_results)} touches. "
                f"Consider tighter timing or adding intermediate touches to maintain momentum."
            )
        elif total_days < 5 and len(sim.step_results) >= 3:
            suggestions.append(
                f"Sequence has {len(sim.step_results)} touches in {total_days} days — this may feel aggressive. "
                f"Consider spreading out timing to avoid overwhelming the prospect."
            )
        
        return suggestions


def compare_sequences(
    prospect: ProspectModel,
    sequence_a: ContentSequence,
    sequence_b: ContentSequence,
    model: str = "gpt-4o",
) -> dict:
    """
    Compare two sequences for the same prospect.
    
    Returns side-by-side comparison with recommendation.
    """
    simulator = InteractionSimulator(model=model)
    
    sim_a = simulator.simulate_sequence(prospect, sequence_a)
    sim_b = simulator.simulate_sequence(prospect, sequence_b)
    
    # Determine winner
    score_a = (
        -sim_a.cumulative_objection_reduction * 2 +
        sim_a.pains_addressed_count +
        -sim_a.new_risks_surfaced
    )
    score_b = (
        -sim_b.cumulative_objection_reduction * 2 +
        sim_b.pains_addressed_count +
        -sim_b.new_risks_surfaced
    )
    
    if score_a > score_b + 0.5:
        winner = sequence_a.name
        reason = "Better objection handling and pain coverage"
    elif score_b > score_a + 0.5:
        winner = sequence_b.name
        reason = "Better objection handling and pain coverage"
    else:
        winner = "tie"
        reason = "Both sequences perform similarly"
    
    return {
        "prospect": prospect.name,
        "sequence_a": {
            "name": sequence_a.name,
            "rating": sim_a.sequence_rating,
            "objection_delta": sim_a.cumulative_objection_reduction,
            "pains_addressed": sim_a.pains_addressed_count,
            "risks_surfaced": sim_a.new_risks_surfaced,
            "suggestions": sim_a.improvement_suggestions,
        },
        "sequence_b": {
            "name": sequence_b.name,
            "rating": sim_b.sequence_rating,
            "objection_delta": sim_b.cumulative_objection_reduction,
            "pains_addressed": sim_b.pains_addressed_count,
            "risks_surfaced": sim_b.new_risks_surfaced,
            "suggestions": sim_b.improvement_suggestions,
        },
        "recommendation": {
            "winner": winner,
            "reason": reason,
        },
        "full_simulations": {
            "a": sim_a.to_dict(),
            "b": sim_b.to_dict(),
        },
    }
