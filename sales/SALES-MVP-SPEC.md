# Sales Intelligence MVP — Content Simulation Spec

## Core Idea

Current state: Monte Carlo simulation for win probability based on scenario changes (rep change, objection resolved, etc.)

**New MVP**: Upload sales content (email sequences, pitch decks, call scripts) and simulate how that content will interact with each prospect's world model to predict state evolution.

---

## What Makes This Different

Current sales tools answer: "What's my win probability?"

We answer: "If I send this 5-email nurture sequence to Nina Petrov, how will her **objections evolve**, which **pain points get addressed**, and what **new concerns might surface**?"

This is **content ↔ world model interaction**, not static forecasting.

---

## Architecture

### 1. Prospect World Model (Already Exists)
```
ProspectModel:
  - pain_points: [{description, severity, category, evolution}]
  - objections: [{text, strength, type, response_effectiveness}]
  - buying_signals: [{signal, strength, timing}]
  - personality_traits: ["analytical", "risk-averse", ...]
  - communication_style: "prefers data over stories"
  - decision_style: "consensus-driven"
  - stakeholders: [{name, role, influence, sentiment}]
```

### 2. Content Model (NEW)

```python
@dataclass
class SalesContent:
    """A piece of sales content that can be simulated."""
    id: str
    type: Literal["email", "call_script", "pitch_deck", "proposal", "case_study"]
    name: str
    content: str  # Raw text/transcript
    
    # Extracted from content
    claims: list[str]  # Value props, promises
    proof_points: list[str]  # Case studies, stats
    pain_addressed: list[str]  # Which pains this content addresses
    cta: str  # What action it asks for
    tone: str  # Professional, casual, urgent
    
@dataclass
class ContentSequence:
    """A sequence of content (e.g., nurture campaign)."""
    id: str
    name: str
    items: list[SalesContent]
    timing: list[int]  # Days between each piece
```

### 3. Interaction Simulator (NEW)

```python
class ContentInteractionSimulator:
    """
    Simulates how content interacts with a prospect's world model.
    
    Uses LLM to predict state changes based on:
    - Content claims vs prospect pain points
    - Content proof points vs prospect objections
    - Content tone vs prospect communication style
    - Content complexity vs prospect decision style
    """
    
    def simulate_interaction(
        self,
        prospect: ProspectModel,
        content: SalesContent,
    ) -> InteractionResult:
        """
        Predict how this prospect will react to this content.
        
        Returns predicted changes to:
        - Objection strengths
        - Pain point resonance
        - Buying signal emergence
        - Next action likelihood
        """
        ...
    
    def simulate_sequence(
        self,
        prospect: ProspectModel,
        sequence: ContentSequence,
    ) -> SequenceSimulation:
        """
        Simulate a full content sequence, tracking state evolution.
        
        Returns:
        - State after each content piece
        - Key inflection points
        - Risk moments (where deal could stall)
        - Optimal send timing adjustments
        """
        ...
```

### 4. Interaction Result Model (NEW)

```python
@dataclass
class InteractionResult:
    """Predicted result of content ↔ prospect interaction."""
    
    # State changes
    objection_changes: list[dict]  # [{objection: str, delta: float, reason: str}]
    pain_resonance: list[dict]  # [{pain: str, resonance_score: float}]
    new_signals: list[str]  # New buying signals that might emerge
    new_concerns: list[str]  # New objections that might surface
    
    # Engagement prediction
    open_probability: float  # 0-1
    response_probability: float  # 0-1
    predicted_response_type: str  # "positive", "objection", "clarification", "ghost"
    
    # Timing
    optimal_send_time: str  # "morning", "afternoon", etc.
    optimal_day_of_week: str
    
    # Next best action
    recommended_followup: str
    
@dataclass
class SequenceSimulation:
    """Full simulation of a content sequence."""
    
    # Per-step results
    step_results: list[InteractionResult]
    
    # Aggregate metrics
    cumulative_objection_reduction: float  # How much total objection strength dropped
    pain_points_addressed: int  # How many pains were meaningfully addressed
    new_risks_surfaced: int  # How many new concerns came up
    
    # Trajectory
    trajectory: list[dict]  # [{step: int, win_prob: float, state_summary: str}]
    
    # Recommendations
    sequence_rating: str  # "strong", "moderate", "weak", "risky"
    improvement_suggestions: list[str]
```

---

## User Flow

### 1. Upload Content
- Upload email templates, sequences, call scripts
- System extracts: claims, proof points, pains addressed, tone

### 2. Select Prospect
- Pick a prospect from pipeline
- See their current world model state

### 3. Run Simulation
- "What happens if I send this 5-email sequence to Nina?"
- See step-by-step state evolution:
  - After Email 1: Budget objection ↓15%, "needs ROI data" concern emerges
  - After Email 2: ROI concern addressed, technical eval signal ↑
  - After Email 3: Ghost risk HIGH (too much content too fast)
  - ...

### 4. Compare Sequences
- "Compare 'ROI-focused' vs 'Pain-focused' sequence for this prospect"
- Side-by-side trajectory comparison
- "ROI-focused wins for analytical prospects like Nina"

### 5. Optimize
- "Generate optimal sequence for Nina's profile"
- System suggests content ordering, timing, customizations

---

## Implementation Plan

### Phase 1: Content Upload & Extraction (2 days)
- [ ] `SalesContent` and `ContentSequence` models
- [ ] Content upload endpoint (txt, pdf, docx)
- [ ] LLM extraction: claims, proof points, pains addressed, tone
- [ ] Content library UI

### Phase 2: Interaction Simulator (3 days)
- [ ] `ContentInteractionSimulator` class
- [ ] Prompt engineering for interaction prediction
- [ ] `InteractionResult` generation
- [ ] Single content ↔ prospect simulation API

### Phase 3: Sequence Simulation (2 days)
- [ ] `SequenceSimulation` with trajectory tracking
- [ ] State evolution across multiple interactions
- [ ] Risk detection (ghost, objection escalation)
- [ ] Sequence simulation API

### Phase 4: UI (2 days)
- [ ] Content library page
- [ ] Simulation wizard (pick prospect → pick content → run)
- [ ] Trajectory visualization
- [ ] Comparison view

### Phase 5: Polish (1 day)
- [ ] Sequence optimization suggestions
- [ ] Export results
- [ ] Demo data for common sequences

---

## Differentiation

| Feature | Gong/Chorus | Clari | Us |
|---------|-------------|-------|-----|
| Call transcription | ✅ | ❌ | ✅ |
| Win probability | ✅ | ✅ | ✅ |
| Per-customer world model | ❌ | ❌ | ✅ |
| Content simulation | ❌ | ❌ | ✅ |
| "What if I send this?" | ❌ | ❌ | ✅ |

**Our wedge**: Not just "what's the deal health?" but "what should I actually send/say next, and what will happen?"

---

## Decay / Validity (From Earlier Discussion)

Each piece of state in the world model should have:

```python
@dataclass
class TemporalFact:
    """A fact with temporal validity."""
    value: Any
    first_observed: float
    last_confirmed: float
    half_life_hours: float  # Expected validity duration
    verification_due: float | None  # When to re-verify
    confidence: float  # 0-1, decays over time
    
    def current_confidence(self) -> float:
        """Confidence decayed based on time since last confirmation."""
        hours_elapsed = (time.time() - self.last_confirmed) / 3600
        decay_factor = 0.5 ** (hours_elapsed / self.half_life_hours)
        return self.confidence * decay_factor
```

Examples:
- "Nina is frustrated with current vendor" → half_life: 720h (30 days)
- "Budget approved for Q1" → half_life: 2160h (90 days), verification_due: Q1 end
- "In a meeting right now" → half_life: 2h

This lets the simulation account for stale information: "Nina's budget objection was observed 60 days ago — confidence decayed to 35%, recommend re-qualifying."

---

## Open Questions

1. **Calibration**: How do we know the simulations are accurate? Need to track actual outcomes vs predictions.

2. **Personalization depth**: How much do we model individual communication preferences? Current: broad traits. Future: learned patterns from their actual responses.

3. **Content generation**: Phase 2 could generate optimized content, not just simulate existing content.

4. **Multi-stakeholder**: Simulate content hitting multiple stakeholders with different profiles.
