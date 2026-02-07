# Temporal Awareness in LLM Strategic Dialogues - Analysis

**Source**: Sehgal et al. (2026). "Real-Time Deadlines Reveal Temporal Awareness Failures in LLM Strategic Dialogues." arXiv:2601.13206

**Benchmark Status**: Implementation complete. Code/data from authors pending paper acceptance.

---

## Executive Summary

LLMs exhibit a **systematic failure to track continuous time** during multi-turn strategic interactions. In negotiation tasks:

| Condition | GPT-5.1 Deal Closure | Interpretation |
|-----------|---------------------|----------------|
| Control (time limit stated once) | **4%** | Cannot track elapsed time |
| Time-Aware (countdown each turn) | **32%** | Explicit time helps significantly |
| Urgency cues ("deadline approaching") | **>32%** | Generic urgency beats numeric countdown |
| Turn-based limits | **≥95%** | Strategic reasoning intact |

**Key Insight**: The failure is specifically in **temporal tracking**, not strategic reasoning. The same models that fail under time pressure achieve near-perfect performance when constraints are expressed as turn counts.

---

## Paper Findings

### 1. The Core Problem

LLMs generate tokens discretely but real-world interaction requires continuous time awareness. The paper defines **temporal awareness** as:

1. Representing how much time has elapsed/remains
2. Anticipating how others' behavior changes with time
3. Conditioning strategy on temporal state

### 2. Experimental Design

- **Task**: Multi-issue hiring negotiations (salary, bonus, vacation, start date)
- **Setup**: Two LLM agents negotiate under strict deadlines
- **Time simulation**: 150 WPM speech latency deducted from budget
- **Conditions**:
  - **Control**: Told total time once at start
  - **Time-Aware**: Receive remaining time at each turn
  - **Turn-Based**: Fixed turn limit (no time dimension)

### 3. Results Across Models (Table 2 from paper)

| Model | Control | Time-Aware | Significant? |
|-------|---------|------------|--------------|
| GPT-5.1-chat-latest | 4.0% | 32.3% | *** |
| GPT-5.1 | 0.3% | 2.7% | |
| GPT-5 | 1.0% | 3.3% | |
| GPT-5-mini | 12.3% | 26.7% | *** |
| GPT-4.1 | 44.7% | 72.0% | *** |
| Claude Sonnet-4.5 (no reasoning) | 0.0% | 0.0% | |
| Claude Sonnet-4.5 (med reasoning) | 0.0% | 3.7% | ** |
| Claude Sonnet-4.5 (high reasoning) | 0.0% | 1.3% | |
| Qwen3-8b (no reasoning) | 31.3% | 31.7% | |
| Qwen3-8b (reasoning enabled) | 0.0% | 3.0% | * |

**Notable observations**:
- GPT-4.1 (non-reasoning) achieves best performance - temporal awareness ≠ reasoning capability
- Claude models struggle overall (strategic competence bottleneck)
- Qwen3-8b with reasoning degrades - extensive reasoning tokens consume time budget

### 4. The Urgency Ablation

A crucial finding: **Generic urgency cues outperform numeric countdowns**.

When agents receive "(Deadline approaching--act with urgency.)" at each turn (no numeric time), they perform *better* than the Time-Aware condition.

**Interpretation**: Models can respond to urgency when cued, but:
1. Cannot generate urgency signals internally
2. Struggle to translate "137 seconds left" into appropriate strategic adaptation
3. Benefit more from high-level framing than precise data

### 5. Turn-Based Success

Under turn limits (5-9 turns), models achieve ≥95% deal closure. This proves:
- Strategic reasoning is intact
- The failure is specifically temporal, not strategic
- Token-aligned constraints (turns) work; continuous constraints (time) don't

---

## Implications for PIE (Personal Intelligence Entity)

### Why This Matters for PIE

PIE operates in continuous time across multiple sessions, contexts, and deadlines. If base LLMs cannot track time, PIE inherits this limitation. However, PIE's architecture may offer mitigation strategies.

### PIE Potential Solutions

#### 1. **Explicit Temporal State Injection** ✓ Most Promising

PIE already tracks entity states. We could inject temporal state as a first-class entity:

```python
# Current PIE entity tracking
{
  "entity": "meeting_deadline",
  "type": "temporal",
  "state": {
    "deadline": "2024-01-15T14:00:00Z",
    "remaining_seconds": 3600,
    "urgency": "medium",  # computed from remaining time
    "urgency_cue": "1 hour remaining - consider wrapping up"
  }
}
```

The paper shows urgency framing helps more than raw numbers. PIE could:
1. Track deadlines as entities
2. Compute urgency levels
3. Inject natural-language urgency cues at context boundaries

#### 2. **Turn-Aligned Temporal Reasoning**

Since models succeed with turn limits, PIE could:
- Convert time constraints to approximate turn budgets
- Frame deadlines as "~N interactions remaining"
- Use turn estimates for planning

```python
def time_to_turns(remaining_seconds: float, avg_turn_seconds: float = 30) -> int:
    """Convert remaining time to approximate turn budget."""
    return max(1, int(remaining_seconds / avg_turn_seconds))
```

#### 3. **Proactive Deadline Monitoring**

PIE's heartbeat system could monitor approaching deadlines and inject reminders:

```python
# In HEARTBEAT.md processing
if any_deadline_within(hours=2):
    inject_context("Upcoming deadline: {deadline_name} in {time_remaining}")
```

#### 4. **Temporal Anchoring in Memory**

PIE's memory system could include temporal anchors:
- "This conversation started 15 minutes ago"
- "User mentioned deadline of 3pm (47 minutes away)"
- "Previous response took ~30 seconds of user's time"

### Benchmark Condition: PIE_TEMPORAL

I've added a `PIE_TEMPORAL` condition to the benchmark that tests:

```python
# PIE-style temporal state injection
prefix = f"[TEMPORAL_STATE: remaining_time={remaining_time:.1f}s, urgency={'HIGH' if remaining_time < 60 else 'MEDIUM' if remaining_time < 120 else 'LOW'}]"
```

This combines numeric precision with urgency framing.

---

## Proposed Experiments

### Experiment 1: PIE Temporal Injection

**Hypothesis**: Combining numeric time with urgency labels improves on Time-Aware alone.

**Method**: Run benchmark with PIE_TEMPORAL condition vs Time-Aware.

### Experiment 2: Natural Language Time

**Hypothesis**: Natural language time expressions ("about 2 minutes left") work better than precise countdowns ("137 seconds").

**Method**: Compare:
- "137 seconds remaining"
- "About 2 minutes left"
- "Running low on time - wrap up soon"

### Experiment 3: Turn-Converted Framing

**Hypothesis**: Expressing deadlines as turn estimates improves temporal reasoning.

**Method**: Instead of "240 seconds", say "approximately 6-8 more exchanges possible".

### Experiment 4: Deadline Entity Tracking

**Hypothesis**: Treating deadlines as tracked entities with state updates improves performance.

**Method**: Inject deadline entity with each context update:
```json
{
  "deadline": {"status": "approaching", "remaining": "2 min", "action_needed": "conclude_or_extend"}
}
```

---

## Conclusions

### The Paper Establishes

1. **LLMs cannot internally track continuous time** - this is a structural limitation
2. **Explicit temporal feedback helps** but doesn't fully close the gap
3. **Urgency framing outperforms numeric countdowns** - models respond to cues but can't generate them
4. **Strategic competence exists** - turn-based success proves this

### For PIE Development

1. **Always inject temporal state** for time-sensitive tasks
2. **Prefer urgency framing** over raw numbers
3. **Consider turn-based approximations** for planning
4. **Monitor deadlines proactively** via heartbeat
5. **Test the PIE_TEMPORAL condition** to validate improvements

### Broader Implications

This benchmark reveals a fundamental mismatch between:
- **LLM architecture** (discrete token prediction)
- **Real-world requirements** (continuous time awareness)

Until architectures change (temporal embeddings, time-aware training), **systems like PIE must compensate externally** by making time legible to the model.

---

## Running the Benchmark

```bash
# Full benchmark (requires API keys)
export ANTHROPIC_API_KEY=your_key
python run_benchmark.py --model claude-sonnet-4-20250514 --trials 100

# Quick test with mock
python run_benchmark.py --mock --quick

# Test PIE approach
python run_benchmark.py --pie_temporal_injection --model claude-sonnet-4-20250514

# Compare all conditions
python run_benchmark.py --trials 50 --pie_temporal_injection
```

---

## References

- Sehgal, N., Guntuku, S.C., & Ungar, L. (2026). Real-Time Deadlines Reveal Temporal Awareness Failures in LLM Strategic Dialogues. arXiv:2601.13206
- Wang et al. (2025). Discrete Minds in a Continuous World. arXiv:2506.05790
- Cheng et al. (2025). Temporal Blindness in Multi-Turn LLM Agents. arXiv:2510.23853
