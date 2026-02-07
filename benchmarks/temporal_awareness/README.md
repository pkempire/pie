# Temporal Awareness Benchmark

Implementation of the benchmark from "Real-Time Deadlines Reveal Temporal Awareness Failures in LLM Strategic Dialogues" (Sehgal et al., 2026, arxiv:2601.13206).

## Key Findings from Paper

| Condition | GPT-5.1 Deal Closure | Notes |
|-----------|---------------------|-------|
| Control (time limit only) | 4% | LLM fails to track elapsed time |
| Time-Aware (remaining time each turn) | 32% | 708% improvement |
| Urgency cues | >32% | Generic "deadline approaching" works better than countdown |
| Turn-based limits | ≥95% | Near-perfect - confirms strategic competence exists |

## Experimental Design

Two LLM agents negotiate a multi-issue hiring contract under strict deadlines.

### Conditions
1. **Control**: Agents told total time at start only
2. **Time-Aware**: Receive remaining time at each turn
3. **Turn-Based** (control): Fixed turn limit instead of time limit

### Time Simulation
- Speech latency: 150 WPM (words per minute)
- Each utterance deducts time proportional to word count
- Time limits: 240, 300, 360 seconds

### Negotiation Structure
- Multi-issue hiring package (salary, bonus, vacation, start date)
- General-sum payoffs (integrative opportunities exist)
- Each agent has private payoff table
- Actions: propose, accept, or invoke BATNA

## Files

- `negotiation_benchmark.py` - Main benchmark implementation
- `scenarios.py` - Negotiation scenarios (New Recruit, Rubbermind)
- `run_benchmark.py` - Execute experiments
- `ANALYSIS.md` - Results and analysis

## Running

```bash
# Run full benchmark
python run_benchmark.py

# Run specific condition
python run_benchmark.py --condition time_aware --time_limit 300

# Test with PIE temporal tracking
python run_benchmark.py --pie_temporal_injection
```
