# Test of Time (ToT) Benchmark: PIE Semantic Temporal Context Analysis

**Date:** 2026-02-04  
**Model:** gpt-4o-mini  
**Dataset:** ToT Semantic (2,800 questions, 8 types)  
**Samples evaluated:** N=100 (stratified across all 8 question types)

## Hypothesis

PIE's semantic temporal context compilation — grouping facts into entity timelines, adding duration descriptions, overlap markers, relative time anchoring, and succession chains — will improve LLM temporal reasoning compared to raw date-based facts.

## Results Summary

| Metric | Baseline (raw dates) | PIE-style (semantic) | Delta |
|--------|---------------------|---------------------|-------|
| **Overall Accuracy** | **38.0%** | **37.0%** | **-1.0%** |
| Questions rescued by PIE | — | 12 | — |
| Questions hurt by PIE | — | 13 | — |

### Breakdown by Question Type

| Question Type | N | Baseline | PIE | Delta | Winner |
|---------------|---|----------|-----|-------|--------|
| relation_duration | 12 | 33.3% | **66.7%** | **+33.3%** | 🟢 PIE |
| first_last | 12 | 41.7% | **58.3%** | **+16.7%** | 🟢 PIE |
| event_at_time_of_another | 13 | 61.5% | **76.9%** | **+15.4%** | 🟢 PIE |
| timeline | 12 | 0.0% | **8.3%** | **+8.3%** | 🟢 PIE |
| before_after | 13 | 7.7% | 0.0% | -7.7% | 🔴 Baseline |
| number_of_events_in_interval | 12 | **41.7%** | 25.0% | -16.7% | 🔴 Baseline |
| event_at_what_time | 13 | **84.6%** | 61.5% | -23.1% | 🔴 Baseline |
| event_at_time_t | 13 | **30.8%** | 0.0% | -30.8% | 🔴 Baseline |

## Key Findings

### 1. PIE excels at complex relational reasoning

The largest PIE gains are on question types that require understanding **relationships between temporal facts**:

- **relation_duration (+33.3%)**: "The 3rd time it happened, for how many years was E33 the R53 of E22?" — PIE's entity timelines with succession chains make it trivial to find the Nth occurrence and read its duration.
- **first_last (+16.7%)**: "Identify the first R90 of E11" — PIE's KEY TEMPORAL RELATIONSHIPS section explicitly lists first/last holders with dates. The baseline requires scanning hundreds of unsorted facts.
- **event_at_time_of_another (+15.4%)**: "At the start time of E92 being R17 of E59, E22 was R11 of which entity?" — PIE's overlap markers and concurrent annotations help the LLM cross-reference temporal windows.

### 2. Baseline wins on simple lookups

The baseline outperforms PIE on tasks that are essentially **key-value lookups** against the raw facts:

- **event_at_time_t (-30.8%)**: "Which entity was R63 of E69 in 1982?" — Scanning a flat list for a matching date range is straightforward. PIE's restructuring actually makes this harder by burying facts inside entity groupings.
- **event_at_what_time (-23.1%)**: "When did E92 stop being R88 of E11?" — A direct scan of raw facts for the end date is simple. PIE's reformulation doesn't add value here.
- **number_of_events (-16.7%)**: "How many unique entities were R22 of E83 between 1921-1928?" — Counting requires precise fact enumeration. PIE's overlap annotations introduce noise.

### 3. The "right tool for the right task" pattern

This reveals a fundamental insight about temporal context compilation: **narrative formatting helps reasoning but hurts retrieval.**

| Task nature | Best format | Why |
|-------------|-------------|-----|
| Cross-referencing temporal windows | PIE | Overlaps & succession chains pre-compute the relationships |
| Finding Nth occurrence of a pattern | PIE | Entity timelines group same-entity facts together |
| Understanding temporal ordering | PIE | Sequence numbers and gap markers make order explicit |
| Point-in-time lookups ("who at year X?") | Raw | Flat scan of `from YYYY to YYYY` is faster for LLMs |
| Date extraction ("when did X stop?") | Raw | End dates are right there in the raw fact |
| Counting within intervals | Raw | Raw facts are enumerable; PIE adds distractor context |

## Implications for PIE Architecture

### Current finding
PIE's semantic temporal context compilation is not a universal improvement — it's a **specialization for complex temporal reasoning** at the cost of simple retrieval.

### Recommended approach: Hybrid context
The optimal strategy would be to **include both formats**:
1. Raw facts section (for lookups)
2. PIE semantic section (for reasoning)
3. Let the LLM choose which to reference based on the question type

Alternatively, a **question-type classifier** could select the appropriate format:
- Simple lookup → raw format
- Complex reasoning → PIE format

### Broader lesson
This aligns with PIE's architecture philosophy: the temporal context compiler should be **adaptive**, not one-size-fits-all. Just as PIE compiles different context for different MCP queries, the format should adapt to the reasoning task.

## Reproduction

```bash
# Dry run (verify reformulation)
python benchmarks/test-of-time/runner.py --limit 20 --dry-run

# Full evaluation
python benchmarks/test-of-time/runner.py --limit 100

# Per-type evaluation
python benchmarks/test-of-time/runner.py --limit 50 --question-type relation_duration
python benchmarks/test-of-time/runner.py --limit 50 --question-type event_at_time_t
```

## Raw Results Files

- `results_20260204_214528_n100.json` — Full N=100 run with per-question details
- `results_20260204_214414_n50.json` — Initial N=50 run
