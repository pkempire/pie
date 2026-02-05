# Test of Time (ToT) Benchmark: Baselines Analysis

**Date:** 2026-02-04  
**Model:** gpt-4o-mini  
**Dataset:** ToT-Semantic (2,800 questions, 8 types)  
**Samples evaluated:** N=80 (stratified across all 8 question types)

## Three Baselines Tested

1. **baseline**: Raw temporal facts with dates (original dataset prompt)
2. **naive_rag**: Embed facts, retrieve top-20 by cosine similarity, answer
3. **pie_temporal**: PIE-style semantic temporal narrative reformulation (entity timelines, succession chains, overlap markers)

## Results Summary

| Baseline | Correct | Total | Accuracy |
|----------|---------|-------|----------|
| **naive_rag** | **45** | **80** | **56.2%** |
| baseline | 37 | 80 | 46.2% |
| pie_temporal | 25 | 80 | 31.2% |

### Breakdown by Question Type

| Question Type | baseline | naive_rag | pie_temporal | Winner |
|---------------|----------|-----------|--------------|--------|
| event_at_the_time_of_another_event | 90.0% | **100.0%** | **100.0%** | 🟢 tie RAG/PIE |
| event_at_what_time | 80.0% | **100.0%** | 50.0% | 🟢 naive_rag |
| first_last | 60.0% | **100.0%** | 50.0% | 🟢 naive_rag |
| relation_duration | 50.0% | 50.0% | 50.0% | ⚪ tie |
| number_of_events_in_time_interval | 40.0% | **50.0%** | 0.0% | 🟢 naive_rag |
| event_at_time_t | **40.0%** | 0.0% | 0.0% | 🟢 baseline |
| before_after | **10.0%** | 0.0% | 0.0% | 🟢 baseline |
| timeline | 0.0% | **50.0%** | 0.0% | 🟢 naive_rag |

## Key Findings

### 1. Naive RAG is the Best Baseline (56.2%)

Surprisingly, simple semantic retrieval (embedding facts and retrieving top-k by similarity) **significantly outperforms** both raw facts and PIE's semantic reformulation:

- **+10% over baseline** (56.2% vs 46.2%)
- **+25% over PIE** (56.2% vs 31.2%)

This suggests that for temporal reasoning tasks:
1. Not all facts are needed — focused retrieval helps
2. Semantic similarity captures question-relevant facts well
3. Smaller, focused context outperforms large narrative reformulations

### 2. PIE Temporal Underperforms (-15% vs baseline)

The PIE-style semantic reformulation **hurts performance** overall:
- **31.2%** vs baseline's **46.2%**
- Loses critical date precision needed for lookups
- Narrative format may distract from simple temporal arithmetic

However, PIE matches or beats on **relational reasoning** tasks:
- `event_at_the_time_of_another_event`: 100% (same as naive_rag)
- `relation_duration`: 50% (same as others)

### 3. Question Type Analysis

**Where naive_rag dominates:**
- `event_at_what_time`: 100% (simple lookup after retrieval)
- `first_last`: 100% (temporal ordering of retrieved entities)
- `timeline`: 50% (retrieval finds relevant entities)

**Where baseline dominates:**
- `event_at_time_t`: 40% (needs exact date ranges in context)
- `before_after`: 10% (needs full succession chain)

**Where all struggle:**
- `before_after`: Max 10% (requires complex temporal reasoning)
- `timeline`: Max 50% (multi-entity chronological ordering)

### 4. Comparison with Published Results (ToT Paper)

| Source | Model | Accuracy |
|--------|-------|----------|
| **ToT Paper** | GPT-4 Turbo | ~45-55% (ER graphs) |
| **Our baseline** | gpt-4o-mini | 46.2% |
| **Our naive_rag** | gpt-4o-mini | 56.2% |

Our naive_rag baseline **matches or slightly exceeds** GPT-4 Turbo performance using the smaller gpt-4o-mini model, suggesting retrieval augmentation is highly effective for temporal reasoning.

## Implications for PIE Architecture

### The Retrieval > Reformulation Pattern

For ToT-style structured temporal facts:
1. **Retrieval** (naive_rag) works better than **reformulation** (PIE)
2. Focused context beats comprehensive narrative
3. Embedding-based retrieval captures temporal relevance surprisingly well

### Recommendation: Hybrid Approach

For PIE's temporal context compilation, consider:

1. **First retrieve relevant facts** using embedding similarity
2. **Then reformulate only retrieved facts** into PIE's narrative
3. Keep raw dates alongside semantic descriptions for lookup tasks

```
Original approach:
  ALL_FACTS → PIE_REFORMULATION → LLM

Recommended hybrid:
  QUESTION → EMBED → RETRIEVE_TOP_K → PIE_REFORMULATION → LLM
```

This combines the filtering power of RAG with PIE's relational context.

## Reproduction

```bash
# Run all three baselines (N=80)
python baselines_runner.py --limit 80 --model gpt-4o-mini

# Run specific baseline
python baselines_runner.py --limit 50 --baseline naive_rag

# Dry run (show prompts)
python baselines_runner.py --limit 5 --dry-run
```

## Raw Results Files

- `baselines_20260204_232851_n80.json` — Full N=80 run with all baselines
