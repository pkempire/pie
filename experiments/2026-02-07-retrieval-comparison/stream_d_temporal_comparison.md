# Stream D: Temporal-Reasoning Comparison

**Date:** 2026-02-07  
**Status:** PARTIAL - API key expired mid-experiment

## Task

Compare naive_rag vs pie_temporal on temporal-reasoning questions specifically.

## Results

### Naive RAG on Temporal-Reasoning

**Sample:** 80/133 questions (60% of category) before API key expired

| Metric | Value |
|--------|-------|
| Questions Evaluated | 80 |
| Correct (1.0) | 51 |
| Partial (0.5) | 1 |
| Wrong (0.0) | 28 |
| **Accuracy** | **64.4%** |

**Accuracy Progression (from terminal logs):**
- After 20 questions: ~75%
- After 40 questions: ~71%  
- After 60 questions: ~71%
- After 80 questions: **64.4%**

The accuracy declined as more questions were evaluated, suggesting the early questions may have been easier.

### PIE Temporal on Temporal-Reasoning

**Status:** ❌ Could not run - API key expired before starting

## Analysis

### What We Know

1. **Naive RAG: 64.4% on temporal-reasoning** (80 questions sampled)
   - This is LOWER than naive_rag's overall performance (~72% on mixed categories)
   - Temporal questions appear harder for naive RAG

2. **Temporal questions are challenging**
   - Questions like "How many days passed between X and Y?"
   - Questions like "What is the order of events A, B, C?"
   - Require understanding time relationships, not just content retrieval

### Key Question: Does PIE's Temporal Approach Help?

**Hypothesis:** PIE's temporal retrieval should excel here because:
- It explicitly stores timestamps with memories
- It can filter by time periods  
- It should handle "X days/weeks/months ago" questions better

**Cannot verify without running pie_temporal.** The comparison would be:
- naive_rag: 64.4% (observed)
- pie_temporal: ??? (not run)

If pie_temporal scores significantly higher (e.g., 75%+), it validates the temporal-aware design.
If pie_temporal scores similar or lower, the temporal features may not be helping as intended.

## Limitations

1. Only 60% of temporal-reasoning questions were evaluated
2. pie_temporal baseline was not run
3. API key expired mid-experiment
4. Results are preliminary estimates from terminal output

## Recommendation

Re-run this experiment when API access is restored:
```bash
# Full naive_rag on temporal
python3 -m benchmarks.longmemeval.runner --baseline naive_rag --category temporal-reasoning --output experiments/temporal_naive_rag_full

# pie_temporal comparison  
python3 -m benchmarks.longmemeval.runner --baseline pie_temporal --category temporal-reasoning --limit 50 --output experiments/temporal_pie
```

## Raw Data

The partial results (before API failure) showed these sample questions and scores:

| Question Type | Example | naive_rag Result |
|--------------|---------|------------------|
| Duration | "How many days passed between museum visits?" | ✅ Correct |
| Ordering | "Which three events happened in order first to last?" | ✅ Correct |
| Relative time | "How many weeks ago did I meet my aunt?" | ✅ Correct |
| Duration | "How many months since I participated in charity events?" | ❌ Wrong |
| Relative time | "How many days did I spend on solo camping trip?" | ❌ Wrong |

The errors suggest naive_rag struggles when:
- Events aren't directly mentioned together
- Time calculations require cross-referencing multiple memories
- Relative dates ("X ago") need current date context

## Conclusion

**Preliminary finding:** Naive RAG achieves ~64% on temporal-reasoning questions, which is notably lower than its overall benchmark performance. This suggests temporal questions are a weakness for pure semantic retrieval.

**Key insight:** This is exactly where PIE's temporal-aware retrieval should provide the biggest improvement - but we couldn't test it due to API issues.

**Priority:** Re-run pie_temporal comparison when API is restored. This is the most valuable experiment for validating PIE's core temporal hypothesis.
