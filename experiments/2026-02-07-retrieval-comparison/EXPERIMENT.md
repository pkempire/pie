# Retrieval Approach Comparison Experiment

**Date:** 2026-02-07
**Goal:** Establish definitive benchmark numbers for PIE launch

## Hypotheses

1. **H1:** naive_rag achieves 65-75% on LongMemEval (baseline)
2. **H2:** graph_aware + caching achieves higher accuracy on temporal-reasoning questions
3. **H3:** Hybrid approach (graph + RAG) outperforms pure approaches on mixed query types

## Experimental Design

### Approaches to Compare
| Approach | Description | Expected Strength |
|----------|-------------|-------------------|
| naive_rag | Embed chunks, retrieve top-k | Simple facts, preferences |
| pie_temporal | Full extraction → world model → compile | Temporal evolution, state changes |
| graph_aware | LLM intent → graph traversal → bi-temporal filter | Complex multi-hop, temporal ordering |
| hybrid | graph_aware + RAG fallback | Best of both |

### Benchmarks
- **LongMemEval** (500 questions): Primary benchmark
  - single-session-user, single-session-assistant, single-session-preference
  - multi-session, knowledge-update, temporal-reasoning
- **Test of Time** (133 questions): Temporal reasoning focus

### Metrics
- Accuracy (0/0.5/1.0 scoring via GPT-4o judge)
- Latency (ms per question)
- Per-category breakdown

## Parallel Workstreams

### Stream A: Baseline Evaluation (naive_rag)
- Run naive_rag on full LongMemEval (500 questions)
- Establish stable baseline numbers
- ETA: ~30 min

### Stream B: Graph-Aware Optimization
- Pre-cache world models per unique haystack
- Run graph_aware with caching on LongMemEval subset
- Compare latency before/after

### Stream C: Extraction Quality Audit
- Analyze output/world_model.json from amber-seaslug run
- Entity type distribution, temporal coverage
- Identify extraction gaps

### Stream D: Temporal Category Deep-Dive
- Run all approaches on temporal-reasoning subset only
- This is where PIE should shine

## Success Criteria
- naive_rag ≥ 70% (beat Zep)
- graph_aware > naive_rag on temporal-reasoning
- Publishable numbers with confidence intervals
