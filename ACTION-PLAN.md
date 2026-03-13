# PIE Action Plan: Experiments, Benchmarks, and Updates

**Date:** 2026-02-11

---

## Phase 0: Critical Bug Fixes (Do First)

Before any experiments, fix the bugs that will crash the pipeline:

| Bug | File | Fix |
|-----|------|-----|
| Missing `use_sliding_window` field | `pie/config.py` | Add `use_sliding_window: bool = True` to `PIEConfig` |
| Wrong field name `web_ground_entity_types` | `pie/ingestion/pipeline.py:215` | Change to `web_ground_types` |
| Wrong action value `"new"` vs `"create"` | `pie/resolution/resolver.py:77` | Change to `"create"` |
| Invalid model names (`gpt-5-mini`) | `pie/core/llm.py:34,39` | Use actual model names (`gpt-4o-mini`) |
| Lost expected_answer in query_interface | `pie/eval/query_interface.py:465-468` | Append `result_dict` to results |
| Non-existent attribute `self.llm.config` | `pie/resolution/resolver.py:289` | Pass resolution model as init param |

---

## Phase 1: Complete Pipeline Run (Week 1)

### 1a. Resume ingestion
```bash
python3 run.py --input ~/Downloads/conversations.json --skip-batches 25 --limit 203
```
- Process all remaining batches (25–203)
- Estimated: ~$15-30 in API costs
- This gives us the full world model to evaluate against

### 1b. Quality audit
```bash
python3 -m pie.eval.extraction_quality --world-model output/world_model.json
```
- Check entity type distribution
- Check date coverage (should be >80% after prompt fix)
- Check transition type distribution (if all "creation," something's wrong)
- Check alias coverage
- Manual spot-check: pick 10 random entities, verify transition chains make sense

---

## Phase 2: Benchmark Evaluation (Week 1-2)

### 2a. Set up MemoryBench
```bash
git clone https://github.com/supermemoryai/memorybench
cd memorybench && bun install
```

### 2b. Write PIE provider adapter
Create `memorybench/src/providers/pie/index.ts`:
```typescript
export class PIEProvider implements MemoryProvider {
  async ingest(sessions: Session[]) {
    // Run PIE ingestion pipeline on benchmark sessions
  }
  async search(query: string) {
    // Use PIE's context compiler to generate LLM-ready context
  }
  async answer(query: string, context: string) {
    // LLM call with compiled context
  }
}
```

### 2c. Run benchmarks
```bash
# LongMemEval (primary benchmark)
bun run src/index.ts run -p pie -b longmemeval -j gpt-4o

# LoCoMo
bun run src/index.ts run -p pie -b locomo -j gpt-4o

# Test of Time (with proper date extraction)
python3 -m pie.eval.temporal_ablation --benchmark tot --conditions all
```

### 2d. Decision point based on results

| PIE Score on LongMemEval | Action |
|---|---|
| **<75%** | Pivot: adopt Mastra-style observations, keep graph for analytics only |
| **75-85%** | Hybrid: Mastra-style retrieval layer + graph analytics layer |
| **>85%** | Invest: build real graph infrastructure (FalkorDB/Neo4j) |

---

## Phase 3: Task-Adaptive Formatting Validation (Week 2)

This is the most publishable finding. Needs rigorous validation.

### 3a. Systematic ablation study on Test of Time
Run 5 conditions:
1. `raw_timestamps` — ISO dates only
2. `relative_time` — "3 months ago"
3. `semantic_narrative` — full narrative with periods
4. `hybrid_both` — both raw dates AND narrative
5. `task_adaptive` — detect query type, pick format

### 3b. Same study on LongMemEval temporal subset
Extract temporal reasoning questions from LongMemEval and run the same 5 conditions.

### 3c. Statistical validation
- Run each condition 3x (different seeds)
- Report mean ± std
- Fisher's exact test for significance between conditions
- Per-question-type breakdown

---

## Phase 4: Improvements Based on SOTA Research (Week 2-3)

### 4a. Add BM25 for candidate retrieval
```python
# In resolver.py — add Tier 0.5 before embedding similarity
from rank_bm25 import BM25Okapi

class EntityResolver:
    def _build_bm25_index(self):
        """Build BM25 index over entity names + aliases."""
        corpus = []
        for entity in self.world_model.entities.values():
            tokens = entity.name.lower().split() + [a.lower() for a in entity.aliases]
            corpus.append(tokens)
        self.bm25 = BM25Okapi(corpus)
```

### 4b. Implement basic importance scoring
Not full PageRank, but graph-structural importance:
```python
def compute_importance(entity, world_model):
    degree = len(world_model.get_relationships(entity.id))
    transition_count = len(world_model.get_transitions(entity.id))
    recency = exp(-days_since_last_seen / 30)
    return 0.4 * normalize(degree) + 0.3 * normalize(transition_count) + 0.3 * recency
```

### 4c. Add Mastra-style three-date model to extraction
Update extraction prompt to output:
- `observation_date`: when the info was extracted
- `referenced_date`: the actual date being discussed
- `relative_date`: e.g., "last week," "in January"

### 4d. Implement TReMu-style neuro-symbolic temporal reasoning
For temporal queries, generate Python code to compute temporal answers:
```python
# Query: "How long between starting Project X and pivoting?"
# Generate:
start = datetime(2025, 1, 5)  # from transition chain
pivot = datetime(2025, 3, 22)  # from transition chain
duration = (pivot - start).days  # = 76 days
answer = f"{duration} days (about {duration // 30} months)"
```

---

## Phase 5: New Benchmark Creation (Week 3-4)

Current benchmarks don't test what makes state transitions valuable. Create one.

### 5a. Temporal State Reasoning Benchmark (TSRB)
Question categories:
1. **Trajectory reconstruction**: "How has entity X evolved?"
2. **Temporal diff**: "What changed between T1 and T2?"
3. **Cross-entity patterns**: "Do similar entities follow similar lifecycles?"
4. **Contradiction detection**: "Where has the user changed their position?"
5. **Temporal validity**: "Is this information still current?"

### 5b. Data generation
- Use PIE's own world model as ground truth
- Generate questions programmatically from known transition chains
- Create synthetic data with planted patterns for controlled evaluation

### 5c. Evaluate PIE vs. baselines
Run PIE, Mastra-style observations, and naive RAG on TSRB. If PIE doesn't win on these graph-specific tasks, the approach is over-engineering.

---

## Phase 6: Framework Generalization (Week 4+)

### 6a. Extract core temporal memory primitives into standalone library
```python
# temporal_memory/
#   state.py       — Entity, StateTransition, TransitionChain
#   tracker.py     — StateTracker (detect changes, type transitions)
#   compiler.py    — TemporalCompiler (format for LLM consumption)
#   router.py      — QueryRouter (detect query type, pick format)
#   memory.py      — TemporalMemory (high-level API)
```

### 6b. Integration targets
- MCP server for Claude/GPT integration
- LangGraph node for agent pipelines
- Standalone Python package on PyPI

---

## Demo Concepts (Twitter-Ready)

### Demo 1: "Your AI's Memory Timeline" (Most Viral Potential)
Visual demo showing an interactive timeline of how the AI's understanding of you has evolved. Animated graph where entities appear, connect, change color when contradicted, cluster into life periods. Narrated: "Here's what happens when your AI actually understands how your life is changing."

### Demo 2: "The Vegetarian Test"
Side-by-side comparison. Tell ChatGPT "I'm vegetarian" → "I'm pescatarian." Ask "How has my diet changed?" ChatGPT gives current state. PIE gives the trajectory with timing and trigger. Quick, punchy, immediately demonstrates the gap.

### Demo 3: "Planning Fallacy Detector"
Feed an AI your project history. It generates a timeline prediction for a new project. Then show it your actual history of project timelines vs. predictions. The AI's estimate vs. what your temporal patterns predict. Demonstrates practical value of temporal reasoning.

### Demo 4: "Contradiction Radar"
Show a live dashboard of beliefs that have been contradicted over time. "You said X in March but Y in August." Visual radar chart of belief stability vs. volatility. Engaging because it reveals something about yourself.

---

## Priority Order

1. **Bug fixes** (Phase 0) — 1 day
2. **Complete pipeline run** (Phase 1) — 2-3 days
3. **LongMemEval benchmark** (Phase 2) — 3-4 days
4. **Task-adaptive ablation** (Phase 3) — 2-3 days
5. **Demo build** — 2-3 days (can run parallel with Phase 3)
6. **Improvements** (Phase 4) — ongoing
7. **New benchmark** (Phase 5) — 1 week
8. **Framework** (Phase 6) — 2+ weeks

Total to first meaningful result: ~1 week (through Phase 2).
Total to publishable result: ~2-3 weeks (through Phase 3).
