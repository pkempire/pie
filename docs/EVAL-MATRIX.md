# PIE Evaluation Matrix

## Retrieval Approaches (What We Have)

| Approach | Description | Implementation | Status |
|----------|-------------|----------------|--------|
| **naive_rag** | Embedding similarity over raw entity data | `query_interface.retrieve_entities_by_embedding()` | ✅ Working |
| **pie_temporal** | Embedding + compile semantic temporal context | `query_interface.compile_entity_context()` | ⚠️ Working but context uses ingestion timestamps, not event dates |
| **graph_aware** | LLM intent parsing → constrained seeds → BFS traversal | `graph_retriever.retrieve_subgraph()` | ✅ Fixed (now computes embeddings on-the-fly) |
| **hybrid** | Embedding + graph merged | `modal_eval.py` implementation | ✅ Working |
| **rl_retriever** | GRPO-trained policy for subgraph selection | `rl_retriever.py` | 📋 Scaffold only |

---

## Benchmarks (What We Can Run On)

| Benchmark | Focus | Data Location | # Questions | Our Adapter |
|-----------|-------|---------------|-------------|-------------|
| **LongMemEval** | Multi-session, preferences, temporal | `benchmarks/longmemeval/data/` | 500 | ✅ `longmemeval/runner.py` |
| **Test of Time (ToT)** | Temporal arithmetic | `benchmarks/tot/tot_semantic.json` | 800 | ✅ `tot/baselines_runner.py` |
| **LoCoMo** | Long conversation reasoning | `benchmarks/locomo/data/` | 1986 | ✅ `locomo/runner.py` |
| **MSC** | Persona consistency | `benchmarks/msc/data/` | ~500 | ✅ `msc/runner.py` |
| **MemoryAgentBench** | Agent memory tasks | `benchmarks/memoryagentbench/repo/` | varies | ⚠️ Partial |
| **MemoryBench** | General memory | `benchmarks/memorybench/repo/` | varies | ⚠️ Partial |
| **PersistBench** | Long-term persistence | `benchmarks/persistbench/` | varies | ⚠️ Partial |
| **Temporal Awareness** | Negotiation scenarios | `benchmarks/temporal_awareness/` | 280 | ✅ Custom |

---

## Results So Far (Feb 2026)

| Benchmark | naive_rag | pie_temporal | graph_aware | hybrid | Notes |
|-----------|-----------|--------------|-------------|--------|-------|
| LongMemEval | **66.3%** | — | — | — | 500q full run |
| ToT | **56.2%** | 31.2% | — | — | 80q sample |
| LoCoMo | 58% | — | — | — | subset |
| MSC | 46% | — | — | — | 76% partial credit |

**Key Finding:** pie_temporal HURTS on ToT (-25%) because semantic reformulation loses precise dates needed for arithmetic.

---

## What Each Approach Actually Does

### naive_rag
```
1. Embed query: query_emb = llm.embed_single(query)
2. For each entity, compute embedding from name+state (on-the-fly)
3. Score by cosine similarity
4. Return top-k entities as context
5. LLM answers with that context
```

### pie_temporal
```
1-4. Same as naive_rag
5. For each entity, compile_entity_context():
   - "First appeared {X months ago}"
   - "Changed state N times (~velocity/month)"
   - Timeline of transitions with humanized deltas
   - Current state description
   - Relationships
6. LLM answers with compiled narratives
```

### graph_aware
```
1. parse_query_intent(query):
   - Extract entity_types, named_entities, temporal_pattern
   - Extract relationship_types, hop_pattern, query_type
2. select_seeds():
   - Named entity lookup (exact, fuzzy)
   - Embedding search (constrained by intent)
   - Type-based fallback
3. traverse_from_seeds():
   - BFS following relationships
   - Respect hop_pattern (direct/transitive/neighborhood)
   - Penalize irrelevant edge types
4. filter_by_temporal_intent():
   - Re-score based on "evolution", "first", "last", etc.
5. Compile context for selected subgraph
```

### hybrid
```
1. Run naive_rag for top-k/2 entities
2. Run graph_aware for top-k/2 entities
3. Merge and deduplicate
4. Compile context
```

---

## Known Issues

### 1. No Event Entities with Dates
- **Symptom:** 0 entities of type "event" in world model
- **Cause:** Extraction was run before event/date prompt was added
- **Impact:** pie_temporal's "temporal" context is fake — uses ingestion timestamps
- **Fix:** Re-run extraction from scratch (~$20, ~4 hours)

### 2. Embeddings Not Persisted
- **Symptom:** `entity.embedding = None` for all entities
- **Cause:** world_model.py intentionally skips embeddings in JSON (too large)
- **Impact:** Every retrieval must recompute embeddings (slow, $$$)
- **Fix:** Either persist embeddings or cache them in memory on load

### 3. ToT Benchmark Mismatch
- **Symptom:** 0% accuracy when running on ToT with PIE world model
- **Cause:** ToT uses synthetic stories, our world model is from ChatGPT history
- **Impact:** Can't evaluate PIE on ToT unless we build world model from ToT data
- **Fix:** Use ToT's own data format, not our world model

---

## Commands Quick Reference

```bash
# Test naive_rag on ToT (5 questions)
python3 modal_eval.py --benchmark tot --approach naive_rag --n 5

# Test graph_aware (uses fixed embedding computation)
python3 -c "
from pie.retrieval.graph_retriever import retrieve_subgraph
from pie.core.world_model import WorldModel
from pie.core.llm import LLMClient
wm = WorldModel('output/world_model.json')
llm = LLMClient()
result = retrieve_subgraph('What tools have I used?', wm, llm)
for eid in result.entity_ids[:5]:
    print(wm.get_entity(eid).name)
"

# Compare retrievers
python3 -m pie.retrieval.compare_retrievers "How has my tech stack evolved?"

# Full Modal run (all benchmarks, all approaches)
modal run modal_eval.py --all --n 50
```

---

## Evaluation Roadmap

### Phase 1: Fix Foundation (Today)
- [x] Fix graph_retriever embedding computation
- [ ] Re-run extraction with event/date prompts
- [ ] Add embedding caching

### Phase 2: Full Benchmark Suite (This Week)
- [ ] Run all 4 approaches on LongMemEval (500q)
- [ ] Run all 4 approaches on ToT (800q)
- [ ] Run all 4 approaches on LoCoMo (subset)
- [ ] Run all 4 approaches on MSC (500q)

### Phase 3: Analysis
- [ ] Break down by question type
- [ ] Identify where graph_aware helps vs hurts
- [ ] Document task-adaptive findings

### Phase 4: RL Training (Future)
- [ ] Collect training data from benchmark runs
- [ ] Train GRPO policy
- [ ] Evaluate rl_retriever
