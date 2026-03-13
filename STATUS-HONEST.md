# PIE: Honest Status — What's Real vs. Ideas

**Date:** 2026-02-09

---

## ✅ What's Actually Implemented & Working

| Component | File | Status |
|-----------|------|--------|
| **Data Models** | `pie/core/models.py` | ✅ Entity, StateTransition, Relationship, Procedure (dataclasses) |
| **World Model Store** | `pie/core/world_model.py` | ✅ JSON-backed graph with CRUD, indexes, persistence |
| **Ingestion Pipeline** | `pie/ingestion/pipeline.py` | ✅ Daily batch processing, rolling context |
| **Extraction Prompts** | `pie/ingestion/prompts.py` | ✅ Entity/state/relationship extraction with date computation |
| **Entity Resolution** | `pie/resolution/resolver.py` | ✅ 3-tier: string → embedding (cosine) → LLM verify |
| **Context Compiler** | `pie/retrieval/context_compiler.py` | ✅ Graph → markdown with dates + relative time |
| **Query Interface** | `pie/eval/query_interface.py` | ✅ Interactive querying over world model |
| **Extraction Quality Eval** | `pie/eval/extraction_quality.py` | ✅ Noise detection, quality metrics |
| **Visualizer** | `graph_viz.html` | ✅ vis-network force graph |

### Currently Working

```bash
# Query the world model (just tested, works)
python3 -m pie.eval.query_interface --world-model output/world_model.json --query "What are my active projects?"

# Run more ingestion
python3 run.py --input ~/Downloads/conversations.json --skip-batches 25

# Check extraction quality
python3 -m pie.eval.extraction_quality --world-model output/world_model.json
```

---

## ❌ What's NOT Implemented (Just Ideas in Blog/Paper)

| Feature | Mentioned In | Reality |
|---------|--------------|---------|
| **Dreaming Engine** | BLOG-DRAFT.md | ❌ Not implemented. No code. |
| **Consolidation/Forgetting** | BLOG-DRAFT.md | ❌ Not implemented. `importance` field exists but never computed. |
| **Procedural Memory Extraction** | BLOG-DRAFT.md | ❌ `Procedure` dataclass exists, no extraction logic. |
| **Graph-Structural Importance (PageRank)** | BLOG-DRAFT.md | ❌ No PageRank. `importance` always 0.0. |
| **Tiered Forgetting** | ARCHITECTURE-FINAL.md | ❌ Not implemented. |
| **Period Detection** | prompts.py | ⚠️ Partial. Extracted as entities but not used systematically. |
| **MCP Server** | BLOG-DRAFT.md | ❌ Not implemented. |
| **FalkorDB/Neo4j Backend** | README.md | ❌ JSON only. |

---

## 🔧 Entity Resolution: Current vs. Should-Be

### Current (what the code does)

```python
# Tier 1: String match (fuzzy, SequenceMatcher)
# Tier 2: Embedding similarity (cosine, text-embedding-3-large)
# Tier 3: LLM verification (for ambiguous cases)
```

### Missing: BM25 for Candidate Retrieval

You're right — BM25 would be better for initial candidate generation:
- **Cosine similarity** works on dense vectors, good for semantic similarity
- **BM25** is term-based, handles exact name matches and aliases better

**Should add:**
```python
# Tier 0.5: BM25 over entity names + aliases (fast, sparse)
# Then filter candidates with embedding similarity
```

Libraries: `rank_bm25`, or use SQLite FTS5.

---

## 📊 What MemoryBench (Supermemory) Does

**Purpose:** Unified benchmark framework for memory systems.

**Pipeline:**
```
Ingest → Index → Search → Answer → Evaluate
```

**Supports:**
- Benchmarks: LongMemEval, LoCoMo, ConvoMem
- Providers: Supermemory, Mem0, Zep (plug in your own)
- Judges: GPT-4o, Claude, Gemini

**Why useful for PIE:**
Instead of our custom `benchmarks/` code, write a PIE provider adapter for MemoryBench:

```typescript
// src/providers/pie/index.ts
export class PIEProvider implements MemoryProvider {
  async ingest(sessions: Session[]) { /* run ingestion */ }
  async search(query: string) { /* retrieve + compile context */ }
  async answer(query: string, context: string) { /* LLM call */ }
}
```

Then: `bun run src/index.ts run -p pie -b longmemeval`

---

## 🌌 Kosmos vs. PIE — Different Problems

| Aspect | Kosmos | PIE |
|--------|--------|-----|
| **Domain** | Scientific discovery | Personal memory |
| **Input** | Dataset + objective | Conversation history |
| **Output** | Scientific report | World model graph |
| **Duration** | 12 hours, 200 rollouts | Continuous ingestion |
| **Agents** | Data analysis + literature search | Single extraction |

**Kosmos's "World Model":**
- Shared state between data analysis agent and literature search agent
- Enables 42,000 lines of code execution coherently
- Structured knowledge accumulation over long horizon

**PIE's World Model:**
- State transition graph of entities over time
- Enables temporal reasoning ("how has X evolved?")
- Different purpose: memory, not discovery

---

## 🧠 The Three Ways to Infinite Context

This is a fundamental framing. Let me explain each:

### 1. Perfect Memory Querying & Recall (RAG)

**Idea:** Store everything, retrieve only what's relevant for this query.

```
[All Past Data] → Query → [Top-K Relevant Chunks] → LLM
```

**Why imperfect:**
- Retrieval is lossy — wrong chunks = wrong answer
- No cross-chunk reasoning during retrieval
- Temporal/relational queries fail (cosine similarity doesn't understand "before/after")

**PIE's approach:** Use a graph + temporal context compilation instead of raw chunks.

### 2. Perfect Compression into Fixed-Size Recurrent State (RNN/SSM)

**Idea:** Compress all past context into a fixed-size hidden state.

```
                    ┌─────────────────┐
... → h[t-2] → h[t-1] → h[t] → output
                    └─────────────────┘
                     Fixed size state
```

**Examples:** LSTM, GRU, Mamba, RWKV, S4

**Why imperfect:**
- Finite state can't perfectly represent infinite past
- Information gets "forgotten" as state overwrites
- Can't do precise lookup ("what did I say on March 15?")

**Mastra's observation memory is like this:**
- Compress raw messages → dense observations
- Fixed observation budget (~30-40k tokens)
- Reflector further compresses when limit hit

### 3. Perfect Delegation of Sub-Problems (RLM / Agent Systems)

**Idea:** Break problem into independent sub-tasks with smaller context each.

```
Main Agent → [Sub-Agent 1] → Result 1
          → [Sub-Agent 2] → Result 2
          → [Sub-Agent 3] → Result 3
          → Synthesize
```

**Examples:** Kosmos (data + literature agents), agent swarms, tree of thought

**Why imperfect:**
- Coordination overhead
- Information loss at agent boundaries
- Hard to know how to decompose

**Kosmos does this:** Data analysis agent + literature agent share world model, run 200 rollouts.

### The Hybrid Reality

No system is "perfect" at any of these. Best systems combine:
- **RAG** for retrieving relevant context
- **Compression** for maintaining coherent history (observations, summaries)
- **Delegation** for complex multi-step tasks

PIE is attempting:
- Graph structure (better than raw chunks for relational queries)
- Temporal compilation (compression of state history into readable narrative)
- Single agent (no delegation yet)

Mastra's success suggests compression + good formatting beats complex retrieval.

---

## 📝 Prompts You Can Run Right Now

```bash
cd ~/personal-intelligence-system

# 1. Interactive query mode
python3 -m pie.eval.query_interface --world-model output/world_model.json

# 2. Specific queries
python3 -m pie.eval.query_interface -q "How has SRA evolved over time?"
python3 -m pie.eval.query_interface -q "What tools am I using for AI projects?"
python3 -m pie.eval.query_interface -q "What decisions have I made about tech stack?"

# 3. Check extraction quality
python3 -m pie.eval.extraction_quality --world-model output/world_model.json

# 4. View the graph
open http://localhost:8888/graph_viz.html

# 5. Resume pipeline (if conversations.json exists)
python3 run.py --input ~/Downloads/conversations.json --skip-batches 25 --limit 50
```

---

## 🎯 Immediate TODOs (Real Work, Not Ideas)

### This Session

1. [ ] Fix visualizer loading (check if world_model.json path is correct in HTML)
2. [ ] Run a few queries, see what works/breaks
3. [ ] Decide: resume pipeline or test with current 142 entities?

### This Week

1. [ ] Add BM25 for candidate retrieval in `resolver.py`
2. [ ] Implement basic importance scoring (degree + recency, not full PageRank)
3. [ ] Write PIE provider for MemoryBench
4. [ ] Run PIE on LongMemEval (first real benchmark)

### Future (If Benchmarks Look Good)

1. [ ] Implement dreaming (consolidation, procedure extraction)
2. [ ] Add period detection and linking
3. [ ] MCP server for external integration
4. [ ] Consider Mastra-style observation compression as alternative

---

## Key Insight

**Mastra's 94.87% with text-only approach suggests:**

1. Maybe we're overcomplicating it with graphs
2. Dense, well-formatted text + temporal markers might be enough
3. The three-date model (observation/referenced/relative) is clever

**But PIE offers something different:**
- Evolution tracking (state transitions over time)
- Relationship reasoning (who works on what)
- Contradiction detection (beliefs that changed)

These aren't captured by Mastra's observation log. Different tools for different questions.
