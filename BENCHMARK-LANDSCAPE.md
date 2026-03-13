# Benchmark & Methods Landscape

**The Real Picture:** There's a massive ecosystem we should be testing against.

---

## Available Benchmarks

### In MemoryBench (Ready to Run)

| Benchmark | Questions | Categories | What It Tests |
|-----------|-----------|------------|---------------|
| **LongMemEval** | 500 | 6 | Memory across 115k-1.5M token histories |
| **LoCoMo** | 10+ items | 5 | Multi-session, temporal, multi-hop |
| **ConvoMem** | ? | 6 | Conversational memory, changing info |

### LongMemEval Categories
```
single-session-user:      User-stated facts
single-session-assistant: Assistant-stated facts  
single-session-preference: User preferences
multi-session:            Cross-session reasoning
temporal-reasoning:       When did X happen?
knowledge-update:         What's the CURRENT state? (info changed)
```

### LoCoMo Categories
```
single-hop:    Simple fact recall
multi-hop:     Connect facts across sessions
temporal:      Time-based reasoning
world-knowledge: Commonsense
adversarial:   Unanswerable (should say IDK)
```

### Other Benchmarks (Not in MemoryBench Yet)

| Benchmark | Source | What It Tests |
|-----------|--------|---------------|
| **Test of Time** | ICLR 2025 | Pure temporal reasoning (synthetic) |
| **MSC** | Multi-Session Chat | Persona consistency |
| **DMR** | Dialog Memory Recall | Fact recall |
| **ENGRAM-R** | ACL 2025 | Neuro-symbolic temporal reasoning |

---

## Competing Systems (SOTA Leaderboard)

### LongMemEval Results (Most Comprehensive)

| System | Model | Score | Architecture |
|--------|-------|-------|--------------|
| **Mastra OM** | gpt-5-mini | **94.87%** | Text observations, no graph |
| **Mastra OM** | gemini-3-pro | 93.27% | " |
| **Hindsight** | gemini-3-pro | 91.40% | Multi-stage retrieval + reranking |
| **Mastra OM** | gemini-3-flash | 89.20% | " |
| **Emergence** | gpt-4o | 86.00% | Internal (not reproducible) |
| **Supermemory** | gemini-3-pro | 85.20% | Hybrid search |
| **Mastra OM** | gpt-4o | 84.23% | " |
| **Oracle** | gpt-4o | 82.40% | Given correct conversations only |
| **Supermemory** | gpt-4o | 81.60% | " |
| **Zep/Graphiti** | gpt-4o | 71.20% | Temporal KG |
| **Full context** | gpt-4o | 60.20% | No memory system |
| **PIE naive_rag** | gpt-4o | ~66% | Embedding retrieval only |
| **PIE (actual)** | ? | **NOT TESTED** | — |

### LoCoMo Results

| System | Model | Overall | Temporal |
|--------|-------|---------|----------|
| MemMachine | ? | 84.9% | 72.6% |
| Memobase | ? | 75.8% | 85.1% |
| Zep/Graphiti | ? | 75.1% | 79.8% |

---

## Methods We Should Compare

### 1. Retrieval Methods

| Method | How | Libraries |
|--------|-----|-----------|
| **Dense (cosine)** | Embed query + docs, cosine similarity | OpenAI, Cohere |
| **Sparse (BM25)** | Term frequency + IDF | rank_bm25, Elasticsearch |
| **Hybrid** | Dense + Sparse fusion | Supermemory, Pinecone |
| **Reranking** | Two-stage: retrieve then rerank | Cohere Rerank, ColBERT |
| **Graph traversal** | Follow relationships | Neo4j, FalkorDB |

### 2. Memory Architectures

| Architecture | Key Idea | Example |
|--------------|----------|---------|
| **RAG** | Retrieve chunks, inject into prompt | Most systems |
| **Observations** | Compress history into dense notes | Mastra OM |
| **Knowledge Graph** | Entities + relationships | Zep/Graphiti |
| **Temporal KG** | KG + typed state transitions | PIE |
| **Hierarchical** | Working + semantic + episodic | MemGPT |
| **Associative** | Spreading activation retrieval | SYNAPSE |

### 3. Temporal Handling Methods

| Method | What It Does |
|--------|--------------|
| **Raw timestamps** | Pass ISO dates to LLM |
| **Relative time** | "3 weeks ago" |
| **Three-date model** | observation + referenced + relative (Mastra) |
| **State transitions** | Track changes as typed events (PIE) |
| **Period anchoring** | "during Q3", "freshman year" |

---

## Recent Papers (Jan-Feb 2026)

From Awesome-Memory-for-Agents, just in the last 2 months:

| Paper | Key Contribution |
|-------|------------------|
| **TiMem** | Temporal-hierarchical consolidation |
| **SwiftMem** | Query-aware indexing for speed |
| **SYNAPSE** | Episodic-semantic spreading activation |
| **HiMem** | Hierarchical for long-horizon agents |
| **MAGMA** | Multi-graph architecture |
| **Memory-T1** | RL for temporal reasoning |
| **A-MEM** | Agentic memory with RL |
| **Zep** | Temporal KG (closest to PIE) |

---

## What We're NOT Doing (And Should Be)

### 1. Not Running Any Real Benchmarks

We have evaluation code but never ran PIE end-to-end on any benchmark.

### 2. Not Comparing Methods

We use cosine similarity only. Should test:
- BM25 for candidate generation
- Hybrid retrieval
- Reranking

### 3. Not Comparing to Competitors

We haven't run:
- Mem0 on our data
- Zep on our data
- Mastra OM on our data

### 4. Not Testing Temporal Methods

We have context_compiler but never ablated:
- Raw timestamps vs. relative time vs. three-date model
- State transitions vs. flat facts

---

## Concrete Eval Plan

### Phase 1: Setup (Today)

```bash
# 1. Clone MemoryBench
git clone https://github.com/supermemoryai/memorybench
cd memorybench
bun install

# 2. Download benchmark data
bun run src/index.ts download longmemeval
bun run src/index.ts download locomo

# 3. Write PIE provider
# src/providers/pie/index.ts
```

### Phase 2: Baseline Runs (This Week)

```bash
# Run naive_rag baseline
bun run src/index.ts run -p pie-naive-rag -b longmemeval -j gpt-4o

# Run PIE with temporal context
bun run src/index.ts run -p pie-temporal -b longmemeval -j gpt-4o

# Compare to competitors
bun run src/index.ts run -p supermemory -b longmemeval -j gpt-4o
bun run src/index.ts run -p mem0 -b longmemeval -j gpt-4o
```

### Phase 3: Ablations (Next Week)

| Ablation | What to Test |
|----------|--------------|
| **Retrieval** | cosine vs BM25 vs hybrid |
| **Temporal format** | raw dates vs relative vs three-date |
| **Graph vs flat** | Use relationships or not |
| **Context size** | Top-5 vs top-10 vs top-20 entities |

### Phase 4: LoCoMo + Test of Time

```bash
# LoCoMo (multi-hop, temporal)
bun run src/index.ts run -p pie -b locomo -j gpt-4o

# Test of Time (pure temporal, synthetic)
python benchmarks/test-of-time/run_eval.py --method pie_temporal
```

---

## PIE Provider for MemoryBench

```typescript
// src/providers/pie/index.ts
import type { Provider, IngestResult, SearchOptions } from "../../types/provider"
import type { UnifiedSession } from "../../types/unified"

export class PIEProvider implements Provider {
  name = "pie"
  private worldModel: any = null
  
  async initialize(config: ProviderConfig): Promise<void> {
    // Load PIE world model or initialize empty
  }

  async ingest(sessions: UnifiedSession[], options: IngestOptions): Promise<IngestResult> {
    // Convert sessions to PIE conversations
    // Run ingestion pipeline
    // Return document IDs
  }

  async awaitIndexing(result: IngestResult): Promise<void> {
    // PIE indexes synchronously, no-op
  }

  async search(query: string, options: SearchOptions): Promise<unknown[]> {
    // 1. Embed query
    // 2. Retrieve relevant entities
    // 3. Compile temporal context
    // 4. Return compiled markdown as "search result"
  }

  async clear(containerTag: string): Promise<void> {
    // Reset world model
  }
}
```

---

## The Real Question

**Why haven't we been doing this?**

We've been:
- Writing architecture docs
- Building extraction prompts
- Implementing resolution tiers
- Writing blog posts

Without ever validating that any of it helps.

**The fix:**
1. Run benchmarks first
2. Measure what helps
3. Then build more

Start with LongMemEval. If PIE beats naive_rag by 10+%, the graph/temporal stuff is working. If not, pivot.
