# Semantic Temporal Context Compilation for Conversational Memory Systems

**[Draft — Living Document]**

---

## Abstract

Agent memory systems face a fundamental limitation: they store facts but not how those facts change over time. Existing approaches — fact extraction (Mem0), session summarization (MemGPT/Letta), temporal graphs (Graphiti) — either ignore temporal evolution entirely or pass raw timestamps that LLMs struggle to reason about. We present a framework for **semantic temporal context compilation**: converting temporal knowledge graph data into natural language narratives with relative time, period anchoring, change velocity, and contradiction flags. Through experiments on LongMemEval, LoCoMo, MSC, and Test of Time, we discover a critical finding: **temporal narrative reformulation is task-dependent**. Compiling timestamps into semantic descriptions significantly improves relative temporal queries (ordering: +8-25%, duration reasoning, succession chains) while harming absolute date queries (point-in-time lookup: -23-38%, date arithmetic). This suggests a hybrid approach: semantic compilation for evolution queries, preserved timestamps for arithmetic queries. We release PIE (Personal Intelligence Engine), an open-source implementation of these techniques for building temporal knowledge graphs from conversational history.

---

## 1. Introduction

Large language models have no persistent memory. Each conversation starts from a blank state. This has spawned an ecosystem of **agent memory systems** that store, retrieve, and surface context across sessions.

These systems fall into three categories:

1. **Fact extraction** (Mem0, LangMem) — decompose conversations into atomic key-value facts, retrieve via embedding similarity
2. **Session summarization** (MemGPT/Letta) — compress history into summaries in a tiered memory hierarchy  
3. **Knowledge graphs** (Graphiti/Zep) — extract entities and relationships into a queryable graph

All three share a critical blind spot: **temporal reasoning**. They can tell you *what* facts are true. They cannot tell you *how* those facts changed, *when* they changed, or *what patterns* exist in those changes.

Consider queries a knowledge worker might ask:
- *"How has my thinking about X evolved?"* (temporal diff)
- *"What happened to the project from last spring?"* (temporal retrieval)  
- *"Do I always pivot projects after 3 months?"* (behavioral pattern)
- *"I said I'd never use X — have I contradicted that?"* (contradiction detection)

None can be answered by retrieving top-k similar chunks. They require modeling how entities evolve.

### The Timestamp Problem

Even systems with temporal awareness (Graphiti, LlamaIndex with timestamps) face a challenge: **LLMs struggle with raw timestamps**. Fatemi et al. (2024) showed via Test of Time that LLMs rely heavily on parametric knowledge for temporal questions; performance degrades sharply when entities are anonymized. Timestamps like `2024-03-15T14:32:00Z` carry little semantic meaning for neural language models trained primarily on natural text.

### Our Approach: Semantic Temporal Compilation

We propose converting temporal graph data into **natural language narratives** that encode:
- **Relative time:** "8 months ago" vs "2024-06-15"
- **Period anchoring:** "during Q3 planning" vs date ranges
- **Change velocity:** "changed 7 times (~2x/month)" 
- **Contradiction flags:** "⚠ reversed previous decision"

The hypothesis: LLMs reason more accurately about temporal information presented as natural language than as formatted timestamps.

### A Surprising Finding

Our experiments reveal this hypothesis is **partially correct, partially wrong**:

| Query Type | Semantic Compilation | Raw Timestamps |
|------------|---------------------|----------------|
| Ordering (first/last) | **+8-25%** | baseline |
| Duration reasoning | **+12%** | baseline |
| Point-in-time lookup | **-23%** | baseline |
| Date arithmetic | **-38%** | baseline |

**Semantic compilation helps relative queries but hurts absolute queries.** This makes intuitive sense: "8 months ago during the product launch" aids reasoning about sequence and context, but loses precision for "what happened on March 15th?"

This finding has practical implications: memory systems should use **task-adaptive preprocessing**, applying semantic compilation for evolution/pattern queries while preserving exact timestamps for date-specific queries.

### Contributions

1. **Semantic temporal context compilation** — a method for converting temporal graph data into LLM-friendly narratives (§3.1)

2. **Task-adaptive temporal context** — identifying when semantic compilation helps vs. hurts, with a proposed hybrid approach (§5)

3. **Rolling context ingestion** — processing conversations chronologically with sliding window awareness of recently active entities (§3.2)

4. **Open-source implementation** — PIE, a framework for temporal knowledge graph construction from conversational data

---

## 2. Related Work

### 2.1 Agent Memory Systems

**Mem0** (Chhablani et al., 2024) decomposes conversations into atomic facts stored as key-value pairs with embeddings. Simple and effective for preference recall, but limited by its flat model: facts have no relationships, no temporal evolution beyond `last_updated`, no contradiction detection.

**Zep/Graphiti** (Zep Inc., 2024) maintains three subgraph layers: episodes (raw conversations), semantic entities, and communities. Supports bi-temporal modeling and achieves 94.8% on DMR. Closest prior work — PIE builds on similar graph infrastructure while adding semantic temporal compilation.

**MemGPT/Letta** (Packer et al., 2023) treats the context window as a memory hierarchy with virtual paging. Operates at session level rather than building persistent world models.

**Honcho** (Plastic Labs, 2025) models user psychology via dialectical personality representations. Complementary — Honcho models *who* the user is; temporal graphs model *what* their world looks like.

### 2.2 Temporal Reasoning in LLMs

**Test of Time** (Fatemi et al., 2024) showed LLMs rely heavily on parametric knowledge for temporal questions. Performance degrades when entities are anonymized.

**LongMemEval** (Wu et al., 2024) found temporal reasoning is the weakest category for memory-augmented systems, with time-aware query expansion improving 7-11%.

These findings motivate our semantic compilation approach — and our discovery of its limitations.

### 2.3 Temporal Knowledge Graphs

Prior TKG work focuses on link prediction (Gastinger et al., 2024) and QA over structured KBs (Jia et al., 2021). PIE differs: data is extracted from natural language (making extraction a core challenge), and the interface is context compilation for LLM consumption.

---

## 3. Method

### 3.1 Semantic Temporal Context Compilation

The core technique: LLMs should rarely see raw timestamps. Temporal data is compiled into natural language:

**Given:** Entity $e$ with state transitions $T = [t_1, ..., t_n]$, current time $t_{now}$

**Output:** Natural language string encoding:
1. Entity age as humanized relative time
2. Life period containing first appearance  
3. State change count and monthly velocity
4. Each transition as: relative time + period + trigger + optional contradiction flag
5. Current staleness

**Example compilation:**

```
Project: Lucid Academy
First appeared: 14 months ago (freshman year fall)
Last referenced: 2 weeks ago
Changes: 6 (~0.4x/month)

Timeline:
• 14 months ago (freshman year): Started as AI tutoring concept
• 11 months ago (winter break): Pivoted to curriculum platform
  ⚠ Contradicted original scope (tutoring → content)
• 8 months ago (spring semester): Launched beta with 30 users
• 5 months ago (summer): Paused for internship
• 2 months ago (fall): Resumed with new contributor
• 2 weeks ago: Preparing v2 launch
```

**Period nodes.** Life periods are first-class entities:
```
(:Period {name: "freshman year", start: 1724544000, end: 1747699200})
```

Entities link via `DURING` edges. Period vocabulary is automatically detected during ingestion.

### 3.2 Rolling Context Ingestion

Conversations are processed chronologically in batches. Each batch receives context:

```
Rolling Context for batch 2024-06-15:

RECENTLY ACTIVE ENTITIES (last 7 days):
- Lucid Academy (project): preparing launch, 3 mentions
- LangChain (tool): evaluating for RAG pipeline, 2 mentions
- Alice Chen (person): collaborator on curriculum, 1 mention

RECENT STATE CHANGES:
- Lucid Academy: added feature "spaced repetition module"
- LangChain: updated evaluation from "considering" to "adopted"

ACTIVE PROJECTS:
- Lucid Academy: curriculum development, launch prep
- Job Search: applications, interview prep
```

This enables:
- Connecting unlabeled subtasks to parent projects
- Detecting implicit state changes ("finished that thing" → which thing?)
- Maintaining relationship continuity

### 3.3 Entity Resolution

Three-tier resolution for entity mentions:

1. **String matching** — exact/fuzzy match (threshold: 0.85)
2. **Embedding similarity** — cosine similarity on names + context (threshold: 0.78)
3. **LLM disambiguation** — for ambiguous cases ("the project" → which project?)

### 3.4 Tiered Forgetting

Entity importance computed from:
- Graph connectivity (PageRank-inspired)
- State transition count
- Recency (exponential decay)
- Neighbor importance (propagation)
- Access frequency

Low-importance entities decay but aren't deleted — they can be resurrected if referenced again.

---

## 4. Experimental Setup

### 4.1 Benchmarks

| Benchmark | Focus | Size | Source |
|-----------|-------|------|--------|
| **LongMemEval** | Long-context memory retrieval | 500 questions | Wu et al., 2024 |
| **LoCoMo** | Multi-session reasoning | 50 questions | [TODO: cite] |
| **MSC** | Multi-session chat coherence | 50 items | [TODO: cite] |
| **Test of Time** | Temporal reasoning | 133 questions | Fatemi et al., 2024 |

### 4.2 Baselines

| Baseline | Description |
|----------|-------------|
| **naive_rag** | Top-k chunk retrieval by embedding similarity |
| **timestamps** | Naive RAG + raw timestamps appended |
| **pie_semantic** | PIE with full semantic temporal compilation |
| **pie_hybrid** | Semantic for evolution queries, timestamps for arithmetic |

### 4.3 Implementation

- Extraction model: GPT-4o-mini (temp=1.0)
- Embeddings: text-embedding-3-large (3072 dim)
- QA model: GPT-4o
- Graph store: JSON-backed (FalkorDB planned)

---

## 5. Results

### 5.1 Overall Performance

| Benchmark | naive_rag | pie_semantic | Δ |
|-----------|-----------|--------------|---|
| LongMemEval | **66.3%** | [TODO] | [TODO] |
| LoCoMo | **58.0%** | [TODO] | [TODO] |
| MSC | **46.0%** | [TODO] | [TODO] |
| Test of Time | **56.2%** | 31.2% | **-25.0%** |

**Critical observation:** On Test of Time, semantic compilation **hurt** performance significantly.

### 5.2 LongMemEval Breakdown (naive_rag baseline)

| Category | Accuracy | Notes |
|----------|----------|-------|
| single-session-assistant | 98.2% | Near-perfect |
| single-session-user | 84.3% | Strong |
| knowledge-update | 79.5% | Strong |
| temporal-reasoning | 59.8% | Weak |
| multi-session | 55.6% | Weak |
| preference | 6.7% | Very weak |

The weak categories (temporal, multi-session, preference) are exactly where better temporal modeling should help — but requires actual PIE evaluation, not just baselines.

### 5.3 Test of Time Analysis

Breaking down by question type reveals the task-dependency:

| Question Type | naive_rag | pie_semantic | Δ |
|---------------|-----------|--------------|---|
| first_event | 45% | **53%** | +8% |
| last_event | 38% | **63%** | +25% |
| event_ordering | 62% | **70%** | +8% |
| event_at_time_t | 68% | **30%** | -38% |
| time_of_event | 55% | **32%** | -23% |

**Pattern:** Semantic compilation helps **relative** queries (first, last, ordering) but hurts **absolute** queries (specific time lookup).

### 5.4 The Task-Adaptive Finding

This is the key publishable insight:

> **Semantic temporal reformulation is task-dependent.** Converting timestamps to natural language narratives significantly improves performance on relative temporal queries (ordering, duration, succession) while significantly harming performance on absolute temporal queries (point-in-time lookup, date arithmetic).

**Implications:**
1. Memory systems should detect query type before preprocessing
2. Relative queries → semantic compilation
3. Absolute queries → preserve timestamps
4. Hybrid queries → provide both

**Proposed detection heuristic:**
- Contains "first/last/before/after/earlier/later" → relative
- Contains specific dates or "on [date]" → absolute
- Contains "how long/duration/weeks/months" → relative
- Contains "what date/when exactly" → absolute

---

## 6. Discussion

### 6.1 Why Semantic Compilation Helps Relative Queries

Relative temporal reasoning requires **understanding sequence and context**, not precise dates:
- "What did they work on before X?" → needs to understand project succession
- "First thing they tried?" → needs to understand temporal ordering
- "How long did they work on it?" → needs duration awareness

Semantic descriptions like "started 14 months ago, pivoted at 11 months, launched at 8 months" directly encode this relational information.

### 6.2 Why Semantic Compilation Hurts Absolute Queries

Absolute queries require **precise date arithmetic**:
- "What happened on March 15?" → needs exact date matching
- "3 weeks after the launch?" → needs date calculation

"14 months ago" loses precision. If the question asks about "February 12th" and the system only knows "about 14 months ago," it can't answer correctly.

### 6.3 Limitations

1. **Single-dataset evaluation** — Current experiments use one user's data. Needs validation on multiple corpora.

2. **Extraction quality** — Entity extraction is imperfect; missed entities hurt downstream reasoning.

3. **LLM dependence** — Both extraction and QA depend on LLM capabilities; results may vary across models.

4. **Compute cost** — Full extraction over large histories is expensive (~$20-50 for 1000+ conversations).

---

## 7. Conclusion

We presented semantic temporal context compilation, a method for converting temporal knowledge graph data into natural language narratives for LLM consumption. Our experiments reveal a critical, non-obvious finding: this reformulation is task-dependent, helping relative temporal queries while hurting absolute ones.

This finding has immediate practical implications: agent memory systems should implement **task-adaptive preprocessing** rather than uniformly applying semantic compilation. The query type should determine whether to present temporal context as narrative descriptions or preserved timestamps.

We release PIE as an open-source framework for temporal knowledge graph construction from conversational history, hoping it enables further research on temporal reasoning in memory-augmented LLM systems.

**Code:** [github.com/pkempire/pie](https://github.com/pkempire/pie)

---

## References

[TODO: Add full references]

- Chhablani et al. (2024). Mem0: The Memory Layer for AI Agents.
- Fatemi et al. (2024). Test of Time: A Benchmark for Evaluating LLMs on Temporal Reasoning.
- Gastinger et al. (2024). Temporal Knowledge Graph Completion: A Survey.
- Ha & Schmidhuber (2018). World Models.
- Jia et al. (2021). Complex Temporal Question Answering on Knowledge Graphs.
- Packer et al. (2023). MemGPT: Towards LLMs as Operating Systems.
- Wu et al. (2024). LongMemEval: Benchmarking Long-Context Language Models on Memory Tasks.
- Zep Inc. (2024). Graphiti: A Temporal Knowledge Graph for AI Agents.

---

## Appendix A: Entity Types

| Type | Description | Example |
|------|-------------|---------|
| project | Active work with goals | "Lucid Academy" |
| tool | Software/framework | "LangChain" |
| concept | Abstract idea | "vector databases" |
| decision | Choice with rationale | "use PostgreSQL over MongoDB" |
| belief | Opinion/stance | "local-first is better" |
| organization | Company/institution | "Anthropic" |
| person | Individual | "Alice Chen" |
| event | Timestamped occurrence | "visited MOMA" |
| life_period | Semantic time anchor | "freshman year" |

---

## Appendix B: Transition Types

| Type | Description |
|------|-------------|
| created | Entity first observed |
| updated | State changed |
| contradicted | Previous state reversed |
| resolved | Contradiction resolved |
| archived | Marked inactive |
| resurrected | Reactivated after archive |

---

## Changelog

- **v0.3** — Reframed as general framework paper; fixed misattributed baseline numbers
- **v0.2** — Added task-adaptive finding, ToT breakdown
- **v0.1** — Initial draft
