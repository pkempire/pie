# PIE System Overview — What's Actually Happening

## The Pipeline (Step by Step)

### Phase 1: Data Loading
```
Input: conversations.json (ChatGPT export)
         ↓
Parser: pie/core/parser.py
         - Parses JSON structure
         - Groups conversations by date
         - Creates Turn objects with content + timestamp
         ↓
Output: List of Conversation objects grouped into daily batches
```

### Phase 2: Extraction (per batch)
```
Input: Daily batch of conversations + world model context
         ↓
Prompt: pie/ingestion/prompts.py
         - EXTRACTION_SYSTEM_PROMPT tells LLM what to extract
         - Includes CURRENT WORLD STATE preamble (active entities)
         - Should extract: entities, state_changes, relationships
         - NEW: Should extract EVENT type with dates
         ↓
LLM Call: gpt-5-mini with temp=1.0
         ↓
Output: JSON with entities, state_changes, relationships
```

### Phase 3: Resolution
```
Input: Extracted entities (may have duplicates)
         ↓
Resolver: pie/resolution/resolver.py
         - Tier 1: String match (exact name, normalized)
         - Tier 2: Embedding similarity (cosine > 0.85)
         - Tier 3: LLM verification (for ambiguous cases)
         ↓
Output: Deduplicated entities merged into world model
```

### Phase 4: World Model Storage
```
Storage: pie/core/world_model.py
         - entities: dict[id, Entity]
         - transitions: dict[id, StateTransition]
         - relationships: dict[id, Relationship]
         ↓
Persistence: output/world_model.json
```

---

## Current State (as of Feb 7, 2026)

### World Model Contents
```
Entities:     1,057
Transitions:  2,278
Relationships: 1,187

By type:
  concept: 269
  tool: 237
  project: 232
  decision: 154
  organization: 86
  belief: 42
  person: 31
  period: 6
  event: 0  ← MISSING!
```

### ⚠️ CRITICAL ISSUE: No Temporal Data

The current world model was extracted **BEFORE** the event/date extraction fix.

**What SHOULD be in the world model:**
- Event entities with explicit dates (e.g., "MoMA visit" with date: "2024-03-15")
- State transitions with computed dates
- Entities with temporal metadata

**What's ACTUALLY in the world model:**
- Entities with ingestion timestamps (when WE processed them)
- No explicit dates for events
- No way to answer "What happened on March 15?"

**To fix:** Need to re-run extraction from scratch with new prompts.

---

## Retrieval Approaches

### 1. naive_rag (WORKING)
```
Query → embed(query) → cosine_similarity(query_emb, entity_embs) → top-k → context
```
- Uses: pie/eval/query_interface.py :: retrieve_entities_by_embedding()
- Status: ✅ Working, this is the baseline

### 2. pie_temporal (BROKEN — needs re-extraction)
```
Query → embed(query) → top-k entities → compile_temporal_context() → narrative
```
- Uses: pie/eval/query_interface.py :: compile_entity_context()
- Problem: compile_entity_context() uses transition timestamps (ingestion time), NOT event dates
- The "temporal" context is relative to when we ingested, not when events happened
- Status: ⚠️ Structurally working but semantically broken

### 3. graph_aware (NEW)
```
Query → LLM parses intent → constrained seed selection → BFS graph traversal → subgraph
```
- Uses: pie/retrieval/graph_retriever.py :: retrieve_subgraph()
- Steps:
  1. parse_query_intent() — extracts entity_types, named_entities, temporal_pattern, relationship_types
  2. select_seeds() — finds starting entities (named > embedding > type match)
  3. traverse_from_seeds() — BFS following relationships
  4. filter_by_temporal_intent() — re-ranks by temporal constraints
- Status: ✅ Working (but temporal filtering is limited without event dates)

### 4. hybrid (NEW)
```
Query → (embedding retrieval) + (graph retrieval) → merge → dedupe
```
- Uses: modal_eval.py implementation
- Takes top-k/2 from embedding, top-k/2 from graph, merges
- Status: ✅ Working

### 5. rl_retriever (SCAFFOLD ONLY)
```
Query → policy network → retrieval actions → process rewards → GRPO update
```
- Uses: pie/retrieval/rl_retriever.py
- Status: 📋 Scaffold only, no training yet

---

## Benchmarks

### LongMemEval
- **What it tests:** Multi-session memory, preference inference, temporal reasoning
- **Source:** benchmarks/longmemeval/
- **Data:** ~500 questions with conversation history
- **Categories:** single-session-user, single-session-assistant, knowledge-update, temporal-reasoning, multi-session, preference
- **Our results:** naive_rag 66.3%

### Test of Time (ToT)
- **What it tests:** Temporal arithmetic (before/after, duration, ordering)
- **Source:** benchmarks/tot/, benchmarks/test-of-time/
- **Data:** ~800 questions with synthetic stories
- **Categories:** before_after, event_at_time_t, event_at_what_time, first_last, relation_duration, etc.
- **Our results:** naive_rag 56.2%, pie_temporal 31.2%
- **Key insight:** PIE HURTS on date arithmetic, helps on ordering

### LoCoMo
- **What it tests:** Long conversation memory across 100+ turns
- **Source:** benchmarks/locomo/
- **Data:** ~2000 questions
- **Categories:** single_hop, multi_hop, temporal
- **Our results:** naive_rag 58%

### MSC (Multi-Session Chat)
- **What it tests:** Persona consistency across sessions
- **Source:** benchmarks/msc/
- **Data:** ~500 items
- **Our results:** naive_rag 46%, 76% partial credit

### MemoryAgentBench
- **What it tests:** Agent memory capabilities
- **Source:** benchmarks/memoryagentbench/
- **Status:** Adapter exists, not fully run

### MemoryBench
- **What it tests:** General memory tasks
- **Source:** benchmarks/memorybench/
- **Status:** Adapter exists, not fully run

### PersistBench
- **What it tests:** Long-term persistence
- **Source:** benchmarks/persistbench/
- **Status:** Adapter exists, not fully run

### Temporal Awareness Benchmark
- **What it tests:** Negotiation scenarios with temporal dynamics
- **Source:** benchmarks/temporal_awareness/
- **Status:** Custom benchmark, partially run

---

## Context Compilation: What "Semantic Temporal" Actually Means

### Current Implementation (compile_entity_context)

```python
def compile_entity_context(entity, transitions, relationships, all_entities, now):
    # Header with "temporal" metadata (BUT it's ingestion timestamps)
    lines.append(f"## {name} ({etype})")
    lines.append(f"First appeared {first_ago}, last referenced {last_ago}.")
    
    # Timeline of changes (using ingestion timestamps)
    for t in transitions:
        t_ago = humanize_delta(now - t.timestamp)  # ← This is INGESTION time
        lines.append(f"  • {t_ago}: {t.trigger_summary}")
    
    # Relationships
    for r in relationships:
        lines.append(f"  → {r.type}: {other_name}")
    
    return lines
```

### What It SHOULD Do (with event dates)

```python
def compile_entity_context(entity, transitions, relationships, all_entities, now):
    # For EVENT entities, use the actual event date
    if entity.type == "event":
        event_date = entity.state.get("date")  # "2024-03-15"
        event_ago = humanize_date(event_date, now)  # "about 10 months ago"
        lines.append(f"## {name} (event)")
        lines.append(f"Occurred {event_ago} on {event_date}")
    
    # For other entities, show evolution with actual dates
    for t in transitions:
        if "date" in t.to_state:
            lines.append(f"  • {t.to_state['date']}: {t.trigger_summary}")
```

---

## Example: Subgraph → Natural Language

Let's trace what happens for a query:

### Query: "How has my approach to databases evolved?"

### Step 1: Intent Parsing (graph_aware)
```json
{
  "entity_types": ["tool", "decision", "belief"],
  "named_entities": ["database", "databases"],
  "temporal_pattern": "evolution",
  "relationship_types": ["uses", "replaces"],
  "hop_pattern": "transitive",
  "query_type": "temporal"
}
```

### Step 2: Seed Selection
```
Found seeds:
- Neo4j (tool) — embedding match 0.87
- FalkorDB (tool) — embedding match 0.84
- PostgreSQL (tool) — embedding match 0.82
```

### Step 3: Graph Traversal
```
Neo4j
  → uses: Knowledge Graph / Multimodal RAG project
  → replaces: SQLite
  
FalkorDB
  → replaces: Neo4j
  → uses: PIE project
```

### Step 4: Context Compilation
```
## Neo4j (tool)
First appeared ~11 months ago, last referenced ~6 months ago.
Changed state 3 times (~0.5x/month)
• 11 months ago: Started using for knowledge graph experiments
• 8 months ago: Production deployment for SRA
• 6 months ago: Evaluating alternatives due to performance issues

Relationships:
  → uses: Knowledge Graph / Multimodal RAG
  → replaces: SQLite

## FalkorDB (tool)
First appeared ~5 months ago, last referenced ~2 weeks ago.
• 5 months ago: Discovered as Neo4j alternative
• 3 months ago: Migrated PIE to FalkorDB
• 2 weeks ago: Current production database

Relationships:
  → replaces: Neo4j
  → uses: PIE project
```

### ⚠️ The Problem
All the "X months ago" are relative to INGESTION time, not when the decisions actually happened. If we had proper event dates, we could say:
- "In March 2024, you started using Neo4j"
- "On June 15, 2024, you decided to evaluate alternatives"

---

## What Needs to Happen

### Option 1: Re-run Extraction (Recommended)
```bash
# Delete old world model
rm output/world_model.json

# Re-run with updated prompts that extract events with dates
python3 run.py ingest --input ~/Downloads/conversations.json --batches 203
```
- **Pros:** Gets proper temporal data
- **Cons:** ~$20 in API costs, takes ~4 hours

### Option 2: Post-hoc Date Extraction
- Run a separate pass over transitions to extract dates from trigger_summary
- Parse "started on X" → add date to state
- **Pros:** Faster, cheaper
- **Cons:** Less accurate, won't catch implicit dates

### Option 3: Hybrid Approach for Benchmarks
- For ToT/LongMemEval: use their provided data directly (don't route through our world model)
- For our demo: re-run extraction on a small subset with dates
- **Pros:** Can show results today
- **Cons:** Doesn't fix the underlying issue

---

## Files Reference

### Core Pipeline
- `pie/core/parser.py` — ChatGPT JSON parser
- `pie/core/models.py` — Entity, Transition, Relationship models
- `pie/core/world_model.py` — In-memory graph with JSON persistence
- `pie/core/llm.py` — OpenAI client wrapper
- `pie/ingestion/pipeline.py` — Daily batch orchestrator
- `pie/ingestion/prompts.py` — Extraction prompts (WITH event/date instructions)
- `pie/resolution/resolver.py` — 3-tier entity resolution

### Retrieval
- `pie/eval/query_interface.py` — Query answering, context compilation
- `pie/retrieval/graph_retriever.py` — Intent parsing + graph traversal
- `pie/retrieval/rl_retriever.py` — GRPO scaffold
- `pie/retrieval/compare_retrievers.py` — Side-by-side comparison

### Benchmarks
- `benchmarks/longmemeval/` — LongMemEval adapter + runner
- `benchmarks/tot/` — Test of Time runner
- `benchmarks/test-of-time/` — Alternative ToT implementation
- `benchmarks/locomo/` — LoCoMo adapter + runner
- `benchmarks/msc/` — MSC adapter + runner
- `benchmarks/memoryagentbench/` — MemoryAgentBench adapter
- `benchmarks/memorybench/` — MemoryBench adapter
- `benchmarks/persistbench/` — PersistBench adapter
- `benchmarks/temporal_awareness/` — Custom temporal benchmark

### Evaluation
- `modal_eval.py` — Unified Modal-compatible eval harness
- `EVAL-COMMANDS.md` — Quick reference
