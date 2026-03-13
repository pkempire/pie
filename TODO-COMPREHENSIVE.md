# PIE: Comprehensive Status & Plan

**Date:** 2026-02-09

---

## 🔴 Reality Check

### What Actually Exists

| Component | Status |
|-----------|--------|
| World Model | 142 entities, 222 transitions, 112 relationships |
| Ingestion Pipeline | ✅ Works, processes daily batches with sliding window |
| Entity Resolution | ✅ 3-tier (string → embedding → LLM) |
| Context Compiler | ✅ Converts graph → markdown with BOTH dates + relative time |
| State Transitions | ✅ Typed (creation/update/contradiction/resolution/archival) |
| Extraction Prompts | ✅ Fixed to extract EVENT entities with computed dates |

### What's Never Been Done

| Task | Status |
|------|--------|
| PIE evaluated on LongMemEval | ❌ Only naive_rag baseline exists (66.3%) |
| PIE evaluated on LoCoMo | ❌ Only naive_rag baseline exists (58%) |
| PIE evaluated on MSC | ❌ Only naive_rag baseline exists (46%) |
| Full pipeline run on YOUR data | ❓ Partial — 142 entities from ~25/203 batches |
| Visualizer with your data | ✅ Now running at http://localhost:8765/graph_viz.html |

### The "Task-Dependent" Conclusion Was Wrong

You were right. The ToT results (PIE hurt by -25%) happened because:
1. Extraction prompt didn't ask for dates → 0% of entities had dates
2. After fix, pie_temporal went 0% → 40%
3. We never re-ran the full eval after the fix

**This is not a fundamental limitation of the approach.**

---

## 🆕 Mastra Just Shipped Observational Memory

**94.87% on LongMemEval** with gpt-5-mini. Read the article.

### Their Architecture (simpler than PIE)

```
Message History (raw) → Observer compresses → Observations
                                              ↓
                        Reflector summarizes ← Reflection
```

### Key Insights

1. **Text-only, no graph DB** — "Down with knowledge graphs. Roon was right. Text is the universal interface."

2. **Three-date model** for temporal reasoning:
   - Observation date (when captured)
   - Referenced date (the date mentioned: "my flight is January 31")
   - Relative date (computed: "2 days from today")

3. **Emoji prioritization**: 🔴 high, 🟡 medium, 🟢 low

4. **Stable, cacheable context** — no per-turn retrieval, just append-only observations

5. **Results:**
   | Model | LongMemEval |
   |-------|-------------|
   | gpt-5-mini | 94.87% |
   | gemini-3-pro | 93.27% |
   | gpt-4o | 84.23% |

### Comparison to PIE

| Feature | Mastra OM | PIE |
|---------|-----------|-----|
| Storage | Text log | Graph DB |
| Temporal | 3-date model | Transitions + semantic compilation |
| Retrieval | None (full context) | Subgraph retrieval |
| Complexity | Simple | Complex |
| Benchmarked | ✅ 94.87% | ❌ Never run |

---

## 🧠 What PIE Actually Does (The Good Parts)

### 1. State Transition Model

From `context_compiler.py`:

```markdown
## Science Research Academy
**Type:** project
**First seen:** 2024-03-15 (~22 months ago)
**Last seen:** 2026-01-20 (3 weeks ago)

**History (7 changes):**
_Change velocity: 0.3/month_
• **2026-01-20** (3 weeks ago): Pranay now running day-to-day
⚠️ **2025-05-15** (9 months ago): Pivoted from mentoring to curriculum
  _This contradicted prior state_
• **2025-02-01** (12 months ago): Launched at scifair.tech with 30 students
```

This is **good** — BOTH raw dates AND humanized context.

### 2. Rolling Context (Sliding Window)

From `world_model.py`:

```python
def build_context_preamble(self, batch_timestamp: float) -> str:
    """
    Build the activity-based context preamble for extraction.
    This is what makes the sliding window work.
    """
    # Shows ACTIVE PROJECTS, RECENTLY ACTIVE ENTITIES, RECENT STATE CHANGES
```

Gives the LLM awareness of what's currently active before extracting.

### 3. Event Extraction with Computed Dates

From `prompts.py`:

```markdown
## CRITICAL: Extracting Events with Dates

- "today" / "just now" / "I just [verb]" → use batch date
- "yesterday" → batch date minus 1 day
- "last week" → batch date minus 7 days

**Event entity format:**
{
  "name": "MoMA visit",
  "type": "event",
  "state": {
    "date": "YYYY-MM-DD",  // REQUIRED
    "description": "what happened"
  }
}
```

---

## 📋 Action Plan

### Phase 1: Get Current State Working (Today)

1. **Visualizer** ✅
   - Running at http://localhost:8765/graph_viz.html
   - Open in browser to see your world model

2. **Check world model quality**
   ```bash
   cd ~/personal-intelligence-system
   python3 -c "
   import json
   with open('output/world_model.json') as f:
       d = json.load(f)
   for eid, e in list(d['entities'].items())[:10]:
       print(f\"{e['name']} ({e['type']}): {e.get('current_state', {}).get('description', '')[:100]}\")
   "
   ```

3. **Resume pipeline if needed**
   ```bash
   python run.py --input ~/Downloads/conversations.json --skip-batches 25 --limit 50
   ```

### Phase 2: Evaluate PIE on LongMemEval (This Week)

We need to actually test PIE, not just naive_rag.

1. **Set up MemoryBench** (cleaner than our custom harness)
   ```bash
   git clone https://github.com/supermemoryai/memorybench
   cd memorybench
   bun install
   ```

2. **Write PIE provider adapter** for MemoryBench
   - Implement `ingest`, `search`, `answer` methods
   - Use PIE's context compiler for retrieval

3. **Run full eval**
   ```bash
   bun run src/index.ts run -p pie -b longmemeval -j gpt-4o
   ```

### Phase 3: Learn from Mastra (This Week)

Key things to steal:

1. **Three-date model** — We have 2/3:
   - ✅ Timestamp (first_seen/last_seen)
   - ✅ Relative date (humanized)
   - ❌ Missing: "referenced date" (the date mentioned in the text)

2. **Emoji prioritization** — Simple, effective signal

3. **Simpler architecture?** — Maybe we don't need a graph for retrieval. Use graph for extraction/understanding, text for context.

### Phase 4: Decision Point

After running evals, decide:

| If PIE scores... | Then... |
|------------------|---------|
| >85% on LongMemEval | Ship it, iterate |
| 70-85% | Hybrid approach: PIE extraction + Mastra-style context |
| <70% | Consider simpler approach, or PIE is for different use case |

---

## 🔧 Immediate Fixes Needed

### 1. Run Remaining Pipeline Batches

```bash
cd ~/personal-intelligence-system
python run.py --input ~/Downloads/conversations.json --skip-batches 25 --save-every 5
```

### 2. Add "Referenced Date" to Extraction

In `prompts.py`, update event extraction to capture:
```json
{
  "state": {
    "date": "2025-03-15",           // computed from "yesterday"
    "referenced_date": "2025-03-15", // explicit if mentioned
    "description": "..."
  }
}
```

### 3. Create Benchmark Runner for PIE

```python
# benchmarks/pie_provider.py
class PIEProvider:
    def ingest(self, sessions):
        # Run ingestion pipeline on sessions
        
    def search(self, query):
        # Retrieve relevant subgraph
        # Compile to markdown via context_compiler
        return compiled_context
        
    def answer(self, query, context):
        # LLM call with compiled context
```

---

## 📊 Current Benchmark Landscape

| System | LongMemEval | Approach |
|--------|-------------|----------|
| **Mastra OM** | 94.87% | Text observations, no graph |
| **Hindsight** | 91.40% | Multi-stage retrieval + reranking |
| **Emergence** | 86.00% | Internal, not reproducible |
| **Supermemory** | 85.20% | Graph + retrieval |
| **Oracle** | 82.40% | Given only correct conversations |
| **naive_rag** | 66.3% | PIE baseline |
| **PIE** | ??? | Never tested |

---

## 💡 Core Insight

Mastra's success suggests:

> **Dense, well-formatted text with explicit temporal markers beats complex graph retrieval.**

PIE's value might not be in retrieval (Mastra wins that), but in:
1. **Understanding** — extracting what matters from raw conversations
2. **Evolution tracking** — seeing how things change over time
3. **Procedural memory** — patterns across entity lifecycles

These are different capabilities than "answer questions about past conversations."

---

## Next Steps

1. [ ] Open visualizer, look at your world model
2. [ ] Run remaining pipeline batches
3. [ ] Clone MemoryBench, write PIE adapter
4. [ ] Run PIE on LongMemEval
5. [ ] Read Mastra's implementation in detail
6. [ ] Decide: iterate PIE, hybrid approach, or simpler system?
