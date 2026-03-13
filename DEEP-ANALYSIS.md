# Deep Analysis: Is PIE Worth Building?

**The Hard Question:** What does a graph give us that flat text doesn't?

---

## The Brutal Honest Assessment

### What PIE's "Graph" Actually Is

```python
# world_model.json structure
{
  "entities": { entity_id: {...}, ... },      # dict
  "transitions": { transition_id: {...}, ... }, # dict
  "relationships": { rel_id: {...}, ... }       # dict
}
```

This is **not** a graph database. It's three hashmaps with foreign keys.

**What we get from this "graph":**
1. O(1) entity lookup by ID
2. Index for name → entity_id
3. Manual traversal of relationships via iteration

**What we DON'T get:**
1. Graph traversal algorithms (no Cypher, no Gremlin)
2. Path queries ("how is X connected to Y?")
3. Neighborhood queries without full scan
4. Graph analytics (PageRank, community detection)
5. Multi-hop reasoning

### What Graph DBs Actually Provide

```cypher
-- Neo4j: Find all projects I worked on that used tools I later abandoned
MATCH (me:Person)-[:WORKS_ON]->(p:Project)-[:USES]->(t:Tool)
WHERE (t)-[:HAS_TRANSITION {type: 'archival'}]->()
RETURN p, t
```

```cypher
-- Find belief contradictions
MATCH (b:Belief)-[t1:TRANSITION {type: 'update'}]->(s1)
       -[t2:TRANSITION {type: 'contradiction'}]->(s2)
WHERE t2.timestamp > t1.timestamp
RETURN b, s1, s2
```

**PIE can't do this efficiently.** We'd have to:
1. Load all entities into memory
2. Iterate through transitions
3. Filter and join manually

### So Why Did We Build This?

The honest answer: **proof of concept before committing to a graph DB.**

JSON is:
- Zero setup cost
- Easy to debug (just look at the file)
- Good for iteration

But if we're serious, we need real graph infrastructure.

---

## What Actually Matters for Temporal Reasoning

Let me think from first principles about what capabilities matter.

### Capability 1: Point-in-Time State Reconstruction

**Query:** "What did I believe about X on March 15th?"

**Requires:** Replay transitions up to timestamp T, compute state.

```python
def state_at(entity_id, timestamp):
    transitions = sorted(get_transitions(entity_id), key=lambda t: t.timestamp)
    state = {}
    for t in transitions:
        if t.timestamp > timestamp:
            break
        state = apply_transition(state, t)
    return state
```

**PIE status:** Could implement, but O(n) per entity.
**Graph DB:** Still O(n) per entity, but can index timestamps.

### Capability 2: Diff Between Time Periods

**Query:** "What changed between Q1 and Q3?"

**Requires:** 
1. Snapshot at T1
2. Snapshot at T2
3. Diff

**PIE status:** Expensive (reconstruct all entities at both times).
**Graph DB:** Same, unless you maintain snapshots.

### Capability 3: Temporal Pattern Detection

**Query:** "Do I always pivot projects after 3 months?"

**Requires:**
1. Find all projects
2. Find their state transitions
3. Detect pattern: created → ... → pivoted (within 3 months)

```python
def detect_early_pivot_pattern():
    projects = get_entities_by_type('project')
    pivots = []
    for p in projects:
        transitions = get_transitions(p.id)
        created = next((t for t in transitions if t.type == 'creation'), None)
        pivot = next((t for t in transitions if 'pivot' in t.trigger_summary.lower()), None)
        if created and pivot:
            days = (pivot.timestamp - created.timestamp) / 86400
            if days < 120:  # within 4 months
                pivots.append((p, days))
    return pivots
```

**PIE status:** Can implement with full scan.
**Graph DB:** Can implement with indexed traversal.

### Capability 4: Multi-Hop Relationship Queries

**Query:** "Who are the people 2 hops away from my current project?"

```
Me -> works_on -> Project -> collaborates_with -> Person -> works_on -> Other Project -> ...
```

**PIE status:** Manual BFS, expensive.
**Graph DB:** Native, fast.

---

## The Mastra Insight: Why Did Simple Text Win?

Mastra got 94.87% on LongMemEval with:
1. Dense text observations
2. No graph traversal
3. Three-date temporal model

**Why did this work?**

### LongMemEval Questions Are Mostly Retrieval

Looking at the question types:
- `single-session-user`: "What did the user say about X?"
- `single-session-assistant`: "What did the assistant say about X?"
- `knowledge-update`: "What is the current state of X?"
- `temporal-reasoning`: "When did X happen?"
- `multi-session`: "Combine info from sessions A and B"

**None of these require graph traversal.**

They require:
1. Finding relevant context (retrieval)
2. Presenting it with temporal markers
3. LLM reasoning over the context

Mastra's observation format does this perfectly:
```
🔴 12:10 User is building a Next.js app, due January 22nd 2026
  🔴 12:10 App uses server components with client-side hydration
  🟡 12:12 User asked about middleware configuration
```

### What LongMemEval DOESN'T Test

- Multi-hop reasoning ("How is X connected to Y through Z?")
- Pattern detection ("Do I always do X before Y?")
- Contradiction tracking ("Have I reversed my position on X?")
- Procedural memory ("What's my usual workflow for X?")

**These are where graph structure would help.**

---

## Real Enterprise Use Cases for Temporal State Tracking

Let me think about where this actually matters.

### 1. Sales Pipeline Intelligence

**Problem:** CRM contains deals, but no understanding of deal evolution.

**Temporal queries:**
- "Which deals stalled after initial contact?" (pattern detection)
- "What changed in this deal since last review?" (diff)
- "Which sales behaviors correlate with closed-won?" (pattern extraction)

**Why graph helps:**
```
Deal → contacted → qualified → demo'd → negotiated → closed
       ↓           ↓           ↓
    Stakeholder  Stakeholder  Competitor mentioned
```

Multi-hop: "Find deals where a competitor was mentioned within 2 weeks of stalling"

### 2. Incident/Change Tracking (IT Ops)

**Problem:** CMDB + incident logs, but no causal reasoning.

**Temporal queries:**
- "What configuration changes preceded this incident?" (causal)
- "Is there a pattern in our deployment failures?" (pattern)
- "What's the state of service X at time of incident?" (reconstruction)

**Why graph helps:**
```
Change → affects → Service → depends_on → Service → incident
```

### 3. Research Knowledge Management

**Problem:** Literature + experiments, but no evolution tracking.

**Temporal queries:**
- "How has the field's understanding of X evolved?" (belief evolution)
- "Which of my hypotheses have been contradicted by later findings?" (contradiction)
- "What's the research trajectory that led to this breakthrough?" (path)

**This is Kosmos territory.** Their world model tracks experiment results and paper findings.

### 4. Compliance/Audit Trail

**Problem:** Policy changes, access changes, need full history.

**Temporal queries:**
- "What was this user's access on date X?" (reconstruction)
- "Who approved this policy change?" (provenance)
- "Show all changes to sensitive data handling over last year" (audit)

**Why graph helps:** Provenance chains, relationship to policies, organizational hierarchy.

### 5. Personal Knowledge Worker (PIE's Current Focus)

**Problem:** ChatGPT history, notes, docs — no coherent world model.

**Temporal queries:**
- "How has my approach to X evolved?" (evolution)
- "What are my active projects and their states?" (current state)
- "When did I last work on Y?" (temporal lookup)
- "What patterns do I follow when starting projects?" (procedural)

---

## The Architecture Choice: What Should We Actually Build?

### Option A: Double Down on Graph

**Implement:**
- FalkorDB or Neo4j backend
- Real graph queries (Cypher)
- PageRank for importance
- Multi-hop retrieval
- Pattern mining algorithms

**Pro:** Enables queries that text can't do.
**Con:** Complexity, may not beat Mastra on standard benchmarks.

### Option B: Mastra-Style Observations

**Implement:**
- Observation log with three-date model
- Compression/reflection cycle
- No graph, just well-formatted text

**Pro:** Proven 94.87% on LongMemEval, simpler.
**Con:** Loses evolution tracking, contradiction detection.

### Option C: Hybrid — Graph for Storage, Observations for Context

**Implement:**
- Keep graph for structured storage (entities, transitions)
- Generate Mastra-style observations from graph
- Use observations for LLM context
- Use graph for analytics/pattern queries

```python
def generate_observations_from_graph(time_window):
    """Convert recent graph activity to observation format."""
    observations = []
    for transition in get_recent_transitions(time_window):
        entity = get_entity(transition.entity_id)
        obs = format_observation(
            entity=entity,
            transition=transition,
            priority=compute_priority(entity),  # 🔴/🟡/🟢
        )
        observations.append(obs)
    return observations
```

**Pro:** Best of both worlds — structured storage + proven context format.
**Con:** Additional complexity of maintaining both.

---

## The Benchmarking Problem

### What MemoryBench Does

```typescript
interface Provider {
  ingest(sessions: UnifiedSession[]): Promise<IngestResult>
  search(query: string): Promise<unknown[]>
  // ...
}

// Pipeline:
// 1. Load benchmark (LongMemEval, LoCoMo, etc.)
// 2. Ingest sessions into provider
// 3. For each question: search → answer → evaluate
```

**The issue:** This tests retrieval + answer, not graph reasoning.

### What We'd Need to Evaluate Graph Value

A benchmark that tests:
1. **Multi-hop queries:** "How is X connected to Y?"
2. **Temporal diff:** "What changed between T1 and T2?"
3. **Pattern detection:** "What recurring patterns exist in my behavior?"
4. **Contradiction tracking:** "Where have I changed my position?"

**These benchmarks don't exist** for personal memory. We'd have to create them.

---

## Recommendation

### Don't Build More PIE Features Yet

The unimplemented features (dreaming, consolidation, procedural memory) assume the graph is valuable. We haven't proven that yet.

### Instead:

1. **Run PIE on LongMemEval** — see if our context compilation beats naive RAG
2. **If it doesn't beat Mastra** — consider pivoting to observation-style
3. **If it does** — then invest in graph infrastructure

### The Minimum Viable Experiment

```bash
# 1. Write PIE provider for MemoryBench
# 2. Run on LongMemEval
# 3. Compare to Mastra's 84.23% (gpt-4o)

bun run memorybench/src/index.ts run -p pie -b longmemeval -j gpt-4o
```

If PIE scores:
- **<80%:** Graph isn't helping for this task. Pivot.
- **80-85%:** Marginal. Consider hybrid.
- **>85%:** Graph helps. Invest more.

### The Harder Experiment (Prove Graph Value)

Create a benchmark that tests graph-specific capabilities:
1. Generate synthetic data with known patterns
2. Create questions that require multi-hop or pattern detection
3. Evaluate PIE vs. Mastra on these questions

If graph matters, PIE should win on these. If not, it's over-engineering.

---

## The Honest Conclusion

**PIE is currently a JSON store with nice formatting.** The graph structure isn't being used for anything a flat list couldn't do.

The bet is that temporal evolution tracking, contradiction detection, and procedural memory matter for personal AI. But we haven't proven they matter for real queries yet.

**Mastra's success suggests:** For most memory tasks, dense well-formatted text with temporal markers is sufficient. Graph complexity may not be worth it.

**The path forward:**
1. Benchmark PIE against Mastra
2. If PIE loses, adopt observation-style approach
3. If PIE wins, build real graph infrastructure
4. Either way, create new benchmarks that test graph-specific capabilities

Don't build more features until we know the foundation is worth building on.
