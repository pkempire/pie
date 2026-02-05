# PIE: Personal Intelligence Engine — Architecture v2

> Revised after deep-diving Honcho, Graphiti/Zep, Mem0, Basic Memory, Jean Memory, Google ADK, Late Chunking, Cognee, and SuperMemory. Incorporates SOTA techniques and Parth's raw notes.

---

## 0. What's Changed From v1

v1 was designed from first principles. v2 is grounded in what's actually working in production:

- **Graphiti's bi-temporal model** solves the temporal encoding question — track both when something happened AND when you learned it
- **Honcho's "dreaming"** is the right metaphor for background consolidation — reason exhaustively, not just retrieve
- **Google ADK's "context as compiled view"** is the correct mental model — context is compiled, not concatenated
- **Late Chunking (Jina)** solves contextual embedding for conversations — don't chunk then embed, embed then chunk
- **Basic Memory's MCP + markdown** proves the interop pattern works — already shipping
- **Graph + Vector hybrid** is consensus — every serious project converges here

**Key insight from the research:** The projects that work best (Honcho, Graphiti) don't just retrieve — they **reason over memory**. Simple RAG recall is table stakes. The value is in the reasoning/inference layer that surfaces what was implied, not just what was said.

---

## 1. Design Principles (Updated)

1. **Context is compiled, not concatenated** (Google ADK) — Every LLM call gets a purpose-built view, not a dump
2. **Reason, don't just retrieve** (Honcho) — Background processes that make deductions from raw data
3. **Bi-temporal everything** (Graphiti) — Track event time AND ingestion time for every fact
4. **Graph + Vector, not either/or** — KG for relationships/traversal, vectors for semantic similarity
5. **MCP as the universal interface** — Any LLM reads/writes through one protocol
6. **Markdown as materialized views** — Human-readable snapshots of graph state, not the source of truth
7. **Local-first, zero cloud dependency** — Everything runs on your machine

---

## 2. System Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                        DATA SOURCES                                │
│  ChatGPT│Claude│OpenClaw│Gmail│Calendar│GitHub│Twitter│Browser│... │
└────────┬───────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────┐
│   INGESTION LAYER   │    Parsers + Extractors + Classifiers
│                     │    Real-time (webhooks/watchers)
│   Late Chunking     │    + Batch (exports)
│   Entity Extraction │
│   Relation Mining   │
│   Temporal Tagging  │
└────────┬────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│                    STORAGE LAYER                             │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │  KNOWLEDGE   │  │   VECTOR     │  │   EPISODE        │  │
│  │  GRAPH       │  │   INDEX      │  │   STORE          │  │
│  │              │  │              │  │                   │  │
│  │  Neo4j /     │  │  ChromaDB /  │  │  SQLite +        │  │
│  │  FalkorDB    │  │  Qdrant      │  │  Markdown files  │  │
│  │              │  │              │  │                   │  │
│  │  Entities    │  │  Semantic    │  │  Raw episodes     │  │
│  │  Relations   │  │  embeddings  │  │  with metadata   │  │
│  │  Temporal    │  │  of facts,   │  │  Decay scores    │  │
│  │  edges       │  │  episodes,   │  │  Source links    │  │
│  │  (bi-temp)   │  │  beliefs     │  │                   │  │
│  └──────┬───────┘  └──────┬───────┘  └────────┬──────────┘  │
│         │                 │                    │              │
│         └─────────┬───────┘────────────────────┘              │
│                   │                                           │
└───────────────────┼───────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│                 REASONING LAYER                              │
│                                                              │
│  ┌─────────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │   DREAMING      │  │  CONFLICT    │  │  PATTERN      │  │
│  │   ENGINE        │  │  DETECTOR    │  │  MINER        │  │
│  │                 │  │              │  │               │  │
│  │  Background     │  │  Finds       │  │  Recurring    │  │
│  │  deductions     │  │  contradicts │  │  themes,      │  │
│  │  from stored    │  │  Flags for   │  │  evolving     │  │
│  │  episodes.      │  │  resolution  │  │  interests,   │  │
│  │  "What does     │  │  Temporal    │  │  blind spots  │  │
│  │  this imply?"   │  │  invalidation│  │               │  │
│  └─────────────────┘  └──────────────┘  └───────────────┘  │
│                                                              │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              CONTEXT ASSEMBLY PIPELINE                       │
│          (Google ADK "Compiled View" Model)                  │
│                                                              │
│  Query → [Identity] → [Active State] → [Retrieval] →       │
│          [Graph Walk] → [Temporal Filter] → [Rerank] →      │
│          [Budget Trim] → Compiled Context                    │
│                                                              │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    MCP SERVER                                │
│                                                              │
│  Tools exposed to any MCP client:                           │
│                                                              │
│  READ:                                                       │
│  - search_memory(query, filters)     Semantic + keyword     │
│  - get_entity(name)                  Entity profile          │
│  - traverse_graph(start, hops)       Multi-hop exploration  │
│  - get_active_context()              Current state/focus     │
│  - get_timeline(entity, range)       Temporal queries        │
│  - get_related(entity, relation)     Graph neighbors         │
│                                                              │
│  WRITE:                                                      │
│  - add_episode(content, source)      Ingest new data        │
│  - add_fact(entity, relation, fact)  Direct KG write        │
│  - update_belief(entity, old, new)   Track belief changes   │
│  - log_decision(what, why, context)  Decision journal       │
│  - add_project(name, description)    Project tracking       │
│                                                              │
│  META:                                                       │
│  - get_conflicts()                   Unresolved conflicts   │
│  - get_patterns()                    Detected patterns      │
│  - run_consolidation()               Trigger dreaming       │
│                                                              │
│  Clients: Claude Desktop, Cursor, OpenClaw, ChatGPT*,      │
│           VS Code, any MCP-compatible tool                   │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│              MATERIALIZED VIEWS (Obsidian Vault)            │
│                                                              │
│  Auto-generated markdown snapshots of graph state.          │
│  Human-readable. Editable (edits sync back to graph).       │
│  NOT the source of truth — the graph is.                    │
│                                                              │
│  vault/                                                      │
│  ├── entities/{people,projects,orgs,concepts}/*.md          │
│  ├── beliefs/*.md                                            │
│  ├── patterns/*.md                                           │
│  ├── episodes/YYYY/MM/*.md                                   │
│  ├── daily-briefs/YYYY-MM-DD.md                             │
│  └── active-context.md                                       │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Knowledge Graph Design (Graphiti-Inspired)

### 3.1 Bi-Temporal Model

Every edge in the graph has TWO timestamps:
- **valid_at** — when the fact became true in the real world
- **ingested_at** — when the system learned about it

This enables:
- "What did I believe about X on date Y?" (point-in-time queries)
- "What changed since I last checked?" (incremental updates)
- "When did my opinion on X shift?" (belief evolution tracking)

### 3.2 Node Types

```
Person          — people in Parth's life/work
Project         — things being built or explored
Organization    — companies, schools, teams
Concept         — ideas, technologies, frameworks
Event           — timestamped occurrences
Decision        — choices made with reasoning
Belief          — opinions, preferences, stances (mutable!)
Task            — actionable items
Source          — data sources (ChatGPT, email, etc.)
```

### 3.3 Edge Types

```
# Relationships
KNOWS / WORKS_WITH / MENTORS
INTERESTED_IN / BELIEVES / PREFERS
FOUNDED / CONTRIBUTED_TO / USES
CONTRADICTS / SUPPORTS / EVOLVED_FROM
DECIDED / BECAUSE_OF
RELATED_TO / PART_OF / LEADS_TO

# Temporal
HAPPENED_AT / LEARNED_AT / CHANGED_ON
PRECEDED / FOLLOWED / CONCURRENT_WITH
```

### 3.4 Conflict Handling (Graphiti's Temporal Edge Invalidation)

```python
# When new info contradicts existing:
# 1. Don't delete the old edge — invalidate it with a timestamp
edge.invalidated_at = now()
edge.invalidated_by = new_episode_id

# 2. Create new edge with current validity
new_edge = Edge(
    source=entity,
    relation="BELIEVES",
    target=new_belief,
    valid_at=now(),
    confidence=0.8,
    source_episode=episode_id
)

# 3. Both coexist — enables "belief evolution" queries
```

---

## 4. Ingestion Pipeline (Detailed)

### 4.1 ChatGPT Bootstrap (Priority 0)

```python
# Phase 1: Parse
conversations = parse_chatgpt_export("export.json")
# → List[Conversation] with messages, timestamps, titles

# Phase 2: Pre-filter (cheap, no LLM)
filtered = [c for c in conversations if len(c.messages) > 2]
# Skip: "What's the capital of France?" type throwaway queries
# Keep: multi-turn discussions, project work, decisions, opinions

# Phase 3: Extract (batched, cheap model)
for batch in chunk(filtered, size=10):
    extraction = await extract_batch(
        batch,
        model="gpt-4.1-mini",  # or haiku
        schema=ExtractionSchema(
            entities=[],        # people, projects, concepts mentioned
            facts=[],           # objective statements
            opinions=[],        # subjective beliefs/preferences
            decisions=[],       # choices made with reasoning
            questions=[],       # recurring questions/curiosities
            topics=[],          # topic tags
            importance=0,       # 1-10
            temporal_refs=[],   # dates/times mentioned
        )
    )

# Phase 4: Late Chunking for embeddings
# DON'T: chunk each message, embed separately
# DO: embed full conversation with long-context model, THEN chunk
token_embeddings = embed_long(full_conversation_text)  # jina-v2 8k context
chunk_embeddings = late_chunk(token_embeddings, boundaries=topic_boundaries)

# Phase 5: Graph construction
for extraction in all_extractions:
    for entity in extraction.entities:
        graph.upsert_node(entity)
    for fact in extraction.facts:
        graph.add_edge(fact.subject, fact.relation, fact.object,
                      valid_at=extraction.timestamp,
                      source=extraction.source_conversation)

# Phase 6: Vector indexing
for chunk in all_chunks:
    vector_store.add(chunk.embedding, metadata=chunk.metadata)
```

### 4.2 Conversation-Level Intelligence Extraction

For each conversation, extract multiple layers:

```yaml
# Layer 1: Surface (what was discussed)
topics: [knowledge-graphs, agent-memory, chunking]
entities: [Graphiti, Neo4j, Honcho]

# Layer 2: Intent (why was this discussed)
intent: "researching architecture for personal memory system"
context: "building PIE project"

# Layer 3: Beliefs (opinions expressed or implied)
beliefs:
  - "knowledge graphs are better than pure vector for relationships"
  - "static chunking doesn't work for conversational data"

# Layer 4: Decisions (choices made)
decisions:
  - what: "use bi-temporal model for graph edges"
    why: "need to track belief evolution over time"
    alternatives_considered: [simple timestamps, no temporal tracking]

# Layer 5: Meta (how Parth thinks)
reasoning_style: "explores broadly first, then converges"
curiosity_signals: [asked 3 follow-up questions about temporal GNNs]
```

### 4.3 Stream-of-Consciousness Notes Handling

Parth raised this explicitly — unstructured notes with rapid context switching.

```python
# Problem: "knowledge graphs for everyone" followed by
#          "need to buy milk" followed by
#          "what about temporal GNNs"

# Solution: Semantic segmentation BEFORE chunking
def segment_stream_of_consciousness(text):
    """
    Use embedding distance between consecutive sentences
    to detect context switches.
    """
    sentences = split_sentences(text)
    embeddings = embed_batch(sentences)

    segments = []
    current_segment = [sentences[0]]

    for i in range(1, len(sentences)):
        similarity = cosine(embeddings[i-1], embeddings[i])
        if similarity < CONTEXT_SWITCH_THRESHOLD:  # e.g., 0.3
            segments.append(current_segment)
            current_segment = [sentences[i]]
        else:
            current_segment.append(sentences[i])

    return segments
    # → Each segment is a coherent thought, regardless of length
```

This is the **dynamic/context-based chunking** Parth was asking about. Key insight: chunk boundaries should be semantic, not size-based.

### 4.4 Real-Time Processing

For live conversations (OpenClaw, new ChatGPT sessions):

```
New message arrives
  │
  ├─ FAST PATH (< 1 sec)
  │   → Append to episode store
  │   → Update active-context.md
  │   → Quick entity detection (NER, no LLM needed)
  │
  └─ SLOW PATH (background, < 5 min)
      → Full extraction (entities, beliefs, decisions)
      → Graph updates (new nodes/edges)
      → Vector index update (incremental)
      → Conflict detection
      → Trigger dreaming if threshold met
```

---

## 5. The Dreaming Engine (Honcho-Inspired)

This is the **secret sauce**. Not just retrieval — active reasoning over stored memory.

### 5.1 What Dreaming Does

```python
class DreamingEngine:
    """
    Background process that reasons over stored episodes
    to make deductions not explicitly stated in any single episode.
    """

    async def dream_cycle(self):
        # 1. SYNTHESIZE — Combine related episodes
        related_episodes = self.find_related_unreasoned_episodes()
        for cluster in related_episodes:
            synthesis = await self.llm.reason(
                "What can you conclude from these related interactions? "
                "What's implied but never explicitly stated?",
                context=cluster
            )
            self.store_deduction(synthesis)

        # 2. CONNECT — Find novel links
        orphan_nodes = self.graph.find_weakly_connected()
        for node in orphan_nodes:
            potential_links = await self.llm.reason(
                f"Given what I know about {node}, what other entities "
                f"in the graph might be related? Why?",
                context=self.graph.get_neighborhood(node, hops=2)
            )
            self.evaluate_and_add_links(potential_links)

        # 3. EVOLVE — Track belief changes
        mutable_beliefs = self.graph.get_nodes(type="Belief")
        for belief in mutable_beliefs:
            history = self.graph.get_temporal_history(belief)
            if len(history) > 1:
                evolution = await self.llm.reason(
                    "How has this belief changed over time? "
                    "What caused the changes?",
                    context=history
                )
                self.store_evolution(belief, evolution)

        # 4. PREDICT — Surface proactive insights
        recent_patterns = self.pattern_miner.get_recent()
        for pattern in recent_patterns:
            insight = await self.llm.reason(
                "Given this pattern, what might the user want to know? "
                "What should they be thinking about?",
                context=pattern
            )
            if insight.confidence > 0.7:
                self.queue_proactive_insight(insight)
```

### 5.2 Dreaming Schedule

- **Micro-dreams** (after each new episode batch): Quick deductions from latest data
- **Deep dreams** (nightly): Full graph traversal, pattern mining, belief evolution
- **Mega-dreams** (weekly): Cross-domain synthesis, long-range connections, stale fact pruning

### 5.3 Cost Control

Dreaming uses LLM calls. Control costs by:
- Using cheap models for micro-dreams (Haiku, GPT-4.1-mini)
- Expensive models only for mega-dreams where reasoning quality matters
- Skip dreaming on low-importance episodes
- Cache reasoning results aggressively

---

## 6. Retrieval System (Multi-Strategy Fusion)

### 6.1 Hybrid Retrieval (Graphiti's Approach)

```python
async def retrieve(query: str, budget_tokens: int = 8000) -> Context:
    # Strategy 1: Semantic search (vector similarity)
    semantic_results = await vector_store.search(
        query, top_k=20,
        filter={"importance": {"$gte": 3}}
    )

    # Strategy 2: Keyword/BM25 (catches exact matches vectors miss)
    keyword_results = await bm25_search(query, top_k=20)

    # Strategy 3: Graph traversal (follows relationships)
    entities = extract_entities(query)
    graph_results = []
    for entity in entities:
        neighbors = await graph.traverse(
            entity, max_hops=2,
            edge_filter=lambda e: e.valid_at is not None  # only current facts
        )
        graph_results.extend(neighbors)

    # Strategy 4: Temporal (recent + relevant)
    if has_temporal_reference(query):
        temporal_results = await episode_store.query_time_range(
            parse_time_reference(query)
        )

    # Strategy 5: LLM-generated regex (Parth's idea!)
    regex_patterns = await llm.generate_search_patterns(query)
    regex_results = await full_text_search(regex_patterns)

    # FUSE + RERANK
    all_results = deduplicate(
        semantic_results + keyword_results +
        graph_results + temporal_results + regex_results
    )

    reranked = await rerank(
        query, all_results,
        model="cross-encoder"  # or LLM-based
    )

    return select_until_budget(reranked, budget_tokens)
```

### 6.2 Context Assembly Pipeline (ADK-Inspired)

```python
class ContextPipeline:
    """
    Processors run in order, each transforming the context.
    Like a compiler — observable, testable, configurable.
    """
    processors = [
        IdentityProcessor(),       # Who is Parth, who am I
        ActiveStateProcessor(),    # Current projects, tasks, focus
        RetrievalProcessor(),      # Multi-strategy retrieval from query
        GraphWalkProcessor(),      # Expand retrieved entities via graph
        TemporalFilterProcessor(), # Filter to relevant time range
        ConflictAnnotator(),       # Flag contradictory facts
        RerankProcessor(),         # Score by relevance to query
        BudgetTrimProcessor(),     # Cut to token budget
        FormatProcessor(),         # Structure for target LLM
    ]

    async def compile(self, query: str, budget: int) -> str:
        ctx = Context(query=query, budget=budget)
        for processor in self.processors:
            ctx = await processor.process(ctx)
        return ctx.render()
```

---

## 7. MCP Server Design

### 7.1 Tool Definitions

```python
# Core read tools
@mcp.tool()
async def search_memory(
    query: str,
    filters: Optional[dict] = None,  # time_range, topics, importance_min
    strategy: str = "hybrid",         # hybrid|semantic|keyword|graph
    limit: int = 10
) -> list[MemoryResult]:
    """Search across all memory layers with multi-strategy retrieval."""

@mcp.tool()
async def get_context(
    query: str,
    budget_tokens: int = 8000,
    include_graph: bool = True,
    include_active: bool = True
) -> str:
    """Get assembled context for a query (compiled view)."""

@mcp.tool()
async def get_entity(name: str) -> EntityProfile:
    """Get full profile of an entity from the knowledge graph."""

@mcp.tool()
async def traverse(
    start: str,
    relation: Optional[str] = None,
    max_hops: int = 2,
    temporal_filter: Optional[str] = None
) -> list[GraphPath]:
    """Walk the knowledge graph from an entity."""

@mcp.tool()
async def get_timeline(
    entity: str,
    start: Optional[str] = None,
    end: Optional[str] = None
) -> list[TimelineEvent]:
    """Get temporal history of an entity or belief."""

# Core write tools
@mcp.tool()
async def add_memory(
    content: str,
    source: str = "manual",
    importance: int = 5,
    topics: list[str] = []
) -> str:
    """Add a new memory/episode. Triggers extraction pipeline."""

@mcp.tool()
async def update_belief(
    entity: str,
    belief: str,
    new_value: str,
    reason: str
) -> str:
    """Track a belief change with reasoning."""

@mcp.tool()
async def log_decision(
    decision: str,
    reasoning: str,
    alternatives: list[str] = [],
    context: str = ""
) -> str:
    """Log a decision with full reasoning chain."""

# Proactive tools
@mcp.tool()
async def get_insights() -> list[Insight]:
    """Get proactive insights from the dreaming engine."""

@mcp.tool()
async def get_daily_brief() -> str:
    """Generate today's context brief."""
```

### 7.2 MCP Resources (Passive Context)

```python
# Resources that MCP clients can subscribe to
@mcp.resource("memory://active-context")
async def active_context() -> str:
    """Current state: active projects, recent decisions, focus areas."""

@mcp.resource("memory://entity/{name}")
async def entity_profile(name: str) -> str:
    """Entity profile from knowledge graph."""

@mcp.resource("memory://brief/today")
async def daily_brief() -> str:
    """Today's auto-generated brief."""
```

---

## 8. Project Auto-Clustering (Parth's Idea)

```python
class ProjectClusterer:
    """
    Automatically groups ideas/episodes into projects.
    Recursive: discovers sub-projects within projects.
    """

    async def cluster(self, max_depth: int = 3):
        # 1. Get all uncategorized episodes
        unlinked = self.episode_store.get_uncategorized()

        # 2. Embed and cluster
        embeddings = embed_batch([e.summary for e in unlinked])
        clusters = hdbscan_cluster(embeddings, min_cluster_size=3)

        # 3. LLM-name each cluster
        for cluster in clusters:
            name = await self.llm.generate(
                "What project or topic connects these ideas?",
                context=[e.summary for e in cluster.episodes]
            )
            project = self.graph.upsert_node(
                name=name, type="Project"
            )

            # 4. Link episodes to project
            for episode in cluster.episodes:
                self.graph.add_edge(episode, "PART_OF", project)

            # 5. Recurse if cluster is large enough
            if len(cluster.episodes) > 10 and max_depth > 0:
                self.cluster_within(project, max_depth - 1)
```

---

## 9. Proactive Agent Layer

### 9.1 Daily Brief Generator

```
Every morning at 7 AM:
1. Query: decisions made yesterday
2. Query: tasks left unfinished
3. Query: calendar events today
4. Query: recent pattern changes
5. Query: new insights from overnight dreaming
6. Compile into brief → push notification + vault/daily-briefs/

"Yesterday you decided to use Graphiti for the KG layer.
 You left the MCP server design unfinished.
 You have a meeting at 2 PM with [person].
 Pattern: you've been thinking about temporal data for 3 days straight.
 Insight: your ChatGPT logs show you explored this same problem in Oct 2024."
```

### 9.2 Connection Detector

```
When a new episode is ingested:
1. Find top-5 most similar past episodes (by embedding)
2. Check if ANY of those episodes are from a different project/context
3. If so → generate an insight card:

"You were thinking about [temporal GNNs] just now.
 6 months ago you explored [graph neural networks for drug discovery] at Sanofi.
 There might be a connection worth exploring."
```

### 9.3 Thinking Evolution Tracker

```
Periodically analyze belief nodes with temporal history:

"Your views on [education]:
 - Oct 2023: Focused on science fair prep, structured curriculum
 - Mar 2024: Shifted to AI-assisted learning, real-time feedback
 - Jan 2025: Now interested in 'personal intelligence' — learning from own data
 - Pattern: moving from teaching others → systems for self-learning"
```

---

## 10. Tech Stack (Final)

| Component | Choice | Why |
|-----------|--------|-----|
| Knowledge Graph | **FalkorDB** (Redis-compatible) | Lighter than Neo4j, Docker one-liner, Graphiti supports it, fast |
| Vector DB | **ChromaDB** (local) | Zero-config, Python-native, good enough for personal scale |
| Episode Store | **SQLite** | Bulletproof, zero-setup, perfect for metadata + episode tracking |
| Embeddings | **jina-embeddings-v3** (local) or **text-embedding-3-small** (API) | Late chunking support for Jina; OpenAI as fallback |
| Extraction LLM | **GPT-4.1-mini** (bulk) / **Claude Sonnet** (synthesis) | Cost-efficient extraction, quality synthesis |
| Reasoning LLM | **Claude Sonnet 4** or **GPT-4.1** | Dreaming needs good reasoning |
| MCP Framework | **Python MCP SDK** | Official SDK, well-documented |
| Graph Library | **Graphiti-core** | Already built temporal KG with hybrid retrieval + MCP server |
| Chunking | **Late Chunking** (Jina) + semantic segmentation | Contextual embeddings, dynamic boundaries |
| Orchestration | **Python scripts + OpenClaw cron** | Already have the infra |
| Markdown Gen | **Jinja2 templates** | Render graph state → Obsidian-compatible markdown |

---

## 11. Build Plan (Revised)

### Phase 0: Foundation (Days 1-2)
- [ ] Set up project structure, virtualenv, dependencies
- [ ] Docker: FalkorDB + ChromaDB
- [ ] Initialize Graphiti with custom entity types for PIE
- [ ] Basic MCP server skeleton with search_memory + add_memory

### Phase 1: ChatGPT Ingest (Days 3-7)
- [ ] ChatGPT JSON parser → Episode format
- [ ] Batch extraction pipeline (entities, beliefs, decisions, topics)
- [ ] Late chunking implementation for conversation embeddings
- [ ] Stream-of-consciousness segmenter
- [ ] Graph construction from extractions
- [ ] Vector index build
- [ ] Run on full 4000+ conversation export
- [ ] Spot-check quality

### Phase 2: Retrieval + MCP (Days 7-10)
- [ ] Multi-strategy retrieval (semantic + BM25 + graph + temporal)
- [ ] Context assembly pipeline
- [ ] Full MCP server with all tools
- [ ] Test with Claude Desktop + Cursor

### Phase 3: Dreaming + Proactive (Days 10-14)
- [ ] Dreaming engine (micro, deep, mega cycles)
- [ ] Conflict detection + resolution tracking
- [ ] Pattern mining
- [ ] Daily brief generator
- [ ] Connection detector
- [ ] OpenClaw integration for auto-ingestion

### Phase 4: Vault + Polish (Days 14+)
- [ ] Markdown materialization (graph → Obsidian vault)
- [ ] Bidirectional sync (vault edits → graph updates)
- [ ] Project auto-clustering
- [ ] Additional data source connectors
- [ ] Thinking evolution tracker

---

## 12. What We're NOT Building (Anti-Scope)

- **Not a product for other people** (yet) — this is personal infra
- **Not a chatbot UI** — the MCP server IS the interface
- **Not cloud-hosted** — everything local
- **Not fine-tuning a model** — extraction + reasoning + retrieval, not parameter updates
- **Not replacing Obsidian** — generating markdown FOR Obsidian as a view layer

---

## 13. Open Questions

1. **FalkorDB vs Neo4j vs just-a-JSON-graph?** FalkorDB is lightest but Neo4j has the most Graphiti support. Graphiti now supports FalkorDB natively.
2. **How aggressive should dreaming be?** More dreaming = more insight but more cost. Need to tune.
3. **ChatGPT API passthrough?** Parth mentioned wanting o3 somehow — could build a proxy that intercepts ChatGPT API calls and tees data to PIE.
4. **Screen recording data?** Parth mentioned this. OCR + temporal extraction is possible but heavy. Park for later.
5. **Continual fine-tuning?** Parth asked about this. Not worth it for a single user — extraction + reasoning is more flexible and doesn't require training infra. Revisit if the extraction quality plateaus.

---

## 14. Competitive Landscape (Post-Research)

| Project | What it does well | What it lacks for PIE |
|---------|------------------|-----------------------|
| **Honcho** | Best reasoning over memory ("dreaming"), SOTA benchmarks, representations | Designed for app builders, not personal use. Cloud-first. |
| **Graphiti** | Best temporal KG, bi-temporal model, hybrid retrieval, has MCP server | No dreaming/reasoning layer. Raw graph, you build the intelligence. |
| **Mem0** | Simple API, good benchmarks, wide integrations | Shallow — extracts preferences, not deep reasoning. No KG for relationships. |
| **Basic Memory** | MCP-native, Obsidian-compatible, simple & working | No graph, no temporal, no proactive reasoning. Just structured markdown. |
| **Jean Memory** | Smart orchestration layer, intent analysis | Built on Mem0 + Graphiti, cloud-focused, not personal-first. |
| **Cognee** | Knowledge graph from docs, "Dreamify" | Enterprise-focused, less personal memory, more document processing. |
| **SuperMemory** | Bookmarks/content ingestion | Surface-level. Save & search, no deep understanding. |

**PIE's edge:** Combines Graphiti's temporal KG + Honcho's dreaming + Basic Memory's MCP + ADK's context compilation. Purpose-built for one person's entire cognitive history, not a generic platform.
