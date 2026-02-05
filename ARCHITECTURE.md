# Personal Intelligence System — Architecture

> A living second brain that ingests, structures, and retrieves personal knowledge across all data sources. Human-readable (Obsidian-compatible), machine-queryable, and continuously evolving.

---

## 1. Design Philosophy

**Core principle:** Memory should work like a human brain, not a database.

Humans don't store raw transcripts. They extract meaning, form connections, forget noise, and strengthen patterns through repetition. This system mirrors that with three cognitive processes:

1. **Encoding** — Transform raw data into structured memory
2. **Consolidation** — Compress, connect, and promote memories over time
3. **Retrieval** — Assemble relevant context on demand

**Non-negotiable constraints:**
- All persistent memory is **plain markdown** (Obsidian-compatible, zero lock-in)
- Must be **LLM-agnostic** — works with any model, not just one provider
- Must be **human-auditable** — you can read, edit, and override anything
- Must **degrade gracefully** — if the vector DB dies, markdown files still work

---

## 2. Memory Architecture (3-Tier Cognitive Model)

Inspired by Generative Agents (Park et al., 2023), MemGPT (Packer et al., 2023), and human cognitive science (Atkinson-Shiffrin model).

### Tier 1: Working Memory (Context Window)

**What:** The active context assembled for each interaction.
**Analogy:** What you're thinking about right now.

```
┌─────────────────────────────────────┐
│          CONTEXT WINDOW             │
│                                     │
│  Identity / Soul  (fixed)           │
│  User Profile     (fixed)           │
│  Active Project   (session-scoped)  │
│  Retrieved Facts  (query-scoped)    │
│  Recent Episodes  (time-scoped)     │
│  Current Task     (turn-scoped)     │
└─────────────────────────────────────┘
```

- **Budget:** ~30-50% of context window for retrieved memory
- **Assembly:** Automatic, ranked by relevance + recency + importance
- **Key insight:** Don't stuff everything in. The context assembler is a retrieval system, not a dump.

### Tier 2: Short-Term / Episodic Memory

**What:** Recent events, conversations, observations — timestamped and linked.
**Analogy:** What happened today/this week.
**Storage:** `vault/episodes/YYYY/MM/DD/`

Each episode is a structured extraction from a raw interaction:

```yaml
---
type: episode
source: chatgpt | openclaw | claude | manual
timestamp: 2026-02-02T16:49:00-05:00
topics: [personal-intelligence-system, architecture]
entities: [Dev, Parth]
importance: 8  # 1-10, LLM-assessed
decay: 0.95    # memory strength multiplier per day
---

## Summary
Parth and Dev designed the architecture for a personal intelligence system...

## Key Points
- Decided on 3-tier memory model
- Chose Obsidian-compatible markdown as base format
- ...

## Extracted Facts
- [[Parth]] wants LLM-agnostic memory system
- [[Parth]] prefers systems that degrade gracefully

## Decisions Made
- Use markdown as source of truth, vector DB as index
- ...

## Open Questions
- How to handle conflicting information across sources?
- ...

## Raw Reference
<!-- link to original conversation if available -->
```

**Retention policy:**
- Episodes decay over time (importance × recency)
- High-importance episodes get promoted to Tier 3 during consolidation
- Low-importance episodes get archived (not deleted) after 90 days

### Tier 3: Long-Term / Semantic Memory

**What:** Distilled knowledge, beliefs, patterns, stable facts.
**Analogy:** What you *know* — your mental model of the world.
**Storage:** `vault/knowledge/`

This is the Obsidian vault — structured, interlinked, evergreen.

```
vault/knowledge/
├── entities/
│   ├── people/
│   │   ├── parth-kocheta.md        # Self-profile
│   │   ├── pranay-kocheta.md       # Brother
│   │   └── ...
│   ├── projects/
│   │   ├── science-research-academy.md
│   │   ├── personal-intelligence-system.md
│   │   └── ...
│   ├── organizations/
│   │   ├── umd.md
│   │   ├── sanofi.md
│   │   ├── cmu-airlab.md
│   │   └── ...
│   └── concepts/
│       ├── applied-ai.md
│       ├── agent-memory.md
│       └── ...
├── beliefs/
│   ├── technical-opinions.md       # "React > Vue", "Python for ML"
│   ├── life-principles.md          # Decision frameworks, values
│   ├── communication-style.md      # How Parth thinks/writes
│   └── ...
├── patterns/
│   ├── thinking-patterns.md        # How Parth reasons through problems
│   ├── recurring-interests.md      # Topics that keep coming up
│   ├── decision-patterns.md        # How Parth makes choices
│   └── blind-spots.md              # Things Parth consistently misses
├── timelines/
│   ├── career.md                   # Linear history
│   ├── projects.md                 # Project timeline
│   └── ...
└── meta/
    ├── schema.md                   # Memory format specs
    ├── sources.md                  # Connected data sources
    └── consolidation-log.md        # When/what was last processed
```

**Key properties of Tier 3:**
- **Bidirectional links** (`[[entity]]` syntax) — Obsidian-native graph
- **Versioned** — git-tracked, so you can see how beliefs evolve
- **Conflict-aware** — when new data contradicts existing knowledge, flag it
- **Source-attributed** — every fact traces back to an episode or source

---

## 3. Knowledge Graph Layer

On top of the markdown files, maintain a lightweight graph index:

```
┌──────────────────────────────────────────────┐
│                KNOWLEDGE GRAPH               │
│                                              │
│  Nodes: entities, concepts, projects, people │
│  Edges: relationships, temporal, causal      │
│                                              │
│  Storage: vault/graph/graph.json             │
│  Format: adjacency list, human-readable      │
│  Sync: rebuilt from markdown [[links]]       │
└──────────────────────────────────────────────┘
```

**Why a separate graph?**
- Markdown links give you the graph implicitly, but explicit graph enables:
  - Multi-hop reasoning ("What projects involved people from CMU?")
  - Cluster detection ("What topics are you most interested in?")
  - Temporal queries ("How has your focus shifted over 2 years?")

**Implementation:** Keep it simple — a JSON adjacency list rebuilt from markdown wikilinks. No heavyweight graph DB needed initially.

---

## 4. Ingestion Pipeline

### 4.1 Data Sources (Priority Order)

| Source | Format | Automation | Priority |
|--------|--------|------------|----------|
| ChatGPT export | JSON (conversations) | One-time bulk + periodic re-export | P0 — bootstrap |
| OpenClaw sessions | Live conversations | Automatic (already happening) | P0 — ongoing |
| Claude conversations | JSON/API | Export or API integration | P1 |
| GitHub repos | Code + commits + issues | `gh` CLI polling | P1 |
| Calendar | Events | Google Calendar API via `gog` | P2 |
| Email | Threads | Gmail API via `gog` | P2 |
| Twitter/X | Posts, bookmarks, likes | API or export | P3 |
| Browser history | URLs + titles | Export | P3 |
| Notes (Obsidian/Notion) | Markdown/API | File sync or API | P2 |
| PDFs/papers | Text extraction | On-demand | P3 |

### 4.2 Ingestion Flow

```
Raw Data
  │
  ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│   PARSER    │────▶│  EXTRACTOR   │────▶│   CLASSIFIER    │
│             │     │              │     │                 │
│ Normalize   │     │ Key points   │     │ Importance 1-10 │
│ to common   │     │ Entities     │     │ Topic tags      │
│ format      │     │ Decisions    │     │ Memory tier     │
│             │     │ Opinions     │     │ Decay rate      │
│             │     │ Questions    │     │                 │
└─────────────┘     └──────────────┘     └─────────────────┘
                                                │
                                                ▼
                                    ┌───────────────────┐
                                    │     LINKER        │
                                    │                   │
                                    │ Match to existing │
                                    │ entities/concepts │
                                    │ Create [[links]]  │
                                    │ Detect conflicts  │
                                    └───────────────────┘
                                                │
                                                ▼
                                    ┌───────────────────┐
                                    │     WRITER        │
                                    │                   │
                                    │ Episode → Tier 2  │
                                    │ Facts → Tier 3    │
                                    │ Index → Vector DB │
                                    │ Graph → Update    │
                                    └───────────────────┘
```

### 4.3 ChatGPT Bulk Ingestion (Bootstrap)

For 4000+ conversations, process in batches:

1. **Parse** — Split JSON into individual conversations
2. **Pre-filter** — Skip trivial interactions (<3 turns, generic questions)
3. **Batch extract** — Run LLM extraction on groups of 10-20 conversations
   - Use a cheaper/faster model for bulk extraction (GPT-4o-mini, Haiku)
   - Reserve expensive models for ambiguous/important cases
4. **Cluster** — Group extracted facts by topic/entity
5. **Synthesize** — Merge clustered extractions into Tier 3 knowledge files
6. **Review** — Human spot-check on synthesized knowledge (you verify key conclusions)

**Estimated cost/time for 4000 conversations:**
- Average conversation: ~2K tokens → ~8M tokens total input
- Extraction output: ~500 tokens per conversation → ~2M tokens output
- Using GPT-4o-mini: ~$1.20 input + $2.40 output ≈ **~$4-5 total**
- Using Claude Haiku: similar range
- Processing time: ~2-4 hours with parallelized batches

---

## 5. Consolidation Engine (Memory Maintenance)

This is the "sleep cycle" — periodic processing that strengthens, connects, and prunes memory.

### 5.1 Processes

**Daily (lightweight):**
- New episodes from today → extract and file
- Update "active context" files (current projects, tasks, focus)
- Decay scores on old episodes

**Weekly (medium):**
- Review week's episodes → update Tier 3 knowledge files
- Detect emerging patterns ("Parth asked about X five times this week")
- Update knowledge graph from new links
- Rebuild vector index with new content

**Monthly (deep):**
- Re-read all Tier 3 knowledge → check for contradictions, staleness
- Synthesize pattern changes ("Interest in X is declining, Y is rising")
- Archive low-value episodes
- Generate "monthly reflection" document
- Prune dead links in knowledge graph

### 5.2 Conflict Resolution

When new information contradicts existing knowledge:

```
New fact: "Parth prefers Vue.js"
Existing: "Parth prefers React" (source: chat from 2024-06)

→ Flag in vault/meta/conflicts.md
→ Keep BOTH with timestamps and sources
→ Mark newer as "current belief" with note about change
→ On next human review, resolve permanently
```

---

## 6. Retrieval System

### 6.1 Multi-Strategy Retrieval

No single retrieval method works for everything. Use a fusion approach:

```
Query: "What does Parth think about education startups?"
         │
         ├──▶ Semantic Search (vector DB)
         │    → Top 10 similar chunks by embedding distance
         │
         ├──▶ Graph Traversal
         │    → [[Parth]] → interested-in → [[education]]
         │    → [[Parth]] → founded → [[Science Research Academy]]
         │    → [[education]] → related → [[startups]]
         │
         ├──▶ Keyword/Tag Search
         │    → topics: [education, startups, edtech]
         │
         ├──▶ Temporal Search
         │    → Most recent episodes tagged education + startups
         │
         └──▶ RERANKER (LLM-based)
              → Score all results for actual relevance
              → Assemble top-K into context window
              → Return with source attribution
```

### 6.2 Context Assembly

When any LLM needs context about Parth, the retrieval system assembles:

```python
def assemble_context(query, budget_tokens=8000):
    # Always include (fixed cost)
    identity = load("vault/identity.md")          # ~200 tokens
    profile = load("vault/knowledge/entities/people/parth-kocheta.md")  # ~500 tokens
    active = load("vault/active-context.md")      # ~300 tokens

    # Dynamic retrieval (fills remaining budget)
    remaining = budget_tokens - fixed_cost
    results = retrieve(query, strategies=["semantic", "graph", "temporal"])
    ranked = rerank(results, query)
    context = select_until_budget(ranked, remaining)

    return identity + profile + active + context
```

### 6.3 Vector Index

- **Engine:** Local (ChromaDB or LanceDB) — no cloud dependency
- **Embedding model:** `text-embedding-3-small` or local alternative (e5-small)
- **Chunk strategy:** Per-episode summary + per-fact (not raw conversations)
- **Metadata:** timestamp, source, importance, topics, entities
- **Rebuild:** Weekly full rebuild, daily incremental

---

## 7. Cross-LLM Interop

The system must work across ChatGPT, Claude, OpenClaw, and any future model.

### 7.1 Universal Context Protocol

Any LLM interface can query the vault via:

```
1. API endpoint (local server)
   GET /context?query=...&budget=8000
   → Returns assembled markdown context

2. File-based (for tools that read files)
   vault/active-context.md  — auto-updated, always current
   vault/briefing.md        — daily briefing for new sessions

3. MCP server (Model Context Protocol)
   → Expose vault as MCP resources + tools
   → Any MCP-compatible client gets full access
```

### 7.2 Conversation Routing

```
┌──────────┐  ┌──────────┐  ┌──────────┐
│ ChatGPT  │  │  Claude   │  │ OpenClaw │  ... any LLM
└────┬─────┘  └────┬─────┘  └────┬─────┘
     │              │              │
     ▼              ▼              ▼
┌──────────────────────────────────────┐
│        CONVERSATION COLLECTOR        │
│                                      │
│  Webhooks / Exports / API polling    │
│  Normalize to common format          │
│  Feed into ingestion pipeline        │
└──────────────────────────────────────┘
```

---

## 8. Vault Directory Structure

```
vault/
├── identity.md                    # Who Parth is (agent-readable)
├── active-context.md              # Current focus, projects, tasks
├── briefing.md                    # Auto-generated daily briefing
│
├── episodes/                      # Tier 2: Episodic memory
│   └── 2026/
│       └── 02/
│           └── 02/
│               ├── chatgpt-session-abc.md
│               ├── openclaw-session-xyz.md
│               └── ...
│
├── knowledge/                     # Tier 3: Semantic memory
│   ├── entities/
│   │   ├── people/
│   │   ├── projects/
│   │   ├── organizations/
│   │   └── concepts/
│   ├── beliefs/
│   ├── patterns/
│   ├── timelines/
│   └── meta/
│
├── graph/
│   └── graph.json                 # Knowledge graph index
│
├── index/
│   ├── vectors/                   # Vector embeddings (ChromaDB/LanceDB)
│   └── tags.json                  # Tag → file mapping
│
├── inbox/                         # Raw data waiting for processing
│   ├── chatgpt-export.json
│   ├── claude-export.json
│   └── ...
│
├── archive/                       # Decayed/archived episodes
│
├── automations/
│   ├── ingest.py                  # Ingestion pipeline
│   ├── consolidate.py             # Consolidation jobs
│   ├── retrieve.py                # Retrieval/context assembly
│   ├── server.py                  # Local API/MCP server
│   └── config.yaml                # Pipeline configuration
│
└── .obsidian/                     # Obsidian config (themes, plugins, etc.)
```

---

## 9. Tech Stack

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Storage | Markdown files + Git | Human-readable, versionable, zero lock-in |
| Vector DB | ChromaDB (local) or LanceDB | No cloud dependency, embeds with data |
| Embeddings | OpenAI text-embedding-3-small | Good quality/cost, swap later if needed |
| Graph | JSON adjacency list → NetworkX | Start simple, upgrade to Neo4j only if needed |
| Extraction LLM | GPT-4o-mini or Haiku (bulk), Opus/Sonnet (synthesis) | Cost-efficient for scale, quality for synthesis |
| API layer | FastAPI local server | Lightweight, serves context to any client |
| Interop | MCP server | Future-proof for Claude, Cursor, etc. |
| Orchestration | Python scripts + OpenClaw cron | Consolidation as scheduled jobs |
| UI | Obsidian | Already designed for this; graph view, backlinks, search |

---

## 10. Build Phases

### Phase 0: Foundation (Week 1)
- [ ] Set up vault directory structure
- [ ] Define markdown schemas for episodes + knowledge files
- [ ] Build ChatGPT JSON parser
- [ ] Bootstrap with web-researched knowledge (already done partially)

### Phase 1: ChatGPT Bootstrap (Weeks 1-2)
- [ ] Ingest 4000+ ChatGPT conversations
- [ ] Extract → classify → link pipeline
- [ ] Generate initial Tier 3 knowledge files
- [ ] Human review pass on key synthesized facts
- [ ] Build initial vector index

### Phase 2: Retrieval (Week 2-3)
- [ ] Multi-strategy retrieval engine
- [ ] Context assembly with budget management
- [ ] Local API server for context queries
- [ ] Integration with OpenClaw (auto-context for sessions)

### Phase 3: Automations (Week 3-4)
- [ ] Daily consolidation job
- [ ] Weekly synthesis job
- [ ] Conversation collector for ongoing OpenClaw sessions
- [ ] Active context auto-updater

### Phase 4: Cross-LLM (Week 4+)
- [ ] MCP server for Claude/Cursor/etc.
- [ ] ChatGPT custom GPT with API retrieval
- [ ] Additional data source connectors (GitHub, Calendar, Email)

### Phase 5: Intelligence Layer (Ongoing)
- [ ] Pattern detection ("your interests are shifting toward X")
- [ ] Proactive suggestions ("based on your history, you might want to...")
- [ ] Contradiction detection and resolution
- [ ] Self-model accuracy tracking

---

## 11. Open Design Questions

1. **Privacy tiers** — Some memories should be more restricted than others. How granular?
2. **Multi-user** — Will anyone else ever query this system, or strictly single-user?
3. **Hosting** — All local? Or cloud-backed for availability across devices?
4. **Real-time vs batch** — How fast should new conversations be ingested? Minutes? Hours?
5. **Obsidian-first vs API-first** — Which is the primary interface for humans?
6. **Cost ceiling** — What's the monthly budget for embedding/LLM calls for maintenance?

---

## 12. What Makes This Different from Existing Tools

| Tool | What it does | What it lacks |
|------|-------------|---------------|
| Obsidian + plugins | Great PKM, manual linking | No automatic ingestion, no LLM integration |
| Mem.ai | AI-powered notes | Cloud-locked, no raw data access |
| Rewind.ai | Records everything | No semantic structure, retrieval is basic |
| Custom RAG | Vector search on docs | No consolidation, no knowledge graph, no evolution |
| MemGPT/Letta | LLM memory management | Conversation-scoped, not life-scoped |

**This system combines:** Obsidian's human-readable structure + RAG's retrieval power + cognitive science's memory consolidation + knowledge graph's relationship reasoning. All open, local, and LLM-agnostic.
