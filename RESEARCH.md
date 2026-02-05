# Personal Intelligence System — Deep Research Document

> **Generated**: 2026-02-02  
> **Purpose**: Comprehensive analysis of memory/context infrastructure for AI agents, informing the architecture of our personal intelligence system.

---

## Table of Contents

1. [Project Analyses](#1-project-analyses)
   - [1.1 Honcho (Plastic Labs)](#11-honcho-plastic-labs)
   - [1.2 Basic Memory](#12-basic-memory)
   - [1.3 Jean Memory (formerly Your Memory)](#13-jean-memory-formerly-your-memory)
   - [1.4 Mem0](#14-mem0)
   - [1.5 Graphiti (Zep)](#15-graphiti-zep)
2. [Platform & Product Analyses](#2-platform--product-analyses)
   - [2.1 Graphlit](#21-graphlit)
   - [2.2 SuperMemory](#22-supermemory)
3. [Architecture Deep-Dives](#3-architecture-deep-dives)
   - [3.1 Google ADK Context Engineering](#31-google-adk-context-engineering)
   - [3.2 Thinking Machines — Tinker / LoRA Infrastructure](#32-thinking-machines--tinker--lora-infrastructure)
4. [Research Topics](#4-research-topics)
   - [4.1 Temporal GNNs for Link Prediction in Knowledge Graphs](#41-temporal-gnns-for-link-prediction-in-knowledge-graphs)
   - [4.2 Dynamic / Context-Based Chunking](#42-dynamic--context-based-chunking)
   - [4.3 Multi-Hop RAG over Knowledge Graphs (GraphRAG)](#43-multi-hop-rag-over-knowledge-graphs-graphrag)
   - [4.4 Re-Ranking Approaches for RAG](#44-re-ranking-approaches-for-rag)
5. [Comparative Matrix](#5-comparative-matrix)
6. [Key Architectural Insights for Our System](#6-key-architectural-insights-for-our-system)
7. [Novel Ideas Worth Stealing](#7-novel-ideas-worth-stealing)

---

## 1. Project Analyses

---

### 1.1 Honcho (Plastic Labs)

**Repository**: https://github.com/plastic-labs/honcho  
**Docs**: https://docs.honcho.dev  
**Paper/Benchmarks**: https://evals.honcho.dev

#### What It Does

Honcho is an open-source **memory library and managed service** for building stateful AI agents. It goes beyond simple RAG — it *reasons* over stored data using formal logic to extract latent information that was never explicitly stated. The key differentiator is that Honcho treats memory as a reasoning problem, not a retrieval problem.

#### Architecture / How It Works

Honcho runs as a **server with background workers**:

1. **Storage Primitives**: Workspaces → Peers → Sessions → Messages (hierarchical)
2. **Ingestion**: Messages are written to the API, stored immediately, and enqueued for background processing
3. **Reasoning Pipeline** (the "Deriver"):
   - **Explicit extraction**: Custom fine-tuned models (Neuromancer XR) extract premises from messages
   - **Deductive reasoning**: Formal logic draws certain conclusions from premises
   - **Inductive reasoning**: Pattern recognition across multiple messages
   - **Abductive reasoning**: Infers simplest explanations for observed behavior
   - **"Dreaming"**: Background process that reasons across already-ingested messages and prior conclusions to make further deductions
4. **Query**: Chat endpoint with a research agent that calls tools to find the best answer. Uses `claude-haiku-4-5` by default for the chat endpoint, `gemini-2.5-flash-lite` for ingestion.

**The "Peer Paradigm"**: Both users and agents are modeled as "Peers" — first-class entities that can observe and reason about each other. This is unique — most systems only model users.

**Perspective-Taking**: Peers can have different representations of each other based on what they've actually observed. Alice's representation of Bob is built only from sessions Alice participated in with Bob.

#### Memory Model

- **Episodic**: Session-based message history with temporal boundaries
- **Semantic**: Conclusions, summaries, and peer cards stored as peer "representations"
- **Procedural**: Not explicitly supported
- **Meta-cognitive**: "Dreaming" — background reasoning that generates new insights from existing reasoning

#### Storage

- **PostgreSQL** with pgvector for vector search
- Supports Turbopuffer and LanceDB as alternative vector stores
- All reasoning artifacts indexed in vector collections

#### Retrieval Strategy

- **Research agent approach**: The chat endpoint runs a model that uses tools to search representations, summaries, peer cards, and raw messages
- Not simple RAG — it's agent-directed retrieval with reasoning
- Median 5%, mean 11% of total tokens used for correct answers (extremely token-efficient)

#### Temporal Handling

- Sessions have temporal boundaries
- Background reasoning can track changes over time through deductive chains
- "Knowledge update" questions scored 94.9% on LongMem S
- Temporal reasoning is acknowledged as a weak point (88.7% on LongMem, 77% on LoCoMo temporal questions)
- Active development on multi-hop and temporal reasoning improvements

#### Strengths

- **SOTA benchmark performance**: 90.4% on LongMem S, 89.9% LoCoMo, top scores on BEAM (10M tokens)
- **Token efficiency**: 60% cost reduction over raw context injection with Gemini 3 Pro
- **Reasoning depth**: Formal logic extracts insights never explicitly stated
- **Peer paradigm**: Unique entity-centric model where both users and agents are peers
- **Perspective-taking**: Different peers can have different representations of each other
- **Scales beyond context windows**: No drop-off in recall up to 1M tokens, minimal at 10M
- **Custom fine-tuned models**: Neuromancer XR for ingestion — cheaper and faster than frontier models

#### Weaknesses/Gaps

- **Complexity**: Requires PostgreSQL + background workers + multiple LLM API keys
- **Latency**: Background reasoning is async — not instant after message write
- **Temporal reasoning**: Still a weak point across all benchmarks
- **No explicit knowledge graph**: Reasoning is stored as flat vector-indexed conclusions, not as a graph structure
- **Limited file type support**: Currently focused on conversational data
- **Cost**: Requires running inference for reasoning on every message batch

#### Novel Ideas Worth Stealing

1. **"Dreaming"** — background reasoning that generates new conclusions from existing conclusions, not just from raw data
2. **Formal logic framework** — deductive → inductive → abductive reasoning chain, each building on the previous
3. **Peer paradigm** — modeling agents as first-class entities with their own representations
4. **Perspective-based segmentation** — different entities see different "memories" based on what they observed
5. **Neuromancer XR** — custom fine-tuned small models for the reasoning task (cheaper than frontier)
6. **Token-efficient reconstruction** — using 5-11% of tokens to match or beat full-context performance

---

### 1.2 Basic Memory

**Repository**: https://github.com/basicmachines-co/basic-memory  
**Docs**: https://docs.basicmemory.com

#### What It Does

Basic Memory enables persistent knowledge through conversations with LLMs, storing everything in **local Markdown files** that both humans and AI can read/write. It uses the **Model Context Protocol (MCP)** to integrate with Claude Desktop, VS Code, and other MCP-compatible tools.

#### Architecture / How It Works

1. **Markdown files as source of truth**: All knowledge stored as Markdown with frontmatter (title, permalink, tags)
2. **Semantic patterns in Markdown**:
   - **Observations**: `- [category] content #tag (optional context)` — facts about entities
   - **Relations**: `- relation_type [[WikiLink]] (optional context)` — links between entities
3. **SQLite database**: Indexes the Markdown for fast search. Files → Entity objects with Observations and Relations
4. **Knowledge graph**: Derived from the file-based relations, traversable by the LLM
5. **MCP Server**: Exposes tools for content management, graph navigation, search, and visualization
6. **Sync**: Real-time file watching (`basic-memory sync --watch`) keeps the index in sync
7. **Cloud option**: Bidirectional sync for cross-device access

**Key MCP Tools**:
- `write_note`, `read_note`, `edit_note` — CRUD operations
- `build_context(url, depth, timeframe)` — graph traversal via `memory://` URLs
- `search_notes` — full-text + metadata search
- `canvas` — knowledge graph visualization

#### Memory Model

- **Semantic**: The primary mode — structured knowledge with entities, observations, and relations
- **Episodic**: Limited — recent activity tracking with `recent_activity()` tool
- **Procedural**: Not supported

#### Storage

- **Markdown files** (local filesystem, ~/basic-memory by default)
- **SQLite** database for indexing and search
- No vector database — relies on text search and graph traversal
- Optional cloud sync via BasicMemory.com

#### Retrieval Strategy

- **Graph traversal**: Follow `[[WikiLinks]]` to build context chains
- **Text search**: SQLite full-text search across notes
- **Temporal filtering**: `recent_activity(timeframe)` for time-scoped queries
- **memory:// URLs**: Custom URI scheme for referencing entities across tools

#### Temporal Handling

- **Minimal**: Frontmatter can include dates, `recent_activity` tracks modification times
- No explicit temporal reasoning or versioning of facts
- No contradiction detection when facts change over time

#### Strengths

- **Local-first**: All data stays on your machine as human-readable Markdown
- **Bi-directional**: Both humans and AI read/write the same files — Obsidian-compatible
- **Simplicity**: No vector DB, no cloud dependency, just files + SQLite
- **MCP integration**: Works natively with Claude Desktop, VS Code, any MCP client
- **Transparency**: You can see exactly what the AI knows by reading the files
- **Graph traversal**: Wiki-link style relations create a traversable knowledge graph

#### Weaknesses/Gaps

- **No vector search**: Relies on keyword/full-text search — misses semantic similarity
- **No reasoning**: Just stores and retrieves — doesn't derive new conclusions
- **Manual knowledge creation**: AI must be explicitly asked to write notes
- **Limited scalability**: SQLite + file-based approach may struggle at large scale
- **No embedding-based retrieval**: Can't find semantically similar but lexically different content
- **No temporal reasoning**: Can't handle "what did I think about X last month vs now?"
- **Fragile structure**: Markdown patterns must be followed exactly for parsing to work

#### Novel Ideas Worth Stealing

1. **Markdown as the universal interchange format** — human-readable, AI-writable, editor-compatible
2. **`memory://` URL scheme** — clean way to reference entities across tools and sessions
3. **MCP-first architecture** — designed for the emerging standard of AI tool integration
4. **Observation/Relation pattern** — simple but powerful: `[category] content` for facts, `relation [[Entity]]` for links
5. **File-system as knowledge graph** — each file is an entity, links create edges, no external graph DB needed

---

### 1.3 Jean Memory (formerly Your Memory)

**Repository**: https://github.com/jean-technologies/jean-memory  
**Docs**: https://docs.jeanmemory.com  
**Note**: Renamed from "Your Memory" by jonathan-politzki. Built on top of mem0 and graphiti.

#### What It Does

Jean Memory provides a **persistent intelligent memory layer** for AI applications with a focus on easy integration (React SDK, Python SDK, Node SDK). It emphasizes "context engineering" — dynamically analyzing intent, saving information, and retrieving relevant context. The product is a managed API service with self-hosting option.

#### Architecture / How It Works

**Two-layer architecture**:

1. **Orchestration Layer** (`jean_memory` tool): 
   - Analyzes user's message and conversation history
   - Determines optimal context strategy
   - Calls core tools to gather information
   - Synthesizes results into a "context package"
   - Handles background memory saving
   
2. **Core API** (Granular Tools):
   - `add_memories` — store new information
   - `search_memory` — find relevant memories
   - `deep_memory_query` — complex retrieval
   - Exposed via REST API for custom integration

**Built on**: mem0 (memory extraction) + Graphiti (knowledge graphs) — combines both as underlying engines.

**Stack**: Docker Compose with Supabase (Postgres), OpenAI/Gemini for LLM, web UI dashboard.

#### Memory Model

- **Semantic**: Via mem0's memory extraction and graph relationships from Graphiti
- **Episodic**: Conversation history tracking
- **Procedural**: Not explicitly supported

#### Storage

- **Supabase** (PostgreSQL) for relational data
- **Vector DB** (via mem0) for semantic search
- **Knowledge graph** (via Graphiti/Neo4j) for entity relationships

#### Retrieval Strategy

- **AI-powered orchestration**: An LLM analyzes the query to determine the best retrieval strategy
- **Multi-strategy**: Combines vector search (from mem0) with graph queries (from Graphiti)
- **Context packages**: Synthesized bundles of relevant context, not raw retrieved chunks

#### Temporal Handling

- Inherits temporal capabilities from Graphiti's bi-temporal model
- Not deeply customized beyond what the underlying libraries provide

#### Strengths

- **Easy integration**: Drop-in React component (`<JeanChat />`), 5-minute setup
- **Smart orchestration**: AI decides the retrieval strategy, not the developer
- **Dual engine**: Combines mem0's vector memory with Graphiti's graph memory
- **Full-stack**: UI dashboard, OAuth flows, frontend + backend SDKs

#### Weaknesses/Gaps

- **Black box orchestration**: Hard to understand or debug what the orchestrator is doing
- **Dependency on two complex systems**: mem0 + Graphiti add complexity
- **Proprietary additions**: Open-source base (mem0) but Jean's additions are proprietary
- **Less control**: The "magic" orchestration trades transparency for convenience
- **Early stage**: Less battle-tested than mem0 or Graphiti alone

#### Novel Ideas Worth Stealing

1. **Orchestration layer that decides retrieval strategy** — AI analyzing intent before choosing which memory system to query
2. **"Context packages"** — pre-assembled bundles of relevant context rather than raw retrieval results
3. **React drop-in component** — making memory integration as easy as adding a widget

---

### 1.4 Mem0

**Repository**: https://github.com/mem0ai/mem0  
**Paper**: https://mem0.ai/research (arXiv:2504.19413)  
**Docs**: https://docs.mem0.ai

#### What It Does

Mem0 is a **universal memory layer for AI agents** that extracts, consolidates, and retrieves important information from conversations. It's the most widely-adopted open-source AI memory system (50k+ developers, YC-backed). The key insight: compress conversation history into dense memory representations to cut tokens while preserving context fidelity.

#### Architecture / How It Works

1. **Memory Extraction**: When `memory.add(messages, user_id)` is called, an LLM extracts key facts from the conversation
2. **Memory Consolidation**: New memories are compared against existing ones — conflicts are resolved, duplicates merged
3. **Memory Search**: `memory.search(query, user_id)` retrieves relevant memories via vector similarity
4. **Multi-level Memory**: Supports User, Session, and Agent state scopes

**Mem0ᵍ** (Graph variant): Layers a graph-based store on top:
- Extracts entities and relationships from every memory write
- Stores embeddings in vector DB AND mirrors relationships in a graph backend (Neo4j)
- On retrieval, vector search narrows candidates while graph returns related context

**Memory Compression Engine**: Intelligently compresses chat history into optimized representations — cuts prompt tokens by up to 80%.

#### Memory Model

- **Semantic**: Core competency — extracted facts and preferences
- **Episodic**: Session-level memory with conversation tracking
- **Procedural**: Not explicitly supported
- **Hierarchical**: User → Session → Agent memory scopes

#### Storage

- **Vector DB**: Supports Qdrant, Chroma, Pinecone, Weaviate, Milvus, PGVector, and more
- **Graph DB**: Neo4j, AWS Neptune, Kuzu (for Mem0ᵍ graph memory)
- **Relational**: Optional for metadata

#### Retrieval Strategy

- **Vector similarity search**: Primary retrieval via embeddings
- **Graph traversal** (Mem0ᵍ): Follows entity relationships for richer context
- **Re-ranking**: Not explicitly mentioned but memory consolidation acts as a quality filter
- **Scoped search**: Filter by user_id, session_id, agent_id

#### Temporal Handling

- **Timestamped memories**: Every memory versioned and timestamped
- **Memory consolidation**: When new info contradicts old, the system resolves conflicts
- **No explicit temporal reasoning**: Can't answer "what did the user think about X last month?"
- **TTL support**: Memory observability with time-to-live tracking

#### Strengths

- **Simplicity**: `pip install mem0ai`, 3 lines of code to start
- **Benchmark performance**: +26% accuracy over OpenAI Memory on LOCOMO, 91% faster, 90% fewer tokens
- **Wide ecosystem**: Works with OpenAI, LangGraph, CrewAI, and more
- **Production-ready**: SOC 2, HIPAA compliant, BYOK, deploy anywhere
- **Token efficiency**: 80% reduction in prompt tokens via compression
- **Graph memory option**: Mem0ᵍ adds relationship-aware retrieval
- **Massive adoption**: 50k+ developers, proven at scale

#### Weaknesses/Gaps

- **Shallow reasoning**: Extracts explicit facts but doesn't reason about latent information
- **No "dreaming"**: Doesn't generate new insights from existing memories
- **Temporal reasoning limited**: Timestamps exist but no temporal inference
- **Memory quality depends on LLM**: Extraction quality varies with the model used
- **No perspective-taking**: All memories are flat per user — no multi-entity reasoning
- **Graph memory is add-on**: Core system is vector-only; graph is optional and adds complexity

#### Novel Ideas Worth Stealing

1. **Memory Compression Engine** — converting verbose conversations into dense, token-efficient memory representations
2. **Memory consolidation with conflict resolution** — comparing new memories against existing ones, resolving contradictions
3. **Multi-level memory scopes** (User / Session / Agent) — different retention and retrieval semantics per scope
4. **Graph memory as an overlay** — Mem0ᵍ layers graph on top of vector without replacing it

---

### 1.5 Graphiti (Zep)

**Repository**: https://github.com/getzep/graphiti  
**Paper**: arXiv:2501.13956 — "Zep: A Temporal Knowledge Graph Architecture for Agent Memory"

#### What It Does

Graphiti is an open-source framework for building **temporally-aware knowledge graphs** for AI agents. It's the core engine behind Zep's commercial platform. The key innovation: a **bi-temporal data model** that tracks both when events occurred and when they were ingested, enabling accurate point-in-time queries.

#### Architecture / How It Works

1. **Episodes**: Data enters as "episodes" — units of information (text, structured JSON, conversation turns)
2. **Entity Extraction**: LLM extracts entities (nodes) and relationships (edges) from episodes
3. **Temporal Tracking**: 
   - **Event time**: When something actually happened
   - **Ingestion time**: When the system learned about it
   - This bi-temporal model enables historical queries and contradiction detection
4. **Knowledge Graph Construction**: Entities and relationships stored in a graph DB
5. **Contradiction Handling**: Temporal edge invalidation — when new info contradicts old, the old edge gets an end timestamp rather than being deleted
6. **Retrieval**: Hybrid search combining semantic embeddings, BM25 keyword search, and graph traversal

**Custom Entity Definitions**: Developers define entity schemas via Pydantic models — the graph is structured by your domain ontology.

**Supported Graph DBs**: Neo4j, FalkorDB, Kuzu, Amazon Neptune

#### Memory Model

- **Episodic**: Episodes as the fundamental unit of data ingestion
- **Semantic**: Entity-relationship knowledge graph
- **Temporal**: Bi-temporal tracking is the core differentiator
- **Procedural**: Not explicitly, but graph structure can encode process knowledge

#### Storage

- **Graph DB**: Neo4j (primary), FalkorDB, Kuzu, Amazon Neptune
- **Vector embeddings**: Stored alongside graph nodes/edges for semantic search
- **No separate vector DB**: Embeddings live in the graph DB layer

#### Retrieval Strategy

- **Hybrid**: Semantic embeddings + BM25 keyword + graph traversal
- **Graph distance reranking**: Search results reranked by their proximity in the graph
- **Search recipes**: Predefined search patterns for common query types
- **Custom entity search**: Query by entity type, relationship, temporal range
- **Sub-second latency**: Designed for real-time agent use

#### Temporal Handling ⭐

This is Graphiti's crown jewel:
- **Bi-temporal model**: Every fact has both event_time and ingestion_time
- **Temporal edge invalidation**: Contradictions handled by timestamping edges with valid_from/valid_to
- **Point-in-time queries**: "What did we know about X as of date Y?"
- **Historical accuracy**: No information is ever truly deleted — old states are accessible
- **Up to 18.5% accuracy improvement** over baselines on temporal reasoning tasks (LongMemEval)
- **90% latency reduction** compared to baseline implementations

#### Strengths

- **Best-in-class temporal reasoning**: Bi-temporal model is unique and powerful
- **Production-proven**: Powers Zep's commercial platform
- **Real-time incremental updates**: No batch recomputation needed
- **Flexible ontology**: Pydantic-based entity definitions
- **Multiple graph DB backends**: Neo4j, FalkorDB, Kuzu, Neptune
- **MCP server**: Ready for Claude/Cursor integration
- **Contradiction handling**: Elegant temporal invalidation instead of destructive updates
- **Sub-second queries**: Optimized for agent use

#### Weaknesses/Gaps

- **Requires graph DB**: Neo4j is heavyweight; simpler alternatives (FalkorDB, Kuzu) exist but are less mature
- **LLM-dependent extraction**: Entity/relationship extraction quality depends on the model
- **No background reasoning**: Doesn't generate new insights — just stores what it extracts
- **Rate limit sensitivity**: High concurrency design can hit LLM rate limits
- **Complexity**: Running a graph DB adds operational burden
- **No user management built in**: Must build your own user/session management around it

#### Novel Ideas Worth Stealing

1. **Bi-temporal data model** — tracking event_time AND ingestion_time separately is essential for temporal queries
2. **Temporal edge invalidation** — don't delete old facts, just timestamp them as no-longer-valid
3. **Episode-based ingestion** — treating data as discrete episodes rather than continuous streams
4. **Graph distance reranking** — using proximity in the knowledge graph to rerank search results
5. **Custom entity definitions via Pydantic** — developer-defined ontology, not one-size-fits-all
6. **Hybrid search (semantic + keyword + graph)** — three retrieval modalities combined

---

## 2. Platform & Product Analyses

---

### 2.1 Graphlit

**Website**: https://www.graphlit.com  
**Docs**: https://docs.graphlit.dev

#### What It Does

Graphlit is a **complete context layer platform** for AI agents — one API for ingestion, extraction, storage, and retrieval. It positions itself as an alternative to assembling 7+ services (vector DB, document parsers, entity extraction, embedding models, storage, search, OAuth connectors).

#### Architecture / How It Works

- **30+ data feed connectors**: Slack, Gmail, GitHub, S3, RSS with automatic sync (polling every 30 seconds to hours)
- **Multi-format processing**: Audio (transcription + diarization), video, documents (Vision OCR), web crawling
- **Knowledge graph**: Schema.org-based entities + relationships with temporal context
- **Hybrid search**: Vector + graph + keyword with advanced filtering (geo-spatial, image similarity, entity-based, temporal, boolean)
- **Workflows**: Customizable multi-stage pipelines (preparation + extraction stages)
- **Per-user isolation**: `userId` parameter scopes all data
- **SDKs**: TypeScript, Python, C#

#### Memory Model

- **Semantic**: Knowledge graph with entities and relationships
- **Episodic**: Conversation sessions with RAG
- **Content-centric**: Organizes around ingested content objects

#### Storage

- Fully managed — vector + graph + keyword indices
- Per-user data isolation

#### Strengths

- **Complete platform**: Replaces 7+ services with one API
- **Data ingestion**: 30+ connectors with automatic sync — far beyond what any open-source tool offers
- **Multi-modal**: Audio, video, images, documents, web
- **Schema.org knowledge graph**: Standards-based entity extraction
- **Production-ready**: Per-user isolation, collections, specifications

#### Weaknesses/Gaps

- **Closed platform**: No self-hosting, vendor lock-in
- **Black box**: Can't inspect or modify the extraction/indexing pipeline internals
- **Cost**: Managed platform pricing not transparent
- **Limited memory reasoning**: Focus is on content extraction and retrieval, not deep reasoning

#### Novel Ideas Worth Stealing

1. **30+ automatic feed connectors with sync** — the ingestion problem is massively underserved in open-source
2. **Schema.org entity extraction** — using a standard ontology rather than custom schemas
3. **Multi-stage extraction workflows** — preparation → extraction pipeline pattern
4. **Per-user knowledge graph isolation** — each user gets their own graph context

---

### 2.2 SuperMemory

**Website**: https://supermemory.ai  
**Docs**: https://docs.supermemory.ai

#### What It Does

SuperMemory is a **memory and context infrastructure API** for AI agents, combining long-term/short-term memory, content extraction, connectors, and managed RAG in a single platform. Claims state-of-the-art performance on LongMemEval and LoCoMo benchmarks.

#### Architecture / How It Works

1. **Ingestion**: Send text, files, chats, videos
2. **Intelligent indexing**: Builds a "semantic understanding graph" per entity (user, document, project, org)
3. **Three retrieval modes**:
   - **Memory API**: Learned user context — extracted facts that evolve in real-time
   - **User Profiles**: Static + dynamic facts about the user (always-know vs episodic)
   - **RAG**: Advanced semantic search with metadata filtering and contextual chunking

**User Profiles** distinguish between:
- **Static**: Information the agent should always know (name, preferences, role)
- **Dynamic**: Episodic information from recent conversations

#### Memory Model

- **Semantic**: Graph-based memory with entity relationships
- **Episodic**: Dynamic facts from recent conversations
- **User profiles**: Combination of static identity and dynamic state

#### Strengths

- **User profiles (static + dynamic)** — smart distinction between always-relevant and situational context
- **Graph-based indexing with sub-300ms recall**
- **Contextual chunking** — content-aware, not static-size
- **Multi-modal**: Text, files, videos

#### Weaknesses/Gaps

- **Managed service only**: No open-source self-hosting
- **Limited documentation**: Docs are thin compared to competitors
- **Early stage**: Less proven than Mem0 or Graphiti

#### Novel Ideas Worth Stealing

1. **Static vs Dynamic user profiles** — a developer-configurable split between persistent identity and episodic context
2. **Per-entity semantic understanding graph** — building separate graphs per user/project/org
3. **Contextual chunking** — adapting chunk strategy to content type

---

## 3. Architecture Deep-Dives

---

### 3.1 Google ADK Context Engineering

**Source**: https://developers.googleblog.com/en/architecting-efficient-context-aware-multi-agent-framework-for-production/

#### Core Thesis

> "Context is a compiled view over a richer stateful system."

This is the most important architectural insight in the entire research. Google's Agent Development Kit (ADK) treats context not as a string buffer but as a **compiled output** of a processing pipeline.

#### The Tiered Model

```
Working Context  →  What the LLM sees THIS call (ephemeral, recomputed)
    ↑ compiled from
Session          →  Durable log of all events (user msgs, agent replies, tool calls)
    ↑ enriched by
Memory           →  Long-lived searchable knowledge (cross-session)
    ↑ references
Artifacts        →  Large binary/text data (files, logs, images) — addressed by name, not pasted
```

#### Key Design Principles

1. **Separate storage from presentation**: Session (ground truth) vs Working Context (derived view). Evolve them independently.
2. **Explicit transformations**: Context built through named, ordered processors — not ad-hoc string concatenation. Observable and testable.
3. **Scope by default**: Every model call sees the MINIMUM context required. Agents reach for more via tools.

#### Context Processing Pipeline

LLM Flows maintain ordered lists of **processors**:
- **Selection**: Filter the event stream to drop irrelevant events
- **Transformation**: Flatten remaining events into Content objects with correct roles
- **Injection**: Write formatted history into the LLM request

#### Context Compaction

When a threshold is reached (e.g., number of invocations):
- LLM summarizes older events over a sliding window
- Summary written back to Session as a "compaction" event
- Original detailed events can be pruned
- Benefits cascade: session stays manageable, contents processor works on already-compacted data

#### Artifacts as Handles

Large data (CSVs, PDFs, API responses) stored as **named, versioned objects**:
- Default: agents see only a lightweight reference (name + summary)
- On-demand: `LoadArtifactsTool` temporarily loads content into working context
- **Ephemeral expansion**: After the task, artifact is offloaded from working context

#### Memory: Searchable, Not Pinned

Two retrieval patterns:
- **Reactive recall**: Agent recognizes knowledge gap, explicitly calls `load_memory_tool`
- **Proactive recall**: Pre-processor runs similarity search on latest input, injects relevant snippets before model invocation

#### Multi-Agent Context Scoping

Two patterns:
- **Agents as Tools**: Root agent calls sub-agent with focused prompt, gets result. Callee sees only the specific input.
- **Agent Transfer**: Root agent hands off entire conversation to a specialist. Context is scoped — maybe just latest query + one artifact.

#### Ideas Worth Stealing

1. **"Context as a compiled view"** — the foundational mental model
2. **Processor pipeline** — named, ordered, composable context transformations
3. **Compaction with sliding window** — summarize old events, keep recent ones detailed
4. **Artifacts as handles** — reference large data by name, load on-demand, offload after use
5. **Reactive vs proactive memory recall** — agent-directed vs system-injected
6. **Static prefixes for cache optimization** — keep system instructions stable for inference caching
7. **Minimum context by default** — agents opt-in to more, not opt-out of noise

---

### 3.2 Thinking Machines — Tinker / LoRA Infrastructure

**Source**: https://thinkingmachines.ai/tinker/, https://thinkingmachines.ai/blog/lora/

#### What It Is

Tinker is a **training API for researchers** that handles GPU infrastructure while exposing low-level primitives (`forward_backward`, `optim_step`, `sample`, `save_state`). It uses LoRA for efficient fine-tuning.

#### Key Research Findings ("LoRA Without Regret")

1. **LoRA matches FullFT** for small-to-medium instruction-tuning and reasoning datasets when done right
2. **Apply LoRA to ALL layers** — especially MLP/MoE, not just attention (contrary to original paper)
3. **Batch size sensitivity**: LoRA pays a larger penalty for large batch sizes than FullFT
4. **Optimal LR is 10x higher** for LoRA than FullFT
5. **RL requires very low capacity** — even small LoRA ranks work for reinforcement learning

#### Relevance to Our System

- **Custom model training**: If we build custom reasoning models (like Honcho's Neuromancer XR), LoRA via Tinker could be the infrastructure
- **Multi-tenant serving**: One base model, many LoRA adapters for different users/domains
- **Cost efficiency**: LoRA + managed infrastructure = accessible post-training
- **Models available**: Llama, Qwen, DeepSeek, GPT-OSS series

#### Novel Ideas

1. **LoRA for per-user personalization adapters** — each user gets a tiny fine-tuned adapter
2. **Four-primitive API** (forward_backward, optim_step, sample, save_state) — clean abstraction for training
3. **Multi-tenant serving** — one base model, many adapters in memory simultaneously

---

## 4. Research Topics

---

### 4.1 Temporal GNNs for Link Prediction in Knowledge Graphs

**Key Sources**: 
- arXiv:2403.04782 — "A Survey on Temporal Knowledge Graph: Representation Learning and Applications"
- arXiv:2502.21185 — "A Survey of Link Prediction in Temporal Networks"
- IJCAI 2023 — "Temporal Knowledge Graph Completion: A Survey"

#### Overview

Temporal Knowledge Graphs (TKGs) extend static KGs by associating facts with timestamps: (subject, relation, object, time). The link prediction task: given a partial temporal fact, predict the missing element.

#### Key Approaches

1. **Timestamp-based methods**: Embed temporal information directly
   - **TTransE, HyTE**: Extend TransE with time-aware projections
   - **TNTComplEx, DE-SimplE**: Extend tensor decomposition with temporal components
   
2. **Sequence-based methods**: Model temporal evolution as sequences
   - **RE-NET**: Uses recurrent networks to model event sequences per entity
   - **CyGNet**: Copy-generation network that predicts based on historical repetition patterns
   - **Know-Evolve**: Temporal point processes for continuous-time modeling

3. **Graph Neural Network methods**:
   - **RE-GCN**: Recurrent graph convolution over temporal snapshots
   - **GHNN (Graph Hawkes Neural Network)**: Uses Hawkes process to capture how past facts influence future facts
   - **EvoKG**: Evolution-aware graph neural networks

4. **Continuous-time models**:
   - **TGAT (Temporal Graph Attention)**: Attention over temporal neighborhoods
   - **TGN (Temporal Graph Networks)**: Memory modules that maintain state per node, updated on each interaction
   - **DyRep**: Dynamic representation learning for temporal interactions

#### Relevance to Our System

- **TGN's per-node memory modules** are directly analogous to Honcho's per-peer representations — but learned, not rule-based
- **Hawkes process modeling** could predict what memories will be needed based on temporal patterns
- **Temporal attention** could weight recent interactions more heavily in retrieval
- **Link prediction** could suggest new connections between entities before they're explicitly stated

#### Key Insight

> The most promising direction is **continuous-time temporal GNNs** (TGN, TGAT) that maintain per-node state and update it incrementally — this maps perfectly to a personal intelligence system where entities evolve over time.

---

### 4.2 Dynamic / Context-Based Chunking

**Key Sources**:
- ACL 2025 — "The Chunking Paradigm: Recursive Semantic for RAG Optimization"
- Weaviate blog — "Chunking Strategies for RAG"
- Firecrawl — "Best Chunking Strategies for RAG in 2025"
- Springer 2025 — "Max-Min semantic chunking of documents for RAG"

#### The Problem

Static chunking (fixed-size splits) ignores content structure, fragments semantic units, and creates up to a **9% gap in recall** vs optimal approaches.

#### Chunking Strategies (Ranked by Sophistication)

1. **Size-based** (character/token count): Simplest, worst quality. Ignores all structure.
2. **Sentence-based**: Respects sentence boundaries. Better, still blind to topic shifts.
3. **Recursive character splitting**: Tries separators in order (paragraphs → lines → sentences → words). **Best default for 80% of use cases.** 88-89% recall with 400-token chunks.
4. **Page-level**: Highest accuracy in NVIDIA 2024 benchmarks for PDFs. Preserves full page context.
5. **Semantic chunking**: Generates embeddings per sentence, splits at points of low similarity. Content-aware. Costs API calls.
6. **LLM-based chunking**: LLM analyzes document structure to determine optimal splits. Highest quality, highest cost.
7. **Adaptive chunking**: Dynamically adjusts chunk size and overlap based on content density, structure, and type.

#### Key Research Findings

- **Chunk size sweet spot**: 100-600 tokens depending on LLM and use case
- **Overlap**: 10-20% overlap recommended to preserve cross-boundary context
- **Recursive semantic chunking** (RSC) outperforms all other approaches in ACL 2025 paper across multiple datasets
- **Max-Min semantic chunking**: Uses both maximum and minimum similarity thresholds to ensure chunks are coherent but not too granular
- **Contextual chunking** (SuperMemory's approach): Adapts strategy to content type (code vs prose vs tables)

#### Relevance to Our System

For a personal intelligence system ingesting diverse data (conversations, documents, notes, web pages), we need:
- **Content-type-aware chunking**: Different strategies for conversations vs documents vs code
- **Semantic boundary detection**: Don't split mid-thought
- **Metadata preservation**: Keep source, timestamp, author through the chunking process
- **Adaptive sizing**: Dense technical content → smaller chunks; narrative → larger chunks

---

### 4.3 Multi-Hop RAG over Knowledge Graphs (GraphRAG)

**Key Sources**:
- arXiv:2408.08921 — "Graph Retrieval-Augmented Generation: A Survey"
- ACL Findings 2025 — "GRAG: Graph Retrieval-Augmented Generation"
- arXiv:2506.00054 — "Retrieval-Augmented Generation: A Comprehensive Survey"

#### Overview

GraphRAG addresses a fundamental limitation of vector-only RAG: it can't follow chains of reasoning across multiple documents. Traditional RAG retrieves individual chunks; GraphRAG retrieves **subgraphs** that capture multi-hop relationships.

#### The GraphRAG Workflow

```
Graph-Based Indexing → Graph-Guided Retrieval → Graph-Enhanced Generation
```

1. **Graph-Based Indexing**: 
   - Entity extraction from documents
   - Relationship extraction between entities
   - Community detection (clusters of related entities)
   - Hierarchical summarization of communities

2. **Graph-Guided Retrieval**:
   - **Subgraph retrieval**: Given a query, find the relevant subgraph
   - **Path retrieval**: Find paths connecting query entities
   - **Community-based**: Retrieve at the community level for broad queries
   - **Hybrid**: Combine graph traversal with vector similarity

3. **Graph-Enhanced Generation**:
   - Retrieved subgraph/paths fed to LLM as structured context
   - Entity-relationship structure provides better reasoning scaffold than raw text
   - **Dual-Pathway KG-RAG**: Parallel structured (graph) + unstructured (corpus) retrieval — reduces hallucinations by 18% in biomedical QA

#### Multi-Hop Reasoning Patterns

- **Bridge entities**: Entity A → Entity B → Entity C (A connects to C through B)
- **Comparison**: Retrieve facts about entities A and B from different parts of the graph
- **Temporal chains**: Event sequence tracking through graph edges
- **Constraint satisfaction**: Find entities matching multiple relationship constraints

#### Relevance to Our System

A personal intelligence system must answer questions like:
- "Who introduced me to the person working on that project I mentioned last week?" (3-hop)
- "What's the connection between my meeting notes and that research paper?" (bridge entity)
- "How has my opinion on X evolved?" (temporal chain)

These require **graph structure** — vector similarity alone can't do it.

---

### 4.4 Re-Ranking Approaches for RAG

**Key Sources**:
- NVIDIA Blog — "Enhancing RAG Pipelines with Re-Ranking"
- Medium — "Re-Ranking Mechanisms in RAG Pipelines"
- Elastic Docs — "Ranking and reranking"

#### Why Re-Ranking Matters

Initial retrieval (vector similarity or BM25) is fast but imprecise. Re-ranking adds a precision layer:
1. **Broad retrieval**: Get top-k candidates (k=20-100) via fast methods
2. **Precise re-ranking**: Use a cross-encoder or specialized model to score query-document relevance
3. **Final selection**: Take top-n (n=3-10) highest-scoring results for generation

#### Re-Ranking Methods

1. **Cross-Encoders**: Feed (query, document) pair through a transformer. Most accurate but slowest. Models: BGE-reranker, Cohere Rerank, NVIDIA NeMo Retriever.

2. **Late-Interaction Models** (ColBERT-style): Encode query and document separately, then compute fine-grained token-level similarity. Faster than cross-encoders, nearly as accurate.

3. **LLM-as-Reranker**: Use an LLM to score relevance. Flexible but expensive.

4. **Reciprocal Rank Fusion (RRF)**: Combine rankings from multiple retrieval sources without a model. Simple and effective for hybrid search.

5. **Graph Distance Reranking** (Graphiti's approach): Re-rank by proximity in the knowledge graph. Novel and powerful for entity-centric queries.

#### AllenAI / Semantic Scholar Approaches

AllenAI's retrieval research has pioneered several approaches relevant to our system:
- **SPECTER / SPECTER2**: Document-level embeddings that capture citation relationships
- **Re-query strategies**: Reformulate the query based on initial results, then re-retrieve
- **Aspect-based retrieval**: Different embeddings for different aspects of the same document (background, method, result)
- **Multi-vector representations**: Represent documents with multiple vectors for different facets

#### Relevance to Our System

For a personal intelligence system:
- **Two-stage retrieval** (broad → precise) is essential for large knowledge bases
- **Graph distance reranking** should be combined with semantic reranking
- **Re-query**: If initial retrieval is poor, reformulate and try again
- **Aspect-based**: Different queries need different facets of the same memory (what happened vs how I felt vs what I learned)

---

## 5. Comparative Matrix

| Feature | Honcho | Basic Memory | Jean Memory | Mem0 | Graphiti | Graphlit | SuperMemory |
|---|---|---|---|---|---|---|---|
| **Memory Type** | Reasoning-based | File-based KG | Orchestrated | Vector + Graph | Temporal KG | Content + KG | Graph + RAG |
| **Reasoning** | ⭐ Formal logic | ❌ None | ⚡ Via orchestrator | ⚡ Extraction only | ⚡ LLM extraction | ⚡ Extraction | ⚡ Extraction |
| **Knowledge Graph** | ❌ Flat vectors | ⚡ Wiki-links | ✅ Via Graphiti | ✅ Optional (Mem0ᵍ) | ⭐ Full temporal KG | ✅ Schema.org | ✅ Semantic graph |
| **Temporal** | ⚡ Sessions | ❌ File dates | ⚡ Via Graphiti | ⚡ Timestamps | ⭐ Bi-temporal | ✅ Temporal context | ⚡ Evolving |
| **Local-first** | ❌ Server | ⭐ Markdown files | ❌ Cloud/Docker | ⚡ Can self-host | ⚡ Can self-host | ❌ Cloud only | ❌ Cloud only |
| **Token Efficiency** | ⭐ 5-11% of context | N/A | N/A | ⭐ 80% reduction | ✅ Sub-second | N/A | ⚡ Sub-300ms |
| **Setup Complexity** | High | Low | Medium | Low | High | Low (managed) | Low (managed) |
| **MCP Support** | ❌ | ⭐ Native | ❌ | ❌ | ✅ MCP server | ✅ MCP server | ❌ |
| **Benchmark SOTA** | ⭐ BEAM 10M | N/A | N/A | ⭐ LOCOMO | ⭐ DMR, LongMemEval | N/A | Claims SOTA |
| **Multi-entity** | ⭐ Peer paradigm | ❌ Single-user | ⚡ User-scoped | ⚡ User/Session/Agent | ⚡ Groups | ✅ Per-user | ✅ Per-entity |
| **Open Source** | ✅ | ✅ AGPL | ⚡ Partial | ✅ Apache 2.0 | ✅ | ❌ | ❌ |

**Legend**: ⭐ = best-in-class, ✅ = good, ⚡ = partial/basic, ❌ = absent

---

## 6. Key Architectural Insights for Our System

### Insight 1: Context Engineering > RAG

The Google ADK paper crystallizes the shift: the problem isn't retrieval, it's **context compilation**. Our system should have a context pipeline with explicit, named processors — not ad-hoc prompt construction.

### Insight 2: Memory Needs Reasoning, Not Just Storage

Honcho proves that reasoning over stored data dramatically outperforms simple retrieval (90.4% vs 62.6% on LongMem S). Our system should extract latent information through inference, not just store what was explicitly said.

### Insight 3: Temporal is Essential and Hard

Every system struggles with temporal reasoning. Graphiti's bi-temporal model is the best approach found — track both when things happened and when the system learned about them. This is non-negotiable for a personal intelligence system.

### Insight 4: Graph + Vector is Better Than Either Alone

Mem0ᵍ, Graphiti, and GraphRAG research all show that combining vector similarity with graph traversal significantly improves retrieval. The graph captures relationships that embeddings miss; embeddings capture semantic similarity that graphs miss.

### Insight 5: Multi-Entity Modeling is Underexplored

Only Honcho models both users and agents as first-class entities with their own evolving representations. A personal intelligence system should model the person, their contacts, their projects, their AI agents — all as interconnected entities.

### Insight 6: The Ingestion Problem is Massively Underserved

Graphlit's 30+ connectors highlight a gap: most open-source tools assume data is already clean and ready. A real personal intelligence system needs to ingest from email, chat, documents, web, calendar, code repos, etc.

### Insight 7: Token Efficiency is a First-Class Concern

Both Honcho (5-11% of context tokens) and Mem0 (80% token reduction) prove that intelligent compression beats brute-force context stuffing. Our system should optimize for information density per token.

---

## 7. Novel Ideas Worth Stealing

### Tier 1: Build Into Our Architecture

| Idea | Source | Priority |
|---|---|---|
| Context as compiled view + processor pipeline | Google ADK | **P0** — foundational architecture |
| Bi-temporal data model (event_time + ingestion_time) | Graphiti | **P0** — essential for temporal queries |
| Formal logic reasoning chain (explicit → deductive → inductive → abductive) | Honcho | **P0** — our core differentiator |
| Graph + vector hybrid retrieval | Graphiti, Mem0ᵍ | **P0** — required for multi-hop queries |
| Memory compression into dense representations | Mem0 | **P0** — token efficiency |
| Entity-centric model (people, projects, ideas as first-class entities) | Honcho | **P0** — personal intelligence core |

### Tier 2: Implement Early

| Idea | Source | Priority |
|---|---|---|
| "Dreaming" — background reasoning over existing conclusions | Honcho | **P1** — generates novel insights |
| Static vs dynamic user profiles | SuperMemory | **P1** — clean retrieval separation |
| Temporal edge invalidation (don't delete, timestamp) | Graphiti | **P1** — elegant contradiction handling |
| Perspective-based segmentation (different views per entity) | Honcho | **P1** — multi-context support |
| Graph distance reranking | Graphiti | **P1** — better retrieval relevance |
| Content-type-aware chunking | Research, SuperMemory | **P1** — diverse data ingestion |

### Tier 3: Explore Later

| Idea | Source | Priority |
|---|---|---|
| Per-user LoRA adapters for personalization | Thinking Machines | **P2** — personalized model behavior |
| MCP-first tool integration | Basic Memory | **P2** — ecosystem compatibility |
| Orchestration layer that decides retrieval strategy | Jean Memory | **P2** — intelligent routing |
| Reactive vs proactive memory recall | Google ADK | **P2** — when to inject vs when to fetch |
| TGN-style per-node memory modules | Temporal GNN research | **P2** — learned entity representations |
| Aspect-based multi-vector representations | AllenAI research | **P2** — different facets of same memory |

### Tier 4: Ambitious / Research

| Idea | Source | Priority |
|---|---|---|
| Continuous-time temporal GNN over personal knowledge graph | Research | **P3** — learned temporal dynamics |
| Hawkes process for predictive memory pre-loading | GHNN paper | **P3** — anticipate what memories will be needed |
| Markdown as universal interchange format | Basic Memory | **P3** — human-readable knowledge |
| Custom fine-tuned reasoning models (like Neuromancer XR) | Honcho | **P3** — cost-efficient reasoning at scale |

---

## Appendix: Sources Consulted

### Repositories
1. https://github.com/plastic-labs/honcho — Honcho v3
2. https://github.com/basicmachines-co/basic-memory — Basic Memory
3. https://github.com/jean-technologies/jean-memory — Jean Memory (formerly Your Memory)
4. https://github.com/mem0ai/mem0 — Mem0
5. https://github.com/getzep/graphiti — Graphiti

### Documentation
- https://docs.honcho.dev — Honcho docs (Overview, Reasoning, Representations)
- https://docs.mem0.ai — Mem0 docs (Platform, Graph Memory)
- https://docs.graphlit.dev — Graphlit docs
- https://docs.supermemory.ai — SuperMemory docs
- https://docs.basicmemory.com — Basic Memory docs

### Papers & Research
- arXiv:2501.13956 — "Zep: A Temporal Knowledge Graph Architecture for Agent Memory"
- arXiv:2504.19413 — "Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory"
- arXiv:2408.08921 — "Graph Retrieval-Augmented Generation: A Survey"
- arXiv:2403.04782 — "A Survey on Temporal Knowledge Graph: Representation Learning and Applications"
- arXiv:2502.21185 — "A Survey of Link Prediction in Temporal Networks"
- arXiv:2510.27246 — "BEAM: BEyond A Million Tokens" benchmark
- ACL 2025 — "The Chunking Paradigm: Recursive Semantic for RAG Optimization"
- IJCAI 2023 — "Temporal Knowledge Graph Completion: A Survey"

### Blog Posts & Articles
- https://blog.plasticlabs.ai/research/Benchmarking-Honcho — Honcho benchmarks
- https://developers.googleblog.com/en/architecting-efficient-context-aware-multi-agent-framework-for-production/ — Google ADK context engineering
- https://thinkingmachines.ai/blog/lora/ — "LoRA Without Regret"
- https://developer.nvidia.com/blog/enhancing-rag-pipelines-with-re-ranking/ — NVIDIA re-ranking
- https://www.firecrawl.dev/blog/best-chunking-strategies-rag-2025 — Chunking strategies
- https://weaviate.io/blog/chunking-strategies-for-rag — Weaviate chunking guide

### Websites
- https://mem0.ai — Mem0 platform
- https://supermemory.ai — SuperMemory
- https://www.graphlit.com — Graphlit
- https://thinkingmachines.ai/tinker/ — Tinker training API
