# Temporal Reasoning Is the Unsolved Problem in Agent Memory

*Every memory system in production today stores facts. None of them model how those facts evolve — and that's the bottleneck.*

---

## The Gap

Agent memory has a temporal reasoning problem. Not a retrieval problem, not an embedding problem, not a context window problem — a *reasoning about change over time* problem.

Consider a concrete failure mode. An agent with access to a user's full conversation history is asked: *"How has my approach to this project evolved since we started?"* The system retrieves the five most semantically similar chunks to the query. It returns a pastiche of excerpts — maybe something from last month, maybe something from six months ago — with no causal ordering, no sense of what changed when, no model of the trajectory. The answer is technically grounded in source material but is fundamentally useless for the question being asked.

This isn't a cherry-picked example. [LongMemEval](https://arxiv.org/abs/2410.10813) (Wu et al., ICLR 2025) demonstrates that every major memory system — including full-context baselines with frontier models — degrades severely on temporal reasoning and knowledge-update questions compared to single-session fact retrieval. [LoCoMo](https://arxiv.org/abs/2402.17753) (Maharana et al., ACL 2024) reports the same pattern: temporal reasoning is consistently the weakest category across all evaluated systems. [Test of Time](https://arxiv.org/abs/2406.09170) (Fatemi et al., ICLR 2025) goes further, showing that LLMs largely *fake* temporal reasoning through parametric knowledge — when entities are anonymized to prevent shortcutting, accuracy on temporal ordering and duration tasks collapses.

The pattern is clear: agent memory systems have converged on a design that handles *what* but not *when* or *how things changed*. This is not an incremental gap. It is a structural one.

This post introduces **PIE** (Personal Intelligence Engine), an open-source framework for building temporal agent memory systems. PIE models memory not as a fact store or embedding index but as a **temporal state machine** — a graph of entities, typed state transitions, semantic time anchors, and procedural patterns extracted from transition sequences. The core claim: if you want an agent to reason about change over time, you need to give it a data structure that explicitly represents change over time. Embeddings don't do this. Timestamps on facts don't do this. State transition chains do.

---

## Why Current Approaches Fail at Temporal Reasoning

The agent memory landscape has matured rapidly since 2023. But every major system has converged on one of a few architectures, and all of them share the same blind spot.

### Mem0: The Flat Fact Store

[Mem0](https://github.com/mem0ai/mem0) (Chhikara et al., [arXiv 2025](https://arxiv.org/abs/2504.19413)) extracts salient facts from conversations and stores them as key-value pairs with vector embeddings for retrieval. It solves a real problem — basic personalization ("user prefers dark mode," "user's name is Alice") — and does it well enough that it's the most popular open-source memory layer by star count.

The architectural limitation is structural: Mem0 maintains no relationships between facts and no model of how facts change. When a user says "I just switched to PostgreSQL" after previously mentioning "I use MySQL," Mem0's `v0.1` architecture either overwrites the old fact or maintains both with no explicit conflict signal. The [Mem0^g variant](https://arxiv.org/abs/2504.19413) adds a graph layer, but the graph captures entity relationships at a single point in time — there is no transition history, no typed change detection, and no mechanism for reasoning about how the graph itself evolves.

For a query like *"How has my tech stack changed this year?"* Mem0 can return current facts about the tech stack. It cannot reconstruct the trajectory.

### Zep/Graphiti: The Temporal Knowledge Graph (Closest Prior Work)

[Graphiti](https://github.com/getzep/graphiti), the knowledge graph engine underlying [Zep](https://arxiv.org/abs/2501.13956) (Rasmussen et al., 2025), is architecturally the closest system to PIE and represents the current state of the art for graph-based agent memory. It organizes knowledge into three subgraph layers — episodic (raw conversation data), semantic entity (extracted entities and relationships), and community (clusters of related entities) — with a bi-temporal model that separately tracks event time and ingestion time.

Graphiti's results are strong: 94.8% on the [DMR benchmark](https://blog.getzep.com/state-of-the-art-agent-memory/) (vs. MemGPT's 93.4%), and up to 18.5% accuracy improvement over baselines on [LongMemEval](https://arxiv.org/abs/2501.13956) with 90% latency reduction.

Where Graphiti stops short is in how it *models* temporal change. Edges carry timestamps and can be invalidated when new information arrives, but there is no explicit state transition model — no typed transitions (creation → update → contradiction → resolution → archival), no transition chains that capture the full evolution of an entity, and no mechanism for extracting *patterns* across entity lifecycles. The bi-temporal model tracks *when facts were known* and *when they were recorded*, but not *what kind of change occurred* or *what triggered it*.

For a query like *"What patterns do I follow when evaluating new technologies?"*, Graphiti can retrieve entities related to technology evaluation. It cannot identify recurring sequences across those entities' lifecycles, because it doesn't model lifecycles as first-class objects.

### MemGPT/Letta: The OS Memory Metaphor

[MemGPT](https://arxiv.org/abs/2310.08560) (Packer et al., 2023) introduced the influential analogy of LLM context as virtual memory, with paging between in-context and out-of-context storage. The insight — that agents need to self-manage what's in their working memory — is correct and has been widely adopted.

But MemGPT operates at the session level. It manages what's in context *right now*, not how knowledge accumulates over time. It is a context management system, not a memory architecture. The [Letta](https://github.com/letta-ai/letta) framework extends MemGPT with persistent storage but inherits the same structural limitation: no graph, no state transitions, no temporal model beyond timestamps.

### Honcho: The User Psychology Model

[Honcho](https://github.com/plastic-labs/honcho) (Plastic Labs) takes a fundamentally different approach — instead of storing facts or building knowledge graphs, it models the user's psychology through dialectical representations that predict behavior and preferences. This is a legitimate and complementary capability: understanding *who the user is* is different from understanding *what the user's world looks like*.

Honcho doesn't attempt temporal state tracking, entity resolution, or procedural memory extraction. It answers "what kind of response does this user want?" not "what happened to this user's project last quarter?" The approaches are orthogonal.

### Generative Agents: Memory Without Structure

The [Generative Agents](https://arxiv.org/abs/2304.03442) architecture (Park et al., 2023) — the "Smallville" paper — established the retrieval-based memory pattern that most subsequent systems build on: store observations as natural language with timestamps, retrieve by a weighted combination of recency, relevance (embedding similarity), and importance. Observations are periodically synthesized into "reflections" — higher-level abstractions.

This architecture has a fundamental limitation for temporal reasoning: observations are atomic and flat. There is no model of how entities persist and change across observations. Two observations — "started Project X" and "Project X launched" — are just two strings in a vector store. The system has no representation of Project X as an entity that underwent a state transition from "started" to "launched," no model of what happened in between, and no ability to compare Project X's trajectory to Project Y's.

### The Common Failure Mode

The shared weakness across all these systems is the same: **they conflate temporal metadata with temporal understanding.**

Adding a `timestamp` field to a fact, an edge, or an observation is temporal *metadata*. It tells you when something was recorded. Temporal *understanding* requires modeling:

- **State transitions**: what changed, from what to what, and what triggered the change
- **Transition types**: was this an incremental update, a contradiction of prior state, a resolution of a conflict, or a creation of something new?
- **Temporal context**: not "2024-03-15T14:30:00Z" but "about 10 months ago, during the user's second internship, around the time they started exploring graph databases"
- **Change patterns**: does this entity change frequently or rarely? Is the change velocity increasing or decreasing? Do entities of this type follow similar lifecycles?
- **Procedural patterns**: across many entities' transition chains, what recurring sequences emerge?

No production system models any of this. PIE does.

---

## A Taxonomy of Agent Memory

Before describing PIE's architecture, it's useful to ground the discussion in cognitive science. Human long-term memory is not a monolith — it's a set of interacting systems, each with different properties and different computational analogues.

[Tulving's](https://psycnet.apa.org/record/1973-08477-007) (1972) episodic–semantic distinction and [Squire's](https://whoville.ucsd.edu/PDFs/206_Squire_etal_AnnuRevPsych_1993.pdf) (1992) declarative–nondeclarative taxonomy provide the standard framework. Mapping this to agent memory systems:

| Memory Type | Cognitive Science Origin | What It Stores | Computational Analogue | Ideal Data Structure | Current System Support |
|---|---|---|---|---|---|
| **Episodic** | Tulving (1972) | Specific experiences with spatiotemporal context | Full conversation logs, interaction records | Indexed document store, chunked + embedded | ✅ Most RAG systems, MemGPT |
| **Semantic** | Tulving (1972) | Decontextualized facts and knowledge | "User lives in NYC," "User prefers Python" | Key-value store, property graph | ✅ Mem0, ChatGPT Memory, Graphiti |
| **Procedural** | Squire (1992) | How to do things; behavioral patterns | "User evaluates tools by researching 3–5 options, deep-diving one, then committing" | State machine, pattern graph | ❌ No production system. [ExpeL](https://arxiv.org/abs/2308.10144) (Zhao et al., 2023) extracts task-level procedures but not cross-entity lifecycle patterns. |
| **Autobiographical/Temporal** | Conway & Pleydell-Pearce (2000) | Life narrative; when things happened and how they connect | Entity evolution chains anchored to life periods | Temporal knowledge graph with typed transitions | ⚠️ Graphiti partially (timestamps on edges). No system models typed transition chains. |
| **Working** | Baddeley (1992) | Currently active information | What's in the context window right now | Context window + paging | ✅ MemGPT/Letta |
| **Prospective** | Einstein & McDaniel (1990) | Intentions and future plans | Scheduled actions, pending follow-ups | Task queue, agenda | ⚠️ Various assistants, none robust |

The pattern is stark: the memory types that are well-supported (episodic, semantic, working) are the ones with obvious computational mappings — store text, store key-value pairs, manage a buffer. The memory types that are unsupported (procedural, autobiographical/temporal) are the ones that require *modeling change and pattern over time*. This isn't a coincidence. It reflects a genuine architectural gap, not just an engineering oversight.

PIE's contribution is on the right side of this table: procedural memory and temporal/autobiographical memory, implemented as typed state transition chains over a temporal knowledge graph.

---

## The Temporal Reasoning Problem: A Formal View

Why do embeddings fail for temporal queries? The answer is precise and structural.

### Why Embeddings Can't Encode Time

Text embeddings encode **semantic similarity** — the degree to which two pieces of text are "about the same thing." The query *"What was I working on last March?"* gets embedded into a vector that is close to texts containing words like "working," "March," "projects." The embedding has no mechanism to:

1. **Resolve "last March" to an absolute time range** relative to the current date
2. **Filter by temporal window** — cosine similarity has no temporal axis
3. **Distinguish ordering** — "X happened before Y" and "X happened after Y" may have nearly identical embeddings
4. **Model state change** — "Project X is in design" and "Project X launched" are semantically similar (both about Project X), but the retrieval system has no way to represent that the second *supersedes* the first

[Khoj's Timely model](https://blog.khoj.dev/posts/timely_date_aware/) is a notable attempt to build date-awareness into embeddings, but fundamentally, temporal reasoning requires structured data — orderings, intervals, state transitions — that vector spaces are not designed to represent.

### What the Benchmarks Show

Three recent benchmarks quantify this gap with precision:

**[Test of Time](https://arxiv.org/abs/2406.09170)** (Fatemi et al., ICLR 2025) isolates temporal reasoning from confounds by anonymizing all entities (E1, E2, R1, R2). Key findings:
- LLMs perform well on temporal tasks when entities are real (can leverage parametric knowledge) but collapse when entities are anonymized — demonstrating that "temporal reasoning" in standard benchmarks is often just *memorized temporal facts*, not reasoning.
- Performance degrades sharply with graph complexity: as the number of temporal facts increases, accuracy on ordering, duration, and interval queries drops, even for frontier models.
- The order in which temporal facts are presented in the prompt significantly affects accuracy — models are sensitive to *narrative order*, not *logical temporal order*.

This has a direct implication for memory systems: if you retrieve temporal facts and present them to an LLM, the model's ability to reason about their temporal relationships depends heavily on *how you present them* — which means the context compilation layer (how you convert retrieved data into LLM-readable text) is not just formatting; it's a core reasoning bottleneck.

**[LongMemEval](https://arxiv.org/abs/2410.10813)** (Wu et al., ICLR 2025) benchmarks chat assistants on five memory abilities with 500 questions across conversation histories of 115K–1.5M tokens. Results across all tested systems:
- **Information extraction** (fact recall): 60–80% accuracy depending on system and scale
- **Temporal reasoning**: 15–40% lower than fact recall across all systems
- **Knowledge updates** (recognizing that information has changed): consistently the hardest category
- Time-aware indexing (augmenting retrieval with temporal filtering) improves temporal reasoning scores by 7–11%, but still leaves a large gap vs. non-temporal categories

The "knowledge update" finding is particularly relevant: every system struggles to recognize that *information has changed* — that the user's current state contradicts their previous state. This is precisely the problem that explicit state transition modeling solves.

**[LoCoMo](https://arxiv.org/abs/2402.17753)** (Maharana et al., ACL 2024) evaluates very long-term conversational memory (300 turns, up to 35 sessions). Published results on temporal reasoning questions:

| System | Single-hop | Temporal | Multi-hop | Overall |
|--------|-----------|----------|-----------|---------|
| MemMachine | 0.933 | 0.726 | 0.805 | 0.849 |
| Memobase | 0.709 | 0.851 | 0.469 | 0.758 |
| Zep/Graphiti | 0.741 | 0.798 | 0.660 | 0.751 |

*(Scores are LLM-judge accuracy; data from [MemoryBench](https://github.com/supermemoryai/memorybench) and published system reports.)*

The temporal column is consistently one of the weakest — and these benchmarks test relatively simple temporal reasoning (ordering, recency). None of them test the harder cases: contradiction detection, state trajectory reconstruction, or procedural pattern extraction across entity lifecycles.

### Three Specific Failure Modes

To make this concrete, here are three query types that *no current system handles well*, and what they require architecturally:

**1. Trajectory reconstruction:** *"How has my position on X evolved?"*
Requires: a chain of state transitions for the belief entity X, with typed changes (especially contradictions) and causal triggers. Current systems store the latest state (Mem0) or timestamps on edges (Graphiti) but don't maintain the full transition chain with typed change semantics.

**2. Temporal diff:** *"What changed in my projects between Q1 and Q3?"*
Requires: reconstructing world state at two points in time and computing a diff. This needs snapshot reconstruction from transition chains — you have to "replay" the transitions up to each time point. No current system supports world-state snapshots at arbitrary time points.

**3. Cross-entity pattern extraction:** *"Do I follow a pattern when starting new projects?"*
Requires: procedural memory — analyzing state transition sequences across multiple entity lifecycles to identify recurring patterns. This is structurally impossible without explicit transition chains, because there is no "lifecycle" to analyze in a flat fact store or a graph that only tracks current relationships.

---

## PIE's Architecture

PIE is built around a single architectural principle: **the unit of memory is not the fact or the chunk — it's the state transition.**

### Core Data Model

The world model contains three primitive types:

**Entities** are typed, persistent graph nodes representing anything that recurs across conversations — people, projects, beliefs, decisions, tools, concepts, organizations, and temporal periods. Each entity carries a canonical name, aliases (for entity resolution), current state, and vector embedding.

**State Transitions** are the core temporal primitive. Every time an entity's state changes, the previous state is *not overwritten* — instead, a typed transition is recorded:

```
Entity: [project-entity]
Transition Type: contradiction
From State: "mentoring model for students"
To State: "research curriculum platform"
Trigger: "User discussed how mentoring wasn't scaling and proposed curriculum approach"
Source Conversation: [conversation-id]
Timestamp: [unix timestamp]
Period: "freshman year"
Confidence: 0.92
```

The `transition_type` field carries crucial semantic information:
- **creation** — entity first appears
- **update** — state evolves naturally (project gained users, tool version updated)
- **contradiction** — new state conflicts with prior state (belief changed, decision reversed)
- **resolution** — a previously flagged contradiction was resolved with new information
- **archival** — entity marked inactive by the consolidation engine

The transition chain for an entity IS the temporal model. Replaying transitions gives you the entity's full history. Comparing transition chains across entities gives you lifecycle patterns.

**Procedures** are extracted from cross-entity transition patterns during a post-ingestion analysis phase. They represent recurring behavioral sequences — "how the user does things" — grounded in observed state transitions across multiple entity lifecycles:

```
Procedure: Technology Evaluation Pattern
Description: "Evaluates new technologies through broad research, comparison of 3-5 alternatives, deep-dive on one candidate, then rapid commitment"
Evidence: [entity-id-1, entity-id-2, entity-id-3, ...]
Times Observed: 8
Domain: "technology evaluation"
```

Procedures aren't programmed or declared. They emerge from pattern analysis over transition sequences. This is a direct computational analogue of procedural memory — learned behavioral patterns extracted from repeated experience.

### Semantic Temporal Context Compilation

A critical design decision: **LLMs never see raw timestamps.** All temporal data from the graph is compiled into natural language narratives before being passed as context.

The temporal context compiler converts graph data into text that an LLM can reason about:

```
[Entity Name] — first appeared about 14 months ago (during [period]),
  last referenced 2 days ago.
  Changed state 5 times (~0.4x/month).
    • 14 months ago ([period]): Created as [initial state description]
    • 11 months ago ([period]): [transition description]
    • 8 months ago ([period]): [transition description]
      ⚠ This contradicted the previous state.
    • 4 months ago ([period]): [transition description]
    • 2 days ago: [most recent transition]
```

This format gives the LLM: temporal ordering, semantic time anchoring (periods, not dates), change velocity, contradiction flags, causal flow, and recency — all in natural language it's trained to understand.

The hypothesis is specific: this compiled temporal context enables better temporal reasoning than raw timestamps, formatted dates, or relative time expressions. This is testable, and we plan to test it. (See [Hypotheses & Experiments](#hypotheses--experiments) below.)

### Rolling Context Ingestion

PIE processes conversations in **chronological order**, and each conversation is extracted with awareness of the current world model state built from all prior conversations. This is the "rolling context" approach.

For each conversation, the ingestion pipeline:

1. **Detects entity mentions** — an LLM pass identifies which entities in the conversation correspond to existing or new entities in the world model
2. **Retrieves relevant world context** — the current state of mentioned entities, their recent transitions, and their relationships (not the full world model — just the relevant subgraph)
3. **Extracts with context** — a structured extraction LLM call receives the conversation *plus* the relevant world model context, and produces: new entities, state changes to existing entities, new relationships, beliefs, decisions, a significance score, and a period assignment
4. **Resolves entities** — extracted entity mentions are matched to existing graph nodes through a hybrid resolution pipeline (string matching → embedding similarity → LLM verification)
5. **Updates the world model** — the graph is updated with new entities, transitions, and relationships, maintaining full provenance back to source conversations

The key property: each conversation is processed in the context of everything that came before it. When conversation #3,000 mentions "the project," the system knows what projects are currently active and can correctly attribute the reference — even if the project name is never mentioned. This is **activity-based context**, not mention-based retrieval.

This design choice has a cost: ingestion is sequential and slow (each conversation depends on the accumulated state). But it produces dramatically better entity resolution and state tracking than independent, parallelized extraction.

### Entity Resolution: Hybrid String + Embedding + LLM

Entity resolution — the problem of determining that "SRA," "Science Research Academy," "scifair.tech," "the science fair platform," and "that project Pranay and I started" all refer to the same entity — is a critical and under-discussed component of memory systems.

PIE uses a three-stage resolution pipeline:

1. **String matching**: exact match on canonical name or known aliases. Fast, handles the easy cases.
2. **Embedding similarity**: for non-exact matches, compute embedding similarity between the extracted mention and all candidate entities of the same type. Retrieve top-k candidates above a threshold.
3. **LLM verification**: for ambiguous cases (similarity > 0.7 but < 0.95), present the extracted mention and the candidate entity's state to an LLM and ask for a match/no-match decision with reasoning.

Resolved entities have their alias lists updated, creating a feedback loop: each successful resolution improves future resolution by expanding the alias set.

This is more expensive than pure embedding-based resolution (used by Graphiti) but handles the long tail of informal references that embedding similarity alone misses — especially references that are context-dependent ("that thing we discussed last week") rather than semantically similar to the entity name.

### Importance from Graph Structure

Not all entities are equally important, and treating them equally produces a graph dominated by noise. PIE computes importance from graph structure — essentially [PageRank](https://en.wikipedia.org/wiki/PageRank) adapted for personal memory:

- **Connectivity**: how many other entities is this connected to? (highly connected entities are structurally important)
- **Transition count**: how many state changes has this entity undergone? (entities that change are actively relevant)
- **Recency**: when was this entity last referenced? (exponential decay over months)
- **Neighbor importance**: are this entity's neighbors important? (importance propagates through the graph)
- **Access frequency**: has this entity been retrieved in queries? (reinforcement through use)

The weighted combination produces a continuous importance score that drives tiered memory management:
- **Active (high importance)**: full transition history preserved, always surfaced in relevant queries
- **Summarized (medium)**: individual transitions merged into summaries, entity remains in graph
- **Archived (low)**: removed from active graph, stored in cold archive, recoverable but never surfaces

This replaces heuristic importance scoring (message count, keyword matching) with a measure grounded in graph topology. The claim: graph-structural importance correlates more strongly with human judgment of "what matters" than surface-level heuristics.

### Consolidation and Forgetting

Periodic consolidation maintains graph quality as it grows:

1. **Merge**: find entity pairs with high embedding similarity that should be a single entity; merge them (preserving all transitions from both)
2. **Summarize**: for low-importance entities with many transitions, compress the transition chain into a summary
3. **Archive**: move stale, low-importance entities to cold storage
4. **Recompute importance**: after structural changes, recompute all importance scores
5. **Resolve contradictions**: find unresolved contradictions and attempt resolution through LLM analysis
6. **Update period boundaries**: as more data accumulates, refine the boundaries of temporal period nodes

Forgetting is not deletion — it is tiered compression. The graph remains complete in cold storage; active memory is kept focused.

### Dreaming Engine

A background process that runs asynchronously to surface insights not explicit in any single conversation:

- **Micro-dreams** (after each ingestion batch): check recently updated entities for unexpected connections to distant parts of the graph
- **Deep dreams** (daily): full consolidation pass, procedure extraction, belief evolution analysis
- **Mega-dreams** (weekly): cross-domain synthesis using the full world model — what high-level patterns span multiple life areas?

This is inspired by [Generative Agents'](https://arxiv.org/abs/2304.03442) reflection mechanism but operates over structured graph data rather than raw observation text, and produces structured outputs (new relationships, updated procedures) rather than natural language reflections.

### Query Layer (MCP)

PIE exposes its world model through the [Model Context Protocol](https://modelcontextprotocol.io/) (MCP), making it accessible to any LLM client — Claude, GPT, local models, custom agents.

Core query operations:
- **`query_world(query)`**: natural language query → relevant subgraph retrieval → temporal context compilation → LLM-ready text
- **`get_entity_timeline(entity)`**: full evolution history of an entity as compiled temporal narrative
- **`diff_periods(period_a, period_b)`**: what changed in the world model between two time periods
- **`get_procedures(domain)`**: extracted procedural patterns, optionally filtered by domain
- **`get_contradictions()`**: unresolved contradictions in the world model
- **`ingest_episode(content)`**: ingest new information in real-time (for live conversations)

The query classifier routes queries to the appropriate retrieval strategy: temporal diff queries trigger state reconstruction, procedural queries trigger pattern retrieval, relationship queries trigger graph traversal, and general queries use hybrid vector + graph retrieval.

---

## What's Novel (And What's Not)

To be precise about contributions:

**Not novel:**
- Extracting entities and relationships from text using LLMs (standard NER/RE pipeline)
- Storing knowledge in a graph database (knowledge graphs are well-established)
- Using embeddings for entity retrieval (standard vector search)
- Tiered memory management (introduced by MemGPT)

**Novel (claims to be tested):**

1. **Typed state transitions as the core temporal primitive.** No existing system models entity changes as typed transitions (creation/update/contradiction/resolution/archival) with full provenance and trigger descriptions. Graphiti tracks temporal edges; PIE tracks *typed change semantics*.

2. **Semantic temporal context compilation.** Converting graph temporal data into LLM-readable narratives with period anchoring, change velocity, and contradiction flags — rather than passing raw timestamps or dates. Testable claim: this measurably improves temporal reasoning accuracy.

3. **Procedural memory from cross-entity lifecycle patterns.** [ExpeL](https://arxiv.org/abs/2308.10144) (Zhao et al., 2023) extracts procedures from single-task execution traces. PIE extracts procedures from *patterns across entity lifecycles* over extended time periods — a fundamentally different signal.

4. **Rolling context ingestion.** Processing conversations chronologically with accumulating world model context, enabling activity-based entity attribution that doesn't require explicit mention of entity names. Testable claim: this produces measurably better entity resolution and state change detection than independent extraction.

5. **Graph-structural importance for memory management.** Computing importance from graph topology (connectivity, transitions, neighbor importance) rather than surface heuristics. Testable claim: this correlates more strongly with human importance judgments.

---

## Hypotheses & Experiments

PIE makes specific, testable claims. We are designing experiments to validate or refute each one. Here is the evaluation plan:

### H1: Semantic Temporal Context > Raw Timestamps

**Claim:** LLMs reason more accurately about temporal information when it's compiled into natural language narratives ("10 months ago, during the internship, changed 3 times") vs. raw timestamps, formatted dates, or bare relative time.

**Design:** 50 temporal reasoning questions over a fixed knowledge base. Four conditions: (A) raw timestamps, (B) formatted dates, (C) relative time, (D) full semantic temporal context with periods, velocity, and contradiction flags. Measured across GPT-4o, Claude Opus, and Llama 3.3.

**Expected:** D >> C > B > A. The richer the temporal language context, the better the reasoning.

[TODO: *Results will go here. This is the core hypothesis — if D does not significantly outperform A, the entire semantic temporal compilation approach loses its justification. We expect large effect sizes (>15% absolute accuracy improvement) on temporal ordering and contradiction detection tasks, with smaller effects on simple recency queries.*]

### H2: Rolling Context Ingestion > Batch Isolation

**Claim:** Processing conversations chronologically with accumulating world model context produces better entity resolution and state change detection than processing each conversation independently.

**Design:** 50 consecutive conversations. Condition A: independent extraction (no context). Condition B: chronological with rolling world model context. Metrics: entity resolution F1, duplicate entity rate, state change detection recall (against manually annotated ground truth).

**Expected:** B significantly fewer duplicates, higher state change recall.

[TODO: *Results will go here. Preliminary observation from 10-conversation test runs: independent extraction produces ~3x more duplicate entities and misses ~40% of implicit entity references (conversations about a project that don't name the project). Rolling context should close both gaps.*]

### H3: Graph-Structural Importance > Heuristic Importance

**Claim:** Importance computed from graph topology correlates more strongly with human judgment than heuristic approaches (message count, recency, keyword matching).

**Design:** Build world model from 200+ conversations. Compute importance both ways. Human annotator ranks 100 entities by actual importance. Spearman rank correlation for each method.

[TODO: *Results will go here.*]

### H4: Procedural Memory Emerges from Transition Chains

**Claim:** Meaningful behavioral patterns can be automatically extracted from state transition sequences across entity lifecycles, and these are rated as more accurate than procedures extracted from raw conversation text.

**Design:** Extract procedures from transition chains and from raw text. Human evaluation (accuracy, completeness, actionability) on 1–5 scale.

[TODO: *Results will go here. This is the hardest hypothesis to validate because the ground truth is subjective. We expect the transition-chain approach to produce more specific and more grounded procedures, while the text-based approach produces vaguer, more generic patterns.*]

### H5: PIE > Standard RAG for Temporal Queries

**Claim:** Retrieval from a temporal knowledge graph with state transitions and compiled temporal context produces better answers to temporal queries than standard vector-search RAG over conversation chunks.

**Design:** Build both systems over the same data. 100 queries across five categories: current state, temporal diff, temporal ordering, contradiction detection, procedural. LLM-judge + human evaluation.

**Expected:** PIE significantly better on temporal diff, ordering, contradiction, and procedural queries. Comparable on current-state queries.

[TODO: *Results will go here. We will also benchmark against Zep/Graphiti and Mem0 on [LongMemEval](https://github.com/xiaowu0162/LongMemEval) and [LoCoMo](https://github.com/snap-research/locomo), using the [MemoryBench](https://github.com/supermemoryai/memorybench) evaluation harness for standardized comparison. Expected advantages: knowledge-update and temporal reasoning categories.*]

---

## Planned Benchmarks

[TODO: *This section will contain results on established benchmarks. Here's what we're running:*

**LongMemEval** (Wu et al., ICLR 2025)
- 500 questions across 5 memory abilities
- PIE as the retrieval/memory layer, frontier LLM as the reader
- Key metric: accuracy per ability category, especially temporal reasoning and knowledge updates
- Comparison against published baselines: Zep (up to 18.5% improvement over baselines), Supermemory (claims SOTA on temporal), full-context GPT-4o

**LoCoMo** (Maharana et al., ACL 2024)
- 1,986 questions across 4 reasoning categories (excl. adversarial)
- Special focus on temporal reasoning (category 2) and multi-hop (category 1)
- Event summarization task — PIE's state transition chains directly produce event graphs comparable to LoCoMo's annotated `event_summary` ground truth
- Comparison against published scores: MemMachine (0.849), Zep (0.751), Mem0 (~0.70)

**Custom ablations:**
- Semantic temporal context compilation: four-condition experiment (H1 above)
- Rolling context vs. isolated extraction (H2)
- Graph-structural vs. heuristic importance (H3)
- State transition chain procedures vs. text-based procedures (H4)
- PIE vs. standard RAG (H5)

*Target: demonstrate SOTA on temporal reasoning and knowledge-update subcategories, with competitive or superior overall scores.*]

---

## Open Source

[TODO: *This section will contain the repo announcement. Here's what we plan to release:*

- *Full PIE framework: ingestion pipeline, extraction prompts, entity resolution, state transition model, consolidation engine, MCP server*
- *Data model schemas (entity types, state transitions, procedures)*
- *Evaluation harness with MemoryBench integration for LongMemEval and LoCoMo*
- *Temporal context compiler (the semantic time compilation layer)*
- *Example configurations for FalkorDB, with migration support for Neo4j and Kuzu*
- *Dreaming engine (micro/deep/mega dream cycles)*

*The goal: anyone building agent memory systems should be able to add temporal state tracking, semantic time compilation, and procedural memory to their stack with minimal integration effort. PIE is infrastructure, not an application.*

*Repo link, documentation, and quickstart guide will go here on release.*]

---

## Further Reading

- **Benchmarks:** [LongMemEval](https://arxiv.org/abs/2410.10813) (Wu et al., ICLR 2025) · [LoCoMo](https://arxiv.org/abs/2402.17753) (Maharana et al., ACL 2024) · [Test of Time](https://arxiv.org/abs/2406.09170) (Fatemi et al., ICLR 2025) · [MemoryBench](https://github.com/supermemoryai/memorybench) (Supermemory, 2025)
- **Systems:** [Graphiti/Zep](https://arxiv.org/abs/2501.13956) (Rasmussen et al., 2025) · [Mem0](https://arxiv.org/abs/2504.19413) (Chhikara et al., 2025) · [MemGPT](https://arxiv.org/abs/2310.08560) (Packer et al., 2023) · [Honcho](https://github.com/plastic-labs/honcho) (Plastic Labs) · [Generative Agents](https://arxiv.org/abs/2304.03442) (Park et al., 2023)
- **Related work:** [ExpeL](https://arxiv.org/abs/2308.10144) (Zhao et al., 2023) — procedural learning from task execution · [Kosmos](https://arxiv.org/abs/2511.02824) (Edison Scientific, 2025) — world models for autonomous research agents · [TGB 2.0](https://tgb.complexdatalab.com/) — temporal graph benchmarking
- **Cognitive science:** [Tulving (1972)](https://psycnet.apa.org/record/1973-08477-007) — episodic vs. semantic memory · [Squire (1992)](https://whoville.ucsd.edu/PDFs/206_Squire_etal_AnnuRevPsych_1993.pdf) — declarative vs. nondeclarative memory taxonomy
- **Repos:** [Graphiti](https://github.com/getzep/graphiti) · [Mem0](https://github.com/mem0ai/mem0) · [Letta](https://github.com/letta-ai/letta) · [LongMemEval](https://github.com/xiaowu0162/LongMemEval) · [LoCoMo](https://github.com/snap-research/locomo)

---

*PIE is open source: [TODO: repo link]*

*[TODO: contact / discussion links]*
