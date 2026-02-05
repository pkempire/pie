# Paper Summary: "Memory in the Age of AI Agents"

> **Paper:** Hu, Liu, Yue, Zhang et al. (47 authors), arXiv:2512.13564v2, Dec 2025 / Jan 2026
> **Pages:** 107 | **References:** 300+ | **GitHub:** [Agent-Memory-Paper-List](https://github.com/Shichun-Liu/Agent-Memory-Paper-List) (1k+ stars)
> **Status:** Featured as HuggingFace Daily Paper #1 (Dec 16, 2025)

---

## Executive Summary

This is the most comprehensive survey of agent memory systems to date. The paper argues that **memory is a first-class primitive** in agentic AI—not an afterthought or plugin. It moves beyond the simplistic "short-term vs long-term" dichotomy that dominated earlier work, proposing instead a **three-lens taxonomy**: Forms (how memory is physically stored), Functions (what purpose memory serves), and Dynamics (how memory changes over time).

The key thesis: **the transition from stateless LLMs to autonomous agents requires persistent, self-evolving memory systems** that learn from interaction, maintain consistency, and improve over time without constant retraining.

---

## 1. Memory Taxonomy — Forms × Functions × Dynamics

### The Three-Lens Framework

The paper's central contribution is organizing the fragmented agent memory landscape through three orthogonal dimensions:

```
┌─────────────────────────────────────────────────┐
│           AGENT MEMORY TAXONOMY                 │
├─────────────┬───────────────┬───────────────────┤
│   FORMS     │  FUNCTIONS    │    DYNAMICS        │
│  (Storage)  │  (Purpose)    │   (Lifecycle)      │
├─────────────┼───────────────┼───────────────────┤
│ Token-level │ Factual       │ Formation          │
│ Parametric  │ Experiential  │ Evolution          │
│ Latent      │ Working       │ Retrieval          │
└─────────────┴───────────────┴───────────────────┘
```

### Distinguishing Agent Memory from Related Concepts

The paper carefully delineates four often-conflated concepts:

| Concept | Scope | Persistence | Evolves? |
|---------|-------|-------------|----------|
| **LLM Memory** | Model's parametric knowledge from pretraining | Static | No (until retrained) |
| **RAG** | External knowledge retrieval at inference | Static corpus | No (corpus changes separately) |
| **Context Engineering** | Optimizing what goes into the context window | Per-request | No |
| **Agent Memory** | Persistent, self-evolving system across sessions | Cross-session | **Yes** — learns and adapts |

**Key distinction:** Agent Memory is defined as a persistent, self-evolving system that enables an agent to maintain consistency, coherence, and adaptability over time (t→∞). Context Engineering manages the immediate window; Agent Memory manages the agent's *identity* and *accumulated knowledge*.

---

## 2. Memory Forms — Storage Mechanisms

### 2.1 Token-Level Memory (Most Prevalent)

Information stored as **discrete, human-readable units** (text chunks, JSON, knowledge triples) in external databases.

**Three organizational structures:**

- **Flat:** Linear logs, append-only lists, conversation histories. Simple but hard to search at scale.
- **Planar:** Graphs and trees within a single layer. Knowledge graphs, entity-relationship graphs, tree-structured summaries. Enables relational queries.
- **Hierarchical:** Multi-layer pyramids with different abstraction levels. Raw events → summarized episodes → abstract knowledge. Most powerful but most complex.

**Storage backends covered:**
- **Vector databases** (Pinecone, Weaviate, Chroma) — for embedding-based similarity search
- **Graph databases** (Neo4j, or custom KG stores) — for relational/structural queries
- **Relational databases** — for structured factual data
- **Document stores** — for unstructured text chunks
- **Hybrid stores** — combining multiple backends (e.g., Zep's temporal knowledge graph + vector search)

**Key systems:**
- **MemGPT** — OS-like memory management with paging between tiers
- **Mem0** — Production-ready scalable long-term memory (26% improvement over OpenAI's memory on LoCoMo)
- **HippoRAG** — Neurobiologically inspired (hippocampal indexing theory)
- **GraphRAG** — Microsoft's graph-based approach to query-focused summarization
- **Zep** — Temporal knowledge graph architecture

**Pros:** High interpretability, easily editable, auditable
**Cons:** Retrieval latency, lossy compression, requires explicit retrieval step

### 2.2 Parametric Memory

Information **embedded directly into model weights** through training or editing.

**Two subtypes:**

- **Internal (Model Editing):** Techniques like ROME, MEMIT, AlphaEdit that surgically modify specific factual associations in the model's weights. Mimics biological "instinct."
- **External (Adapters):** LoRA adapters, K-Adapters, MemLoRA that add trainable parameters alongside the base model. Can be swapped, merged, or composed.

**Key systems:**
- **ROME/MEMIT** — Rank-one model editing / mass editing memory in transformers
- **WISE** — Rethinking knowledge memory for lifelong model editing
- **MemLoRA** — Distilling expert adapters for on-device memory
- **CharacterGLM/Character-LLM** — Role-playing agents with personality in weights

**Pros:** No retrieval step needed, fast inference, compact
**Cons:** Catastrophic forgetting, high update costs, hard to audit, can corrupt base model capabilities

### 2.3 Latent Memory

Information stored as **continuous vector representations or KV-cache states**.

**Types:**
- **KV-cache compression** (SnapKV, Scissorhands, H2O) — selectively retaining important attention states
- **Learned memory tokens** (Titans, MemoryLLM, M+) — dedicated trainable tokens that encode long-term context
- **Compressed context** (Gist tokens, AutoCompressor, ICAE) — distilling long contexts into dense representations

**Key systems:**
- **Titans** — Learning to memorize at test time with surprise-driven memory updates
- **MemoRAG** — Global memory-enhanced retrieval using latent representations
- **MemoryVLA** — Perceptual-cognitive memory for robotic manipulation
- **MEM1** — Synergizing memory and reasoning for long-horizon agents

**Pros:** High information density, machine-native, fast access
**Cons:** Low interpretability (black box), hard to edit/debug, not human-readable

---

## 3. Memory Functions — Why Agents Need Memory

### 3.1 Factual Memory (Declarative Knowledge)

The agent's **knowledge base** — what it knows about the world, the user, and itself.

**Contents:**
- **User profiles:** Preferences, demographics, interaction history ("The user prefers Python", "Lives in NYC")
- **Environmental state:** World knowledge, task context, file locations
- **Self-knowledge:** Agent's own capabilities, persona, goals
- **Entity relationships:** How people, places, concepts relate to each other

**Purpose:** Ensures **consistency and coherence** across sessions. Without factual memory, agents can't maintain a stable identity or personalized service.

**Key challenge:** Temporal validity — facts change over time. "User works at Company X" may become stale.

### 3.2 Experiential Memory (Procedural Knowledge)

**How to do things** — the most novel and important category in this taxonomy.

**Three subtypes:**

1. **Case-based:** Raw interaction trajectories stored for replay. "Last time the user asked about X, we did Y and it worked." Analogous to episodic memory in cognitive science.

2. **Strategy-based:** Abstracted workflows, insights, and heuristics. "When debugging async code, first check the event loop." Distilled from multiple experiences into generalizable rules.

3. **Skill-based:** Executable code, tool APIs, reusable functions. The agent discovers and stores *actual callable skills*. Systems like SkillWeaver and CREATOR let agents create new tools from experience.

**Key insight:** Experiential memory is what enables agents to **improve performance on novel tasks without parameter updates**. Rather than just logging conversations, the agent distills successful debugging sessions into reusable workflows.

**Key systems:**
- **ExpeL** — LLM Agents as Experiential Learners (distills experiences into rules)
- **Reflexion** — Verbal reinforcement learning through self-reflection
- **Agent Workflow Memory** — Storing and reusing successful multi-step workflows
- **SkillWeaver** — Self-improving web agents that discover and hone skills
- **Buffer of Thoughts** — Thought templates as reusable reasoning patterns

### 3.3 Working Memory (Active Context Management)

The agent's **scratchpad** for the current task — what's actively being reasoned about.

**Not just the context window.** Working memory is an actively managed subset of all available memory, curated for the current reasoning step.

**Key operations:**
- Context compression (keeping relevant info, discarding noise)
- Active retrieval decisions (when to pull from long-term memory)
- Attention management (what to focus on right now)

**Key systems:**
- **MemAgent** — RL-based multi-conversation memory management
- **AgentFold** — Proactive context management for web agents
- **ACON** — Optimizing context compression for long-horizon tasks
- **Sculptor** — Active context management with cognitive agency

---

## 4. Memory Dynamics — The Lifecycle

### 4.1 Memory Formation (Writing)

Memory formation is **not passive logging** — it's an active transformation:

```
M_form = F(M_t, φ_t)
```

Where raw interaction traces φ_t are compressed, summarized, or distilled into useful artifacts.

**Formation methods:**

1. **Semantic Summarization:** Transform linear streams into narrative blocks. Recursive summarization, sliding-window compression.

2. **Structured Construction:** Parse interactions into structured formats — knowledge graphs, entity-relation triples, timeline events.

3. **Reflection-based Formation:** The agent actively reflects on what happened and what's worth remembering. Systems like Generative Agents (Park et al. 2023) use periodic reflection to generate higher-level insights.

4. **Pre-storage Reasoning:** Shifting inference burden to memory time — reason about what's important BEFORE storing, not just at retrieval time.

### 4.2 Memory Evolution (Consolidation, Updating, Forgetting)

This addresses the **"Stability-Plasticity" dilemma** — how to integrate new information without destroying old knowledge.

**Three evolutionary operations:**

1. **Consolidation:** Merging fragmented traces into coherent schemas. Local-to-global generalization. Multiple similar experiences → one abstract rule.
   - Example: Five debugging conversations about async Python → one "async debugging strategy" memory

2. **Updating:** Resolving conflicts when new facts contradict old ones. The user changed jobs — update the profile. A tool API changed — update the skill.
   - Challenge: Detecting *when* information is stale vs. when the new info is just an exception

3. **Forgetting:** Deliberately removing outdated or low-utility information.
   - **Temporal decay:** Older memories fade (like Ebbinghaus forgetting curve)
   - **Frequency-based:** Rarely accessed memories are pruned
   - **Importance-based:** Low-utility memories removed regardless of age
   - **Interference-based:** Conflicting memories resolved by removing less-supported ones

**Key shift:** From **heuristic-based evolution** (LRU caches, fixed decay rates) to **learning-based approaches** where the agent itself predicts the long-term utility of a memory trace using RL.

**Notable systems:**
- **Mem-α** — Learning memory construction via reinforcement learning
- **Memory-R1** — RL-based memory management
- **MemEvolve** — Meta-evolution of agent memory systems
- **RGMem** — Renormalization group-based memory evolution (physics-inspired hierarchical consolidation)

### 4.3 Memory Retrieval (Reading)

Retrieval is framed as a **dynamic decision process** with two key sub-decisions:

1. **When to retrieve** (timing) — Not every step needs memory access. Some systems use RL to learn when retrieval adds value.

2. **What to retrieve** (intent) — Moving from simple similarity search to intent-aware, context-sensitive retrieval.

**Retrieval methods (from simple to sophisticated):**

1. **Similarity-based:** Vector embedding cosine similarity (vanilla RAG)
2. **Keyword/sparse:** BM25, TF-IDF based matching
3. **Hybrid:** Combining dense + sparse retrieval
4. **Multi-hop:** Following chains of related memories (graph traversal)
5. **Temporal-aware:** Weighting by recency, considering time-sensitive validity
6. **Generative retrieval:** The agent actively *synthesizes* a memory representation tailored to the current reasoning context, rather than just fetching stored text

**Key insight:** The frontier is **generative retrieval** — agents don't just look up memories, they reconstruct relevant context on-demand, mirroring the reconstructive nature of biological memory.

---

## 5. Memory Formats — How Memories Are Structured

The paper identifies multiple formats, which can be mixed:

| Format | Examples | Pros | Cons |
|--------|----------|------|------|
| **Natural language** | Summaries, narratives, reflections | Human-readable, flexible | Verbose, hard to query precisely |
| **Structured data** | JSON, key-value pairs, tables | Queryable, compact | Rigid schema, lossy |
| **Knowledge triples** | (entity, relation, entity) | Relationship-rich, composable | Hard to capture nuance |
| **Knowledge graphs** | Full KG with typed edges | Multi-hop reasoning, temporal edges | Complex to build/maintain |
| **Embeddings** | Dense vectors | Fast similarity search | Not interpretable |
| **KV-cache states** | Attention key-value pairs | Direct model integration | Model-specific, opaque |
| **Code/functions** | Executable skills | Directly usable | Narrow applicability |
| **Timelines** | Temporally ordered event sequences | Supports temporal reasoning | Can grow unbounded |

**The paper recommends hybrid approaches** — combining structured extraction (KG triples for facts) with narrative summaries (for experiential context) and embeddings (for similarity retrieval).

---

## 6. Temporal Reasoning in Memory

One of the paper's key identified **open problems**. Current systems struggle with:

### Current Approaches

1. **Timestamp tagging:** Simple — attach timestamps to memories, use recency as a retrieval signal.

2. **Timeline-based management:** Systems like "Towards Lifelong Dialogue Agents via Timeline-based Memory Management" organize memories along temporal axes.

3. **Temporal Knowledge Graphs:** Zep implements a temporal KG where facts have validity periods. Edges include temporal metadata (start_time, end_time, confidence_decay).

4. **Semantic time modeling:** TSM (Temporal Semantic Memory) builds a "semantic timeline" rather than a dialogue one, consolidating temporally continuous and semantically related information into "durative memories."

5. **Ebbinghaus-inspired forgetting curves:** MemoryBank and similar systems use mathematical decay functions based on time since last access.

### Key Gaps

- **Temporal conflict resolution:** When was X true? Is it still true? How to handle contradicting temporal facts?
- **Durative vs. point events:** "User lived in NYC" (durative, has a timespan) vs. "User graduated in 2020" (point event)
- **Temporal reasoning across memories:** Connecting events that happened at different times to infer causality or patterns
- **Future prediction:** Using temporal patterns in memory to anticipate needs

---

## 7. Benchmarks for Agent Memory

The paper compiles the most comprehensive benchmark survey for memory evaluation:

### Major Benchmarks

| Benchmark | Focus | Scale | Key Metrics |
|-----------|-------|-------|-------------|
| **LoCoMo** | Long-term conversational memory | Multi-session dialogues, 1,986 questions across categories | Question answering accuracy across memory types |
| **LongMemEval** | Long-term memory evaluation | Up to 1.5M tokens of context | Accuracy across 5 memory-intensive tasks |
| **MemBench** | Continual learning from feedback | Multi-turn + feedback | Tests if systems can use feedback without forgetting |
| **MemoryBench** | Memory agent performance | Information extraction, multi-hop reasoning | Memory utilization effectiveness |
| **LOOM-Scope** | Multi-task session-aware evaluation | Unified templates, efficiency metrics | Cross-model comparability |
| **LoCoMo-V** | Multimodal memory (visual) | Visual + textual conversations | VQA accuracy with persistent memory |
| **StoryBench** | Narrative memory evaluation | Story-following with memory | Coherence and factual consistency |

### Evaluation Dimensions

The paper identifies these key evaluation dimensions for memory systems:

1. **Factual accuracy** — Does the agent remember facts correctly?
2. **Temporal consistency** — Does the agent respect when facts were true?
3. **Multi-hop reasoning** — Can the agent connect multiple memories?
4. **Forgetting resistance** — Does the agent retain important info over time?
5. **Update capability** — Can the agent incorporate corrections?
6. **Personalization quality** — Does memory improve personalized responses?
7. **Scalability** — How does performance degrade as memory grows?

### Open-Source Frameworks Covered

- **Mem0** — Production-ready scalable memory
- **MemGPT/Letta** — OS-like memory management
- **LangChain/LlamaIndex** — Memory modules in orchestration frameworks
- **A-MEM** — Agentic memory with self-organizing structure
- **Zep** — Temporal knowledge graph memory

---

## 8. Key Findings and Open Problems

### Main Conclusions

1. **Memory is not optional for agents** — It's the foundation for long-horizon reasoning, personalization, and self-improvement. Stateless agents hit a ceiling.

2. **The long-term/short-term dichotomy is insufficient** — The Forms × Functions × Dynamics framework captures the real diversity of memory systems.

3. **Experiential memory is the underexplored frontier** — Most work focuses on factual memory (user profiles, knowledge). The real breakthrough comes from agents that learn *how to do things better* from experience.

4. **Heuristic memory management is being replaced by learned policies** — RL-driven memory operations (read/write/forget as learnable actions) significantly outperform hand-crafted rules.

5. **Retrieval is evolving from search to synthesis** — Moving from "find the most similar memory" to "reconstruct the most relevant context for the current reasoning step."

6. **Trustworthiness is a critical unsolved problem** — Memory creates new attack vectors (persistent prompt injection, privacy leakage) and compounding errors (hallucinated memories corrupting future reasoning).

### Open Problems (Research Frontiers)

1. **Memory Automation:** How to automatically decide what to store, when to consolidate, and what to forget — currently requires significant manual engineering.

2. **RL Integration:** Treating memory operations as actions in an RL framework. Systems like Mem-α and Memory-R1 are early explorations.

3. **Multimodal Memory:** Moving beyond text-only to storing visual, audio, and sensor memories. MemoryVLA shows early promise for robotics.

4. **Multi-Agent Memory:** How do agents share, merge, and collaborate through shared memory? G-Memory, MIRIX explore this.

5. **Trustworthiness:**
   - **Privacy:** Persistent memory creates vectors for data leakage across sessions
   - **Hallucination compounding:** If the model hallucinates during formation/consolidation, it corrupts its own long-term knowledge
   - **Right to be forgotten:** How to reliably purge specific memories
   - **Adversarial robustness:** Prompt injection that persists in memory

6. **Temporal Reasoning:** No good solutions exist for complex temporal queries, durative facts, or temporal conflict resolution.

7. **Evaluation:** Current benchmarks are narrow. Need comprehensive evaluation of memory quality, not just downstream task performance.

---

## 9. Practical Recommendations for Building Memory Systems

Based on the survey's analysis, here are the key takeaways for implementation:

### Architecture Recommendations

1. **Use hybrid storage:** Combine vector DB (for semantic similarity) + knowledge graph (for structured relations) + document store (for raw context). No single storage type is sufficient.

2. **Implement hierarchical organization:** Raw events → summarized episodes → abstract knowledge. Three-tier minimum for non-trivial applications.

3. **Separate factual from experiential memory:** Different storage, different update policies, different retrieval strategies.

4. **Active memory formation, not passive logging:** Process and structure information at write-time, not just at read-time. Pre-storage reasoning reduces retrieval burden.

5. **Build in forgetting from day one:** Memory systems without forgetting become unbounded and noisy. Implement temporal decay, importance scoring, and explicit pruning.

### Retrieval Recommendations

6. **Hybrid retrieval is essential:** Combine dense (embedding) + sparse (keyword) + structural (graph traversal) retrieval. No single method handles all query types.

7. **Add temporal awareness:** Weight memories by recency and validity period. A fact from yesterday may be more relevant than a more semantically similar fact from a year ago.

8. **Consider generative retrieval:** Instead of returning stored text verbatim, have the agent synthesize a contextually relevant memory representation.

### Evolution Recommendations

9. **Implement consolidation cycles:** Periodically merge related memories into higher-level abstractions. Don't just accumulate — distill.

10. **Use conflict detection:** When new information contradicts stored facts, flag for resolution rather than silently overwriting or ignoring.

11. **Consider RL for memory management:** If you can define a reward signal (task success, user satisfaction), let the agent learn what's worth remembering.

### Safety Recommendations

12. **Separate memory from model:** Keep memory external and auditable. Parametric memory is powerful but hard to inspect and correct.

13. **Implement access controls:** Not all memories should be retrievable in all contexts. Especially important for multi-user or multi-agent systems.

14. **Plan for memory corruption:** Have mechanisms to detect and recover from hallucinated or poisoned memories.

---

## 10. Application to Personal Memory Systems (4000+ ChatGPT Conversations)

Here's how the paper's framework maps to building a personal intelligence system from massive conversation history:

### Direct Applicability

**Memory Forms for Personal Data:**
- **Token-level** is the primary form — conversations are text, and need to be stored as structured text artifacts
- **Hierarchical organization is critical:** Raw conversation turns → conversation summaries → topic-level knowledge → life-domain abstractions
- **Knowledge graph** for entity relationships (people, projects, ideas, preferences)

**Memory Functions Mapping:**

| Paper's Category | Your System Equivalent |
|-----------------|----------------------|
| **Factual Memory** | User profile, preferences, beliefs, project details, people mentioned, factual knowledge extracted from conversations |
| **Experiential Memory (Case-based)** | Complete conversation trajectories indexed for retrieval — "what happened when I tried X" |
| **Experiential Memory (Strategy-based)** | Extracted decision patterns, problem-solving approaches, learning insights — "how I think about Y" |
| **Experiential Memory (Skill-based)** | Code snippets, workflows, prompt templates discovered through conversations |
| **Working Memory** | The current session's context — what's actively relevant right now |

### Recommended Architecture

Based on the paper's analysis, for a 4000+ conversation personal memory system:

1. **Ingestion Pipeline (Memory Formation):**
   - Parse conversations into turns with timestamps
   - Extract entities, topics, and key facts per conversation
   - Generate per-conversation summaries (semantic summarization)
   - Build knowledge triples from factual statements
   - Identify experiential insights and decision patterns
   - **Pre-storage reasoning:** Have an LLM assess importance and categorize before storing

2. **Storage Layer (Hybrid Forms):**
   - **Vector store** for embedding-based retrieval of conversation chunks and summaries
   - **Knowledge graph** for entities, relationships, temporal facts (Neo4j or similar)
   - **Timeline store** for chronological navigation and temporal queries
   - **Structured profile store** for consolidated user preferences and facts

3. **Memory Evolution:**
   - **Consolidation:** Periodically merge related conversation insights into topic summaries
   - **Profile maintenance:** Keep a living user profile that updates as preferences change
   - **Temporal tracking:** Mark facts with validity periods, detect when information becomes stale
   - **Importance scoring:** Weight memories by frequency of reference, emotional salience, and practical utility
   - **Forgetting:** Prune low-utility memories, merge redundant ones

4. **Retrieval Strategy:**
   - **Multi-signal retrieval:** Combine semantic similarity + temporal recency + entity matching + importance score
   - **Context-aware:** What the user is currently working on should influence what memories surface
   - **Multi-hop:** Follow entity connections to surface related but not directly similar memories

### Key Insights from the Paper for This Use Case

1. **Don't just store conversations — transform them.** Raw conversation logs are the lowest-value memory format. Extract structured knowledge, generate summaries, identify patterns.

2. **Experiential memory is the killer feature.** Knowing *what* the user discussed is table stakes. Knowing *how they think*, *what approaches they prefer*, and *what worked before* is transformative.

3. **Temporal coherence matters enormously** for personal memory. A user's beliefs, preferences, and situation change over time. The system must track *when* things were true, not just *what* was true.

4. **Hierarchical abstraction prevents drowning in detail.** 4000 conversations × ~20 turns each = 80,000+ memory candidates. Must abstract into manageable layers.

5. **The consolidation cycle is critical.** Without periodic consolidation, the system accumulates redundant, contradictory, and noisy memories. Schedule regular "reflection" passes that merge related memories into higher-level insights.

6. **Privacy and safety are paramount for personal data.** This is literally someone's thought history. Implement strict access controls, encryption, and the ability to selectively delete memories.

---

## Appendix: Key Referenced Systems

| System | Year | Key Contribution |
|--------|------|-----------------|
| **Generative Agents** (Park et al.) | 2023 | Reflective memory with importance scoring and periodic consolidation |
| **MemGPT** | 2023 | OS-like memory management with context paging |
| **Reflexion** | 2023 | Self-reflection as verbal reinforcement learning |
| **ExpeL** | 2023 | Experiential learning — distilling task experience into rules |
| **HippoRAG** | 2024 | Neurobiologically inspired memory (hippocampal indexing) |
| **GraphRAG** (Microsoft) | 2024 | Graph-based RAG for query-focused summarization |
| **Agent Workflow Memory** | 2024 | Storing and reusing successful multi-step workflows |
| **Mem0** | 2025 | Production-ready scalable long-term memory |
| **Zep** | 2025 | Temporal knowledge graph for agent memory |
| **A-MEM** | 2025 | Agentic self-organizing memory |
| **Mem-α** | 2025 | RL-learned memory construction policies |
| **Memory-R1** | 2025 | RL-based memory management |
| **MemEvolve** | 2025 | Meta-evolution of memory systems |
| **Titans** | 2025 | Test-time memorization with surprise-driven updates |

---

## Appendix: Paper Structure

For reference, the paper is organized as:

1. **Introduction** — Motivation and scope
2. **Preliminaries** — Formal definitions, distinguishing from RAG/Context Engineering
3. **Forms** (Section 3) — Token-level, Parametric, Latent
4. **Functions** (Section 4) — Factual, Experiential, Working
5. **Dynamics** (Section 5) — Formation, Evolution, Retrieval
6. **Benchmarks & Frameworks** (Section 6) — Evaluation landscape
7. **Positions & Frontiers** (Section 7) — Future directions
8. **Conclusion**

---

*Summary compiled from: arXiv abstract, ArXivIQ Substack analysis, alphaXiv overview, GitHub paper list, and benchmark research. Last updated: 2026-02-02.*
