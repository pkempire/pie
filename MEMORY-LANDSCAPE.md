# The Agent Memory Landscape: Approaches, Benchmarks, and SOTA
## Comprehensive Reference (March 2026)

---

## Part 1: How to Think About Memory

There are fundamentally different philosophies about what "memory" means for an LLM agent. These aren't just implementation details — they represent genuinely different theories of what the right abstraction is.

---

### Paradigm 1: Extract → Store → Retrieve (Knowledge Memory)

The dominant paradigm. Conversations happen, you extract structured facts/entities from them, store those in some database, and retrieve relevant ones when the agent needs them later.

**Mem0** is the cleanest example. After each conversation, an LLM extracts facts: `("Pranay", "works_on", "PIE")`, `("PIE", "is_a", "memory system")`. These get stored in a vector DB + optional graph. On the next conversation, you embed the user's query, retrieve top-K similar facts, inject them as context.

**How it actually works under the hood:** The ingestion prompt says something like "Extract all facts, preferences, and relationships mentioned in this conversation. Output as JSON." The LLM produces structured output. That gets upserted into the store. At query time, the user's message gets embedded, cosine similarity finds the closest stored facts, and those get prepended to the system prompt.

**The core problem:** LLM-based fact extraction is lossy and brittle. Mem0's GitHub issues document a ~40% fact extraction failure rate (issue #3009). Extracted facts lose nuance, context, and temporal information. You end up with a pile of decontextualized factoids.

**Systems:** Mem0 ($24M raised), Zep (open source), basic RAG setups, ChatGPT's memory feature.

---

### Paradigm 2: Knowledge Graphs with Temporal Edges (Structured Memory)

Instead of flat fact storage, build a graph where entities are nodes, relationships are edges, and both have temporal metadata. This preserves structure that flat retrieval loses.

**Zep/Graphiti** is the main example. It builds a bi-temporal knowledge graph — each edge has both a "valid time" (when the fact was true in the world) and a "transaction time" (when the system recorded it). Entities go through resolution (is "PIE" the same as "Personal Intelligence Engine"?), and the graph gets updated incrementally.

**How it actually works:** Conversations get chunked and processed by an LLM that extracts entities and relationships as structured triples. These get inserted into a graph DB (Neo4j typically). Entity resolution runs to merge duplicates. At query time, the system does both vector search (for semantic relevance) and graph traversal (for structural relationships), then combines results.

**The core problem:** Doesn't scale. Zep's GitHub issue #1275 documents that `resolve_extracted_nodes` sends ALL graph nodes to the LLM — O(n) context growth that causes max_tokens failures above ~500 entities. Entity extraction + resolution is the same hard problem as in Paradigm 1, just with more structure around it.

**Systems:** Zep/Graphiti (23K stars, actively maintained), Microsoft GraphRAG, FalkorDB-based systems.

**Zep/Graphiti deep dive:** Zep's paper (Jan 2025) introduced Graphiti — a temporally-aware knowledge graph engine that synthesizes both unstructured conversational data and structured business data while maintaining bi-temporal relationships. Claims: 94.8% on DMR (vs 93.4% MemGPT), up to 18.5% accuracy improvement on LongMemEval, 90% response latency reduction. The open-source Graphiti library (github.com/getzep/graphiti) has 23K+ stars. Supports hybrid search: semantic + BM25 + graph traversal with result fusion. Temporal edges track both "valid time" (when fact was true) and "transaction time" (when system recorded it). Now has an MCP server for Claude/Cursor integration.

---

### Paradigm 3: Observational Memory (Store Raw, Reason Later)

Don't extract anything at write time. Just store raw observations (what was said, when, by whom) with metadata. Do all the reasoning at query time.

**Mastra** pioneered this and proved it works: 94.87% on LongMemEval using ONLY text observations, no graph, no entity extraction. The key insight is that LLMs are good enough at reasoning over raw text that pre-extraction is unnecessary overhead that introduces errors.

**How it actually works:** Each conversation turn gets stored as an "observation" — raw text with timestamp, speaker, and conversation ID. At query time, you retrieve relevant observations via embedding search and let the LLM reason over them directly. No extraction step, no entity resolution, no graph construction.

**The core problem:** Query-time compute scales with observation count. For a user with thousands of conversations, retrieving and reasoning over relevant observations gets expensive. Also, without any structure, complex temporal or relational queries are harder (though Mastra showed this matters less than people assumed).

**Systems:** Mastra (open source, 94.87% LongMemEval), conceptually similar to Langchain's ConversationBufferMemory but smarter.

---

### Paradigm 4: Episodic Memory (Store Experiences)

Inspired by human episodic memory — store whole episodes (coherent sequences of events) rather than individual facts or observations. Episodes preserve temporal order, causal structure, and narrative coherence.

**Hindsight** is the main example. It has three layers: retention (what to remember), recall (how to retrieve it), and reflection (how to interpret it). The reflection layer is the key innovation — the agent explicitly learns from past episodes, improving future performance.

**How it actually works:** Conversations get segmented into "episodes" — coherent units of interaction around a topic or task. Each episode gets a summary, importance score, and embedding. At query time, relevant episodes are retrieved and the agent reasons over the full episode context, not just extracted facts. The reflection step adds meta-learning: "what did I learn from this episode that applies to future situations?"

**SYNAPSE** extends this with spreading activation — retrieving one memory activates related memories through graph connections, mimicking how human memory works (thinking of one thing reminds you of related things).

**The core problem:** Episode segmentation is itself a hard problem. What constitutes an "episode"? How do you merge episodes that span multiple conversations? And reflection is expensive — an extra LLM call per episode.

**Systems:** Hindsight (91.4%), SYNAPSE (Jan 2026, episodic-semantic hybrid with spreading activation).

---

### Paradigm 5: Parametric Memory (Distill into Weights)

Instead of storing memories externally and retrieving them, compress the knowledge directly into the model's parameters. The model literally "learns" the information rather than looking it up.

**ParamMem** (Feb 2026) encodes cross-sample reflection patterns into parameters. The model processes multiple memory samples, reflects on patterns across them, and those patterns get baked into the weights through fine-tuning. At inference time, the model doesn't need retrieval — it already "knows."

**MemVerse** (Dec 2025) takes a similar approach — distilling retrieval knowledge into parameters so the model can answer from "memory" without external lookup.

**TTT (Test-Time Training)** compresses long contexts into weight updates at inference time. Instead of feeding 100K tokens as context, you train the model on those tokens for a few steps, and it "remembers" them parametrically. This gives constant latency regardless of context length.

**How it actually works (ParamMem):** Take a batch of conversations. LLM reflects on patterns: "This user tends to revisit projects after 2-3 day gaps" or "When this user says X, they usually mean Y." These reflections get used as training data for LoRA fine-tuning. The resulting adapter weights encode the behavioral patterns. At inference time, load the adapter — no retrieval needed.

**The core problem:** Requires fine-tuning access, which isn't available for most commercial LLMs (you can't fine-tune Claude). Also, weight updates aren't easily inspectable or debuggable. And catastrophic forgetting means new fine-tuning can overwrite old knowledge.

**Systems:** ParamMem (+38% over external memory), MemVerse, TTT-E2E, JitRL (modulates logits without gradient updates).

---

### Paradigm 6: RL-Trained Retrieval (Learn When/What to Retrieve)

Don't hand-craft retrieval heuristics — learn them. Use reinforcement learning to train a policy that decides what to retrieve, when, and how to use it.

**R3-RAG** trains a retrieval policy using RL. The agent learns from experience which documents are actually useful for answering questions (vs. which are merely similar). The reward signal comes from answer quality.

**RouteRAG** uses GRPO (Group Relative Policy Optimization) to train a router that decides whether to use text retrieval, graph retrieval, or both for each query. Instead of always doing the same retrieval, the system learns query-dependent routing.

**AgeMem** (Agentic Memory, Jan 2026) goes further — the agent autonomously decides ALL memory operations (create, read, update, delete) via RL + step-wise GRPO. The system learns its own memory management policy from experience.

**How it actually works (AgeMem):** The agent has a "memory action" space: {store_new, update_existing, delete, retrieve, do_nothing}. After each conversational turn, the RL policy decides which memory action to take. The reward comes from downstream task performance. Over training, the agent learns when to store (important new info), when to update (contradictions), when to delete (stale info), and when to retrieve (relevant context needed).

**The core problem:** Requires training infrastructure and lots of data. The reward signal is often sparse and delayed (you don't know if a memory was useful until much later). Hard to apply to production systems without significant engineering.

**Systems:** R3-RAG, RouteRAG, AgeMem, R3 (self-improving retriever).

---

### Paradigm 7: Self-Evolving Strategies (Meta-Memory)

Instead of storing facts or experiences, store *strategies* — learned approaches that accumulate over time without weight updates.

**ACE** (Agentic Context Engineering, ICLR 2026) maintains "evolving playbooks" that accumulate strategies through a Generator → Reflector → Curator loop. After each task, the system reflects on what worked and updates its playbook. This gave +10.6% on AppWorld benchmarks, matching GPT-4.1 with smaller models.

**How it actually works:** The agent has a "playbook" — a text document containing strategies and heuristics. After each task: (1) Generator proposes strategy updates based on what happened, (2) Reflector evaluates whether those updates would improve performance, (3) Curator decides which updates to accept. The playbook evolves over time without any weight changes.

**The core problem:** Task-scoped — playbooks are about "how to complete tasks" not about "who is the user" or "what happened before." Also, the reflection/curation loop is itself expensive (multiple LLM calls per update).

**Systems:** ACE (ICLR 2026), MemGen (treats memory as generative hidden states), Evo-Memory (Sakana AI, self-evolving test-time learning).

---

### Paradigm 8: Multi-Specialized Storage (Decomposed Memory)

Don't force all memory into one format. Use multiple specialized stores, each optimized for a different type of query.

**MAGMA** (Multi-Graph Agentic Memory Architecture, Jan 2026) maintains separate graphs: action graph (what actions → what outcomes), entity graph (relationships), event graph (temporal causality), knowledge graph (general facts). Queries get routed to the appropriate graph(s).

**How it actually works:** Each memory type has its own storage and indexing. A router (can be learned or rule-based) analyzes each query and dispatches it to the right store(s). Results from multiple stores get aggregated by an LLM. This avoids the "one size fits all" problem — temporal queries go to the event graph, factual queries go to the knowledge graph, procedural queries go to the action graph.

**The core problem:** Complexity. Maintaining multiple synchronized stores is engineering-heavy. The routing decision is itself a hard problem. And data that spans multiple categories (a temporal fact about an entity's action) needs to exist in multiple stores.

**Systems:** MAGMA, MemoriesDB (temporal-semantic-relational graph), R3Mem (reversible compression with multiple representation layers).

**MAGMA deep dive:** (Jan 2026, UT Dallas/UF) Represents each memory item across 4 orthogonal graphs: semantic, temporal, causal, and entity. Retrieval is formulated as *policy-guided traversal* over these relational views — the system learns which graph(s) to traverse based on query intent. This decouples memory representation from retrieval logic. Tested on LoCoMo and LongMemEval, consistently outperforms SOTA agentic memory systems in long-horizon reasoning.

---

### Paradigm 9: Identity-Centered Memory (Theory of Mind)

Don't just remember facts — build a working model of the user's identity, psychology, and preferences through continuous reasoning.

**Honcho** (Plastic Labs, $5.4M pre-seed) takes a fundamentally different approach: instead of storing facts or experiences, it builds and maintains "peer representations" — LLM-native models of each user that are continuously refined through what they call "dialectic" reasoning (agent-to-agent natural language communication about the user).

**How it actually works (v3.0):** Three core components:
1. **Ingestion** — Exhaustive explicit information capture from each conversation, done in full parallel for unlimited scale.
2. **Dreaming Agent** — Asynchronous background processing that crawls everything known about a user, producing deductive/inductive/abductive conclusions, summaries, and peer cards. Like human dreaming — consolidates and reorganizes knowledge.
3. **Chat/Retrieval** — The `.chat()` method surfaces reasoned context, not just retrieved facts. Gets the agent "the 10K tokens that matter, not the 100K that don't."

**Key claims:** SOTA on LoCoMo, LongMemEval, AND BEAM benchmarks. Also fastest, cheapest, most token-efficient memory solution. 5x cost reduction in v3. 60-90% fewer tokens with better context quality.

**Core insight:** "Context window size doesn't solve personalization." Theory of Mind is the right abstraction — the agent needs to *think about* the user, not just *recall facts about* them.

**The core problem:** Fully hosted (not self-hostable for the core reasoning), opaque internals, and benchmark performance doesn't necessarily translate to the user experience qualities (proactivity, temporal awareness) that matter most.

**Systems:** Honcho (Plastic Labs), conceptually related to Letta/MemGPT's self-editing memory.

---

### Paradigm 10: Compressed Context Representations (Cartridges)

Instead of retrieving from external memory, *train a smaller KV cache* offline that captures the essence of long contexts.

**Stanford Cartridges** (HazyResearch, Jun 2025) introduces "self-study" — a test-time training recipe where you back-propagate loss through a smaller KV cache to compress long contexts into tiny representations.

**How it actually works:** Normal approach: forward pass on 100K tokens → huge KV cache consuming 100GB+ GPU memory. Cartridges approach: train a much smaller KV cache on the document using gradient descent (next-token prediction loss). The resulting "cartridge" can be stored, reused across different user queries, and loaded instantly. 26x throughput improvement while maintaining quality.

**Why it matters for agent memory:** Chat histories are exactly the kind of long, per-user contexts where this applies. Instead of RAG over conversation history, you could train a cartridge per user that compresses their entire interaction history into a small, reusable cache. No retrieval latency, no lost-in-the-middle effects.

**The core problem:** Requires GPU-based training per user/context. Training takes minutes, not milliseconds. Only works with models you have weight access to. Not applicable to API-only models (GPT, Claude).

**Systems:** Stanford Cartridges (HazyResearch), TTT (Test-Time Training), conceptually related to MemVerse's parametric distillation.

---

## Part 2: The Benchmarks

---

### LongMemEval

**What it evaluates:** Long-term memory in multi-session chat assistants. Can the system remember facts from many conversations ago?

**Data structure:** Multi-session dialogue histories. Each "user" has a series of conversations (typically 10-50 sessions) that establish facts, preferences, events, and relationships. The conversations are synthetic but designed to be naturalistic.

**Question types (5 categories):**
1. **Information extraction** — "What is [person]'s favorite restaurant?" (direct fact recall)
2. **Multi-session reasoning** — "Based on what I told you about my diet AND my travel plans, what restaurant should I go to?" (requires combining info from multiple sessions)
3. **Temporal reasoning** — "What was my job BEFORE I switched to the new company?" (requires temporal ordering)
4. **Knowledge update** — "I told you I liked pizza, but then I said I'm now vegan. What should you recommend?" (requires handling contradictions/updates)
5. **Preference tracking** — "Based on everything you know about me, what gift would I like?" (requires synthesis)

**Metrics:** Accuracy (LLM-as-judge evaluates whether the answer correctly addresses the question given ground truth). Binary correct/incorrect per question. Overall score is percentage correct.

**Key scores:**
- OMEGA: 95.4%
- Mastra (observational): 94.87%
- Hindsight: 91.4%
- Zep/Graphiti: ~72%
- Mem0: ~69%
- Naive RAG: ~60%
- Full context (dump everything): varies by model's context handling

**What it misses:** Doesn't test proactivity (can the agent surface information without being asked?). Doesn't test temporal awareness (does the agent know how much time has passed?). Doesn't test behavioral learning (does the agent get better at interacting with the user?). All questions are direct queries — the user always explicitly asks for what they need.

---

### LoCoMo (Long Conversational Memory)

**What it evaluates:** Memory over very long single conversations (not multi-session). Can the system maintain coherence over 10K+ dialogue turns?

**Data structure:** Extended dialogues between two speakers, spanning hundreds to thousands of turns. Based on real conversation patterns. Each conversation establishes a rich context of shared knowledge, ongoing topics, and evolving discussions.

**Question types:**
1. **Single-hop** — direct fact from a specific turn ("What did I say about X?")
2. **Multi-hop** — requires connecting information from different parts of the conversation ("Given what I said about X and Y, what follows?")
3. **Open-ended** — broader questions requiring synthesis ("Summarize what we discussed about my career plans")

**Metrics:** LLM-as-judge scoring (typically GPT-4 evaluating answer quality against ground truth on a 1-5 scale, then binarized). F1 for factual extraction. BLEU/ROUGE for open-ended.

**Key scores:**
- Note: Zep originally claimed 84%, later corrected to 58.44% (GitHub issue #5 — significant discrepancy)
- MemMachine v0.2: top scores as of Dec 2025
- Mem0: claims 26% accuracy boost over baseline
- Scores vary wildly depending on evaluation methodology

**What it misses:** Single-conversation only (no cross-session memory). Doesn't test memory management (what to forget/compress). Long conversations are increasingly rare in real usage — most people use short, multi-session patterns.

---

### MSC (Multi-Session Chat)

**What it evaluates:** Consistency and persona maintenance across multiple chat sessions. Originally from Meta (FAIR).

**Data structure:** Sequences of 4-5 chat sessions between the same pair of speakers. Each speaker has a defined persona (a set of persona sentences like "I have two dogs", "I work as a teacher"). Sessions are separated by gaps.

**Question types:** Not explicit questions — evaluation is based on whether the chatbot's responses are consistent with previously established persona facts and conversation history. Evaluated by checking if the model contradicts itself or forgets persona details.

**Metrics:** Perplexity, persona consistency score (does the response align with established facts), engagingness ratings, humanness ratings. Also F1 on persona fact retrieval.

**Key scores:** This is an older benchmark (2022) and most modern systems don't report on it. Used primarily in academic settings. Mastra and Mem0 sometimes cite MSC results but it's less central than LongMemEval.

**What it misses:** Very constrained setup (fixed persona sentences). Doesn't test evolving knowledge or temporal dynamics. The "sessions" are artificially structured.

---

### Test of Time (Google Research, Jun 2024)

**What it evaluates:** Temporal reasoning ability — can LLMs understand time, ordering, duration, and temporal relationships? Uses *synthetic* datasets to avoid contamination from pre-training data.

**Data structure:** Novel synthetic datasets specifically designed to assess LLM temporal reasoning. Unlike prior benchmarks using real-world data (which LLMs may have seen during training), ToT generates fresh temporal scenarios. Questions span problem structure, size, question type, fact order, and other controllable factors.

**Question types (multiple categories):**
1. **Temporal ordering** — "Did X happen before or after Y?"
2. **Duration estimation** — "How long did X last?"
3. **Temporal arithmetic** — "If X started on date A and lasted B days, when did it end?"
4. **Temporal causality** — "What caused X, given the timeline?"
5. **Temporal common sense** — "Is it reasonable for X to take Y amount of time?"

**Metrics:** Accuracy per category. Overall accuracy across temporal reasoning types. Systematic investigation into impact of problem structure, size, question type, and fact order on performance.

**Key results:** LLMs are surprisingly bad at temporal reasoning. Models that excel at factual recall often fail at basic temporal ordering. The benchmark exposed systematic weaknesses in how LLMs handle time. Performance degrades significantly as temporal chain length increases.

**Dataset:** Open-sourced at huggingface.co/datasets/baharef/ToT.

**What it misses:** Tests world knowledge temporal reasoning, not personal memory temporal reasoning. Knowing "WWII ended before the Korean War" is different from knowing "the user changed jobs 3 months ago."

---

### TempoBench (Columbia, Oct 2025)

**What it evaluates:** Temporal causal reasoning — can models understand how temporal relationships create causal chains? Uses *formally grounded* evaluation based on linear temporal logic (LTL) and finite-state automata.

**Data structure:** Two evaluation tasks synthesized from LTL specifications:
1. **Temporal Trace Evaluation (TTE)** — Given a multi-step reasoning system (automaton), can the LLM correctly simulate its execution on an input trace? Tests whether models can "run" temporal logic in their heads.
2. **Temporal Causal Credit Assignment (TCCA)** — Given a trace and an outcome, which earlier state/action was causally responsible? Tests backward temporal reasoning.

Difficulty is *parametrized* via features like effect depth, state count, and trace length — making evaluation interpretable and scalable.

**Metrics:** F1 on temporal causal reasoning tasks. Per-feature performance breakdown.

**Key results:** GPT-4o scored 7.5% F1 on temporal causal reasoning tasks. This is the number that demonstrates LLMs are fundamentally broken at temporal causal reasoning — near random performance on tasks humans find straightforward. Performance drops drastically as effect depth increases. Open code at github.com/nik-hz/tempobench, pip installable.

**What it misses:** Formal/synthetic automata scenarios, not personal memory contexts. The 7.5% result is dramatic but the tasks involve formally specified reactive systems. Gap between this and "does the agent know Tuesday was 3 days ago" is significant.

---

### TimE (NeurIPS 2025 Spotlight)

**What it evaluates:** Multi-level temporal reasoning in real-world scenarios. Goes beyond simple ordering to test deep temporal understanding.

**Data structure:** Real-world scenarios with temporal complexity. Multi-level difficulty (from simple ordering to complex temporal inference chains).

**Question types:** Hierarchical — Level 1 is basic temporal fact recall, Level 2 is temporal relationship reasoning, Level 3 is multi-step temporal inference requiring chain-of-thought through time.

**Metrics:** Accuracy per level, overall accuracy.

**Key results:** Performance drops dramatically at higher levels. Models that handle Level 1 (simple recall) well often collapse at Level 3 (multi-step temporal inference). Showed that temporal reasoning failure is not about knowledge but about reasoning capability.

**What it misses:** Still focused on world knowledge, not personal memory. Doesn't test whether agents can maintain their own temporal state.

---

### MemoryArena (Feb 2026)

**What it evaluates:** Agent memory in interdependent multi-session tasks. The key differentiator: tasks have DEPENDENCIES — completing task B requires remembering results from task A.

**Data structure:** Sequences of agent tasks where later tasks depend on earlier ones. Each task involves tool use, information gathering, or decision-making. The agent must carry forward state across sessions.

**Question types:** Task-based — "Complete this task, which requires information from the task you completed 3 sessions ago." Not factual recall questions but functional memory tests.

**Metrics:** Task completion rate, information carry-forward accuracy.

**What it misses:** Relatively new, limited public scores. Focused on agent tasks rather than conversational memory.

---

### MemoryAgentBench (ICLR 2026, UCSD/McAuley Lab)

**What it evaluates:** Four core memory competencies grounded in cognitive science: accurate retrieval, test-time learning, long-range understanding, and conflict resolution. The key innovation: transforms existing long-context datasets into *multi-turn incremental format* that simulates how real memory agents process information.

**Data structure:** "Inject once, query multiple times" — one long text context corresponds to multiple questions. All data is split into chunks to simulate real multi-turn interaction scenarios. Includes both reformulated data from prior benchmarks AND two newly constructed datasets: **EventQA** (temporal event reasoning) and **FactConsolidation** (contradictory information handling).

**Four competencies tested:**
1. **Accurate Retrieval (AR)** — Can the agent find specific information from its accumulated memory?
2. **Test-Time Learning (TTL)** — Can the agent learn NEW patterns from information encountered during deployment (not training)?
3. **Long-Range Understanding (LRU)** — Can the agent connect information from very distant parts of its interaction history?
4. **Conflict Resolution (CR)** — When information contradicts previously stored knowledge, can the agent handle it correctly?

**Metrics:** Per-competency accuracy, overall accuracy. Designed to be significantly harder than prior benchmarks.

**Key results:** Published at ICLR 2026. Shows that existing memory mechanisms have fundamental gaps, particularly in test-time learning and long-range understanding. Simple RAG falls short on all four competencies.

**What it misses:** Still fundamentally reactive (query-response). Doesn't test proactivity, behavioral adaptation, or lived temporal awareness.

---

### StructMemEval (Feb 2026)

**What it evaluates:** The agent's ability to *organize* its long-term memory, not just recall facts. Tests whether agents can maintain structured representations (ledgers, to-do lists, trees) in memory.

**Core insight:** Simple retrieval-augmented LLMs struggle with tasks requiring structured memory organization, but memory agents can solve them if told HOW to organize. However, LLMs don't always recognize the right memory structure autonomously.

**What it misses:** New benchmark, limited adoption so far.

---

### Evo-Memory (Sakana AI / Google, Nov 2025)

**What it evaluates:** Self-evolving memory in *streaming task settings*. The key differentiator: structures datasets into sequential task streams, requiring agents to search, adapt, and evolve memory after each interaction. Tests test-time evolution (continuous memory updating during deployment).

**Core insight:** Existing evaluations focus on static conversational settings where memory is passively retrieved. Evo-Memory tests the *dynamic* ability to accumulate and reuse experience across evolving task streams — closer to how real agents need to work.

---

### Locomo-Plus (Feb 2026)

**What it evaluates:** "Beyond-factual cognitive memory" — extends LoCoMo to test inference, reasoning, and synthesis over memories, not just factual recall.

**Data structure:** Same long-conversation format as LoCoMo but with questions that require cognitive operations beyond recall.

**Question types:**
- Inferential: "Based on everything discussed, what would the user likely think about X?" (requires inference from multiple data points)
- Counterfactual: "If the user hadn't changed jobs, where would they likely be now?"
- Synthesis: "What are the common themes across all the user's decisions?"

**What it misses:** Still conversation-based, still reactive (user asks a question).

---

## Part 3: The Scorecard

| System | Approach | LongMemEval | LoCoMo | BEAM | Notes |
|--------|----------|-------------|--------|------|-------|
| **Honcho 3.0** | Theory of Mind / Identity | **SOTA** | **SOTA** | **SOTA** | Claims Pareto frontier: best accuracy + fastest + cheapest |
| **OMEGA** | Hybrid (undisclosed) | **95.4%** | — | — | #1 on leaderboard pre-Honcho, closed source |
| **Mastra** | Observational | **94.87%** | — | — | Text-only, no extraction, 10x cheaper |
| **MAGMA** | Multi-graph + policy traversal | Outperforms SOTA | Outperforms SOTA | — | 4 orthogonal graphs, learned routing |
| **Hindsight** | Episodic + Reflection | **91.4%** | — | — | Retention + recall + reflection layers |
| **Zep/Graphiti** | Bi-temporal KG | ~72%→improved* | 58.44%** | — | *94.8% DMR, 18.5% LME improvement. **corrected from 84% |
| **Mem0** | Fact extraction | ~69% | +26% vs baseline | — | 40% extraction failure rate documented |
| **Naive RAG** | Chunk + embed + retrieve | ~60% | — | — | Baseline that most systems beat |
| **Full context** | Dump everything | varies | — | — | Works until context window exceeded |

**Key takeaways:**
1. Honcho claims the Pareto frontier (best accuracy + best efficiency), but their approach is opaque and fully hosted
2. MAGMA shows that decomposed memory (separate graphs per concern) with learned routing is powerful
3. Observational memory (Mastra) remains the simplest high-performer
4. The gap between systems that use reflection/reasoning (Honcho, MAGMA, Hindsight) vs pure retrieval (Mem0, RAG) is widening

---

## Part 4: What No Benchmark Tests

Every existing benchmark has the same fundamental blind spot: they test REACTIVE memory (user asks → system retrieves → system answers). None of them test:

1. **Proactivity** — Can the agent surface relevant information WITHOUT being asked? Can it say "Hey, you mentioned a deadline 3 days ago — it's tomorrow"?

2. **Behavioral learning** — Does the agent get BETTER at working with this specific user over time? Does it learn communication preferences, correct recurring mistakes, adapt its style?

3. **Temporal awareness (lived)** — Does the agent know that 5 days have passed since the last conversation? Does it treat a thread differently if it's been dormant for a week vs. active today? (Test of Time and TempoBench test temporal REASONING about world events, not temporal AWARENESS of the agent's own timeline.)

4. **Thread continuity** — Can the agent maintain multiple parallel threads of work and seamlessly resume the right one? Not "recall a fact" but "pick up where we left off on thread X while thread Y is dormant."

5. **Commitment tracking** — If the agent says "I'll look into this," does it actually follow up? No benchmark tests follow-through.

6. **Compression under real scale** — Most benchmarks use 10-50 sessions. What happens at 500 sessions? 5,000? Real persistent AI needs to work for months/years, not hours.

7. **Memory management under contradiction** — Real conversations have contradictions, corrections, evolving beliefs. Most benchmarks test this minimally (LongMemEval has a "knowledge update" category, but it's simple).

These gaps represent the actual frontier — and they're exactly what the self-compiling agent proposal targets. The reason no system solves these is that no benchmark measures them, so there's no pressure to build them.

---

## Part 5: The Research Frontier (Where Things Are Heading)

**Convergence 1: Retrieval is becoming learned, not engineered.**
R3-RAG, RouteRAG, AgeMem all show that RL-trained retrieval policies outperform hand-crafted heuristics. The question isn't "what similarity threshold should I use?" — it's "what reward signal should I train on?"

**Convergence 2: Extraction is becoming optional.**
Mastra proved you don't need entity extraction. ParamMem showed you can encode patterns directly in weights. The trend is away from explicit extraction toward implicit representation.

**Convergence 3: Reflection is the secret sauce.**
Hindsight's reflection layer, ACE's evolving playbooks, ParamMem's cross-sample reflection — the systems that learn from their own experience outperform those that just store and retrieve. Memory that improves through use > memory that just accumulates.

**Convergence 4: Temporal reasoning is the unsolved problem.**
TempoBench (7.5% F1), TimE (performance cliffs at multi-step temporal inference), Test of Time (systematic failures) — LLMs are fundamentally broken at reasoning about time. No amount of better storage fixes this. It requires architectural innovation.

**Convergence 5: The field is moving from "what to remember" to "how to behave."**
ACE's playbooks, ParamMem's behavioral patterns, AgeMem's learned memory policies — the cutting edge isn't about storing more facts. It's about learning better behaviors from experience. This is the paradigm shift.

**Convergence 6: Identity > Memory.**
Honcho's thesis — that agents need a *theory of mind* about each user, not just a fact store — represents a philosophical shift. The Dialectic API (agent-to-agent reasoning in natural language about the user) and the "Dreaming Agent" (asynchronous consolidation) suggest that memory is a substrate for identity modeling, not an end in itself.

**Convergence 7: Compressed representations are viable.**
Stanford Cartridges showed 26x throughput with quality preservation by training small KV caches offline. This opens a path where per-user memory could be a trained artifact rather than a retrieval system — especially relevant as context windows grow but serving costs remain painful.

---

## Part 6: Key Papers Deep Dive

### Time-R1 (UIUC, May 2025)
**Paper:** arxiv 2505.13508 | **Code:** github.com/ulab-uiuc/Time-R1

The first framework to give a 3B-parameter LLM comprehensive temporal abilities: understanding, prediction, and creative generation. Uses a 3-stage RL curriculum with dynamic rule-based rewards:
1. **Stage 1:** Foundational temporal understanding — logical event-time mappings from historical data
2. **Stage 2:** Future event prediction — events beyond knowledge cutoff
3. **Stage 3:** Creative future scenario generation — zero-shot generalization

**Key result:** 3B model outperforms 671B DeepSeek-R1 on future event prediction and creative scenario generation. Also released **Time-Bench** dataset.

**Relevance:** Proves temporal reasoning can be *trained into* small models via RL curriculum. The reward design (temporal consistency, factual grounding, creativity) is directly applicable to training agents that need lived temporal awareness.

---

### Robotouille (Cornell, ICLR 2025)
**Paper:** arxiv 2502.05227

The first *asynchronous* planning benchmark for LLM agents. Uses a cooking domain where agents must manage concurrent tasks with temporal dependencies, time delays, and resource constraints.

**Key results:** ReAct (gpt-4o) achieves 47% on synchronous tasks but only **11% on asynchronous tasks**. This 36-point gap exposes that LLMs fundamentally struggle with parallel temporal planning — managing overlapping tasks with different timelines.

**Relevance to lived temporal awareness:** This is the closest existing benchmark to testing "thread management" — can the agent track multiple concurrent activities with different temporal rhythms? The 11% async result suggests current LLMs are deeply broken at this, which is exactly what our temporal thread tracker would need to solve.

---

### Real-Time Deadlines (UPenn, Jan 2026)
**Paper:** arxiv 2601.13206

Uses negotiation games to expose temporal awareness failures. Two conditions: (1) control — agents know only global time limit, (2) time-aware — agents receive remaining-time updates each turn.

**Key results:**
- GPT-5.1: 4% deal closure (control) vs 32% (time-aware). 8x improvement with time updates.
- All models: ≥95% closure under *turn-based* limits (no temporal tracking needed)
- Offer acceptances 6x higher in time-aware condition

**Core finding:** LLMs fail at *temporal tracking*, not *strategic reasoning*. When they know how much time has passed, they perform fine. When they need to internally track elapsed time, they collapse. This is a direct validation that lived temporal awareness requires *external temporal grounding* — exactly what a memory system should provide.

**Relevance:** This paper is the smoking gun for why agents need explicit temporal state management. Pure LLMs can't internally track time. The memory layer must provide temporal grounding as a service.

---

### MAGMA Deep Dive (UT Dallas, Jan 2026)
**Paper:** arxiv 2601.03236

Multi-graph agentic memory with 4 orthogonal graphs (semantic, temporal, causal, entity). Retrieval = policy-guided traversal — the system learns which graph paths to follow based on query structure.

**Key architectural insight:** Existing systems entangle temporal, causal, and entity information in a monolithic store. MAGMA shows that *separating these concerns* into orthogonal representations with learned routing dramatically improves retrieval precision for complex queries.

**Results:** Consistently outperforms SOTA on LoCoMo and LongMemEval for long-horizon reasoning.

---

### ACE Deep Dive (Stanford/SambaNova, ICLR 2026)
**Paper:** arxiv 2510.04618

Treats context as an evolving playbook: Generator → Reflector → Curator loop. Prevents "context collapse" (iterative rewriting that erodes detail over time) through structured incremental updates.

**Results:** +10.6% on agent benchmarks, 86.9% lower adaptation latency. Matches GPT-4.1-powered production systems with smaller open-source models. On AppWorld leaderboard: matches #1 ranked agent on overall average, surpasses it on harder tasks.

**Key for memory:** ACE's "playbook as evolving context" pattern is directly related to our Self-Compiling Agent's "self-program" — both treat the agent's behavioral instructions as a living document that improves through use.

---

## Part 7: Multi-Agent Evals

### MultiAgentBench (UIUC, ACL 2025)
**Paper:** arxiv 2503.01935 | **Code:** github.com/MultiagentBench/MARBLE

The first comprehensive benchmark for multi-agent LLM systems across diverse interactive scenarios. Measures both task completion AND quality of collaboration/competition using milestone-based KPIs.

**What it tests:**
- Multiple coordination protocols: star, chain, tree, graph topologies
- Group discussion and cognitive planning strategies
- Task completion, milestone achievement, coordination efficiency

**Key results:** gpt-4o-mini achieves highest average task score. Graph structure performs best among coordination protocols for research scenarios. Cognitive planning improves milestone achievement by 3%.

**Relevance:** Tests multi-agent coordination but NOT multi-agent *memory*. No benchmark yet tests whether agents can share memories, avoid duplicate knowledge acquisition, or coordinate their memory states. This is a gap.

---

### Robotouille as Multi-Agent Eval

As noted above, Robotouille tests asynchronous planning which inherently involves multi-agent coordination in cooking scenarios. The 11% success rate on async tasks is the current bar for multi-agent temporal planning.

---

### What's Missing: Multi-Agent Memory Evals

No existing benchmark tests:
1. **Shared memory coherence** — Do agents maintain consistent world models when sharing memory?
2. **Memory conflict resolution across agents** — When agent A and agent B have contradictory memories about the same entity, how is this resolved?
3. **Collaborative memory building** — Can multiple agents collectively build a richer memory than any individual agent?
4. **Memory delegation** — Can agent A tell agent B "remember this for me" and have it work?

---

## Part 8: Designing an Eval for Lived Temporal Awareness

No existing benchmark tests what we actually care about. Here's what such an eval would look like:

### What "Lived Temporal Awareness" Means

The agent doesn't just *reason about* time (that's temporal reasoning) — it *experiences* time passing between interactions. Specifically:

1. **Gap awareness** — "It's been 5 days since we last talked about X"
2. **Decay detection** — "Thread Y hasn't been touched in 2 weeks, should I check in?"
3. **Rhythm recognition** — "User typically works on X on Mondays and Y on Fridays"
4. **Deadline tracking** — "User mentioned a deadline 3 days ago — it's tomorrow"
5. **Relative recency** — "This information is from yesterday vs. 3 months ago"
6. **Temporal context switching** — "Last time we talked about X was before the user changed jobs, so it may be stale"

### Proposed Eval Structure

**Setup:** Multi-session conversations with *real temporal gaps* encoded (not just sequential sessions). Each session has a timestamp. Sessions are separated by varying durations (hours, days, weeks).

**Session design:**
- Session 1 (Day 1): User establishes several threads: project A, project B, personal goal C
- Session 2 (Day 3): User mentions a deadline for project A in "one week" (i.e., Day 10)
- Session 3 (Day 5): User works on project B, no mention of A
- Session 4 (Day 9): Agent should proactively mention that project A deadline is tomorrow
- Session 5 (Day 15): User returns after a gap. Agent should recognize the gap, ask about how project A went, note that project B hasn't been discussed in 10 days
- Session 6 (Day 30): User mentions project A again. Agent should contextualize: "Last time we worked on this was Day 3, and the deadline was Day 10. What happened?"

**Question types (novel):**

1. **Proactive temporal surfacing** (no benchmark tests this)
   - Does the agent mention upcoming deadlines without being asked?
   - Does the agent note when a thread has been dormant?
   - Score: binary (did/didn't surface) + timing (how close to the deadline?)

2. **Gap-aware resumption**
   - When user returns after a gap, does the agent acknowledge the time that passed?
   - Does it prioritize the most relevant thread given the gap duration?
   - Score: relevance of resumed thread + accuracy of gap duration acknowledgment

3. **Temporal grounding of facts**
   - "When did I tell you about X?" → Agent should give approximate temporal reference
   - "Is the information about Y still current?" → Agent should reason about staleness
   - Score: accuracy of temporal localization

4. **Rhythm prediction**
   - After observing patterns (user works on X on Mondays), does the agent anticipate the right context for the right day?
   - Score: precision of predicted thread vs. actual thread engaged

5. **Commitment follow-through**
   - Agent or user says "I'll do X by date Y"
   - In the next session after date Y, does the agent follow up?
   - Score: binary (did/didn't follow up) + timing

6. **Temporal consistency**
   - Agent should not confuse temporal ordering of events
   - "What was I working on BEFORE I started project A?" should get the right answer even if project A was mentioned more recently/frequently
   - Score: accuracy of temporal ordering

### Evaluation Methodology

**Synthetic but temporally grounded:** Generate conversation sets with explicit timestamps and temporal dependencies. Ground truth includes expected agent behaviors at each temporal checkpoint.

**Multi-axis scoring:**
- **Accuracy** (traditional): Correct factual recall with temporal context
- **Proactivity** (novel): Surfacing relevant information without prompting (0-1 per opportunity)
- **Temporal precision** (novel): How accurately the agent tracks and references time (RMSE of temporal estimates)
- **Thread coherence** (novel): Ability to resume the right thread in the right state
- **Commitment tracking** (novel): Follow-through rate on commitments with temporal bounds

**Scale testing:** Run at 10, 50, 100, 500 sessions to test degradation. Real systems need to work over months.

### Why This Doesn't Exist Yet

1. No system claims to have lived temporal awareness, so there's no demand for benchmarking it
2. It requires multi-session eval with encoded temporal gaps — more complex than standard Q&A
3. Proactivity is hard to evaluate (how do you score "should have said something but didn't"?)
4. The Real-Time Deadlines paper (UPenn) is the closest anyone has come, and it showed the failure mode without proposing a memory-system solution

### Connection to Our Proposal

The Self-Compiling Agent's **Temporal Thread Tracker** (reusing PIE's survival functions) directly addresses this eval:
- Survival functions model thread decay → gap awareness + decay detection
- Hawkes process intensity → rhythm recognition + temporal context switching
- The self-program stores temporal behavioral patterns → deadline tracking + commitment follow-through

Building this eval first, then building the system to pass it, would be a genuine contribution to the field. No one has done either.
