# Semantic Temporal Context Compilation for Personal Knowledge Graph Construction from Conversational History

**[Draft — Living Document]**

---

## Abstract

We present PIE (Personal Intelligence Engine), a system that constructs a temporal knowledge graph — a *world model* — from 4,758 longitudinal ChatGPT conversations spanning 33 months (April 2023–January 2026). Unlike existing agent memory systems that store flat key-value facts (Mem0), session summaries (MemGPT/Letta), or user psychological profiles (Honcho), PIE models the user's world as a graph of typed entities with explicit state transitions, enabling temporal reasoning over how knowledge, projects, beliefs, and decisions evolve. We introduce four techniques: (1) **semantic temporal context compilation**, which converts raw temporal graph data into natural language narratives for LLM consumption, replacing timestamp-passing with period-anchored, change-velocity-enriched descriptions; (2) **rolling context ingestion** with activity-based sliding windows, where each conversation batch is processed with awareness of recently active entities in the accumulated world model; (3) **procedural memory extraction** from state transition patterns across entity lifecycles; and (4) **tiered forgetting** with graph-structural importance scoring inspired by PageRank. We evaluate on LongMemEval, LoCoMo, and Test of Time, with ablation studies isolating the contribution of semantic temporal compilation. [TODO: need experiment results — fill in specific numbers after benchmarks]

---

## 1. Introduction

Large language models have no persistent memory. Each conversation begins from a blank state, requiring users to re-establish context — who they are, what they're working on, what they've decided — every time. This limitation has spawned an ecosystem of *agent memory systems*: infrastructure that stores, retrieves, and surfaces personal context across sessions.

The current landscape falls into three broad categories. **Fact extraction systems** like Mem0 (Chhablani et al., 2024) decompose conversations into atomic key-value facts and retrieve them via embedding similarity. **Session summarization systems** like MemGPT/Letta (Packer et al., 2023) compress conversation history into summaries stored in a tiered memory hierarchy. **Profile systems** like Honcho (Plastic Labs, 2025) model user psychology — dialectical personality representations that predict behavior and preferences.

All three approaches share a critical blind spot: **temporal reasoning**. They can tell you *what* facts are true about a user. They cannot tell you *how* those facts have changed over time, *when* they changed, *why* they changed, or *what patterns exist* in those changes. A fact-extraction system stores "Parth is working on Lucid Academy" but not "Parth started Lucid Academy 8 months ago, pivoted from mentoring to curriculum 3 months in, and follows the same project pattern he used for Science Research Academy." The temporal dimension — arguably the most important axis of personal knowledge — is either ignored or reduced to a `last_updated` timestamp.

This matters because temporal understanding is foundational to useful personal AI. Consider the queries a knowledge worker might ask an assistant with access to their full history:

- *"How has my thinking about knowledge graphs evolved?"* (temporal diff)
- *"What happened to the project I was working on last spring?"* (temporal retrieval)
- *"Do I always pivot projects after 3 months?"* (procedural/pattern)
- *"I said I'd never use cloud services — have I contradicted that?"* (contradiction detection)

None of these can be answered by retrieving the top-k most similar text chunks. They require a structured model of how entities in the user's world have evolved.

We present PIE, a system that ingests a knowledge worker's full conversational history into a temporal knowledge graph. PIE models the user's world as a set of **typed entities** (people, projects, beliefs, decisions, tools, concepts, organizations, life periods) connected by relationships, where each entity carries a chain of **state transitions** recording every change — with typed transition categories (creation, update, contradiction, resolution, archival), causal provenance linking back to source conversations, and semantic temporal descriptions compiled for LLM consumption.

Our contributions:

1. **Semantic Temporal Context Compilation.** We propose a method for converting raw temporal graph data (timestamps, transition chains, period metadata) into natural language narratives enriched with relative time, period anchoring, change velocity, and contradiction flags. We hypothesize and test that LLMs reason more accurately about temporal information presented this way than as raw or formatted timestamps (§4.1, §5.1).

2. **Rolling Context Ingestion with Activity-Based Windows.** We process conversations chronologically in daily batches, where each batch receives a context preamble of recently active entities, recent state changes, and active project summaries — enabling the extraction model to connect unlabeled subtasks to parent projects and detect implicit state changes (§4.2).

3. **Procedural Memory from State Transition Patterns.** By analyzing state transition sequences across entity lifecycles (e.g., how multiple projects evolve from inception through pivots to completion), we extract recurring behavioral patterns — the user's habitual approaches to technology evaluation, project management, and decision-making (§4.4).

4. **Tiered Forgetting with Graph-Structural Importance.** We compute entity importance from graph connectivity, state transition count, recency, neighbor importance, and access frequency — a PageRank-inspired approach that separates signal from noise without hand-crafted heuristics (§4.3).

---

## 2. Related Work

### 2.1 Agent Memory Systems

**Mem0** (Chhablani et al., 2024) is the most widely adopted open-source memory layer. It decomposes conversations into atomic facts stored as key-value pairs with vector embeddings, retrieving relevant facts via cosine similarity at query time. Mem0 is simple and effective for preference recall ("user likes dark mode") but fundamentally limited by its flat data model: facts have no relationships to each other, no temporal evolution beyond a `last_updated` field, and no mechanism for contradiction detection or state tracking. When a user's preference changes, Mem0 overwrites the old fact — the history of change is lost.

**Zep/Graphiti** (Zep Inc., 2024) represents the closest prior work to PIE. Graphiti is a temporal knowledge graph framework built on FalkorDB that maintains three subgraph layers: an episode subgraph (raw conversation data), a semantic entity subgraph (extracted entities and relationships), and a community subgraph (entity clusters). Graphiti supports bi-temporal modeling (event time vs. ingestion time) and achieves strong results on the DMR benchmark (94.8%) and LongMemEval (up to 18.5% improvement over baselines). However, Graphiti was designed as a general-purpose agent memory infrastructure, not for deep temporal reasoning over personal knowledge. It lacks: (a) typed state transitions with semantic descriptions, (b) semantic temporal context compilation (it passes formatted timestamps), (c) procedural memory extraction, and (d) life-period anchoring. PIE builds on Graphiti's graph infrastructure while adding these capabilities.

**Honcho** (Plastic Labs, 2025) takes a fundamentally different approach: rather than storing facts or building knowledge graphs, it models the user's *psychology*. Using dialectical personality representations derived from conversation patterns, Honcho predicts user behavior, preferences, and cognitive style. This is complementary to PIE's approach — Honcho models who the user *is*, while PIE models what the user's *world* looks like and how it evolves. The two could be combined, with Honcho providing personality context and PIE providing world-state context.

**MemGPT/Letta** (Packer et al., 2023) introduced the concept of virtual context management for LLMs, treating the context window as a memory hierarchy analogous to OS page management. While influential in establishing the tiered-memory paradigm, MemGPT operates at the session level — it manages what information is paged into the current context — rather than building a persistent structured model of the user's world.

### 2.2 World Models in AI Systems

**Kosmos** (UIUC, 2025) demonstrated that LLM agents can construct and maintain "world models" — structured, evolving representations of the state of a task environment — and that doing so dramatically improves multi-step reasoning. In Kosmos, the world model tracks the state of a research process: what papers have been found, what hypotheses are active, what experiments have been run. PIE extends this concept from bounded task environments to the unbounded, continuous domain of a knowledge worker's entire professional and personal life. Where Kosmos builds a world model over hours for a single research run, PIE builds one over years from thousands of conversations.

The term "world model" in the context of AI agents typically refers to an internal representation that allows prediction of future states given actions (Ha & Schmidhuber, 2018). PIE's world model is closer in spirit to this definition than to a simple knowledge base: by tracking state transitions and extracting procedural patterns, PIE can support predictions about likely future state changes ("based on past patterns, this project is likely to pivot within 2 months").

### 2.3 Temporal Knowledge Graphs

Temporal knowledge graphs (TKGs) extend static knowledge graphs with temporal annotations on facts, enabling reasoning about when facts are valid and how the graph evolves (Cai et al., 2023). Prior work on TKGs has focused primarily on link prediction — predicting future edges given temporal patterns (Gastinger et al., 2024) — and temporal question answering over structured knowledge bases (Jia et al., 2021).

PIE differs from the TKG literature in two ways. First, PIE's temporal data is *extracted from natural language conversations*, not structured from existing databases — making the extraction pipeline (entity resolution, state change detection, contradiction identification) a core system challenge. Second, PIE's primary interface is *context compilation for LLM consumption*, not direct graph querying. The system must translate graph-structural temporal information into natural language that downstream LLMs can reason with effectively.

### 2.4 Temporal Reasoning in LLMs

Recent work has highlighted that LLMs struggle with temporal reasoning despite parametric knowledge of temporal facts. Fatemi et al. (2024) showed via the Test of Time benchmark that LLMs rely heavily on parametric knowledge for temporal questions and performance degrades sharply when entities are anonymized — suggesting that apparent temporal reasoning is often memorization. LongMemEval (Wu et al., 2024) found that temporal reasoning is the weakest category for most long-context and memory-augmented systems, with time-aware query expansion improving results by 7-11%.

These findings directly motivate PIE's semantic temporal context compilation: if LLMs struggle with raw temporal data, converting that data into natural language narratives — with relative time, period anchoring, and change-velocity descriptions — should improve temporal reasoning quality.

### 2.5 Personal Knowledge Management

The goal of building a structured model from a user's conversational history connects to the broader field of personal knowledge management (PKM). Tools like Obsidian, Roam Research, and Notion allow users to manually build knowledge graphs. PIE automates this process: the system produces a knowledge graph that could, in principle, be materialized as an Obsidian vault — but constructed automatically from conversation history rather than through manual curation.

---

## 3. Data

### 3.1 Dataset Characteristics

Our primary dataset is the complete ChatGPT conversation history of a single knowledge worker (the first author) spanning 33 months: April 2023 through January 2026.

| Statistic | Value |
|-----------|-------|
| Total conversations | 4,758 |
| Total tokens (estimated) | ~16M |
| User messages | ~17,000 |
| Assistant messages | ~24,000 |
| Time span | 33 months |
| Models used | GPT-3.5, GPT-4, GPT-4o, GPT-o1, GPT-5 |
| Avg conversations/day | ~4.8 |

The data exhibits characteristics common to knowledge workers who use LLMs as thinking partners: a mix of project planning, technical problem-solving, conceptual exploration, debugging, content creation, and personal decision-making. Critically, the same entities (projects, people, tools, concepts) recur across many conversations over months and years, with their states evolving — making this dataset well-suited for temporal knowledge graph construction.

### 3.2 Data Format

ChatGPT exports conversations as JSON with a tree structure (message nodes with parent/children pointers), which we linearize by walking from the `current_node` backwards to root. We filter to substantive text content, excluding system messages, code execution outputs, browsing artifacts, and reasoning traces. Conversations are sorted chronologically by creation timestamp, which is critical for rolling context ingestion.

### 3.3 Ethical Considerations

This dataset is a single individual's personal conversation history, processed with that individual's full consent (as the first author). No other individuals' data was collected. Entity names in examples throughout this paper have been anonymized where they refer to real individuals other than the first author. The system is designed for self-hosted, local-first deployment — the world model never leaves the user's machine.

---

## 4. Method

### 4.1 Semantic Temporal Context Compilation

The core insight of PIE's temporal approach is that LLMs should never see raw timestamps. Instead, temporal graph data is compiled into natural language narratives that encode:

- **Relative time:** "22 months ago" rather than "2024-03-15"
- **Period anchoring:** "during freshman year" rather than a date range
- **Change velocity:** "changed 7 times (~0.3x/month)" encoding both total changes and rate
- **Staleness signals:** "last referenced 3 weeks ago" indicating recency
- **Contradiction flags:** "⚠ This contradicted the previous state" marking belief/state reversals

**Formal definition.** Given an entity $e$ with state transition history $T = [t_1, ..., t_n]$ and the current time $t_{now}$, the temporal context compiler $\mathcal{C}$ produces a natural language string:

$$\mathcal{C}(e, T, t_{now}) \rightarrow s \in \Sigma^*$$

where $s$ encodes: (1) entity age as humanized relative time, (2) the life period $p$ containing $e$'s first appearance, (3) total state change count and monthly velocity, (4) each transition $t_i$ described as relative time + period + trigger summary + optional contradiction flag, and (5) current staleness.

**Period nodes.** Life periods are semantic time anchors maintained in the graph as first-class entities:

```
(:Period {name: "UMD freshman year", start: 1724544000, end: 1747699200})
```

Entities link to periods via `DURING` edges. The period vocabulary is automatically detected during ingestion and refined during consolidation passes.

**Example compiled output** (what a downstream LLM receives):

```
Science Research Academy — first appeared about 22 months ago
  (high school senior year), last referenced 3 weeks ago.
Changed state 7 times (~0.3x/month).
  • 22 months ago (high school senior year): Founded with Pranay
    as educational platform for science fair students
  • 18 months ago (summer before UMD): Launched at scifair.tech,
    30 students signed up
  • 14 months ago (UMD freshman year): Pivoted from mentoring to
    research curriculum
    ⚠ This contradicted the previous state.
  • 9 months ago (UMD freshman year): Expanded to 100+ students
```

### 4.2 Rolling Context Ingestion

PIE processes conversations chronologically in daily batches with an activity-based context preamble. This design addresses a fundamental limitation of per-conversation extraction: when users work on subtasks without naming the parent project (e.g., "write a curriculum outline" when the curriculum belongs to Lucid Academy), isolated extraction fails to make the connection.

**Daily batch formation.** Conversations from the same calendar day are grouped into a single extraction batch. The median daily batch in our dataset contains ~6 conversations (~17K characters), fitting comfortably within a single LLM extraction call with context.

**Activity-based context preamble.** Each batch receives a preamble containing:

1. **Recently active entities:** All entities with state transitions in the last 3 days, regardless of whether they are mentioned in the current batch, with name, type, and current state summary.
2. **Recent state changes:** The last 5–10 state transitions across the entire graph, providing a "what's been happening" signal.
3. **Active project summaries:** Current state of all project-type entities with importance > 0.3.

This preamble is *activity-based, not mention-based* — a critical distinction. The extractor doesn't need to see "Lucid Academy" in the conversation to connect a curriculum-writing conversation to the Lucid Academy project; it sees "Lucid Academy: AI summer program, currently in curriculum design phase" in the preamble and makes the connection.

**Extraction schema.** For each daily batch, a structured extraction call produces:

- **Entities** (new or existing): typed nodes with current state descriptions
- **Relationships** between entities
- **State changes** on existing entities, with old/new state aspects and contradiction flags
- **Beliefs** expressed or updated
- **Decisions** made, with reasoning and alternatives considered
- **Significance score** (0–1) for the batch, based on content importance
- **User cognitive state** (exploratory, decisive, frustrated, etc.)

**Entity resolution.** After extraction, each extracted entity is resolved against the existing graph via embedding similarity search (top-5 candidates) followed by LLM verification for candidates above a 0.7 similarity threshold. This two-stage approach balances precision (LLM verification catches false positives from embedding search) with cost (most entities resolve quickly without LLM calls).

### 4.3 Tiered Forgetting with Graph-Structural Importance

Not all knowledge should be retained at full fidelity indefinitely. PIE implements a three-tier archival system governed by an importance score computed from graph structure:

**Importance function.** For entity $e$ with degree $d(e)$, transition count $|T_e|$, neighbor set $N(e)$, recency $r(e)$ (days since last reference), and access count $a(e)$:

$$I(e) = 0.25 \cdot \frac{d(e)}{d_{max}} + 0.20 \cdot \min\left(\frac{|T_e|}{10}, 1\right) + 0.20 \cdot \frac{1}{1 + r(e)/30} + 0.20 \cdot \overline{I(N(e))} + 0.15 \cdot \min\left(\frac{a(e)}{5}, 1\right)$$

where $\overline{I(N(e))}$ is the mean importance of neighboring entities. This recursive definition (importance depends on neighbor importance) is analogous to PageRank and is computed iteratively until convergence.

**Tier definitions:**

| Tier | Condition | Treatment |
|------|-----------|-----------|
| Active | $I(e) > 0.3$ or $r(e) < 90$ days | Full detail. All transitions preserved. Surfaces in queries. |
| Summarized | $I(e) \in [0.1, 0.3]$ and $r(e) > 90$ days | Transitions merged into LLM-generated summary. Entity remains in graph with compressed history. |
| Archived | $I(e) < 0.1$ and $r(e) > 180$ days | Removed from active graph. Stored in cold archive. Recoverable but never surfaces in queries. |

**Consolidation passes** run periodically (every 100 conversations during ingestion, nightly post-ingestion) and perform: entity merging (for high-similarity pairs), transition summarization, archival, importance recomputation, contradiction resolution, and period boundary refinement.

### 4.4 Procedural Memory Extraction

After initial ingestion, PIE analyzes state transition sequences across entity lifecycles to extract recurring behavioral patterns — procedures.

**Approach.** For each entity type with meaningful lifecycles (primarily projects, but also beliefs and decisions), PIE collects ordered state transition sequences and submits them to an LLM with instructions to identify recurring patterns across entities. For example, given the transition histories of 15 project entities, the system might extract:

- *"Technology evaluation pattern: deep research → compare alternatives → narrow to one → build fast"* (observed in 8/15 projects)
- *"Project trajectory: ambitious scope → reality check → scope reduction → focused build"* (observed in 6/15 projects)

Procedures are stored as first-class graph entities with evidence links (which entity lifecycles exhibit the pattern), observation count, temporal bounds (when the pattern was first and last observed), and confidence score (higher with more observations).

**Distinction from ExpeL** (Zhao et al., 2023). ExpeL extracts procedures (called "insights") from single task episodes — learning from trial-and-error on one problem. PIE extracts procedures from *patterns across entity lifecycles over months* — meta-level behavioral patterns visible only in longitudinal data. The temporal dimension is essential: "Parth used to evaluate tools by trying many in parallel, but shifted to deep-dive-one-at-a-time after 2024" is a procedure with its own temporal evolution.

### 4.5 Query Interface

PIE exposes a query layer via Model Context Protocol (MCP), enabling any MCP-compatible LLM client to query the world model. The query pipeline:

1. **Intent classification:** Categorize the query as temporal-diff, current-state, entity-lookup, procedural, relationship, or contradiction.
2. **Entity extraction:** Identify entities referenced in the query.
3. **Subgraph retrieval:** Retrieve the relevant subgraph using intent-appropriate traversal (e.g., temporal range queries for temporal-diff, shortest paths for relationship queries).
4. **Temporal context compilation:** Apply the compiler (§4.1) to convert retrieved graph data into LLM-readable context.
5. **Response generation:** Pass compiled context + query to an LLM for answer generation.

Additionally, PIE supports **world snapshots** (reconstructing full world state at any period by replaying transitions up to that point) and **period diffs** (computing what changed between two life periods).

### 4.6 Dreaming Engine

A background process that performs offline reasoning over the world model to surface insights not explicit in any single conversation. Three tiers:

- **Micro-dreams** (after each ingestion batch): Check for unexpected connections between recently updated entities and distant graph neighbors.
- **Deep dreams** (nightly): Full consolidation pass, procedure re-extraction, belief evolution analysis, stale entity detection.
- **Mega-dreams** (weekly): Cross-domain synthesis using the strongest available model — identifying meta-patterns, blind spots, and predictions about the user's trajectory.

---

## 5. Experiments

### 5.1 Hypothesis 1: Semantic Temporal Context > Raw Timestamps

[TODO: need experiment results]

**Design:** 50 temporal reasoning questions over the same knowledge base, evaluated under four conditions: (A) raw Unix timestamps, (B) formatted dates, (C) relative time strings, (D) full semantic temporal context. Four models: GPT-4o, GPT-5, Claude Opus, Llama 3.3. Metric: answer accuracy (exact match + LLM-judge partial credit).

**Expected result:** D >> C > B > A, with the largest gains on questions requiring temporal ordering and change detection.

### 5.2 Hypothesis 2: Rolling Context Ingestion > Batch Isolation

[TODO: need experiment results]

**Design:** 50 consecutive conversations processed under two conditions: (A) independent extraction per conversation, (B) rolling context with activity-based preamble. Metrics: entity resolution F1 (against manually annotated ground truth), duplicate entity rate, state change detection recall.

**Expected result:** Condition B produces significantly fewer duplicate entities and higher state change recall, particularly for implicit entity references.

### 5.3 Hypothesis 3: Graph-Structural Importance > Heuristic Importance

[TODO: need experiment results]

**Design:** Build world model from 200+ conversations. Compute importance via both graph-structural method and heuristic baselines (message count, recency, keyword matching). Human annotator ranks 100 entities by actual importance. Metric: Spearman rank correlation.

### 5.4 Hypothesis 5: World Model Retrieval > Standard RAG

[TODO: need experiment results]

**Design:** Build both PIE and a standard RAG baseline (chunk conversations, embed, retrieve top-k) over the same data. 100 queries spanning five categories: current-state, temporal-diff, temporal-ordering, contradiction-detection, procedural. Metrics: LLM-judge accuracy + human eval on 30-question subset.

**Expected result:** PIE significantly outperforms on temporal-diff, ordering, and contradiction queries; comparable on current-state.

### 5.5 Benchmark Evaluation: LongMemEval

**Setup:** We evaluated on the full LongMemEval benchmark (500 questions across 6 categories) using our naive_rag baseline, with GPT-4o as the answering model and text-embedding-3-large for retrieval.

**Results (naive_rag baseline):**

| Category | Accuracy | Count |
|----------|----------|-------|
| single-session-assistant | **98.2%** | 56 |
| single-session-user | **84.3%** | 70 |
| knowledge-update | **79.5%** | 78 |
| temporal-reasoning | 59.8% | 133 |
| multi-session | 55.6% | 133 |
| single-session-preference | 6.7% | 30 |
| **Overall** | **66.3%** | 500 |

**Comparison to SOTA:**

| System | Accuracy |
|--------|----------|
| Emergence AI (Internal) | 86.0% |
| Supermemory | 71.4% |
| Zep | 71.2% |
| **Our naive_rag** | **66.3%** |

**Analysis:** Our naive_rag baseline achieves competitive results on single-session and knowledge-update categories, where simple embedding retrieval suffices. The large gap on preference questions (6.7%) reveals a fundamental limitation: preferences must be *inferred* from scattered context, not retrieved directly. Multi-session and temporal-reasoning require cross-session synthesis that chunked retrieval struggles with. These are exactly the categories where PIE's structured world model should provide advantages.

### 5.6 Benchmark Evaluation: LoCoMo

**Setup:** We evaluated on LoCoMo (Maharana et al., 2024), a benchmark of 10 very long-term conversations averaging 300+ turns across 27 sessions each. LoCoMo tests long-horizon memory across 1,986 QA pairs. We ran naive_rag baseline with embedding retrieval.

**Preliminary Results (25/50 questions, evaluation in progress):**

| Baseline | Accuracy (partial) |
|----------|-------------------|
| naive_rag | **62.5%** |

**Note:** Full results pending. LoCoMo's long conversations (588 turns avg, ~73K chars) stress-test retrieval systems. The benchmark includes five question types spanning temporal, causal, and factual reasoning over extended dialogues.

[Results will be updated when full evaluation completes.]

### 5.7 Benchmark Evaluation: Test of Time

**Setup:** We evaluated on Test of Time (Fatemi et al., 2024) with three conditions: (A) raw facts with dates ("baseline"), (B) embedding-based retrieval of relevant facts ("naive_rag"), (C) PIE's semantic temporal reformulation ("pie_temporal"). N=80 questions across relation types, GPT-4o-mini as answering model.

**Results:**

| Baseline | Accuracy |
|----------|----------|
| naive_rag | **56.2%** |
| baseline (raw facts) | 46.2% |
| pie_temporal | 31.2% |

**Critical Finding — Negative Result:** PIE's semantic temporal reformulation *hurts* performance on Test of Time by 25% compared to naive_rag. This was unexpected but highly informative.

**Analysis by question type:**

| Question Type | naive_rag | pie_temporal | Δ |
|---------------|-----------|--------------|---|
| relation_duration | 50% | 75% | **+25%** |
| first_last | 87.5% | 75% | -12.5% |
| event_at_what_time | 100% | 62.5% | **-37.5%** |
| event_at_time_t | 100% | 62.5% | **-37.5%** |

**Interpretation:** PIE's narrative reformulation helps *relative* temporal reasoning (durations, sequences) but severely damages *absolute* temporal lookup (specific dates, point-in-time queries). When the model receives "Entity X was active during Period A (~3 weeks ago)" instead of "Entity X: 2024-01-15", it loses the precision needed for arithmetic questions like "what happened on January 15?"

**Implication for System Design:** Temporal context compilation must be *task-adaptive*. For questions requiring date arithmetic, preserve exact timestamps. For questions about evolution, patterns, and relative ordering, use semantic compilation. A hybrid system that detects question type and selects context format accordingly is needed.

---

## 6. Results

[TODO: need experiment results — this entire section will be populated after running experiments]

### 6.1 Ablation: Semantic Temporal Compilation

### 6.2 Rolling Context vs. Isolated Extraction

### 6.3 Graph-Structural vs. Heuristic Importance

### 6.4 PIE vs. RAG Baseline

### 6.5 LongMemEval Results

### 6.6 LoCoMo Results

### 6.7 Test of Time Results

### 6.8 Qualitative Analysis

[TODO: Include examples of compiled temporal context, extracted procedures, contradiction detection, and period diffs from the actual PIE system running on real data]

---

## 7. Discussion

### 7.1 Why Temporal Matters

[TODO: expand after results — for now, theoretical argument]

The distinction between PIE and prior work is not primarily architectural — it is ontological. Existing memory systems model the user as a *set of facts*. PIE models the user as a *world that evolves through time*. This is not a superficial difference. A set of facts is atemporal: "user works on project X," "user prefers Y." A world model is inherently temporal: "user started X during period A, pivoted it during period B, and it currently looks like Z — and this trajectory matches a pattern seen in two prior projects."

The practical implications are significant. Temporal understanding enables:
- **Proactive assistance:** If the system knows a project follows a familiar trajectory, it can anticipate needs.
- **Contradiction surfacing:** If a user's stated beliefs conflict with their actions (tracked via state transitions), the system can flag this.
- **Decision support:** By showing how past decisions in similar contexts played out, the system provides historically-grounded advice.

### 7.2 Limitations

**Single-user study.** Our evaluation is on a single user's data. While the system architecture is general, we cannot yet claim that results generalize across different conversation patterns, usage frequencies, or topical distributions.

**Extraction quality ceiling.** PIE's world model quality is bounded by the extraction LLM's ability to identify entities, state changes, and relationships. Extraction errors compound — a missed entity in conversation 100 means all subsequent references may also be missed or misresolved.

**Cost.** Full ingestion of 4,758 conversations with GPT-5 is estimated at ~$350–500. This is acceptable for a research prototype but prohibitive for wide deployment without more efficient extraction approaches.

**Personal data sensitivity.** A comprehensive world model of a user's professional and personal life raises significant privacy concerns. PIE is designed for local-first deployment, but the model itself represents a high-value target.

### 7.3 Future Work

- **Multi-user world models:** Extending PIE to model shared context between collaborators
- **Real-time ingestion:** Live MCP integration where ongoing conversations update the world model incrementally
- **Proactive surfacing:** Using procedural memory and dreaming insights to proactively offer relevant context before the user asks
- **Cross-platform ingestion:** Extending beyond ChatGPT to Slack, email, notes, and other knowledge-worker data sources
- **Benchmark contribution:** A temporal state tracking benchmark for conversational memory systems, addressing gaps identified in existing benchmarks (§2)

---

## 8. Conclusion

[TODO: write after results — will summarize contributions and findings]

We presented PIE, a system for constructing temporal knowledge graphs from longitudinal conversational history. Our four contributions — semantic temporal context compilation, rolling context ingestion, procedural memory extraction, and graph-structural tiered forgetting — address fundamental limitations in how current agent memory systems handle the temporal dimension of personal knowledge. [TODO: fill in key results and implications after experiments]

---

## References

Cai, B., Xiang, Y., Gao, L., Zhang, H., Li, J., & Li, J. (2023). Temporal knowledge graph completion: A survey. *arXiv preprint arXiv:2308.02457*.

Chhablani, G., et al. (2024). Mem0: The memory layer for personalized AI. *GitHub repository*.

Fatemi, B., Halcrow, J., & Perozzi, B. (2024). Test of Time: A benchmark for evaluating LLMs on temporal reasoning. *ICLR 2025*.

Gastinger, J., et al. (2024). TGB 2.0: A benchmark for learning on temporal knowledge graphs and heterogeneous graphs. *NeurIPS 2024 Datasets and Benchmarks Track*.

Ha, D., & Schmidhuber, J. (2018). World models. *arXiv preprint arXiv:1803.10122*.

Jia, Z., Abujabal, A., Saha Roy, R., Strötgen, J., & Weikum, G. (2021). Complex temporal question answering on knowledge graphs. *CIKM 2021*.

Maharana, A., Lee, D., Tulyakov, S., Bansal, M., Barbieri, F., & Fang, Y. (2024). Evaluating very long-term conversational memory of LLM agents. *ACL 2024*.

Packer, C., Wooders, S., Lin, K., Fang, V., Patil, S. G., Stoica, I., & Gonzalez, J. E. (2023). MemGPT: Towards LLMs as operating systems. *arXiv preprint arXiv:2310.08560*.

Plastic Labs. (2025). Honcho: Personalization infrastructure for AI agents. *Documentation*.

Wu, X., et al. (2024). LongMemEval: Benchmarking chat assistants on long-term interactive memory. *ICLR 2025*.

Zhao, A., Huang, D., Xu, Q., Lin, M., Liu, Y.-J., & Huang, G. (2023). ExpeL: LLM agents are experiential learners. *arXiv preprint arXiv:2308.10144*.

Zep Inc. (2024). Graphiti: Build real-time, temporally aware knowledge graphs. *GitHub repository*.

[TODO: Complete and verify all citations — add missing references for Kosmos, Supermemory, StoryBench, MEMTRACK, and others mentioned in text]
