# The Unit of Memory Is the State Transition

*What I learned building a temporal knowledge graph for agent memory — and why every existing system gets consolidation wrong.*

---

Every AI memory system implements some version of three stages: encoding, consolidation, and retrieval. Encoding is solved — LLMs extract salient information from text beautifully. Retrieval is mostly solved — embedding search works. The problem is consolidation, and nobody is even trying.

Consolidation is where the human brain integrates new information with what it already knows. It detects contradictions. It updates beliefs. And crucially, it stores *change itself* as a meaningful signal. When you learn your friend switched jobs, you don't just overwrite "works at Google" with "works at Anthropic." You store the transition — the fact that something changed, when it changed, and what it changed from. That transition is more informative than either endpoint.

Current systems skip this entirely. They store facts. They don't store how facts evolve. I spent the last few months finding out what happens when you take the opposite approach.

---

## The Structural Problem

Start from the substrate. A text embedding maps a string to a point in high-dimensional space. The query "What was I working on last March?" lands near texts containing "working," "March," "projects." But the embedding has no mechanism to resolve "last March" to a concrete date range. Cosine similarity operates on a spatial axis. It has no temporal axis.

This isn't patchable. "X happened before Y" and "X happened after Y" produce near-identical vectors. "Project X is in design" and "Project X launched" are semantically similar — both about Project X — but the second *supersedes* the first, and no retrieval system built on embeddings alone can represent supersession.

You can attach timestamps. But timestamps tell you *when something was recorded*. They don't tell you *what kind of change occurred*. Was this an update? A contradiction of prior state? A resolution of a flagged conflict? The delta is missing.

Here's the concrete version. A user tells their AI assistant "I'm vegetarian" on Monday, then "I'm pescatarian" on Friday. In a flat vector database, both facts coexist as equally valid points. A conflict resolution system overwrites Monday with Friday. Problem solved? No — overwriting loses the most informative signal: *the user's dietary preferences are evolving*. The trajectory is more useful than either snapshot.

And consider temporal validity. "I'm angry" expires in hours. "I'm vegetarian" lasts months. "I have a PhD" is permanent. "Working on Project X" lasts maybe a month. No current memory system makes this distinction. Every fact gets identical treatment regardless of its temporal decay profile. Within a day, a third of conversational facts are stale. Within a month, over half.

---

## The Benchmarks Confirm It

Three recent benchmarks quantify the gap.

**Test of Time** (Fatemi et al., ICLR 2025) tests pure temporal reasoning by anonymizing entities to prevent shortcutting. The finding: LLMs largely *fake* temporal reasoning. The most common error on duration questions was a deviation of *exactly one day* — not random noise, but systematic temporal arithmetic failure. These models aren't reasoning about time. They're pattern-matching against templates.

**LongMemEval** (Wu et al., ICLR 2025) benchmarks chat assistants on 500 questions across conversation histories reaching 1.5M tokens. The pattern: basic fact recall scores 60–80%, while temporal reasoning and knowledge update questions score 15–40% lower. Detecting that information has *changed* is consistently the hardest task.

**TReMu** (ACL 2025 Findings) demonstrated that the fix isn't bigger models — it's better temporal representation. By generating Python code for temporal calculations instead of asking the LLM to do temporal arithmetic, they achieved a 2.6x improvement (29.83 → 77.67 on GPT-4o). The LLM was never the bottleneck. The *representation* was.

Every benchmark, every system, every model: temporal reasoning is the weakest dimension. And these benchmarks only test *simple* temporal reasoning — ordering, recency, basic sequencing. None test trajectory reconstruction, cross-entity pattern detection, or contradiction resolution over time.

---

## What Exists Today

The landscape has matured fast. Each major system solves part of the problem.

**Mem0** stores key-value facts with vector embeddings. Good for basic personalization. When preferences change, it either overwrites (losing history) or stores both versions (creating ambiguity). No model of how or why things changed.

**Zep's Graphiti** builds a temporal knowledge graph with a bi-temporal model — tracking both when an event occurred and when it was ingested. The most sophisticated temporal handling in production. But bi-temporal edges track *when facts were known*, not *what kind of change occurred*.

**Letta (MemGPT)** treats context as virtual memory. The agent pages information in and out as needed. This correctly solves context management. It doesn't address how remembered things change over time.

**Mastra's Observational Memory** deserves special attention. I dug into their codebase — it's fully open source TypeScript. Two background agents run asynchronously: an Observer that fires every ~30K tokens and compresses conversations into dense observations, and a Reflector that fires every ~40K tokens to garbage-collect stale observations. The result is an append-only log — no vector database, no graph, no retrieval at all. They stuff the entire compressed log into context. It works because observations are denser than raw messages, and modern context windows are huge. They hit 94.87% on LongMemEval.

This taught me something important: for standard memory tasks, dense text with good temporal formatting may be sufficient. But ask Mastra "how has my approach to database selection evolved across my last five projects?" and it scans the log linearly, hoping relevant observations are nearby. No entity-level indexing. No cross-entity pattern detection. No structured representation of how things change.

---

## The Core Idea

Don't store facts. Store typed state transitions.

Instead of recording "user is working on Project X," you record:

```json
{
  "entity": "Project X",
  "transition_type": "creation",
  "to_state": "Evaluating for popup implementation",
  "timestamp": "2025-01-05",
  "trigger": "User mentioned exploring popup tools"
}
```

Three days later, an update. A week later, a contradiction — the user discovers Framer's per-editor pricing and their assessment of "cost-effective" flips to "concerned about costs." Five transition types cover the full lifecycle: **creation → update → contradiction → resolution → archival**.

The transition chain for an entity IS the temporal model. Replay it for full history. Compare chains across entities for lifecycle patterns. Detect contradictions for belief evolution. Analyze transition velocity for stability assessment.

I built this in PIE (Personal Intelligence Engine). It processes conversation history chronologically, extracting entities and typed state transitions into a temporal knowledge graph. Each batch is processed in the context of the accumulated world model — the system knows what's active, what's recently changed, and can attribute ambiguous references based on activity patterns.

The current world model from my actual ChatGPT conversation data:

```
  Entities:      873  (275 events, 163 concepts, 142 tools, 136 projects,
                       73 decisions, 41 organizations, 32 beliefs, 9 people)
  Transitions: 1,491
  Relationships: 717
```

That's not synthetic data. That's every project I've discussed, every tool I've evaluated, every decision I've made, every belief I've updated — extracted into a structured temporal graph.

---

## What This Unlocks

### Contradiction Detection

A user's database technology choice evolving over time. Flat vector store sees six independent facts. State transition store sees a trajectory: MySQL → evaluating PostgreSQL → **contradiction** (switched to PostgreSQL) → validated PostgreSQL → **contradiction** (reconsidering MySQL) → **resolution** (PostgreSQL final).

Same query ("What database does the user use?") gets a fundamentally different quality of answer. The flat store returns the most recent fact. The transition store returns: PostgreSQL, *resolved* after deliberate evaluation with 2 contradictions. The system understands not just *what* but *how confident to be* and *why*.

### Procedural Memory

This is where the real value hides. Running pattern extraction across PIE's actual world model data:

```
  Evaluate → Commit     (12 entities, avg 4.3 transitions over 6.0 days)
  High Velocity Iteration (10 entities, avg 4.2 transitions over 2.3 days)
  Project Lifecycle      (18 entities, avg 2.7 transitions over 2.0 days)
```

These patterns don't exist in any single conversation. They emerge from analyzing transition sequences across entities. This is the structural equivalent of procedural memory — "how I typically evaluate technology" or "what my project lifecycle looks like." With 873 entities, the patterns are becoming statistically meaningful.

### Temporal Diff

"What changed in my projects between Q1 and Q3?" requires reconstructing world state at two time points and computing a delta. With state transitions, replay the chain to any timestamp for point-in-time state. With flat stores, re-query everything and hope retrieval surfaces the right facts.

---

## Where This Matters

Personal memory demonstrates the concept. Enterprise applications demonstrate the value.

**Sales intelligence.** Current CRMs track deal *stages*. Temporal memory tracks deal *trajectories*. Budget freezes, champion departures, competitor entries — these are contradictions in a transition chain. Pattern-match against historical deal trajectories and you get predictive risk flagging that's structurally impossible with stage-based tracking.

**Multi-agent orchestration.** When Agent-A produces a finding at 10:10 that contradicts Agent-B's starting assumption from 10:02, the orchestrator needs to detect the contradiction in real time. This is orchestration — reasoning about the evolving state of a distributed system. Stateless task dispatch can't do this.

**Proactivity.** True agent proactivity requires temporal pattern recognition ("your projects hit blockers at month 3"), temporal validity modeling ("your meeting prep from last week is stale"), and future state prediction ("a deal going silent for 14 days after POC matches a high-risk pattern"). All natural consequences of state transition tracking. All impossible with flat fact stores.

---

## The Honest Assessment

Let me be direct about where PIE stands. Real numbers.

**PIE on LoCoMo: 69% overall, 76% on the hardest single conversation.** LoCoMo tests 1,986 questions across 10 long conversations. Our breakdown by type:

```
  Commonsense:   100%    (common knowledge inference)
  Adversarial:    61%    (deliberately misleading speaker attributions)
  Single-hop:     61%    (basic factual recall)
  Multi-hop:      59%    (reasoning across multiple facts)
  Temporal:        52%    (date ordering, "when" questions)
```

The irony: a system built for temporal reasoning scores *lowest* on temporal questions. But the diagnosis is instructive. Temporal questions fail not because the graph can't represent the information — the entities and transitions are there — but because extraction misses precise dates and retrieval sometimes surfaces the wrong entities. The architecture is right. The implementation needs work.

I traced every wrong answer on the best-performing conversation. Five root causes: extraction gaps (~15 questions), adversarial speaker swaps (~15), retrieval misses (~8), date precision (~8), scoring artifacts (~3). Half are fixable with better extraction prompts. The other half need retrieval improvements.

**The competitive landscape:**

```
  CORE             88.24%  LoCoMo      (closed-source, GPT-4o)
  Mastra OM        94.87%  LongMemEval (observation log, no retrieval)
  MemMachine       84.87%  LoCoMo      (open-source knowledge graph)
  Zep/Graphiti     75.14%  LoCoMo      (bi-temporal KG)
  Full Context     ~85%    LoCoMo      (stuff everything in prompt)
  PIE (ours)       69%     LoCoMo      (temporal KG, state transitions)
  Mem0             ~65-70% LoCoMo      (flat fact store)
  Naive RAG        ~55%    LoCoMo      (chunk + embed + top-k)
```

PIE at 69% is behind the leaders. Full context baselines — just stuffing the entire conversation into the prompt — score ~85%. For conversations that fit in a context window, you don't need a knowledge graph.

But full context doesn't scale. A 1.5M token conversation history won't fit in any context window. PIE's graph compresses 10 conversations into a structured world model that retrieves in milliseconds. The question is whether we can close the accuracy gap while keeping that scalability advantage.

After reading Mastra's codebase, I respect their engineering but see the ceiling. Their observation log works beautifully for conversations that compress well. When the log exceeds the context window, you lose information. PIE's graph doesn't have that ceiling.

**Path forward:** Close the accuracy gap on LoCoMo to 80%+. Run LongMemEval head-to-head against Mastra's 94.87%. Replicate Mastra and other methods on our benchmark infrastructure for real apples-to-apples comparison. Build trajectory benchmarks that test what only state transitions can answer. And if PIE can't beat observation logs on *any* benchmark — be honest that the graph is over-engineering.

---

## What's Novel

**Not novel:** Entity extraction with LLMs. Knowledge graphs. Embedding search. Tiered resolution. Hybrid retrieval. These are established techniques.

**Novel:**

**Typed state transitions as memory primitive.** No production system models entity changes as creation/update/contradiction/resolution/archival sequences with full provenance. Graphiti tracks temporal edges. Mastra tracks observation timestamps. PIE tracks typed change semantics.

**Entity retrieval, not chunk retrieval.** Most RAG systems retrieve text chunks. PIE retrieves *entities with their full state history* via reciprocal rank fusion over BM25 keyword scores and embedding similarity. The retrieval unit is a living object with a timeline, not a dead text fragment.

**Procedural memory from cross-entity lifecycle analysis.** Extracting behavioral patterns from transition sequences across multiple entities. Nobody else extracts "how you evaluate technology" from entity lifecycle data.

**Adversarial-aware temporal answering.** LoCoMo taught us that ~15% of benchmark questions deliberately swap speaker names. Our answer pipeline explicitly handles this — no other system we've found builds adversarial robustness into the answering stage.

---

## Where This Goes

The immediate work is closing the accuracy gap. PIE at 69% needs to reach 80%+ before the architectural argument carries weight. The failure analysis shows this is achievable — half the errors are extraction gaps, a quarter are retrieval ranking issues. These aren't architectural limitations. They're engineering debt.

The deeper work is proving that state transitions unlock capabilities observation logs can't. That requires new benchmarks. ConvoMem (75,000 questions, Salesforce 2025) tests memory at scale. MemoryAgentBench tests four competencies — retention, association, generalization, timeliness — that map directly to PIE's architecture. BEAM (10M+ tokens, ICLR 2026) tests extreme-length memory that exceeds any context window. These are the battlegrounds where PIE's graph should shine and observation logs should fail.

The simplest version of the thesis: current memory systems answer "what do I know?" Temporal memory systems answer "what has changed, and what does that imply?"

The delta is where the intelligence lives.

---

**Code:** [github.com/pkempire/pie](https://github.com/pkempire/pie)

**References:**

Tulving, E. "Elements of Episodic Memory." Oxford University Press, 1983.

Wu et al. "LongMemEval." ICLR 2025. arXiv:2410.10813.

Maharana et al. "LoCoMo." ACL 2024. arXiv:2402.17753.

Fatemi et al. "Test of Time." ICLR 2025. arXiv:2406.09170.

Sehgal et al. "Temporal Awareness Failures in LLM Dialogues." arXiv:2601.13206.

TReMu. "Temporal Reasoning in Multi-Session Dialogues." ACL 2025 Findings. arXiv:2502.01630.

Rasmussen et al. "Zep/Graphiti." arXiv:2501.13956.

Packer et al. "MemGPT." arXiv:2310.08560.

Park et al. "Generative Agents." arXiv:2304.03442.

Kosmos. "An AI Scientist for Autonomous Discovery." arXiv:2511.02824.

"Memory in the Age of AI Agents." arXiv:2512.13564.

Mastra AI. "Observational Memory." github.com/mastra-ai/mastra. 2025.

---

*Built in public. Feedback welcome.*
