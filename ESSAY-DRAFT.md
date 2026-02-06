# The Temporal Memory Gap: Why Agent Memory Systems Can't Reason About Change

*Current agent memory stores what happened. It doesn't model how things change. That's the bottleneck.*

---

## The Failure Mode

Ask any AI assistant with memory about its user: "How has my thinking about this project evolved?"

Watch what happens. The system retrieves chunks — semantically similar snippets from different times. Maybe something from last month. Maybe something from six months ago. The response is a collage of excerpts with no causal ordering, no sense of what changed when, no model of trajectory.

The answer is technically grounded in source material. It's also useless for the question asked.

This isn't cherry-picked. LongMemEval (Wu et al., ICLR 2025) shows every major memory system — including frontier models with full context — degrades severely on temporal reasoning compared to basic fact retrieval. LoCoMo (Maharana et al., ACL 2024) reports the same pattern. Test of Time (Fatemi et al., ICLR 2025) goes further: LLMs largely *fake* temporal reasoning through memorized facts. When you anonymize entities to prevent shortcutting, accuracy collapses.

The pattern is clear: agent memory systems handle *what* but not *when* or *how things changed*.

This isn't incremental. It's structural.

---

## Why Embeddings Can't Encode Time

Text embeddings encode semantic similarity — how much two texts are "about the same thing." The query "What was I working on last March?" gets embedded into a vector close to texts containing "working," "March," "projects."

The embedding has no mechanism to:
1. Resolve "last March" to an absolute time range
2. Filter by temporal window (cosine similarity has no time axis)
3. Distinguish "X before Y" from "X after Y" (may have near-identical embeddings)
4. Model supersession — "Project X launched" should invalidate "Project X is in design"

Timestamps on facts don't fix this. You can attach `created_at: 2024-03-15` to a chunk. The retrieval system still can't answer "what changed between Q1 and Q3?" because it has no model of change — just timestamped snapshots with no transition semantics.

---

## The Unit of Memory Is the Transition

The fix requires changing the primitive. Don't store facts. Store state transitions.

Here's a real example from my own ChatGPT history. PIE tracked my relationship with Framer (a design tool) through 6 state transitions:

```
[1] 2025-01-05 | creation
    → "evaluating for popup implementation"

[2] 2025-01-05 | update  
    → "building scroll-triggered slide-in popup"

[3] 2025-01-06 | update
    → "exploring cost-saving strategies for editor access"
    Trigger: "moved from evaluating to worrying about $50/editor fees"
```

This isn't just "uses Framer." It's the trajectory: started evaluating → built a specific feature → hit cost concerns. A flat fact store loses this evolution. A state transition chain preserves it.

The transition chain IS the temporal model. Replay it to reconstruct history. Compare chains across entities to find patterns.

This is what PIE does: extracts entities and typed state transitions from conversation history, building a temporal knowledge graph where change is first-class.

---

## The Unexpected Finding

Here's where it gets interesting.

We built PIE expecting that converting temporal data into rich semantic narratives would uniformly improve reasoning. "14 months ago, during sophomore year, while exploring graph databases" seems more informative than "2024-03-15T14:30:00Z."

We were half right.

Testing on Test of Time (a temporal reasoning benchmark), we broke down results by query type:

| Query Type | Raw Timestamps | Semantic Narrative | Δ |
|------------|---------------|-------------------|---|
| first_event | 45% | 53% | **+8%** |
| last_event | 38% | 63% | **+25%** |
| event_ordering | 62% | 70% | **+8%** |
| event_at_time_t | 68% | 30% | **-38%** |
| time_of_event | 55% | 32% | **-23%** |

Semantic reformulation helps *relative* queries (first, last, ordering) while significantly hurting *absolute* queries (point-in-time lookup, date arithmetic).

Why? 

Relative queries need sequence understanding: "What came before X?" requires knowing the order. "14 months ago, then pivoted at 11 months" directly encodes this.

Absolute queries need precision: "What happened on March 15?" requires exact date matching. "About 14 months ago" loses the information needed to answer.

---

## The Implication

This finding changes how temporal memory should work.

The right architecture isn't "timestamps vs. narratives." It's a hybrid that picks format based on query type:

- Relative queries → semantic compilation
- Absolute queries → preserve timestamps
- Hybrid queries → provide both

Detection is straightforward:
- Contains "first/last/before/after/earlier/later" → relative
- Contains specific dates or "on [date]" → absolute
- Contains "how long/duration/weeks/months" → relative
- Contains "what date/when exactly" → absolute

No existing system does this. They uniformly apply one format or the other.

---

## What This Unlocks

Get temporal memory right and several capabilities become possible:

**Trajectory reconstruction.** "How has my position on X evolved?" Answer requires replaying the belief entity's transition chain — which PIE maintains but flat fact stores don't.

**Temporal diff.** "What changed between Q1 and Q3?" Requires reconstructing world state at two points and computing delta. Needs transition replay, not just filtered retrieval.

**Pattern extraction.** "Do I follow a pattern when starting projects?" Requires analyzing transition sequences across entity lifecycles. Structurally impossible without explicit transition chains.

**Realtime coworking.** An agent that maintains a live model of project state, detecting when new information contradicts prior understanding, flagging drift, predicting bottlenecks. This needs contradiction detection and state tracking — not keyword search.

These aren't marginal improvements. They're capabilities that current architectures literally cannot provide.

---

## Current State

PIE is open-source (github.com/pkempire/pie). Here's what we have:

**Working:**
- Conversation parser for ChatGPT exports (extensible to other formats)
- Daily batch extraction with sliding window context
- Typed entity extraction (people, projects, tools, beliefs, decisions, events)
- Three-tier entity resolution (string → embedding → LLM verification)
- State transition tracking with contradiction detection
- Force-directed graph visualization

**Evaluated (baselines):**
- LongMemEval: 66.3% (naive RAG)
- LoCoMo: 58% (naive RAG)
- Test of Time: 56.2% (naive RAG), 31.2% (PIE semantic — hurt by absolute queries)

**In progress:**
- Task-adaptive query routing (detect query type, pick format)
- Full PIE evaluation on all benchmarks
- Procedural pattern extraction across entity lifecycles

**Honest limitations:**
- Extraction quality depends on LLM capability
- Compute cost is non-trivial (~$20-50 for 1000+ conversations)
- Single-dataset validation so far

---

## The Broader Point

Agent memory has converged on a comfortable abstraction: facts are retrieved by similarity, context is stuffed into prompts, the LLM figures it out.

This works for what. It fails for when and how things changed.

Fixing it requires changing primitives — from facts to transitions, from timestamps to typed change semantics, from uniform formatting to task-adaptive compilation.

The temporal memory gap isn't a feature to add. It's a rearchitecture to do.

PIE is one attempt. The finding about task-adaptive formatting generalizes beyond it.

---

**Code:** github.com/pkempire/pie

---

*Thanks to [collaborators] for feedback on this work.*
