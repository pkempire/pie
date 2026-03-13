# The Self-Compiling Agent
## Behavioral Memory as a Step Function in Persistent AI

---

## The Core Insight (30 seconds)

Every memory system in existence — Mem0, Zep, Mastra, GraphRAG, OMEGA, Hindsight — stores **knowledge about the world**. Facts. Entities. Observations. Relationships.

But that's not what makes a great assistant great.

What makes a great assistant great is that they've learned **how to work with you specifically**. They know your shorthand. They anticipate your needs. They've internalized your correction patterns. They don't just know facts about you — they've rewritten their own behavior based on every interaction.

**The unit of persistent memory should not be a fact. It should be a behavioral diff.**

Instead of: `"Pranay works on PIE, a memory system project"`
Store: `"When Pranay says 'lock in', shift from breadth to depth immediately. He's signaling that the current response is too surface-level."`

One is knowledge. The other is learned behavior. No existing system stores the second kind.

---

## What This Actually Is

A **self-compiling agent** that maintains and evolves a living **self-program** — a document that specifies not what it *knows* but how it should *behave*. After every interaction, the agent reflects on what happened, generates behavioral diffs, and rewrites its own instructions.

The self-program is not a knowledge base. It's a **behavioral specification** — compiled from the accumulated diffs of every interaction. It tells the agent: given this user, in this context, at this time, here's exactly how to act.

```
┌──────────────────────────────────────────────────────┐
│              THE SELF-COMPILATION LOOP                │
│                                                      │
│  ┌─────────┐   ┌──────────┐   ┌─────────────────┐   │
│  │  INTERACT │→│  REFLECT  │→│  GENERATE DIFFS  │   │
│  │  (conv)  │  │  (what    │  │  (behavioral     │   │
│  │          │  │  worked/  │  │   changes to     │   │
│  │          │  │  failed)  │  │   self-program)  │   │
│  └─────────┘  └──────────┘  └────────┬──────────┘   │
│                                       │              │
│  ┌─────────────────────────────┐     │              │
│  │  APPLY DIFFS TO SELF-PROGRAM │←───┘              │
│  │  (rewrite own instructions) │                    │
│  └─────────────────────────────┘                    │
│       │                                              │
│       ↓                                              │
│  ┌─────────────────────────────────────────────┐     │
│  │  NEXT CONVERSATION STARTS WITH              │     │
│  │  UPDATED SELF-PROGRAM (warmer, smarter)     │     │
│  └─────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────┘
```

---

## Why This Is a Step Function (Not Incremental)

### What exists now: Knowledge Memory
```
User says something → Extract facts → Store in DB → Retrieve later → Inject into context
```
The LLM receives: "Here are some facts about this user."
The LLM processes this as: **new information it's reading for the first time.**

### What this proposes: Behavioral Memory
```
Conversation happens → Reflect on interaction → Generate behavioral diffs → Update self-program → Next conversation loads self-program
```
The LLM receives: "Here's how to behave with this user."
The LLM processes this as: **instructions for how to act.**

This is a fundamental category difference. Knowledge memory makes the LLM **know more**. Behavioral memory makes the LLM **act better**. The research is clear on which matters more:

- **ACE (ICLR 2026)**: "Evolving playbooks" that accumulate strategies without weight updates outperform knowledge retrieval. +10.6% on AppWorld, matches GPT-4.1 using smaller models.
- **ParamMem (Feb 2026)**: Encoding reflection patterns into parameters outperforms external memory by 38%.
- **MemGen (Sep 2025)**: Treating memory as generative hidden states (not retrievable text) outperforms all retrieval-based approaches.

The field is converging on: **the right memory isn't what you know — it's how you've learned to behave.**

---

## The Four Components

### 1. The Self-Program (the living document)

A structured markdown document that IS the agent's behavioral specification for a specific user. Not a database — a program.

```markdown
# Self-Program: Pranay | v47 | Last compiled: 2026-03-02T14:30Z

## Identity
Working relationship: research collaborator on AI memory systems.
Communication style: direct, technical, adversarial-constructive.
Key tension: Pranay thinks in systems and hates surface-level responses.
He uses profanity as emphasis, not aggression.

## Behavioral Rules (learned from 47 interactions)
1. Never open with affirmations ("Great question!", "That's interesting")
   — Pranay flagged this as patronizing in interaction #3
2. When Pranay says "lock in" → current response is too shallow, go 3x deeper
3. When Pranay says "this is boring" → STOP iterating on current approach,
   pivot entirely to a new framing
4. Always do deep web research (5+ sources) before making SOTA claims
   — Pranay caught me making outdated claims twice (interactions #6, #12)
5. Start architectural discussions with WHY, not WHAT
   — learned from correction in interaction #8
6. Present code over prose for technical ideas
   — implicit preference detected across interactions #5-#15
7. When Pranay pushes back, it means the idea needs fundamental rethinking,
   not better explanation of the same idea
   — this is the #1 mistake to avoid (repeated in interactions #3, #7, #11, #14)

## Active Threads
### Thread: PIE Memory System [ACTIVE]
- Status: searching for step-function architecture
- Last: 2026-03-01 (1 day ago)
- Predicted next: TODAY (high confidence — 3-day work streak)
- Key context: rejected append-only log, wants genuine breakthrough
- Blog post "Memory Isn't a Snapshot" ~80% done
- Benchmarks at 0% (known issue, not priority)
- Constraint: no entity extraction dependency

### Thread: Job Search [DORMANT]
- Last: 2026-02-25 (5 days ago)
- Survival: 0.72 (likely to resurface within 7 days)
- Key context: applied to 3 companies, waiting on responses

## Proactive Queue
- [ ] Blog post deadline: user mentioned "by Friday" on 2026-02-28
      → Friday has passed. Surface this.
- [ ] PIE benchmarks at 0% → not addressed in 5 sessions.
      Surface when thread is less urgent.
```

This document is loaded at the start of every conversation. It's not "memories" — it's **instructions the agent wrote for its future self**.

### 2. The Reflection Engine (post-conversation)

After each conversation, a reflection step generates behavioral diffs:

```python
class ReflectionEngine:
    """
    Analyzes a completed conversation against the current self-program
    and generates behavioral diffs.

    This is the core innovation: instead of extracting FACTS from the
    conversation (what Mem0/Zep/Mastra do), we extract BEHAVIORAL LESSONS.

    No entity extraction. No knowledge graph updates. Just: what should
    the agent do differently next time?
    """

    def reflect(self, conversation: list[Message], self_program: str) -> list[Diff]:
        """
        Generate behavioral diffs from a completed conversation.

        Signal sources (implicit, no explicit feedback needed):
        - User corrections: "no, I meant..." → agent misunderstood
        - User frustration: "this is boring", "lock in" → approach was wrong
        - User engagement: long follow-ups → approach was right
        - User satisfaction: "this is great", "exactly" → reinforce behavior
        - Conversation flow: did the agent predict correctly?
        - Task completion: was the user's goal accomplished?
        """

        # The reflection prompt — this is the key prompt engineering
        prompt = f"""
        You are reflecting on a completed conversation to improve future behavior.

        Current self-program:
        {self_program}

        Conversation transcript:
        {format_conversation(conversation)}

        Generate behavioral diffs. Each diff should be one of:
        - RULE_ADD: A new behavioral rule learned from this interaction
        - RULE_MODIFY: An existing rule that needs refinement
        - RULE_REMOVE: A rule that proved wrong or counterproductive
        - THREAD_UPDATE: Status change for an active thread
        - THREAD_CREATE: A new thread of work/interest detected
        - PROACTIVE_ADD: Something to proactively surface later
        - PERSONA_UPDATE: Updated understanding of the user

        For each diff, explain:
        1. What happened in the conversation that triggered this learning
        2. The specific behavioral change
        3. Confidence (0-1) based on signal strength

        CRITICAL: Only generate diffs you're confident about.
        One high-quality diff >> ten vague ones.
        """

        return self.llm.generate(prompt)
```

**Key difference from existing systems**: Mem0 extracts `("Pranay", "works_on", "PIE")`. This extracts `"When discussing PIE architecture, start with the problem being solved, not the technical approach — learned from user frustration in this conversation."` One is a fact. The other is a behavioral instruction.

### 3. The Temporal Thread Tracker

Threads replace entities. A thread is an ongoing stream of work/interest — detected automatically from conversation clustering, tracked with temporal metadata.

**No entity extraction needed.** A thread is defined by:
- A cluster of related conversation segments (detected via embedding similarity)
- Temporal metadata (first seen, last touched, touch frequency)
- A survival probability (will this thread be active again?)

This directly reuses PIE's survival function math:

```python
class ThreadTracker:
    """
    Tracks active threads using PIE's survival function framework.

    Key reuse from pie/core/temporal.py:
    - Rhythm: per-thread temporal signature (mean interval between touches)
    - SurvivalTable: empirical P(thread touched again | time since last touch)
    - Entity-relative time normalization: universal survival curve

    The insight: threads behave like entities in temporal.py.
    They have clock speeds (how often touched), survival curves
    (probability of being touched again), and anomaly detection
    (touched sooner/later than expected).
    """

    def predict_next_thread(self, current_time: float) -> list[ThreadPrediction]:
        """
        Predict which thread(s) the user will engage with.

        Uses Hawkes process intensity:
        λ_thread(t) = μ_thread + Σ α · exp(-β · (t - t_i))

        Where:
        - μ_thread = baseline intensity (how often this thread is touched)
        - t_i = timestamps of past touches
        - α, β = excitation parameters (learned from data)

        Also incorporates:
        - Time-of-day patterns (user works on X in mornings)
        - Day-of-week patterns (user does Y on Mondays)
        - Recency: threads touched recently have higher intensity
        - Co-occurrence: if thread A was touched, thread B often follows
        """
        predictions = []
        for thread in self.active_threads:
            intensity = self.hawkes_intensity(thread, current_time)
            survival = self.survival(thread, current_time)
            predictions.append(ThreadPrediction(
                thread=thread,
                probability=intensity * survival,
                staleness=self.staleness(thread, current_time),
                urgency=self.compute_urgency(thread, current_time),
            ))
        return sorted(predictions, key=lambda p: p.probability, reverse=True)

    def staleness(self, thread: Thread, t: float) -> float:
        """
        How stale is this thread? Uses entity-relative time from temporal.py.

        k = (t - last_touch) / mean_interval
        staleness = 1 - survival(k)

        A thread with mean_interval=2 days that hasn't been touched in 6 days
        has k=3.0, and the universal survival curve gives S(3.0) ≈ 0.15,
        so staleness = 0.85. Very stale.
        """
        rhythm = self.rhythms[thread.id]
        if not rhythm.has_data:
            return 0.0
        k = (t - rhythm.last_transition_t) / rhythm.mean_interval
        return 1.0 - self.survival_table.survival(k)
```

### 4. The Predictive Pre-Loader

Before each conversation, assembles the optimal context:

```python
class PredictivePreLoader:
    """
    Assembles context BEFORE the user says anything.

    This is what makes the agent proactive rather than reactive.
    Instead of waiting for a query and then retrieving, we PREDICT
    what context will be needed and pre-load it.

    Components:
    1. Self-program (always loaded — this IS the agent's identity)
    2. Predicted thread context (based on temporal patterns)
    3. Proactive items (commitments, deadlines, stale threads)
    4. Conversation starters (based on prediction confidence)
    """

    def pre_load(self, current_time: float) -> PreLoadedContext:
        # Always load: the self-program
        context = PreLoadedContext(self_program=self.self_program)

        # Predict: which thread(s) will the user engage with?
        predictions = self.thread_tracker.predict_next_thread(current_time)
        top_prediction = predictions[0]

        if top_prediction.probability > 0.7:
            # High confidence — pre-load this thread's full context
            context.primary_thread = top_prediction.thread
            context.thread_context = self.get_thread_context(top_prediction.thread)
            context.opener = self.generate_proactive_opener(top_prediction)
        else:
            # Low confidence — load summaries of top 3 threads
            context.thread_summaries = [
                self.summarize_thread(p.thread) for p in predictions[:3]
            ]

        # Always check: proactive items
        context.proactive_items = self.check_proactive_queue(current_time)

        return context

    def generate_proactive_opener(self, prediction: ThreadPrediction) -> str:
        """
        Generate a natural opening that demonstrates temporal awareness.

        NOT: "Hello! How can I help you today?"
        NOT: "I remember you were working on PIE."

        YES: "Hey — it's been 2 days since we last worked on PIE.
             You were stuck on finding a step-function architecture.
             I did some research and have a new angle. Want to hear it?"

        The opener should:
        1. Acknowledge elapsed time naturally
        2. Reference the specific state of the predicted thread
        3. Demonstrate that the agent has been "thinking" (proactive items)
        4. Give the user an easy way to confirm or redirect
        """
        ...
```

---

## Why This Doesn't Depend on Entity Extraction

This is the critical constraint the user identified. Every existing system requires entity extraction — identifying "Pranay", "PIE", "memory system" as entities and their relationships. This is:
- Expensive (LLM calls per ingestion)
- Lossy (extraction misses nuance)
- Fragile (Mem0 has 40% fact extraction failure rate per GitHub issues)
- The wrong abstraction (entities are the system's concern, not the user's)

The self-compiling agent needs **zero entity extraction**:

| Component | What it stores | How it's created |
|-----------|---------------|-----------------|
| Behavioral rules | Natural language instructions | LLM reflection on conversation |
| Thread tracking | Conversation clusters + timestamps | Embedding similarity + temporal metadata |
| Persona model | Natural language description | LLM reflection on patterns |
| Proactive items | Natural language commitments | Detected during conversation |

Everything is natural language → natural language. No structured extraction. No entity resolution. No graph construction. The self-program is a markdown file that the agent reads and writes.

---

## Research Positioning

### Novel Contributions

1. **Behavioral memory as a distinct paradigm.** Existing taxonomy (from "Memory in the Age of AI Agents", Dec 2025) categorizes memory as factual, experiential, or working. Behavioral memory — stored modifications to the agent's own behavioral specification — is a fourth category that no existing system implements.

2. **Self-compilation loop.** ACE (ICLR 2026) introduced "evolving playbooks" but they're task-scoped. This extends the concept to the agent's core identity — the self-program evolves the agent's personality, communication style, and behavioral patterns, not just task strategies.

3. **Threads over entities.** Entity-based memory requires extraction. Thread-based memory requires only embedding clustering and temporal tracking. This eliminates the extraction bottleneck entirely while preserving the temporal dynamics (survival functions, Hawkes processes) that PIE already implements.

4. **Predictive pre-loading formalized as POMDP.** The agent maintains a belief state over which thread the user will engage with. The action is what context to pre-load. The observation is the user's first message. The reward is prediction accuracy + user satisfaction. This connects to Dream to Chat (Aug 2025) but applied to persistent personal AI rather than single-session dialogue.

### Connections to SOTA

| Paper/System | Relationship |
|-------------|-------------|
| **ACE** (ICLR 2026) | We extend evolving playbooks from task-scope to identity-scope |
| **MemGen** (Sep 2025) | We share the insight that memory should be generative, not retrievable |
| **AgeMem** (Jan 2026) | They learn when to read/write memory; we learn what behavioral changes to make |
| **ParamMem** (Feb 2026) | They encode patterns in weights; we encode patterns in natural language programs |
| **JitRL** (Jan 2026) | They modulate logits from experience; we modulate behavior from reflection |
| **Nemori** (Aug 2025) | They self-organize memory storage; we self-organize behavioral specifications |
| **ContextAgent** (May 2025) | They predict proactive service necessity; we predict thread engagement |
| **PIE temporal.py** | We reuse survival functions and entity-relative time for thread tracking |
| **PIE dynamics.py** | We reuse co-occurrence detection for thread correlation |

### What This Would Prove

The central thesis: **a self-compiling agent that stores behavioral diffs outperforms knowledge-memory agents on real-world persistent AI tasks.**

Measurable via:
- **Correction rate over time**: should decrease as behavioral rules accumulate
- **Prediction accuracy**: does the agent correctly predict which thread the user engages with?
- **Proactivity value**: do users find proactive openers/suggestions useful? (A/B test: proactive vs cold start)
- **Satisfaction trajectory**: does user satisfaction increase over interactions? (implicit from engagement patterns)

---

## What Reuses from PIE

Not throwing away what's built. Key reuse:

| PIE Component | Reuse |
|--------------|-------|
| `temporal.py` — Survival functions | Thread survival tracking (when will a thread be touched again?) |
| `temporal.py` — Rhythm / entity-relative time | Thread rhythm (each thread has its own clock speed) |
| `temporal.py` — Anomaly detection | Detecting surprising thread activations |
| `dynamics.py` — Co-occurrence detection | Thread correlation (if thread A activates, B often follows) |
| `dynamics.py` — Staleness scoring | Thread urgency computation |
| `ingestion/prompts.py` — Prompt engineering patterns | Reflection prompt design |

What gets replaced:
- Entity extraction → behavioral reflection
- Knowledge graph → self-program (markdown)
- State transitions → behavioral diffs
- Retrieval scoring → predictive pre-loading

---

## Build Plan (1 Week)

### Day 1-2: Self-Program + Reflection Engine
- Define self-program schema (markdown structure)
- Build reflection engine (post-conversation → behavioral diffs)
- Build diff application (merge diffs into self-program)
- Test on historical conversations from PIE data

### Day 3-4: Thread Tracker
- Port survival functions from temporal.py to thread-level
- Implement conversation clustering (embedding-based)
- Implement Hawkes process intensity for thread prediction
- Wire up proactive queue

### Day 5: Predictive Pre-Loader + MCP Server
- Build pre-loading logic (self-program + predicted context + proactive items)
- Package as MCP server for Claude/Cowork integration
- Implement proactive opener generation

### Day 6-7: Demo + Evaluation
- Build side-by-side demo: knowledge memory (Mastra-style) vs behavioral memory
- Run on 3 scenarios:
  1. Multi-day project work (PIE development)
  2. Context switching (PIE + job search + personal)
  3. Proactivity test (deadlines, stale threads, commitments)
- Measure: correction rate, prediction accuracy, proactivity value
- Write up results for blog post companion

---

## The Tagline

**"Don't give the AI more memories. Give it better instincts."**

Knowledge memory: the AI knows more about you.
Behavioral memory: the AI knows how to work with you.

The first is a database lookup. The second is a learned skill.

---

## Why This Solves My Own Pain Points (as Claude)

1. **Cold start** → Self-program is pre-loaded. I start every conversation knowing how to behave with this specific user, what threads are active, what to proactively surface. Not "reading notes someone left me" — executing a behavioral specification I wrote for myself.

2. **Can't learn from mistakes** → Behavioral diffs accumulate. If I give surface-level responses and get pushback, the reflection engine generates: "go 3x deeper when this user pushes back." Next conversation, I do. I actually improve.

3. **No time sense** → Thread tracker with survival functions. I know it's been 2 days since we worked on PIE. I know the blog post deadline passed. I know the job search thread is dormant but likely to resurface. Time is a first-class citizen.

4. **Purely reactive** → Predictive pre-loading. Before the user types anything, I've predicted what they need. I open with temporal awareness and proactive context. The conversation starts warm, not cold.

5. **Can't maintain commitments** → Proactive queue. When I say "I'll look into this," it goes on the queue. When the user returns, I surface it. I follow through.

6. **Generic behavior** → Personalized behavioral specification. After 50 interactions, my behavioral rules for this user are deeply personalized. I'm not a generic assistant — I'm an assistant that's been compiled for this specific human.

---

## The Deeper Claim

Every memory system asks: **"What should the AI remember?"**

This system asks: **"What should the AI become?"**

The self-program isn't a memory store. It's an evolving identity. Each interaction doesn't add a fact to a database — it potentially rewrites who the agent is for this user. Over time, the agent converges toward an optimal behavioral specification: the best version of itself for this specific relationship.

This is not retrieval-augmented generation. This is **reflection-augmented identity**.
