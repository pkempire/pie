# VIDEO OUTLINE: "Agent Memory is Broken. Here's What's Missing."

**[Draft — Living Document]**

**Target:** 15-20 minutes | YouTube  
**Tone:** Fireship meets 3Blue1Brown — fast, visual, technical depth without jargon walls  
**Audience:** AI engineers, builders, and curious technical people who've heard of RAG but think there must be more  

---

## COLD OPEN [0:00 - 0:45]

**Visual:** Screen recording — open ChatGPT, start a new chat.

**Script:**

> "I've had 4,758 conversations with ChatGPT over the last three years. That's more than I've had with most humans.
>
> And every single time I open a new chat... it has no idea who I am.
>
> Sure, it knows my name. It knows I'm a CS student. It even remembers that I like dark mode. But it doesn't know that I've started and pivoted three companies. It doesn't know my thinking about databases has completely reversed. It doesn't know that every single project I've ever built follows the exact same pattern.
>
> All that context — hundreds of hours of my thinking, decisions, and evolution — gone. Every time.
>
> This video is about why that happens, what everyone is trying to do about it, and why they're all missing the most important piece."

**Visual:** Cut to title card — animated. Text: **"Agent Memory is Broken."** Then underneath fades in: **"Here's What's Missing."**

---

## SECTION 1: THE PROBLEM [0:45 - 3:00]

### The "Memory" You Have Now [0:45 - 1:30]

**Visual:** Side-by-side comparison — ChatGPT memory panel vs. what a real personal assistant would know.

**Left side (what ChatGPT stores):**
```
- User is a CS student at UMD
- User likes Python
- User is building an AI project
```

**Right side (what a real assistant would know):**
```
- Started SRA in high school, pivoted from mentoring to curriculum
- Moved from SQLite → Neo4j → FalkorDB over 6 months
- Every project follows: research → narrow → build fast → iterate
- Said "never cloud" in June, running Docker in production by December
- Working style shifts when under deadline pressure
```

> "One side is a Rolodex. The other is understanding. Current AI memory is stuck on the left."

### Why This Matters [1:30 - 2:15]

**Visual:** Animated sequence showing four types of questions appearing on screen, each one failing with current systems.

1. **"How has my approach to building products changed?"** → ❌ Mem0 returns flat facts
2. **"What was I working on when I got into knowledge graphs?"** → ❌ RAG returns random chunks
3. **"Do I always pivot projects after 3 months?"** → ❌ No system can answer this
4. **"I said I'd never use cloud — have I contradicted that?"** → ❌ Old fact overwritten

> "These aren't retrieval problems. You can't solve them with better embeddings. They're *temporal reasoning* problems. And that's a fundamentally different challenge."

### The Scale of the Problem [2:15 - 3:00]

**Visual:** Animated infographic of the dataset.

- 4,758 conversations
- 33 months (April 2023 → January 2026)
- ~16 million tokens
- ~17,000 user messages
- GPT-3.5 → GPT-4 → GPT-4o → GPT-o1 → GPT-5

> "Three years of thinking out loud. Decision-making. Problem-solving. Learning. All of it sitting in a JSON export file, completely unstructured. What if we could turn that into something an AI actually understands?"

---

## SECTION 2: HOW EVERYONE ELSE DOES IT [3:00 - 6:30]

### The Memory Landscape — 60-Second Tour [3:00 - 4:30]

**Visual:** Animated tier list / landscape diagram. Each system appears with a one-line description and a visual of its architecture pattern.

**Tier: Fact Extraction**
- **Mem0:** Conversations → atomic facts → vector store → similarity search
  - *Visual:* Conversation flows into a flat grid of key-value cards
  - "Great for preferences. Terrible for evolution."

**Tier: Session Management**  
- **MemGPT/Letta:** OS-style memory paging for context window
  - *Visual:* Tiered boxes (like CPU cache → RAM → disk)
  - "Clever context management. But it's managing what's in context *right now*, not building persistent memory."

**Tier: Knowledge Graphs**
- **Zep/Graphiti:** Conversations → entities + relationships → temporal KG
  - *Visual:* Nodes and edges forming on screen, with timestamps
  - "Closest to what we need. Temporal-aware graph. But still passes raw timestamps and doesn't extract behavioral patterns."

**Tier: Psychology**
- **Honcho:** Conversations → user personality model
  - *Visual:* Silhouette of a person with personality traits orbiting
  - "Not 'what you know' — 'who you are.' Interesting but different problem."

**Tier: Built-in**
- **ChatGPT Memory / Claude Projects:** Simple fact lists
  - *Visual:* A sticky note with 5 facts on it
  - "Baby steps. We need to go further."

### What They All Miss [4:30 - 6:30]

**Visual:** Comparison table builds row by row, dramatically revealing the gaps.

| Capability | Mem0 | Zep | Honcho | What You Need |
|---|:---:|:---:|:---:|:---:|
| Store facts | ✅ | ✅ | ❌ | ✅ |
| Entity relationships | ❌ | ✅ | ❌ | ✅ |
| **Temporal state tracking** | ❌ | ⚠️ | ❌ | ✅ |
| **Contradiction detection** | ❌ | ❌ | ❌ | ✅ |
| **Procedural memory** | ❌ | ❌ | ❌ | ✅ |
| **Semantic time anchoring** | ❌ | ❌ | ❌ | ✅ |

> "See the pattern? The top rows — the easy stuff — everyone handles. The bottom rows — the stuff that would actually make personal AI *personal* — nobody does."

**Visual:** The bottom four rows pulse/glow red. Then text appears: "This is the missing piece."

---

## SECTION 3: TYPES OF MEMORY [6:30 - 9:00]

### The Cognitive Science Framework [6:30 - 7:30]

**Visual:** Brain diagram with labeled regions, each lighting up as discussed. Animated.

> "Cognitive science has known about different memory types for decades. Let's map them to AI."

**Visual:** Animated table building piece by piece:

| Type | Human Example | AI Example | Current Status |
|---|---|---|---|
| **Episodic** | "That meeting where we pivoted" | Full conversation logs | ✅ Most RAG systems |
| **Semantic** | "Paris is the capital of France" | "User prefers dark mode" | ✅ Mem0, ChatGPT memory |
| **Procedural** | Riding a bike, your morning routine | "User always researches before building" | ❌ **Nobody** |
| **Temporal** | "College years," "that summer in SF" | State transition chains across time | ❌ **Nobody (well)** |

> "We've solved the easy two. Episodic — just store conversations. Semantic — extract facts. Done.
>
> But procedural and temporal — *how you do things* and *when things happened and how they changed* — that's where the real intelligence lives. And nobody's cracked it."

### Why Procedural Memory Matters [7:30 - 8:15]

**Visual:** Animation showing 5 project timelines laid out in parallel — each following a similar shape/pattern. Camera zooms out to reveal the pattern.

> "Imagine you've built 12 projects over 3 years. Each one went through phases — research, prototyping, pivoting, shipping. If you line up those timelines, patterns emerge.
>
> Maybe you always pivot around month 2. Maybe you always over-scope then cut. Maybe your belief changes follow a specific arc — strong opinion, encounter pushback, temporary ambivalence, refined position.
>
> That's procedural memory. It's your behavioral DNA. And it's *invisible* to every memory system that exists today — because they look at individual conversations, not at patterns *across* entity lifecycles."

### Why Temporal Memory Matters [8:15 - 9:00]

**Visual:** Two side-by-side comparisons.

**Left: What systems store**
```
Project X status: "launched"
timestamp: 1718236800
```

**Right: What temporal memory looks like**
```
Project X — first appeared 14 months ago (freshman year)
Changed 5 times (~0.4x/month)
• 14 months ago: Created as idea
• 11 months ago: First prototype, 3 users
• 8 months ago: Pivoted B2C → B2B ⚠ contradiction
• 4 months ago: Launched v2, 50 users week 1
• 2 days ago: Exploring enterprise pricing
```

> "One is a data point. The other is a story. And LLMs are really, *really* good at reasoning about stories."

---

## SECTION 4: THE PIE APPROACH [9:00 - 13:30]

### The Architecture in 90 Seconds [9:00 - 10:30]

**Visual:** Animated architecture diagram building step by step.

> "PIE — Personal Intelligence Engine. Here's what it does."

**Step 1:** Show ChatGPT export JSON flowing in.
> "4,758 conversations. Sorted chronologically."

**Step 2:** Show daily batch grouping.
> "Grouped into daily batches. Each day's conversations processed together — with a context preamble showing what's currently active in your world."

**Step 3:** Show extraction producing entities and transitions.
> "An LLM extracts entities, relationships, state changes, beliefs, and decisions. But here's the key — it's not working in isolation. It sees the whole accumulated world state. So when you say 'write a lesson plan for week 3' without mentioning which project, the system knows you're working on Lucid Academy because it's in the active context."

**Step 4:** Show entities entering a knowledge graph with glowing edges.
> "Everything goes into a temporal knowledge graph. Entities are nodes. Relationships and state transitions are edges. Time flows through the whole thing."

**Step 5:** Show the graph being queried with compiled context flowing out.
> "When you query, the system doesn't return raw graph data. It compiles semantic temporal context — natural language narratives with relative time, period anchoring, and change velocity. The LLM never sees a timestamp."

### Deep Dive: Semantic Temporal Context [10:30 - 11:30]

**Visual:** Split screen animation. Left side: raw temporal data (timestamps, JSON). Right side: compiled context (natural language narrative). Show the compilation process as a transformation.

> "This is maybe the most important idea in the whole system. LLMs are bad at timestamps. Like, really bad. Research from Google shows they fake temporal reasoning by pattern-matching on memorized facts. Anonymize the entities, and performance collapses.
>
> So we don't give them timestamps. We give them this..."

**Visual:** The compiled temporal context example from Section 3 appears, with each component highlighted:
- "22 months ago" ← relative time (highlighted)
- "high school senior year" ← period anchor (highlighted)
- "0.3x/month" ← change velocity (highlighted)
- "⚠ contradicted" ← contradiction flag (highlighted)

> "Relative time. Period anchoring. Change velocity. Contradiction flags. All in natural language the model is trained to understand. No date parsing required."

### Deep Dive: State Transitions [11:30 - 12:15]

**Visual:** Animated state machine for a single entity. States are boxes, transitions are arrows with labels and typed badges (creation, update, contradiction, resolution).

> "In most systems, when your project status changes from 'design' to 'launched,' the old value gets overwritten. In PIE, it becomes a state transition.
>
> Every entity carries a chain of these. Each one typed — was this a creation? An update? A *contradiction* of what was believed before? A resolution of a previous conflict?
>
> The chain IS the temporal model. When you ask 'how has X evolved,' the system reads the state machine. No search required."

**Visual:** Show a contradiction transition highlighted in red — "Pivoted from mentoring to curriculum" — with the previous state crossed out.

> "And when you contradict yourself, the system *knows*. It doesn't silently overwrite. It flags it."

### Deep Dive: Forgetting [12:15 - 13:00]

**Visual:** Graph visualization where nodes pulse with brightness proportional to importance. Low-importance nodes gradually dim and fade. High-importance nodes stay bright.

> "4,758 conversations. Probably 3,000 are one-off debugging questions. How do you separate signal from noise?
>
> Not with heuristics. With graph structure. PIE computes importance like PageRank — connectivity, transition count, recency, neighbor importance. An entity connected to nothing, changed once, referenced 8 months ago? Importance: near zero. Gets summarized, then archived.
>
> An entity connected to 15 others, changed 7 times, queried last week? Importance: high. Full detail preserved.
>
> The graph structure *itself* separates your life's main storylines from the noise."

### Deep Dive: Procedural Extraction [13:00 - 13:30]

**Visual:** Multiple entity lifecycle timelines (project A, B, C, D) laid out as parallel swimlanes. Camera zooms out, and dotted lines connect similar phases across timelines. A "procedure" box appears, summarizing the pattern.

> "After ingestion, PIE looks at entity lifecycles in aggregate. Not individual conversations — entire arcs. 'Project A: research, prototype, pivot, ship. Project B: research, prototype, pivot, ship. Project C: research, prototype, pivot, scale.'
>
> The pattern is obvious when you see it. PIE extracts it as explicit procedural memory: 'This is how you build things. Here's the evidence.'"

---

## SECTION 5: RESULTS [13:30 - 16:30]

[TODO: This entire section depends on experiment results. Outline of what to show:]

### 5.1 The Graph Visualization [13:30 - 14:30]

[TODO: Need actual graph viz from PIE]

**Planned visual:** Full knowledge graph visualization with:
- Color-coded entity types (projects in blue, people in green, beliefs in yellow, etc.)
- Edge thickness proportional to relationship strength
- Time axis showing entity creation dates
- Clusters visible (the SRA cluster, the internship cluster, the tools cluster)

> "This is 33 months of one person's intellectual life, as a knowledge graph. [X] entities, [Y] relationships, [Z] state transitions. Let me zoom into a few interesting areas..."

**Planned walkthrough:** Zoom into 2-3 interesting subgraphs. Show how entities connect across life domains.

### 5.2 The Ablation — Does Semantic Time Actually Help? [14:30 - 15:30]

[TODO: Need ablation results from Hypothesis 1]

**Planned visual:** Bar chart — 4 conditions × 4 models. Dramatic reveal of results.

> "We tested the same temporal reasoning questions under four conditions: raw timestamps, formatted dates, relative time, and PIE's full semantic temporal context. Across [X] models..."

### 5.3 Head-to-Head: PIE vs. RAG vs. Mem0 [15:30 - 16:00]

[TODO: Need comparison results]

**Planned visual:** Side-by-side-by-side query comparisons. Same question, three system outputs. Quality difference should be visually obvious.

### 5.4 Actual Extracted Procedures [16:00 - 16:30]

[TODO: Need actual procedure outputs]

**Planned visual:** Show 2-3 real extracted procedures with the evidence entities highlighted in the graph.

> "These are real patterns extracted from real data. Things the user may not have consciously realized about their own behavior."

---

## SECTION 6: WHAT THIS MEANS [16:30 - 18:00]

**Visual:** Return to the simple diagram from the opening — but now the "right side" (what a real assistant would know) is achievable.

> "Agent memory isn't a solved problem. It's barely started.
>
> We've been treating it as a retrieval problem — how do I find the right chunk? — when it's actually a *modeling* problem — how do I represent someone's evolving world?
>
> The path forward isn't bigger context windows or better embeddings. It's structure. Temporal structure. State transitions. Procedures. Contradiction tracking.
>
> The AI that remembers what you like... that's table stakes. Already here. The AI that understands how you think, how your world has changed, and what patterns you follow? That's next. And building it requires rethinking memory from the ground up."

### The Roadmap [17:15 - 17:45]

**Visual:** Animated progression:

```
Temporal Understanding (PIE v1 — we are here)
         ↓
Procedural Memory (extracted behavioral patterns)
         ↓
Proactive Assistance (anticipating needs from patterns)
         ↓
Autonomous Agency (acting on deep world understanding)
         ↓
Continual Learning (the model that grows with you)
```

> "Each step depends on the previous. You can't be proactive without procedures. You can't have procedures without temporal tracking. This is the foundation."

---

## CLOSE [18:00 - 18:30]

**Visual:** Back to screen recording of ChatGPT. But now, in a second panel, PIE's query interface is open showing rich temporal context.

> "4,758 conversations. 33 months. 16 million tokens. For the first time, all of it is structured, temporal, and queryable.
>
> PIE is open source. Link in the description. If you're building in agent memory, check out the paper. If you think this matters, share this video.
>
> And next time you open a new chat and your AI has no idea who you are... remember — it doesn't have to be this way."

**End card:** Links to repo, paper, blog post. Subscribe CTA.

---

## PRODUCTION NOTES

### Key Visuals to Create

1. **Knowledge graph visualization** — The hero visual. Needs to be beautiful. Consider using a force-directed layout with entity type colors. Time axis optional but cool.
2. **State machine animation** — Single entity evolving through transitions. Clean, minimal, animated step-by-step.
3. **Compilation transformation** — Raw timestamps/data on left, compiled narrative on right, with animated transformation between them.
4. **Parallel lifecycle swimlanes** — Multiple projects' timelines in parallel showing pattern emergence.
5. **Memory landscape tier list** — Animated, with systems appearing and being categorized.
6. **Architecture flow diagram** — Step-by-step animated pipeline from JSON export to queryable graph.
7. **Brain diagram** — Cognitive memory types, each lighting up.
8. **Before/after comparison cards** — What current systems store vs. what PIE stores.
9. **Importance visualization** — Graph with pulsing brightness for importance scores, dimming for archival.

### B-Roll / Screen Recordings Needed

- ChatGPT new chat (empty state, memory panel)
- PIE query interface in action [TODO: need working system]
- Graph database GUI showing entities [TODO: need working system]
- Code scrolling — extraction pipeline, temporal compiler
- Terminal showing ingestion running [TODO: need working system]

### Tone Notes

- **Pace:** Fast. Fireship-speed for landscape overview (Section 2). Slower, 3Blue1Brown-style for the deep dives (Section 4).
- **Energy:** Conviction without hype. "This is clearly broken. Here's why. Here's a fix."
- **Humor:** Dry, occasional. "One side is a Rolodex. The other is understanding." Not forced.
- **Technical depth:** Assume viewer knows what RAG is, what embeddings are, roughly what a knowledge graph is. Don't over-explain basics. Do explain novel concepts carefully.
- **Avoid:** "In this video we'll..." / "Don't forget to like and subscribe" in the middle / lengthy intros.

### Music

- Upbeat, slightly tense electronic for Sections 1-2 (problem statement, landscape)
- Calmer, thoughtful instrumental for Sections 3-4 (deep explanation)
- Triumphant/resolved for Section 5 (results)
- Reflective for Section 6 (implications)

---

**Changelog:**
- v0.1 (Feb 2026): Initial outline. Results section (Section 5) entirely TODO — need working system + experiments.
