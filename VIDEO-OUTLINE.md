# VIDEO OUTLINE: "Agent Memory is Broken. Here's What's Missing."

**[Draft — Living Document]**

**Target:** 18-20 minutes | YouTube  
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

## SECTION 5: RESULTS & THE AHA MOMENT [13:30 - 17:00]

### 5.1 The Benchmark Scorecard [13:30 - 14:30]

**Visual:** Animated scorecard building piece by piece. Each benchmark appears with its score and category breakdown.

**LongMemEval: 66.3%**
```
single-session-assistant:  98.2% ████████████████████ ✓
single-session-user:       84.3% ████████████████▌    ✓
knowledge-update:          79.5% ███████████████▉     ✓
temporal-reasoning:        59.8% ███████████▉         ⚠
multi-session:             55.6% ███████████          ⚠
preference:                 6.7% █▎                   ✗
```

> "98% on single-session. Great. 6.7% on preferences. Ouch. See the pattern? The easy stuff — just retrieve what someone said — works fine. The hard stuff — inferring patterns, synthesizing across sessions — that's where it breaks."

**Visual:** Comparison bar chart appearing:
```
Emergence AI:    86.0% ████████████████▉
Supermemory:     71.4% ██████████████▎
Zep:             71.2% ██████████████▎
Our naive_rag:   66.3% █████████████▎
```

> "We're competitive. Not leading. The gap? Mostly in those bottom categories — exactly where PIE's structured approach should help."

**LoCoMo: 58%**
- Temporal reasoning: **35.7%** (worst category, again)

**MSC: 46%** (76% partial credit)

> "Same story everywhere. Temporal is the weakest. Always."

### 5.2 The Experiment That Changed Everything [14:30 - 16:00]

**Visual:** Dramatic setup — three columns appearing: "naive_rag", "baseline", "pie_temporal"

> "Now here's where we ran an experiment that completely changed how we think about this problem."

**Visual:** Results revealed dramatically:
```
naive_rag:     56.2% █████████████████
baseline:      46.2% █████████████▊
pie_temporal:  31.2% █████████▍       ← Wait what?
```

> "PIE's semantic temporal approach... made things worse? By 25 percentage points?"

**Visual:** Confusion animation — question marks, red alerts

> "We spent a week figuring out what went wrong. Turns out... nothing went wrong. We discovered something important."

**Visual:** Per-question-type breakdown table appearing row by row:

| Question Type | naive_rag | pie_temporal | What happened? |
|---------------|-----------|--------------|----------------|
| duration | 50% | **75%** ↑ | Narrative helps! |
| first_last | 87.5% | 75% ↓ | |
| what_time | 100% | 62.5% ↓↓ | Narrative hurts! |
| at_time_t | 100% | 62.5% ↓↓ | Narrative hurts! |

**Visual:** Animation showing the insight:

**Left side:** Query "How long was X the CEO?" → Narrative context "X was CEO for ~3 years during the growth period" → ✓ Model understands duration

**Right side:** Query "What happened on January 15?" → Narrative context "~3 weeks ago, during Period A" → ✗ Model lost the date

> "When you convert '2024-01-15' into 'about 3 weeks ago during the expansion period,' you HELP the model reason about duration and patterns. But you DESTROY its ability to answer 'what happened on January 15?'
>
> This isn't a bug. It's a fundamental tradeoff. And it means the right answer isn't 'semantic context' or 'raw timestamps' — it's BOTH. A hybrid system that picks the right format based on what you're asking."

**Visual:** Split-screen decision tree appearing:
```
Query comes in
     │
     ├── About evolution/patterns? → Semantic narrative
     │   "How has X changed?"
     │   "What patterns do I follow?"
     │
     └── About specific dates? → Preserved timestamps
         "What happened on [date]?"
         "Who was X on [date]?"
```

> "This is, we think, a publishable finding on its own. The temporal context format should be task-adaptive."

### 5.3 The Fix: Date Fallback [16:00 - 16:30]

**Visual:** Before/after comparison

> "We traced the failures to a data problem. Our extraction prompt didn't ask for dates — so 0% of entities had date metadata. The temporal compiler had nothing to work with."

**Visual:** Code diff showing prompt change, then results improvement:
```
Before fix: pie_temporal on date queries → 0%
After fix:  pie_temporal on date queries → 40%
```

> "Adding a fallback to use creation timestamps took us from 0% to 40%. The approach works — it just needs the right data."

### 5.4 Entity Quality — What 866 Entities Look Like [16:30 - 17:00]

**Visual:** Animated pie chart / entity breakdown

```
Entity Types (866 total):
├── Concept:  26% ████████
├── Tool:     23% ███████
├── Project:  22% ███████
├── Decision: 13% ████
└── Other:    16% █████

Quality:
├── Have descriptions: 100% ✓
├── Have aliases:      8.2% (resolution working)
└── Had dates:         0%   (now fixed in prompt)
```

> "Good type distribution. 100% descriptions. 8% have aliases — meaning we're catching 'SRA' equals 'Science Research Academy' equals 'scifair.tech.' The 0% date coverage was the gap that broke temporal reasoning. Now fixed."

---

## SECTION 6: WHAT THIS MEANS [17:00 - 18:30]

**Visual:** Return to the simple diagram from the opening — but now the "right side" (what a real assistant would know) is achievable.

> "Agent memory isn't a solved problem. It's barely started.
>
> We've been treating it as a retrieval problem — how do I find the right chunk? — when it's actually a *modeling* problem — how do I represent someone's evolving world?
>
> The path forward isn't bigger context windows or better embeddings. It's structure. Temporal structure. State transitions. Procedures. Contradiction tracking.
>
> The AI that remembers what you like... that's table stakes. Already here. The AI that understands how you think, how your world has changed, and what patterns you follow? That's next. And building it requires rethinking memory from the ground up."

### The Roadmap [17:45 - 18:15]

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

## CLOSE [18:30 - 19:00]

**Visual:** Back to screen recording of ChatGPT. But now, in a second panel, PIE's query interface is open showing rich temporal context.

> "4,758 conversations. 33 months. 16 million tokens. 866 entities. For the first time, all of it is structured, temporal, and queryable.
>
> But here's what I want you to take away: we ran these experiments expecting to prove that semantic temporal context beats raw timestamps. Instead, we discovered it's not that simple. Duration queries? +25% improvement. Date lookups? -37% disaster. The answer isn't one or the other — it's both, selected based on what you're asking.
>
> That insight — that temporal context should be task-adaptive — that might be the most important thing we found. And now you know it too.
>
> PIE is open source. Link in the description. If you're building in agent memory, check out the paper. If you think this matters, share this video.
>
> And next time you open a new chat and your AI has no idea who you are... remember — it doesn't have to be this way."

**End card:** Links to repo, paper, blog post. Subscribe CTA.

---

## POST-PRODUCTION NOTES (Updated)

### Section 5 is now the emotional core

The "aha moment" where pie_temporal underperforms, followed by the insight about task-adaptive temporal context, should be edited for maximum dramatic effect:

1. **Build tension:** Show the benchmark setup, explain what we expected
2. **The reveal:** pie_temporal at 31.2% — pause, let it sink in
3. **The investigation:** "We spent a week..." — quick montage of debugging
4. **The insight:** Per-question breakdown appears, light bulb moment
5. **The implication:** Split-screen showing hybrid approach

This is the scientific story of the video — hypothesis → unexpected result → deeper understanding. Edit accordingly.

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
10. **Task-adaptive decision tree** — NEW: Split-screen showing query routing: evolution queries → narrative, date queries → timestamps. Animated decision flow.
11. **The "aha moment" reveal** — NEW: Dramatic benchmark results with pie_temporal underperforming, then zooming into per-question breakdown to show the insight. Think "plot twist" energy.
12. **Duration vs. date query comparison** — NEW: Side-by-side showing same entity, two query types, why one format helps and the other hurts. Key visual for explaining the task-adaptive finding.
13. **Benchmark scorecard animation** — NEW: Progress bars filling in for each benchmark category, with color-coding (green for strong, yellow for weak, red for failures like 6.7% preference).

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
- v0.2 (Feb 2026): **Major update** — Added actual benchmark results (LongMemEval 66.3%, LoCoMo 58%, MSC 46%, Test of Time task-adaptive finding). Section 5 completely rewritten with the "aha moment" narrative about task-adaptive temporal context. Added new visuals for the insight reveal. Updated timing to 18-20 min.
