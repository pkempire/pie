# Video: "Your AI Doesn't Know What Day It Is"
## Talking Points + Shot List

---

## VIBE / STYLE NOTES

- **Tone:** You explaining this to a smart friend at 1am. Not a lecture. Not a pitch. You genuinely went down this rabbit hole and came back with something.
- **Pacing:** Start punchy (hook), slow down for the personal story (act 1), build energy through the landscape (act 2), hit hard on the thesis (act 3), leave them thinking (close)
- **Visual style:** Mix of facecam, screen recordings, clean graphics for stats. No corporate polish — this should feel like a builder's journal.
- **Edit style:** Jump cuts on facecam. Quick zooms on stats/numbers. Screen recordings at 1.5-2x speed with voiceover. Occasional full-screen text for the punchline stats.
- **Length target:** 12-18 min. YouTube algo likes 12+, but don't pad.

---

## HOOK (0:00 - 0:45)

### Talking points:
- Open with the core paradox. Every AI company is racing to give their agents better memory. Billions in funding. Hundreds of papers. And yet — ask any AI assistant what day of the week it is relative to your last conversation, and it has no idea.
- The stat: researchers ran LLM agents through negotiations with real-time deadlines. Under turn-based limits, 95% success. Under equivalent time-based limits? As low as 4%. Same model, same task. The only difference: the model couldn't track time passing.
- This isn't a known problem being worked on. It's a blind spot the entire field is ignoring. And I think I know why.

### Shots:
- Open on facecam, mid-sentence energy. No intro, no "hey guys."
- Cut to full-screen stat: "95% → 4%" with a clean animation
- Quick montage: screenshots of Mem0, Zep, Langchain, Honcho landing pages — "everyone building memory" — then cut back to face
- Maybe: screen record of asking ChatGPT/Claude "how long has it been since we last talked?" and getting a useless response

---

## ACT 1: "I Built a Memory System" (0:45 - 4:30)

### Talking points:
- Personal origin. I was interested in a question: what would it look like if an AI actually understood your life over time? Not just remembered facts — understood the dynamics. What you're working on, what's fading, what's coming back.
- So I built PIE — Personal Intelligence Engine. Fed it a year of my personal data. WhatsApp messages, notes, whatever I could get. It processed everything and built a world model — basically a graph of my life. 4,000 entities, 7,000 transitions between them, 3,000 relationships. I could literally browse a map of my own patterns.
- Show the explorer. This is what a year of my life looks like as a knowledge graph. Every project, every person, every topic — connected by when and how they interacted in my data.
- It was cool. And then I sat with it and realized... it wasn't enough. The entity extraction was lossy — maybe 40% of nuance gets lost when you force conversations into structured triples. The graph was static — it told me what happened, not what's happening now. And the temporal stuff — I had survival functions, statistical models tracking how entities recur over time — was technically interesting but felt like I was forcing 1950s statistics onto a problem that needed something else.
- The honest moment: I built what the field says you should build. Knowledge graph, entity extraction, temporal modeling. And it felt like a demo, not a product.

### Shots:
- Facecam for the personal story — let it be conversational
- Screen recording: scrolling through PIE's explorer.html. Zoom into interesting clusters. Show the sheer scale — 4,000 entities.
- Maybe: timelapse of the terminal running PIE's processing pipeline — the "building the world model" moment
- Screen recording: the query interface. Type a query, get back entities with timelines. Show what it CAN do.
- Then facecam for the "it wasn't enough" moment. Let this land.

---

## ACT 2: "Everyone Is Solving the Wrong Problem" (4:30 - 8:30)

### Talking points:
- So I went deep. Read every paper I could find. Tested every system. Built a landscape of everything being done in agent memory.
- Here's what I found: there are like 10 different philosophies about what "memory" even means for an AI agent.
  - **The simple approach:** extract facts from conversations, store them, retrieve them later. Mem0 does this. ChatGPT's memory feature does this. It works for "remember my name" and falls apart for everything else.
  - **Knowledge graphs:** build a structured graph of entities and relationships. Zep/Graphiti does this with 23,000 GitHub stars. Sounds sophisticated. But the entity resolution sends ALL nodes to the LLM, so it breaks above 500 entities. And the temporal edges? They track WHEN facts were recorded, but the model still doesn't know what to DO with that information.
  - **The contrarian result:** Mastra proved you don't need ANY of this. Just store raw conversation text with timestamps. No extraction, no graph, no nothing. Let the LLM reason over raw text at query time. 94.87% on the main benchmark. Nearly matched the #1 system. With dramatically less complexity.
  - **The identity play:** Honcho says forget facts entirely — build a theory of mind about the user. Have a background "dreaming agent" that reasons about who the user IS, not just what they said. Claims SOTA on everything.
- The benchmarks: LongMemEval is the main one. Tests "can you remember facts from 50 conversations ago." The leaderboard is competitive — scores in the 90s. Sounds great.
- But here's what every single benchmark misses — and this is the thing that radicalized me on this topic:
  - Can the agent bring something up WITHOUT being asked?
  - Does the agent know it's been 5 days since your last conversation?
  - If you mentioned a deadline last week, does the agent mention it's approaching?
  - Can the agent manage multiple threads and pick up the right one?
- I surveyed the benchmark landscape. Only 0.4% of ALL agent benchmarks test continuous time awareness combined with proactive behavior. Zero point four percent. The entire field is optimizing for recall and ignoring presence.

### Shots:
- Fast-paced section. Screen recordings of each system's GitHub/website as you mention them, but quick — 2-3 seconds each.
- The scorecard table as a clean graphic: systems ranked by benchmark scores. Let it sit for a few seconds so people can read it.
- The "what no benchmark tests" section: numbered list appearing on screen one by one, with you talking over each.
- The 0.4% stat: full screen, big text, let it breathe. Maybe a beat of silence.
- Optional: screen record of the actual papers/benchmarks. Show you actually read this stuff — scroll through PDFs, highlight sections.

---

## ACT 3: "Time Is the Missing Variable" (8:30 - 13:00)

### Talking points:
- Here's the thesis. It's not about memory. It's about time.
- A transformer — the architecture behind every LLM — is a pure function. Tokens go in, tokens come out. Between API calls, the model doesn't exist. It's not sleeping. It's not idle. It literally ceases to be. There is no clock. There is no state. There is nothing.
- When you start a new conversation with ChatGPT or Claude, the model has no idea how long it's been. A minute? A month? It's exactly the same to the model. The context window might contain your previous messages, but the model has no internal experience of time passing between them.
- This explains everything. Walk through the evidence:
  - **The negotiation result** (Real-Time Deadlines paper): Researchers at UPenn ran LLM agents through negotiations. Turn-based limits: 95% deal closure. Real-time limits: as low as 4%. But — and this is the key — when they just told the model "you have X minutes remaining" each turn, success jumped to 32%. And when they gave qualitative urgency cues instead of numbers, it worked even BETTER. The model CAN reason about time. It just can't track it.
  - **The async planning result** (Robotouille, ICLR 2025): LLM agents managing cooking tasks — basically concurrent project management. Synchronous: 47%. Asynchronous (same task, but now things happen in parallel with real delays): 11%. Managing multiple timelines simultaneously breaks them.
  - **The temporal reasoning result** (TempoBench): GPT-4o on temporal causal reasoning — connecting "A happened, then B happened, so C" — scores 7.5% F1. Near random. On tasks humans find straightforward.
  - **The staleness result** (Tic-Toc): Agents deciding whether cached information is still fresh — less than 65% alignment with human judgment. They can't tell when their own knowledge is stale.
- Every one of these failures has the same root cause: the model has no internal elapsed-time variable. It cannot track time passing. And no amount of better retrieval, bigger context windows, or fancier knowledge graphs gives it one. Those things change what the model KNOWS about time. They don't give it the experience of time.
- The distinction I keep coming back to: "reasoning ABOUT time" versus "operating IN time." Every AI system in the world does the first. Nothing does the second. And the second is what makes a coworker a coworker.

### Shots:
- This is the core of the video. Slow down here. Facecam, direct to camera.
- The "pure function" explanation: maybe a simple animation? Input tokens → black box → output tokens. Then: between calls, the box disappears. No state persists.
- Each evidence point: show the paper title/authors briefly, then the key stat as a full-screen graphic. 95% → 4%. 47% → 11%. 7.5% F1. <65% alignment.
- The "reasoning about time vs operating in time" distinction: maybe two columns on screen. Left: "Reasoning ABOUT time" with examples (answering "when did WWII end", knowing timestamps). Right: "Operating IN time" with examples (noticing it's been 3 days, bringing up an approaching deadline). All current systems live on the left. Nothing lives on the right.
- Consider: a real-world analogy shot. "Imagine a coworker who has perfect memory of every conversation you've ever had. But every time you walk into the room, they have no idea if your last conversation was 5 minutes ago or 5 months ago. That's every AI agent today."

---

## ACT 4: "What I Think the Answer Is" (13:00 - 15:30)

### Talking points:
- So I went through a bunch of possible solutions. Let me be honest about what I found.
- **The uncomfortable truth:** You can't give a transformer a first-class temporal state variable without modifying the architecture. And modifying the architecture doesn't work yet. I ran experiments — training temporal awareness into model weights from scratch hit literally 0% success. Even with RL fine-tuning, it's fragile. The architecture fundamentally resists this.
- **But there's a spectrum.** Level 0: current LLMs, no awareness at all. Level 1: just tell it the time — text injection. Discovery report showed this alone gets you from 4% to 32%. Level 2: a persistent runtime that maintains temporal state and feeds structured briefings to the model. The MODEL is stateless, but the SYSTEM has temporal state. Level 3-4: actual architectural changes (conditioning, state space models) — genuine research, maybe years out.
- **Level 2 is the practical answer.** And it's the robotics pattern — in robotics, the planner doesn't have its own clock. It reads from a shared system clock. Time is an ambient signal that every module receives. LLM agents have no equivalent. Build that equivalent.
- **The briefing approach:** A cheap model pre-processes all temporal metadata — when each thread was last discussed, what deadlines are approaching, how long the user's been away, what day it is and what they usually work on today — and generates a natural language temporal briefing that gets injected into context. Qualitative urgency cues actually work BETTER than numeric timestamps — "your demo is tomorrow and you haven't finished the integration tests" outperforms "deadline in 18.5 hours."
- But here's the thing — before any solution matters, we need to be able to MEASURE temporal awareness. There's no benchmark. No eval. Nobody has defined what "good" looks like.
- **So that's what I built.** TemporalBench — the first benchmark that measures temporal PRESENCE, not temporal REASONING. Not "when did WWII end" — that's a factoid. "Your deadline is tomorrow and you haven't mentioned it" — that's awareness. 50 scenarios, 6 evaluation axes, LLM-as-judge scoring. I tested every major model and memory system against it. And the results are exactly what the thesis predicts.

### Shots:
- The runtime architecture diagram: clean graphic showing the temporal runtime (persistent, maintains state) feeding structured context into the LLM (stateless, receives temporal grounding). Keep it simple.
- For the diffusion analogy: maybe show an actual diffusion process — noisy image → clean image — then draw the parallel. "Same idea, but for time instead of noise."
- For the SSM idea: don't go too deep. Just the concept: "a small module that stays alive between conversations and ages in real time." Maybe visualize a decaying signal.
- For the eval: this is the call to action. What does it test? Show the 6 capabilities on screen.

---

## CLOSE: "The Race Nobody's Running" (15:30 - 17:00)

### Talking points:
- Here's what I find wild. Billions of dollars going into AI agents. Memory is considered one of THE core problems. And yet the most basic temporal capability — knowing what day it is relative to your last conversation — is not being worked on, not being measured, and not being discussed.
- I think whoever solves this unlocks the thing everyone actually wants from AI agents: the feeling that it's been there the whole time. Not just remembering what you said, but knowing where you are in time. Picking up where you left off. Noticing that your deadline is tomorrow. Feeling like a colleague, not a search engine.
- Everything I've built and researched is open. [Link to repo/resources]. If this resonated — go break things. Time is the missing variable. Let's build it.

### Shots:
- Facecam, direct to camera. Genuine energy, not performed.
- End on a clean card: project links, your handles, maybe the core stat one more time (0.4%).

---

## B-ROLL IDEAS

- Terminal sessions: PIE processing, query interface running, debugging the gpt-5-mini empty response bug (real footage of real problems)
- Explorer.html: the world model visualization, zooming into clusters
- Paper browsing: scrolling through arxiv papers, highlighting key findings
- Benchmark tables: clean graphics of the scorecard
- Architecture diagrams: temporal runtime, the "pure function" visualization
- Your actual notes/documents: MEMORY-LANDSCAPE.md, PROPOSAL.md, TEMPORAL-STATE-RESEARCH.md scrolling by — shows the depth of work
- Optional: time-lapse of your screen while working on this project (if you have any screen recordings)

## THUMBNAIL IDEAS

- Split screen: left side = colorful knowledge graph / world model visualization, right side = big text "YOUR AI HAS NO IDEA WHAT TIME IT IS"
- The 95% → 4% stat with a clock/timer visual
- Your face looking at a screen showing the explorer.html world model, with overlay text

## TITLE OPTIONS

- "Your AI Doesn't Know What Day It Is (and Nobody's Fixing It)"
- "I Read Every Paper on AI Memory. Here's What They're All Missing."
- "The Missing Variable in AI Agents"
- "Why AI Agents Can't Be Coworkers Yet"
