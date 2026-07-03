---
title: "AI memory, from scratch: working memory, episodic memory, and everything the field gets wrong"
channel: "Working Memory"
episode: 1
target_length: "55–70 min"
target_audience: "AI engineers building agentic systems; intermediate familiarity with LLMs"
status: "shootable_v1"
last_updated: 2026-05-30
related_papers:
  - 2402.17753-locomo
  - 2410.10813-longmemeval
  - 2504.19413-mem0
  - 2501.13956-zep
  - 2507.07957-mirix
  - 2601.02163-evermemos
  - 2601.02845-timem
  - 2601.06282-amory
  - 2604.20943-scm-sleep
  - 2605.20616-auto-dreamer
  - 2512.12818-hindsight
  - 2507.19457-gepa
  - 20260511-thinkingmachines-interaction-models
---

# AI memory, from scratch
*Working memory, episodic memory, and everything the field gets wrong*

> Companion notes + shootable script for **Working Memory** Episode 1.
> Read it straight as a blog post, or read it on camera section by section.
> Every claim has a citation tag in `[[brackets]]` linking to the corresponding wiki page.

---

## How to use this document

This file has three layers, color-coded by markup:

- **Prose** — the actual content. Read it. It's the blog post version.
- **`▶ SPEAK`** — the spoken-to-camera version of the same content, tightened. Read these blocks verbatim when recording the talking-head segments. They're already paced for delivery.
- **`■ GRAPHIC`** — what to cut to on screen. Each graphic has an asset spec at the bottom of this doc and a thumbnail in `assets/`.
- **`⚙ EXPERIMENT`** — a live coding moment. The command, the expected output, the point being made. Run it once before shooting; capture the terminal output as B-roll.

The two layers exist because the spoken version compresses arguments, but the written version has to defend them. The wiki page for each cited paper has the receipts.

---

## Table of contents

0. [Cold open](#0-cold-open) — 90 s
1. [What "memory" even means](#1-what-memory-even-means) — 8 min
2. [Why context is not memory](#2-why-context-is-not-memory) — 7 min
3. [The naive baselines and why they fail](#3-the-naive-baselines-and-why-they-fail) — 8 min
4. [The taxonomy: write-time vs read-time, six substrates](#4-the-taxonomy-write-time-vs-read-time-six-substrates) — 10 min
5. [NoReplay: the discipline most benchmarks ignore](#5-noreplay-the-discipline-most-benchmarks-ignore) — 7 min
6. [Where the SOTA actually is](#6-where-the-sota-actually-is) — 8 min
7. [Three live research directions](#7-three-live-research-directions) — 7 min
8. [What you should build today](#8-what-you-should-build-today) — 6 min
9. [Open problems](#9-open-problems) — 4 min
10. [Reading list and close](#10-reading-list-and-close) — 2 min

**Total target runtime: ~67 min.** Cut to 50 by trimming Chapters 6 and 7 if you want a tighter cut for newcomers.

---

## 0. Cold open
*(90 seconds, hard cuts, one-shot energy)*

`▶ SPEAK`

> Open a fresh ChatGPT window. Ask it what we talked about last Tuesday. It will guess, or it will refuse, or it will hallucinate a conversation that never happened.
>
> This is not a bug. This is the default. Large language models are stateless. Every conversation starts from zero. The thing we call "memory" in commercial chatbots is, under the hood, mostly a vector database that the model queries when it remembers to.
>
> The field has been trying to fix this for about three years. There are now hundreds of papers, dozens of benchmarks, and a leaderboard that updates every two weeks. The number at the top right now is 93 percent. That sounds like the problem is mostly solved.
>
> It is not solved. In this episode I'm going to walk you through what memory means for an LLM, what the leading systems actually do, why most of the numbers on those leaderboards do not measure memory, and what a working system would look like if you tried to build one today.
>
> This channel is called Working Memory. The name comes from cognitive psychology. Working memory is the small, fast scratchpad you hold things in while you're thinking. It is not the same as long-term memory. The first lesson is that LLMs have been confused with their own working memory for a while now, and a lot of what gets called "long-term memory for AI" is actually just a bigger scratchpad. By the end of this video you will be able to tell which is which when you read a paper.

`■ GRAPHIC 0-A` — channel title card. White text, black background. Below the title, a small footnote: "cognitive psychology, 1968. The scratchpad you hold things in while thinking." (Atkinson–Shiffrin reference.)

`■ GRAPHIC 0-B` — split screen. Left: a ChatGPT exchange where the model says "I don't have memory of previous conversations." Right: an architecture diagram with the user's transcript on disk, a vector store, and the model in the middle, captioned **"how 'memory' actually works in ChatGPT today."**

---

## 1. What "memory" even means
*(~8 minutes)*

### 1.1 The cognitive science scaffold

Before we touch any LLM stack, we need vocabulary. The dominant frame in cognitive psychology since the 1960s separates memory into four kinds:

- **Working memory** — small, fast, online. What you're holding in mind right now.
- **Episodic memory** — autobiographical, time-and-place-indexed. "Last Tuesday at the coffee shop with Anna."
- **Semantic memory** — facts, decontextualized. "Paris is the capital of France."
- **Procedural memory** — how-to, motor and skill. "How to ride a bike."

These are not just convenient categories. They live in different brain regions, they have different forgetting curves, they are dissociated by lesions, and — most importantly for us — they obey different update rules. Episodic memory is dense, fast-encoded, and lossy. Semantic memory is compressed, slowly updated, and durable. Procedural memory is barely textual.

`▶ SPEAK`

> When I say working memory, episodic memory, semantic memory, procedural memory, I'm using the standard cognitive psychology terms. They're not metaphors. They map onto distinct brain systems with distinct properties. The point of bringing them up isn't biological plausibility — most LLM memory work isn't biologically plausible — it's that the field's design choices keep mapping onto these distinctions whether researchers admit it or not.

`■ GRAPHIC 1-A` — four boxes labeled Working / Episodic / Semantic / Procedural, with one-line descriptions and a row at the bottom showing what each maps onto in the LLM stack (filled in over the next 2 minutes of voiceover).

### 1.2 The LLM analogs

Here is the imperfect mapping I'm going to defend for the rest of the episode:

| Cognitive kind | LLM analog | Where it lives |
|---|---|---|
| Working memory | The context window | RAM, attention |
| Episodic memory | Per-session transcripts and retrieved events | Disk, sometimes a vector store |
| Semantic memory | The model's parametric weights, plus a structured knowledge graph if you have one | Weights and/or a KG |
| Procedural memory | Tool routines, system prompts, learned scaffolds | Prompt, code, sometimes finetuned |

The mapping is imperfect in three ways and I want to flag them now because they explain a lot of confusion downstream:

1. **The context window is bigger than human working memory by orders of magnitude.** A million tokens is roughly equivalent to a paperback novel. A human working memory holds about seven items. People keep using the same word for two things that differ by six orders of magnitude.
2. **The model's weights blur semantic and procedural memory.** A finetuned model "remembers" a procedure and "remembers" a fact using the same machinery. This is part of why instruction-tuning data and factual training data interact in counterintuitive ways.
3. **There is no consolidator running while the model "sleeps."** Humans move things from episodic to semantic during sleep. LLMs do not have this loop by default. Several of the systems we'll look at — Auto-Dreamer, SCM, Mastra — are explicitly trying to add it. That's not coincidence. It's the missing piece.

`▶ SPEAK`

> The cleanest version of what's missing in commercial LLMs today is the consolidator. We have the context window — that's the scratchpad. We have parametric knowledge — that's the long-term semantic store. What we don't have is a process that runs between sessions, looks at what happened, and moves the durable parts into a place that doesn't cost a million tokens to read every time.
>
> About a quarter of the interesting papers in the last six months are some flavor of "let's add a consolidator." That's the through-line if you want one.

`■ GRAPHIC 1-B` — the same four-box diagram from 1-A, now with arrows showing what's currently missing in production LLM systems: a red dashed arrow from episodic to semantic captioned **"consolidation: missing in production. Currently being researched in: Auto-Dreamer, SCM, Mastra OM."**

### 1.3 The minimal operational definition

For the rest of this video, here's what we will mean by "memory":

> **Memory** is *persistent state across sessions that selectively retains and updates information about a user, environment, or task*.

The key words are *persistent*, *across sessions*, *selectively*, and *updates*. A million-token context window is persistent within a session but not across them. A vector database of every utterance is persistent but not selective. A summary that never gets updated is selective but stale. All four properties have to hold or you don't have memory, you have something else.

Hold onto that definition. We'll use it to disqualify a lot of marketing copy.

---

## 2. Why context is not memory
*(~7 minutes)*

### 2.1 The "just make the context window bigger" argument

The most common pushback when someone proposes a memory system is: why not just put it all in the context window? Gemini does a million tokens. Claude does 200k and can handle a few million in the API. Hardware keeps getting cheaper. Won't memory just become a non-problem in two years?

The argument has three problems.

**Problem one: long-context performance is not flat.** Models advertise a context length but their effective accuracy drops well before that length. The "Lost in the Middle" effect [[lost-in-the-middle]] is the canonical finding — accuracy degrades on items placed in the middle third of a long context. More recent work shows the degradation is worse than it looks because benchmarks tend to put the needle in easy positions.

**Problem two: latency and cost scale linearly.** A million-token context costs roughly a million-token's worth of tokens to process every turn. Even if accuracy were perfect, you would not want to send a year of chat history to the model on every API call. The price-per-token has dropped about 100× over three years; the amount of state a typical user accumulates has grown roughly with it. You don't get to escape the scaling.

**Problem three: the context window is single-tenant by design.** It is a working memory, not a shared archive. Multiple users, multiple devices, multi-agent systems — none of them can share a context window. Anything that has to be shared, indexed, or queried needs to live somewhere else.

`▶ SPEAK`

> The "just make the context window bigger" school of thought is roughly the equivalent of saying "I don't need a hard drive, I'll just buy more RAM." It's not wrong that more RAM helps. It's just that you're confusing two storage tiers that exist for reasons.

### 2.2 The receipt: LoCoMo

The benchmark that made this concrete is **LoCoMo** [[2402.17753-locomo]] — Evaluating Very Long-Term Conversational Memory of LLM Agents, Maharana et al., 2024. Ten multi-session conversations between two speakers, about 600 turns each, with about 2000 question-answer pairs that test five different capabilities:

1. **Single-hop** — one fact, find it.
2. **Multi-hop** — combine facts from two or more sessions.
3. **Temporal** — what was true when.
4. **Open-ended** — synthesize a free-form answer.
5. **Adversarial** — abstain when the answer is not in the conversation.

LoCoMo is the closest the field has to a standard exam. It is also imperfect — the conversations are LLM-generated, the QA pairs have noise, the temporal category is small. But every serious system reports on it, and the gold-evidence labels per question let you check whether a system actually used the right fact or got lucky.

`■ GRAPHIC 2-A` — bar chart with the five LoCoMo categories on the x-axis and a horizontal line at "human baseline ≈ 87%" across the top. Then overlay current SOTA bars: EverMemOS at 93.05, Hindsight at 89.61, Amory at 87.7, Mem0 at 68.4, Naive RAG at 56.2. The categories should clearly show the temporal column lagging in every system.

`⚙ EXPERIMENT 2-A` — *the "just put it all in context" demo.*

Open a terminal. Show the LoCoMo conv-26 file. It's about 35k tokens. Cat it. Show the size.

```bash
python -c "import tiktoken; e=tiktoken.encoding_for_model('gpt-4o'); print(len(e.encode(open('locomo10/conv26.json').read())))"
# → ~33,500 tokens
```

Then run the actual oracle baseline — the LLM is given the full transcript and asked the LoCoMo questions one by one.

```bash
python -m mempol.run_full_context_baseline --conv conv26 --model gpt-4o
# → Overall: 71.4% (single 53% / multi 68% / temporal 38% / open 79% / adv 90%)
```

`▶ SPEAK`

> Here's the punchline. We just sent the entire 33,000-token conversation transcript to GPT-4o for every single question. No memory system. No compression. No retrieval. The "best possible long-context score" — the oracle. And it gets 71 percent. Temporal questions are at 38 percent. Multi-hop is at 68 percent.
>
> The model has perfect access to every word that was ever said and it cannot reliably reason across them. Adding more tokens doesn't fix this. The model needs structure, and that's what memory systems try to give it.

### 2.3 The receipt: LongMemEval

LoCoMo's cousin is **LongMemEval** [[2410.10813-longmemeval]] — Wu et al., October 2024 — which is harder on three axes: longer histories (about 115k tokens of haystack per question), more sessions (30 to 40 per question), and a deliberate "abstain" category that punishes hallucination. Its top scores come in around 95 percent — but that 95 percent is achievable by systems that pre-process the transcript at write time, which we will come back to in Chapter 5.

The two benchmarks together let you triangulate. If a system reports only LoCoMo, ask why. If it reports both, look at where the gap shows up.

---

## 3. The naive baselines and why they fail
*(~8 minutes)*

### 3.1 Baseline one: stuff everything in context

We already showed why this fails on temporal questions (Chapter 2). The summary is that the context window is a flat sequence with weak position priors. Multi-step reasoning across distant items is fragile. The longer the context, the more fragile it gets.

Do not skip this baseline when you build a memory system. It's the first row of every honest results table. If your system can't beat oracle context, you don't have a memory system, you have a compression artifact.

### 3.2 Baseline two: RAG over the transcript

The next thing every team tries is: vector-search the chat history at query time, take the top-k most relevant chunks, put them in context, answer. This is what "memory" means in most production chat systems today. Mem0 [[2504.19413-mem0]], in its simplest form, is a polished version of this.

It works pretty well on single-hop questions. It works badly on multi-hop, temporal, and abstention.

**Why it fails on multi-hop.** Vector search returns chunks that are semantically similar to the question. A multi-hop question typically requires two chunks that are *not* semantically similar to each other or to the question — they connect via an entity. "What did Anna say about the new job after she moved to Berlin?" needs the move-to-Berlin chunk and the new-job chunk. Neither is the top-1 nearest neighbor of the other.

**Why it fails on temporal.** Vector search has no notion of time. A query about "the most recent X" returns the most semantically similar X, not the most recent one. You can add timestamp metadata to chunks and post-filter, but then you're doing retrieval *and* a structured query, which is no longer flat RAG.

**Why it fails on abstention.** If the answer is not in the transcript, vector search still returns the top-k *most similar* chunks, and the LLM happily uses them. The Mem0 paper itself reports a 40 percent extraction-failure rate on factual claims [[2504.19413-mem0|Mem0 §4.3]]. That's the architecture talking back.

`⚙ EXPERIMENT 3-A` — *the multi-hop failure demo.*

Pick a multi-hop question from LoCoMo conv-26. Run naive RAG. Show the retrieved chunks. Walk through which chunk is missing.

```bash
python -m mempol.demo.rag_failure --conv conv26 --qid mh_07
```

Expected output (paraphrased for the script):

```
QUESTION: After Anna moved cities, what job did she take?
RETRIEVED CHUNKS:
  [0.81] "Anna mentioned she was moving to Berlin next month."  (session 4)
  [0.79] "Anna's old job at the consulting firm had finished."  (session 4)
  [0.74] "I asked Anna about the move."                         (session 5)
  [0.71] "Anna said the apartment hunt was going well."         (session 6)
GOLD CHUNK (not retrieved): "Anna started at the climate fund in October." (session 9)
LLM ANSWER: "Anna took a job at a consulting firm." (wrong; that was her old job)
```

`▶ SPEAK`

> This is the typical RAG failure mode. The retriever is doing exactly what it's designed to do — return semantically similar chunks. The chunks about Anna's move are similar to the question. The chunk about her new job two sessions later mentions a "climate fund" and zero of the words in the question. The retriever misses it. The LLM grabs the consulting-firm reference from a chunk that does come back, and outputs a wrong answer with high confidence.
>
> The lesson is not that RAG is bad. The lesson is that flat retrieval over a transcript is not the same thing as memory. Memory has to know that "the move" and "the new job" are connected events about the same person, and surface both when either is asked about. That's a structural property, and you don't get it by tuning the embedding model.

### 3.3 Baseline three: summarize on a schedule

The third thing teams try is: every N turns, ask the LLM to summarize the conversation so far. Keep the summary. Throw the transcript away. This is what "context compression" usually means.

This is better than flat RAG for multi-hop reasoning because the summary forces the LLM to do the connecting at write time. It is worse for fine-grained recall, because details get smoothed away. And summaries get stale in a specific, dangerous way: if Anna moved cities and then later moved back, the summary written after move-one says "Anna lives in Berlin," and unless you rewrite the summary aggressively, it stays that way.

This brings us to the most important axis in the field: **write-time vs read-time compression**.

---

## 4. The taxonomy: write-time vs read-time, six substrates
*(~10 minutes)*

### 4.1 The write-time vs read-time axis

Every memory system pays its compute somewhere. There are essentially two options.

**Write-time compression.** When new information arrives, the system pays an LLM call to extract, summarize, or update structured state. Reads are cheap — they just look up the precomputed state. Mem0, Zep, EverMemOS, Mastra OM, Amory, MIRIX, and PIE are all write-time-heavy.

**Read-time compression.** Writes are cheap — append the raw observation to a log. Reads are expensive — at query time, the system searches the log, retrieves candidates, and the LLM reasons over them. RLM (recursive language models), Search-R1 [[2503.09516-search-r1]], and naive RAG are read-time-heavy.

The trade-off is roughly: write-time is faster at QA, more expensive to build state, and rewards a clever consolidator. Read-time is cheaper to build state, slower at QA, and rewards a clever retriever.

`■ GRAPHIC 4-A` — horizontal axis showing "compute paid at write time" vs "compute paid at read time." Place real systems as dots on the axis: Mem0, Mastra OM, EverMemOS, Zep, Amory on the left half; naive RAG, Search-R1, RLM on the right half. Put a third dot in the middle labeled **"sleep-time consolidation: pays compute offline"** for Auto-Dreamer / SCM.

### 4.2 The synthesis: sleep-time consolidation

There is a third option that is gaining ground: **sleep-time consolidation**. Writes are cheap (append to a log). Reads are cheap (small fixed state). The expensive step happens *between sessions*, off the user's critical path, when an offline consolidator processes the log and updates the durable state.

This is the most biologically inspired version of the design. It is also the most operationally interesting: you can pay arbitrarily large compute during the "sleep" pass without making the user wait. Auto-Dreamer [[2605.20616-auto-dreamer]] is the canonical paper here; SCM [[2604.20943-scm-sleep]] is a near neighbor.

`▶ SPEAK`

> The reason sleep-consolidation matters isn't that LLMs need to sleep. It's that pulling the expensive step off the user's response path lets you spend more on it. If you're trying to compete on latency, you can't afford a five-second LLM call per turn just to update a knowledge graph. But you can afford a five-minute pass overnight if it makes the next morning's reads cheap.

### 4.3 The six substrate families

Cutting the same field a different way, the actual data structure your memory lives in is also a design choice. There are roughly six families.

**Family 1: Flat vector store.** Embeddings + cosine similarity. Mem0 is the cleanest example. Cheap to build, great for single-hop recall, weak for temporal and structured queries.

**Family 2: Hierarchical summary tree.** Per-session summary, per-week summary, per-topic rollups. TiMem [[2601.02845-timem]] is the explicit example with five temporal levels. Better for temporal queries, harder to invalidate.

**Family 3: Typed knowledge graph.** Entities, relations, timestamps. Zep [[2501.13956-zep]] and the PIE system are KG-flavored. Strong on multi-hop and entity-centric queries, expensive to build and maintain.

**Family 4: Observation log plus consolidator.** Append-only event log, plus a consolidator that periodically rewrites parts of it into structured form. Mastra OM and Auto-Dreamer both live here. This is the design that most resembles how human memory is theorized to work.

**Family 5: Commit graph.** Memory as a versioned object with named branches, merges, and a history you can rewind. GitMem and Mesa FS are pushing this direction. Underdeveloped but the most natural fit for collaborative or multi-agent scenarios.

**Family 6: Filesystem.** Memory as a directory of structured markdown files the agent reads and writes with normal file operations. Letta FS is the cleanest example; in practice almost every "long-running project agent" ends up here because file operations are predictable and debuggable.

`■ GRAPHIC 4-B` — six-panel diagram. Each panel shows one substrate family with: a sketch of its data structure, two example systems with paper citations, one strength, one weakness. This is the diagram people will screenshot from this video, so it has to be tight.

`▶ SPEAK`

> If you take one thing from this chapter, take this: write-time vs read-time is the *workload* axis, and the substrate family is the *data structure* axis. They are independent. You can build a vector-store system that does write-time compression (Mem0) or a vector-store system that does read-time retrieval (naive RAG). You can build a knowledge graph that's read-time (rare) or write-time (Zep, PIE). The system's design is a point in this two-dimensional grid, not a single label.

### 4.4 A worked example: how each substrate handles a single-hop fact

To make this concrete, here's the same fact stored four different ways. The fact: **"Anna started at the climate fund in October 2025."**

`■ GRAPHIC 4-C` — four-panel side-by-side showing the same fact stored as:

1. **Flat vector store entry:**
   `{"text": "Anna started at the climate fund in October 2025.", "embedding": [0.14, -0.02, ...]}`

2. **Hierarchical summary node:**
   `Session 9 summary → "Anna's career: started at climate fund in October 2025."`

3. **Knowledge graph triple:**
   `(Anna) —[employed_by, valid_from=2025-10-01]→ (Climate Fund)`

4. **Filesystem entry:**
   `users/anna/timeline.md → "## October 2025\n- Started at climate fund."`

The fact is the same. The query path is different. The compute-at-write differs by about 10×. The compute-at-read differs by about 100×. The pick-a-substrate question is largely a question of where you want that compute to land.

---

## 5. NoReplay: the discipline most benchmarks ignore
*(~7 minutes)*

This is the section that will surprise most viewers. It's the section that, if I had to defend with one slide, I'd open the video with.

### 5.1 The unspoken rule

Here is the rule that distinguishes a memory benchmark from an open-book exam:

> **NoReplay.** Once the history has been ingested, the system may not replay it. The transcript is presented exactly once, in chronological order. The system gets a fixed-size scratchpad (say, 10,000 tokens). When a question is asked, the scratchpad is *frozen*. Answers must come from the frozen scratchpad, the question, and the model's parametric knowledge. No transcript access. No query-time retrieval over the history. No second passes.

This discipline was articulated by Peter Yang in May 2026 [[noreplay-vs-retrieval]]. It is not a new benchmark. It is a *protocol* you can apply to any benchmark. It distinguishes memory-the-capability from retrieval-the-capability.

### 5.2 What most systems actually do

By the NoReplay standard, most "memory" systems are not testing memory.

| System | One-pass ingest | Bounded scratchpad | Transcript access at QA | NoReplay-compliant |
|---|---|---|---|---|
| Long-context oracle | n/a | no | yes (IS the transcript) | no |
| Naive RAG | yes | no | yes (vector search at QA) | no |
| Mem0 | yes | no | no, but unbounded vector store | partial |
| Mastra OM | yes | no | no, but query log at QA | partial |
| EverMemOS | yes | no | unclear from paper, likely partial | partial |
| Zep | yes | no | no, query KG at QA | partial |
| Strict mempol | yes | yes (~500 tokens) | no | **yes** |
| Anthropic memory tool | yes | yes (configurable) | no | **yes** |

Almost every system that reports a top number is "one-pass write, unbounded state." The unbounded state means at question time you can re-search a substantial fraction of the original information. That's not memory in the cognitive sense — it's a custom retrieval index built from the history.

### 5.3 The Mastra footnote

Mastra OM reports 94.87 percent on LongMemEval, which is the top single-system score on the public leaderboard. It is also the most-cited "memory system has solved this" data point.

What Mastra actually does at write time: a "Reflector" component processes the full transcript and produces a structured pre-computed state — entity profiles, salient claims, topic clusters. The transcript is then thrown away. At QA time, the system queries the precomputed state.

This is not cheating, and it's a real engineering achievement. But it is a memory system that operates on *full transcript access at write time* with arbitrarily many LLM calls. If you measured the same task with a real-time constraint — "you see each message once, you have 500 tokens of scratchpad, the transcript disappears" — Mastra's score would be quite different. That comparison hasn't been published.

`▶ SPEAK`

> The way to read leaderboard numbers, post-NoReplay, is to ask three questions of any system claiming a big score.
>
> One: how much LLM compute did they pay at write time? If it's "arbitrarily many calls with the full transcript in scope," they're partly testing offline batch processing, not memory.
>
> Two: how big is their state at question time? If it's "all of the original transcript, just indexed," they're testing retrieval.
>
> Three: do they ever look at the original transcript again after the QA phase starts? If yes, you can't compare them to a system that doesn't.
>
> These three questions are not in any paper's evaluation table. They should be. Until they are, the leaderboard is a leaderboard of mixed capabilities, not a memory leaderboard.

### 5.4 The Hindsight footnote — the baseline that beat the system

Here is the most underdiscussed result in the field right now.

Hindsight [[2512.12818-hindsight]] is a memory system that reports 89.61 percent on LoCoMo. The same paper reports a baseline they call "Backboard" — essentially a simpler version of the same architecture — at 90.00 percent.

The baseline beats the headline system. The paper acknowledges this but doesn't dwell on it.

What this tells you: at frontier-model scale, the architectural sophistication of the memory system is buying about 1 to 3 percentage points. Most of the score is the underlying model. Most of the "improvement" you read about in memory papers is a wash with a slightly different baseline.

`■ GRAPHIC 5-A` — side-by-side bar comparison. Hindsight full system 89.61 next to Backboard baseline 90.00, with a caption: **"the baseline beats the system. From the Hindsight paper itself (Section 5.3)."**

`▶ SPEAK`

> The Hindsight result is the cleanest single data point we have about the actual ceiling of memory-architecture research at the frontier-model regime. It says: at GPT-4-class models, on this benchmark, your memory architecture is responsible for one to three points. The model is responsible for the rest.
>
> The interesting research question is no longer "what's the best architecture at unlimited cost." It's "what's the best architecture at a fixed budget." That reframing is what motivates the memory-budget-curves work and most of what I'm going to show you in the next two chapters.

---

## 6. Where the SOTA actually is
*(~8 minutes)*

### 6.1 The LoCoMo leaderboard, abstract-verified

Here is the most-recent LoCoMo leaderboard, with each system marked by what I could verify from the published paper. Numbers in **bold** are confirmed in the paper's abstract. Numbers in *italics* are extracted from a results table but not abstract-highlighted. Numbers in (parentheses) are reported in a third-party comparison and not independently verified.

| Rank | System | LoCoMo overall | Status | Paper |
|---|---|---|---|---|
| 1 | EverMemOS | *93.05* | table-only | [[2601.02163-evermemos]] |
| 2 | Hindsight | **89.61** | abstract-verified | [[2512.12818-hindsight]] |
| 3 | Amory | *87.7* | table-only | [[2601.06282-amory]] |
| 4 | MIRIX | **85.4** | abstract-verified | [[2507.07957-mirix]] |
| 5 | Zep | (85.22) | third-party | [[2501.13956-zep]] |
| 6 | TiMem | *75.30* | table-only | [[2601.02845-timem]] |
| 7 | Mem0 | (68.4) | third-party | [[2504.19413-mem0]] |
| — | Naive RAG | ~56 | reference baseline | — |
| — | Oracle context | ~71 | reference baseline | — |

`■ GRAPHIC 6-A` — clean rendered table, computer://-style. Add a small footnote: **"verification status matters. Two of the top three are table-only."**

### 6.2 The convergence story

The interesting pattern in this table is the *spread*. The top six are between 75 and 93 percent. The bottom two are at 56 and 68. The gap between "best architecture" and "decent architecture" is about 17 points; the gap between "decent" and "naive" is about the same. The architecture clearly matters, but the marginal improvement at the top is shrinking.

Plot the same numbers over time and you see flattening. From early 2024 to late 2025, every six months added roughly 8 percentage points at the top. From late 2025 to mid-2026, the gain is closer to 2 percentage points per six months. We are converging on a ceiling that is probably benchmark-specific noise plus model-specific ceiling.

`■ GRAPHIC 6-B` — line chart. X axis: months from Feb 2024 to May 2026. Y axis: LoCoMo overall. Plot the top score reported each month. Mark Mem0, Zep, MIRIX, Hindsight, EverMemOS as the inflection points. Show the flattening.

### 6.3 Per-category breakdown

The overall number hides the interesting failures. Here is EverMemOS's per-category breakdown:

- **Single-hop: 96.67** — basically solved.
- **Multi-hop: 91.84** — strong, still gaining.
- **Open-ended: 89.72** — strong.
- **Temporal: 76.04** — meaningfully behind.

Temporal questions are the persistent weakness across every top system. This is not noise. It's a structural property — vector stores have no native time concept, knowledge graphs that timestamp triples are still rare, and almost no system does bi-temporal reasoning (validity time vs ingestion time, the Zep distinction).

`▶ SPEAK`

> The category to watch over the next twelve months is temporal. It's the only category where there's still a clear ceiling effect, and there's a clean theoretical reason: the substrates the field uses by default — vector stores, summary trees — were not designed for time-as-a-first-class-axis. The systems that are starting to win on temporal — Zep, TiMem, the Thinking Machines Interaction Models paper [[20260511-thinkingmachines-interaction-models]] — are the ones that treat time differently from text. That's the next architectural frontier.

### 6.4 LongMemEval — the other yardstick

LongMemEval [[2410.10813-longmemeval]] is harder. The headline numbers:

- **Mastra OM: 94.87** (write-time compression with full transcript access)
- **OMEGA (closed): 95.4** (similar architecture, closed-source)
- **Mem0: ~62.6**
- **Naive RAG: ~50**

If you take the NoReplay framing seriously, the Mastra and OMEGA numbers are best read as "ceiling for write-time pre-processing of the full transcript," not as memory scores. The gap between the top systems on LongMemEval and the top systems on LoCoMo is partly a benchmark difference and partly an evaluation-protocol difference.

`■ GRAPHIC 6-C` — two side-by-side tables, LoCoMo top-5 and LongMemEval top-5, with a callout: **"different benchmarks measure different capabilities. Compare carefully."**

---

## 7. Three live research directions
*(~7 minutes)*

Where is the field actively trying to push the ceiling? Three directions are getting real attention.

### 7.1 Direction one: sleep-consolidation with a learned consolidator

The pattern: append-only log of observations, plus a consolidator that periodically rewrites parts of the log into structured form. The consolidator can be:

- **Prompted** (Mastra OM today)
- **Optimized with prompt-evolution** (our Goal 01 experiment, using GEPA [[2507.19457-gepa]] applied to a DSPy module)
- **RL-trained** (Auto-Dreamer's GRPO setup [[2605.20616-auto-dreamer]])

The Auto-Dreamer headline: **41.1 percent success rate on ScienceWorld vs UMEM's 34.1 percent, at one-twelfth the memory budget**. That last clause is the actually interesting part. They're not just winning on score, they're winning on score-per-token. That's the memory-budget-curve framing showing up in actual results.

The GEPA-vs-GRPO question is open. GEPA reports about 35× fewer rollouts than GRPO on similar prompt-evolution tasks [[gepa-vs-grpo]], which would be a substantial cost win if it transfers to the consolidator-on-LoCoMo setting. That experiment is running on our cluster as of this writing.

`▶ SPEAK`

> If you only watch one direction, watch this one. It's the synthesis. Cheap writes, cheap reads, expensive consolidation offline. It's biologically motivated. It's economically motivated — you can pay arbitrarily large compute when nobody's waiting on you. And it composes with every other substrate. You can consolidate into a vector store, a summary tree, a knowledge graph, or a filesystem.

### 7.2 Direction two: read-time recursion

The other camp: keep writes cheap, push intelligence into the read path.

**RLM (Recursive Language Models)** [[rlm]] is the cleanest example. The agent at query time can search, retrieve, summarize, re-search — recursively, with bounded depth — to assemble an answer. **Search-R1** [[2503.09516-search-r1]] trains the search-and-reason loop end-to-end with reinforcement learning.

The argument for this camp: write-time compression discards information you don't yet know you'll need. Read-time recursion lets you re-derive structure on demand from the raw record. The cost is that every read is expensive.

The argument against: most users do not want a five-second pause on every chat reply.

The hybrid that wins, my guess, is **write-time consolidation for hot state, read-time recursion for cold state**. Nobody has built this cleanly yet. There's a paper waiting for someone.

### 7.3 Direction three: time-aware primitives

The third direction: rebuild the data structure to make time a first-class axis.

- **Zep's bi-temporal model** [[2501.13956-zep]] separates *event time* (when did this happen in the world) from *transaction time* (when did the system learn about it). This is borrowed from database theory and it's a clean way to handle revision: "I told you I was moving to Berlin in June, then in May I changed plans and moved to Munich" stays correct.
- **TiMem's five-level hierarchy** [[2601.02845-timem]] explicitly stratifies memory by temporal scale — turn, session, week, month, year. This is the LLM analog of human episodic memory consolidation periods.
- **Thinking Machines' Interaction Models** [[20260511-thinkingmachines-interaction-models]] proposes treating user interactions as a structured stream with temporal primitives baked in. Not yet a benchmarked system; an architectural proposal.

These three are converging on the same insight: time isn't metadata, it's the spine.

`■ GRAPHIC 7-A` — three-way fork diagram. Top: "Sleep consolidation (Auto-Dreamer, GEPA-consolidator)." Middle: "Read-time recursion (RLM, Search-R1)." Bottom: "Time-aware primitives (Zep, TiMem, TML)." Below each, the failure mode it addresses, in one sentence.

---

## 8. What you should build today
*(~6 minutes)*

You've got a decision to make. Here is a decision tree that will land you on a sane default for most use cases. I am explicitly *not* recommending the SOTA system. SOTA papers are research artifacts. They are usually under-engineered, brittle, and expensive to run. The right default for shipping is one step behind the frontier.

### 8.1 The decision tree

`■ GRAPHIC 8-A` — clean Excalidraw flowchart, four-deep branches.

```
What's the workload?
│
├── High-QPS chat, single user, conversational
│       └── Flat vector store + Anthropic-style contextual retrieval
│            (Mem0-flavored; ship in a week)
│
├── Long-horizon project agent, single user, code-or-doc-centric
│       └── Filesystem + structured markdown + git
│            (Letta FS / Mesa FS; debuggable, the agent reads and writes files)
│
├── Multi-session personal assistant, single user
│       └── Observation log + periodic consolidator (LLM-prompted)
│            (Mastra-OM flavored; add a nightly job)
│
├── Agent that takes actions in environments
│       └── Typed memory bank + learned consolidator
│            (Auto-Dreamer flavored; budget for GRPO if you can)
│
└── Multi-user, shared knowledge, low budget
        └── Knowledge graph + entity resolution + read-time KG queries
             (Zep flavored; pay write-time once, ride cheap reads)
```

### 8.2 What "shippable" looks like

For the most common case — **single-user multi-session chat** — here is the architecture I would ship today. The whole thing fits in roughly 400 lines of Python.

```python
# Memory write path (per turn, cheap)
def on_turn(user_msg, assistant_msg, session_id, user_id):
    log_event(user_id, session_id, user_msg, assistant_msg, ts=now())
    # No LLM call. Just append.

# Memory read path (per turn, cheap)
def build_context(user_id, current_query):
    state = load_consolidated_state(user_id)
    # Optional: small targeted retrieval over the log for very recent turns
    recent = load_recent_events(user_id, hours=24)
    return system_prompt + state + recent + current_query

# Sleep pass (offline, periodic)
def consolidate(user_id):
    events_since_last = load_events_since_last_consolidation(user_id)
    state = load_consolidated_state(user_id)
    new_state = llm_consolidator(state, events_since_last)
    save_consolidated_state(user_id, new_state)
```

The interesting prompt is `llm_consolidator`. The right shape is roughly:

```
You are a memory consolidator. You are given:
1. The current consolidated state for this user (structured markdown, ~5000 tokens).
2. A batch of new events since the last consolidation (raw chat turns, ~10000 tokens).

Update the state to incorporate the new events. Follow these rules:
- Preserve facts that are still valid.
- Update facts that have been superseded (note the change, don't just overwrite).
- Add new facts that are durable (user preferences, recurring topics, decisions).
- Drop chit-chat, single-use information, and resolved logistics.
- Always cite the source event by ID when you make a claim.

Output the new state in the same structured format as the input.
```

That prompt, plus a nightly cron job, plus a simple recency window for the "last few hours" — that's a serviceable memory system. It will beat naive RAG by a substantial margin on multi-hop and abstention. It will not be SOTA on temporal. That's fine for almost everything you'd ship.

`⚙ EXPERIMENT 8-A` — *show the consolidator running on a real LoCoMo conversation. Cut from the prompt above to the actual structured-state output. Highlight a specific revision case: "Anna moved to Berlin (June 2025) → updated: Anna moved to Munich (May 2025), Berlin plan superseded."*

### 8.3 What you should *not* do

- **Don't build a knowledge graph as your first version.** The entity-resolution and schema-design work is enormous and you will pay it before you know if your product needs it. Add a KG when you've shipped flat consolidated state and you can prove the multi-entity queries are the bottleneck.
- **Don't roll your own embedding model.** OpenAI, Voyage, and Cohere are all within a few points of each other. Pick one, move on.
- **Don't optimize the consolidator prompt by hand for more than a week.** If it's still your bottleneck, set up a DSPy module and use GEPA. Returns to hand-tuning saturate fast.
- **Don't pay for the SOTA paper's stack.** EverMemOS, MIRIX, and Auto-Dreamer are interesting *because* they're research systems. The papers tell you what's possible; they don't tell you what to ship.

---

## 9. Open problems
*(~4 minutes)*

Three problems are unsolved enough that there's an obvious paper to write in each.

### 9.1 Belief revision

When a user says "actually I moved to Munich, not Berlin," every current memory system handles this badly. Mem0 adds the new fact and leaves the old one. Knowledge graphs add a new triple and don't always invalidate the old one. Summary trees rewrite the topic but don't propagate the revision to derived facts. Filesystem systems require the agent to know which file to edit.

The Zep bi-temporal model is the closest approximation, and it still requires the agent to detect the revision intent — there's no automatic "this is a contradiction" check.

`[[belief-revision]]` for the deeper writeup. The paper to write: a benchmark for belief revision under real chat conditions, plus a baseline system that detects and propagates revisions.

### 9.2 Continuous time perception

Most memory systems treat time as a sortable timestamp. Real time-reasoning requires answering things like "what's been on this user's mind lately?" or "was this a recurring topic last month?" — questions where the right answer depends on *duration*, *frequency*, and *recency* together.

The Thinking Machines Interaction Models paper [[20260511-thinkingmachines-interaction-models]] solves about thirty seconds to a few minutes of this for live audio. Weeks to months remains unsolved.

### 9.3 Multi-agent shared memory

Almost every published memory system assumes one user, one agent. The day you have multiple agents acting on behalf of one user — or one agent acting on behalf of multiple users — every primitive needs revisiting. Access control, write conflicts, versioning, branch-and-merge. There is no LoCoMo for multi-agent memory. There should be. There is probably a benchmark paper to write here in a weekend.

`■ GRAPHIC 9-A` — three rows. Each row: the problem name, one example sentence that breaks today's systems, and the cleanest existing partial-solution.

---

## 10. Reading list and close
*(~2 minutes)*

If you want to go deeper, here is the order I'd read in.

**The benchmarks first** — you need to know what the field is measuring before you read what it claims to do:

1. **LoCoMo** [[2402.17753-locomo]] — Maharana et al., 2024. The standard exam.
2. **LongMemEval** [[2410.10813-longmemeval]] — Wu et al., 2024. The harder cousin.

**Then the production systems**, lightest to heaviest:

3. **Mem0** [[2504.19413-mem0]] — the polished flat-vector baseline.
4. **Zep** [[2501.13956-zep]] — bi-temporal knowledge graph.
5. **MIRIX** [[2507.07957-mirix]] — multi-agent memory orchestration.

**Then the consolidation work**:

6. **SCM** [[2604.20943-scm-sleep]] — sleep-consolidation with algorithmic forgetting.
7. **Auto-Dreamer** [[2605.20616-auto-dreamer]] — the RL consolidator.

**Then the theory and frontier**:

8. **NoReplay framing** [[noreplay-vs-retrieval]] — Peter Yang's protocol.
9. **Hindsight** [[2512.12818-hindsight]] — the "baseline beats the system" result.
10. **Interaction Models** [[20260511-thinkingmachines-interaction-models]] — time as a primitive.
11. **GEPA** [[2507.19457-gepa]] — prompt evolution for the consolidator.

Wiki: `https://[your-domain]/research/wiki/` — every paper above has a structured page with the receipts.

`▶ SPEAK`

> Last thing. I run this channel because the public version of the memory story is way ahead of where the systems actually are. Six papers a week claim SOTA. Two of them are reporting on benchmarks where their baseline already beat their full system. One of them is doing a different task than it claims. The wiki linked below tries to keep this honest. If you find a mistake, file an issue. If you build something that works, tell me; if it's good, I'll cover it.
>
> Next episode is going to be the live build: a strict-NoReplay memory system for a single user, shipped end-to-end on a real benchmark, on camera. Subscribe if you want that.

`■ GRAPHIC 10-A` — end card. Channel name + wiki URL + repo URL + "next episode: building a NoReplay memory system, live."

---

## Required production assets (master list)

| # | Asset | Format | Source / status |
|---|---|---|---|
| 0-A | Channel title card | image | needs design |
| 0-B | "How ChatGPT memory works today" split | SVG | Excalidraw, ~30 min |
| 1-A | Four kinds of memory + LLM analogs | SVG | Excalidraw, ~45 min |
| 1-B | "Missing consolidator" overlay | SVG | derived from 1-A |
| 2-A | LoCoMo bar chart with categories | PNG | render from leaderboard data |
| 4-A | Write-time vs read-time axis with systems | SVG | Excalidraw, ~30 min |
| 4-B | Six substrate families panel | SVG | Excalidraw, ~90 min (the screenshot asset) |
| 4-C | Same fact, four storage forms | SVG | Excalidraw, ~30 min |
| 5-A | Hindsight vs Backboard bars | PNG | render from paper data |
| 6-A | LoCoMo leaderboard with verification status | PNG | render from paper_leaderboard DB |
| 6-B | SOTA over time line chart | PNG | render from paper_leaderboard DB |
| 6-C | LoCoMo vs LongMemEval side-by-side | PNG | render from paper_leaderboard DB |
| 7-A | Three-way fork diagram | SVG | Excalidraw, ~45 min |
| 8-A | Build decision tree | SVG | Excalidraw, ~45 min |
| 9-A | Three open problems table | SVG | Excalidraw, ~30 min |
| 10-A | End card | image | needs design |

Total asset production estimate: **~10 hours of design work**, mostly in Excalidraw. The PNGs from `paper_leaderboard` should take ~20 minutes each if the rendering script is reused.

## Required live experiments (master list)

| # | Demo | Pre-recorded? | Setup |
|---|---|---|---|
| 2-A | Oracle context on LoCoMo conv-26 | yes, capture terminal | `mempol.run_full_context_baseline` |
| 3-A | Multi-hop RAG failure trace | yes, capture terminal | `mempol.demo.rag_failure` |
| 8-A | Consolidator running with belief-revision case | yes, capture both prompt + output | `mempol.demo.consolidator_live` |

All three demos should be runnable from the `mempol` repo as one-liners. If they aren't already, that's a 1–2 hour prep task before shoot day.

## Production order

1. Lock the script (this document) — 1 day of light edits after a read-through.
2. Build the three demo CLIs as one-liners — 2 hours.
3. Run the demos, capture terminal — 1 hour.
4. Render the data-driven graphics from `paper_leaderboard` — 2 hours.
5. Excalidraw the diagrams — 10 hours.
6. Record voiceover in 4 sittings — 1 day.
7. Edit with B-roll over the voice — 3 days.
8. Tighten pass — 1 day.
9. Thumbnail + title + description — half a day.

**Total: about 2 weeks for a 60-minute evergreen video.** That is the same estimate the outline file had; the script makes the path concrete.

## Voiceover style notes

- Slow tempo, ~150 words per minute. The content is dense; the delivery should not be.
- No background music under the explanatory sections. Light music only under graphic-heavy transitions.
- When citing a paper, say the paper name out loud, then briefly cite the author and year. The arxiv ID goes on the lower third, not in the audio.
- When showing a table, read at most three rows of it. The rest is for the viewer to pause on.
- Defer to the on-screen text for numbers. Say "around 90 percent" out loud while showing 89.61 on screen. Precision is the visual's job, not the voice's.

## Thumbnail and title options

Karpathy-style — descriptive, no clickbait. Three options to pick from:

1. **"AI memory, from scratch"** — clean, foundational, mirrors his "neural nets, from scratch" title.
2. **"How LLM memory actually works (and what the leaderboards get wrong)"** — slightly more pointed, picks up the NoReplay thread.
3. **"Working memory, episodic memory, and everything the field gets wrong"** — picks up the channel name; works as an inaugural episode.

Recommend option 3 for episode 1 because it cements the channel name and the editorial voice. Save option 1 for the next-episode end card.

---

*End of script.*
