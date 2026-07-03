---
title: "Memory in language models — annotated TOC"
channel: "Working Memory"
episode: 1
audience: "AI researchers and serious practitioners; assumes transformer literacy, basic RL vocabulary, comfort reading papers"
status: "TOC for review — fill out after sign-off"
last_updated: 2026-05-30
inspiration_refs: "Distill.pub posts; Sasha Rush 'Annotated Transformer'; Lilian Weng's notes; Karpathy zero-to-hero; Chris Olah's Anthropic posts"
---

# Memory in language models
*An annotated table of contents.*

> The intro video is **not** a survey. It is one of those Distill-style guided tours where the reader leaves with a working mental model and a code repo. Researcher audience. Bleeding-edge papers cited matter-of-factly. Demos drive the narrative — every chapter has at least one live experiment we run on screen, with the result shown, interpreted, and connected back to a paper.

## The series this episode opens

Three foundational episodes. Each watchable on its own. Together they form the canonical reference.

- **Ep 1 — Memory (this).** Substrates, design axes, training methods, SOTA tour. The data-structure layer.
- **Ep 2 — Temporal reasoning.** Time as a first-class axis: validity vs ingestion, decay, scheduling, bi-temporal queries. Why most current systems are "temporally blind" and what fixes it.
- **Ep 3 — World models and planning.** The action layer. Memory + temporal → an agent that simulates before acting. Dreamer-flavored latent models, LLM-as-world-model, planning.

Ep 1 ends with a clean handoff: *"the consolidator we just built treats time as a sort key. In the next episode we'll see why that's nowhere near enough."*

---

## Episode 1 — table of contents

| # | Chapter | Target time | Demo(s) | Key papers |
|---|---|---|---|---|
| 0 | Cold open: the canonical fail | 2 min | Show ChatGPT fail across sessions, live | — |
| 1 | What we mean by memory (operational defs) | 6 min | None — whiteboard | [[2309.02427-coala]] |
| 2 | Why context isn't memory (mechanistic) | 8 min | Cost curve as f(session length) | [[2402.17753-locomo]], [[2410.10813-longmemeval]] |
| 3 | The naive baselines, profiled | 9 min | Vector-search RAG failure trace | [[2504.19413-mem0]] |
| 4 | The hidden design axes | 9 min | Substrate-axis grid plot | [[write-time-vs-read-time]], [[noreplay-vs-retrieval]], [[memory-budget-curves]] |
| 5 | The substrate zoo, re-derived | 12 min | Same fact, six storage forms, side-by-side | [[2501.13956-zep]], [[2601.02845-timem]], [[2604.20943-scm-sleep]], [[gitmem]] |
| 6 | Sleep consolidation as the synthesis | 8 min | Run a 30-line consolidator on conv-26 | [[2605.20616-auto-dreamer]], [[2604.20943-scm-sleep]] |
| 7 | Learning the consolidator (DSPy + GEPA) | 8 min | Live GEPA optimization, +20pp visible | [[2507.19457-gepa]], [[2508.19828-memory-r1]] |
| 8 | The SOTA tour, in axis-coordinates | 7 min | Render leaderboard pareto plot | [[2601.02163-evermemos]], [[2512.12818-hindsight]], [[2601.06282-amory]], [[2507.07957-mirix]] |
| 9 | Open problems and the next episode | 4 min | None — whiteboard | [[belief-revision]], [[time-aware-memory]] |

**Total: ~73 min.** Cuttable to 55 by tightening Ch 5 to four substrates instead of six and skipping the GEPA live-run in Ch 7 (cut to pre-recorded clip).

---

## Chapter-by-chapter notes

### 0 — Cold open: the canonical fail (2 min)

The single tightest demonstration of why we're here. Two browser windows, two ChatGPT sessions, two days apart in the screen recording. Establish that this is a clean operational gap, not a debate.

- **Demo:** Live screen capture. Session 1: "I'm allergic to penicillin." Session 2 next day, fresh chat: "What antibiotics should I avoid?" The model either refuses or guesses. Cut.
- **Voiceover beat:** "This is the default. Every state-of-the-art deployed LLM forgets you the moment you close the tab. The thing we informally call memory is, behind the scenes, mostly retrieval — and as we'll see, retrieval and memory are not the same."

No editorializing about the field. We're here to build, not litigate.

---

### 1 — What we mean by memory (operational defs) (6 min)

Set the vocabulary we'll use for the next hour. The vocabulary is CoALA's, lightly adapted. The point isn't to lecture the cog-psych framework, it's to make the rest of the video legible.

- **1.1** The CoALA frame: working / episodic / semantic / procedural. One sentence each. Map onto current LLM stacks: context window / per-session log / weights or KG / tool routines and system prompts.
- **1.2** The mapping is imperfect on purpose. Three specific places it breaks: (a) context windows are 6 orders of magnitude bigger than human working memory; (b) the model's weights blur semantic and procedural; (c) **no consolidation loop exists by default** — the missing piece.
- **1.3** The operational definition we'll use for the rest of the episode: *persistent state across sessions that selectively retains and updates information about a user, environment, or task*. Each word load-bearing.

**Papers:** [[2309.02427-coala]] (the cognitive architecture frame).

**Novel angle:** Most videos hand-wave the cog-arch mapping. We use CoALA's exact decomposition so that when we later show MIRIX as a "multi-agent episodic + procedural" system, the viewer has a slot for it. The cog-arch frame earns its place by making downstream taxonomies trivial.

---

### 2 — Why context isn't memory (mechanistic) (8 min)

The argument *"just make the context window bigger"* deserves a serious refutation, not a vibes-based one. Three mechanistic reasons + one cost demo.

- **2.1** Long-context recall isn't flat. Cite the lost-in-the-middle finding and the more recent needle-in-haystack work. Show one curve.
- **2.2** Cost scales linearly with state and is hit on *every* turn. Run the math live for a 50-turn-a-day user with 6 months of history.
- **2.3** Context windows are single-tenant. Multi-user, multi-device, multi-agent — anything shared needs an external substrate.
- **2.4** **Demo:** Plot turn cost in tokens vs session number on the LoCoMo conv-26 transcript. Cumulative cost curve. The curve is the picture that justifies the rest of the video.

**Papers:** [[2402.17753-locomo]], [[2410.10813-longmemeval]] (introduce the benchmarks we'll use throughout; one slide each, defer detail).

**Novel angle:** Most "why not just long context" debates are empirical. We frame it as three orthogonal failure modes: accuracy degradation, cost scaling, tenancy. The third is rarely discussed and is the one that actually matters for production.

---

### 3 — The naive baselines, profiled (9 min)

The two things every team tries first, and exactly where each breaks. We do not say "RAG is bad" — we show *which class of question* each baseline fails on.

- **3.1** Baseline A: stuff-everything-in-context (oracle). We already paid for this in Ch 2. Now we show the score: ~71% on LoCoMo conv-26. Single-hop near-perfect; temporal questions at ~38%. The model has all the words and cannot reason across them.
- **3.2** Baseline B: vector-search over the transcript at QA time. The Mem0-shaped pattern.
- **3.3** **Demo:** A multi-hop failure trace. Print the top-5 retrieved chunks, highlight the gold chunk that wasn't retrieved (the "new job" sentence two sessions after the "moved cities" sentence). Show why semantic similarity is the wrong distance metric for multi-hop.
- **3.4** Baseline C: schedule-based summarization. Better on multi-hop, worse on detail recall. Introduces the *staleness* failure mode that motivates Ch 6.
- **3.5** **Demo:** A belief-revision failure on a summarized state. "Anna moved to Berlin (June)" → user later says "actually it was Munich" → summary still says Berlin → model still says Berlin. This is the canonical worked example we'll thread through the rest of the episode.

**Papers:** [[2504.19413-mem0]] (their own paper reports ~40% extraction-failure rate, cite it).

**Novel angle:** Don't dunk on RAG. Profile it. Show the exact question class where it fails and why the embedding distribution makes that failure structural, not a tuning bug. We use real LoCoMo data so the viewer can rerun.

---

### 4 — The hidden design axes (9 min)

The single most useful chapter for a researcher. The field looks chaotic when you list systems. It clarifies when you decompose the design space into independent axes. Three axes. Four if you count the substrate.

- **4.1** **Axis 1 — write-time vs read-time compression.** Where does the LLM cost land? Mem0, Mastra, EverMemOS, Amory, Zep, PIE pay at write time. Search-R1, RLM, naive RAG pay at read time. Auto-Dreamer pays *offline*, the third position.
- **4.2** **Axis 2 — bounded vs unbounded state.** This is the axis NoReplay [[noreplay-vs-retrieval]] makes explicit. Almost every published "memory system" has unbounded state at QA time, which means they're partly testing retrieval. Show the Yang protocol cleanly.
- **4.3** **Axis 3 — single vs continuous time.** Most systems treat memory as a flat set; a few (Zep, TiMem, TML) treat time as the spine. Foreshadows Ep 2.
- **4.4** **Axis 4 (orthogonal) — substrate.** The data structure. Vector, tree, KG, log, commit-graph, FS. We do these in Ch 5.
- **4.5** **Demo:** A 3-axis cube. Place 8 named systems as dots. The cube is the figure the viewer screenshots.

**Concepts cited:** [[write-time-vs-read-time]], [[noreplay-vs-retrieval]], [[memory-budget-curves]], [[time-aware-memory]].

**Novel angle:** Most public material lists substrates ("vector vs KG") or compares systems pairwise. The 3-axis decomposition is in nobody's YouTube video; we earned it from reading 27 papers and noticing that all the apparent disagreement reduces to position on these axes. This is the chapter people will cite the video for.

---

### 5 — The substrate zoo, re-derived (12 min)

Now that the axes exist, the substrate families fall out as engineering choices, not arbitrary buckets. Walk six. For each: (a) one-paragraph mechanism, (b) where it lands on the axes, (c) the named exemplar paper, (d) what it's good and bad at.

- **5.1 Flat vector store.** Mem0. Cheap writes if pre-extracted; cheap reads via ANN. Great single-hop, structurally bad multi-hop and temporal.
- **5.2 Hierarchical summary tree.** TiMem. Five temporal levels (turn → session → week → month → year). Stronger temporal, harder to invalidate locally.
- **5.3 Typed knowledge graph.** Zep (bi-temporal — preview Ep 2), PIE. Multi-hop wins, but build cost is real.
- **5.4 Observation log + consolidator.** Mastra-OM, Amory, Hindsight. The pattern that most resembles human episodic→semantic. Sets up Ch 6.
- **5.5 Renormalization-group memory.** RGMem [[2510.16392-rgmem]] — a beautiful physics-inspired pattern: coarse-grain stable patterns into higher-level memory blocks the way condensed-matter people coarse-grain spin lattices. Rarely covered. Worth four minutes for the elegance alone.
- **5.6 Commit graph + filesystem.** GitMem, Letta FS, Mesa FS. Memory as a versioned object with branches and merges. Underdeveloped substrate, but the right one for collaborative or multi-agent work.

- **Demo:** A single belief — *"Anna moved to Berlin → Munich"* — stored six ways. Side-by-side panels. Same fact, six query paths. The figure that anchors substrate choice in your viewer's head for the rest of their career.

**Papers:** [[2504.19413-mem0]], [[2601.02845-timem]], [[2501.13956-zep]], [[2601.06282-amory]], [[2512.12818-hindsight]], [[2510.16392-rgmem]], [[gitmem]] (system page).

**Novel angle:** Most surveys list substrates without the connection back to axes. We derive each from the axis position. RGMem's renormalization-group framing is mentioned almost nowhere in public talks and is genuinely worth knowing — we give it real space.

---

### 6 — Sleep consolidation as the synthesis (8 min)

The pattern that resolves the write-time vs read-time tension by moving the expensive step off the user's path. The bio-motivated, operationally interesting third option.

- **6.1** The two-system claim from cognitive neuroscience: fast encoder (hippocampus) + slow consolidator (cortex), bridged during sleep. The CLS literature. One slide of receipts, not a lecture.
- **6.2** What that buys in an LLM stack: cheap writes (append), cheap reads (small fixed state), expensive batched consolidation in the gap. You can spend arbitrarily large compute offline if the user doesn't have to wait.
- **6.3** The two cleanest research examples: **SCM** with algorithmic forgetting curves, and **Auto-Dreamer** with a CLS-inspired fast-slow split trained via GRPO.
- **6.4** **Demo:** Write a 30-line consolidator prompt live. Run it on the conv-26 chunks. Show the structured-state output. Show the consolidator handling the belief-revision case from Ch 3 — "Anna moved to Munich, Berlin plan superseded" — propagating cleanly into the new state.
- **6.5** Why this matters operationally. The cost-vs-performance pareto curve shifts; the latency-vs-quality pareto curve shifts. Both matter for shipping.

**Papers:** [[2605.20616-auto-dreamer]] (the canonical instance), [[2604.20943-scm-sleep]] (the forgetting-curve neighbor), [[sleep-consolidation]] (the concept page).

**Novel angle:** Most public memory content stays at "RAG vs long-context." Sleep consolidation is the next-frontier idea most viewers haven't encountered. We earn it from first principles (the CLS argument), then show it running.

---

### 7 — Learning the consolidator (DSPy + GEPA) (8 min)

The most directly experimental chapter. The consolidator prompt is the right target for optimization because it lives offline (cost-tolerant) and has a clear outcome metric (downstream QA score). Two ways to learn it: prompt evolution and RL.

- **7.1** Why the consolidator is the right thing to optimize. Argument from the design axes: writes are cheap, reads are cheap, the consolidator is where the LLM intelligence actually lives.
- **7.2** Express the consolidator as a DSPy module. Show the signature on screen (~20 lines). The signature *is* the API contract between memory and the optimizer.
- **7.3** **Demo (the headline experiment of the episode):** Run GEPA against the DSPy consolidator on conv-26. Live or replay. Show the baseline mean (60%), show the GEPA-evolved mean (80%), show the prompt diff between original and optimized. Watch the prompt evolve in front of the camera.
- **7.4** The RL alternative: Memory-R1 and Auto-Dreamer train the consolidator with GRPO and outcome reward. 35× more rollouts than GEPA (citing the GEPA paper's own claim). Tradeoff: more compute, potentially more headroom.
- **7.5** Where this lives in your stack today. A flag in your config that swaps consolidator versions. Talk through the engineering.

**Papers:** [[2507.19457-gepa]], [[2508.19828-memory-r1]], [[2605.20616-auto-dreamer]], [[gepa-vs-grpo]] (concept page).

**Novel angle:** Live optimization of a memory component on camera is essentially absent from public content. This is the chapter that ties the whole video to a working repo. The on-screen +20pp delta is the visceral receipt that "the consolidator is the right target."

---

### 8 — The SOTA tour, in axis-coordinates (7 min)

Not a leaderboard. A *map.* For each major system, locate it on the axes from Ch 4 and the substrate from Ch 5. The viewer leaves with coordinates, not a ranking.

- **8.1** **EverMemOS** — write-time, unbounded state, hierarchical with conflict resolution. 93.05% LoCoMo (as of recording). What its self-organizing layer actually does.
- **8.2** **Hindsight** — write-time, observation log + reflective rewrite. 89.61%. The Backboard ablation result is genuinely interesting — we mention it as a one-line *"the simpler version of their own system is within noise of the full thing,"* without making it a polemic.
- **8.3** **Amory** — narrative-driven write-time, observation-log substrate. 87.7%. The narrative-coherence loss is the interesting research contribution.
- **8.4** **MIRIX** — multi-agent orchestration over typed memory. Six memory components, sub-agents that own them. 85.4%. The right thing to point at for Ep 2/3 (multi-agent memory is mostly unbuilt).
- **8.5** **Mastra OM** — write-time, observation log + Reflector. 94.87% LongMemEval. We note its result honestly and flag that its write-time access pattern is different from oracle-context (foreshadows the NoReplay deep-dive episode).
- **8.6** **Demo:** A pareto plot. X-axis: median tokens of state at QA time. Y-axis: LoCoMo overall. Each system as a dot. Color-coded by substrate family. This is the chart we link in the description.

**Papers:** [[2601.02163-evermemos]], [[2512.12818-hindsight]], [[2601.06282-amory]], [[2507.07957-mirix]] + mention of Mastra OM via the relevant tool/system page.

**Novel angle:** Plot in axis-coordinates instead of a flat leaderboard ranking. The viewer can predict where the next paper will land and what trade-off it'll make. This is the chapter that transitions the viewer from "consumer of the field" to "participant in the field."

---

### 9 — Open problems and the next episode (4 min)

End on what isn't solved. Three honest gaps; each is a paper waiting to be written.

- **9.1** **Belief revision.** Cite [[belief-revision]]. The Anna-moved-to-Munich example we've been threading all episode. Even Zep's bi-temporal model requires the agent to detect the revision intent. No clean solution exists. Pointer to the paper-shaped opportunity.
- **9.2** **Continuous time perception.** Cite [[time-aware-memory]], [[2510.23853-temporally-blind]], and [[20260511-thinkingmachines-interaction-models]]. The next-episode preview. *"Most current systems treat time as a sort key. As we'll see in Ep 2, that's nowhere near enough."*
- **9.3** **Multi-agent shared memory.** Almost nobody has built this. MIRIX is the closest production-shape, the multi-agent-delegation concept page is the conceptual frame.
- **9.4** Reading list (8 items, paced for an end-card). Three benchmarks, three architectures, two concepts. All link to wiki pages.

**Papers:** [[belief-revision]], [[time-aware-memory]], [[multi-agent-delegation]], [[2510.23853-temporally-blind]], [[20260511-thinkingmachines-interaction-models]].

**Novel angle:** Don't survey open problems generically. Each one is connected back to a specific demo we ran in the episode that exposes the gap. Belief revision motivated by our Ch 3 + Ch 6 example. Time perception motivated by the temporal sub-score on LoCoMo. Multi-agent motivated by the MIRIX architecture we just located in axis-coordinates.

---

## Demos and experiments — master list

Eight experiments, eight figures. All scripted, all from the `mempol` and `personal-intelligence-system/research` repos. Each can be pre-recorded if live shooting is too brittle.

| # | Demo | Pre-record? | Approx. wall-clock | Repo location |
|---|---|---|---|---|
| 0 | ChatGPT forgets across sessions | live | 90s | screen recording |
| 2 | LoCoMo conv-26 cumulative cost curve | replay | 30s | new util `scripts/footnote_for_video/cost_curve.py` (write) |
| 2-eval | Full-context oracle on conv-26 → 71% | replay | 60s | `mempol.run_full_context_baseline` |
| 3-A | Multi-hop RAG failure trace | replay | 60s | `mempol.demo.rag_failure` (write) |
| 3-B | Belief-revision in summarized state | replay | 45s | `mempol.demo.summary_staleness` (write) |
| 5 | One fact, six storage forms | static | n/a | Excalidraw + code snippets |
| 6 | 30-line consolidator on conv-26 | replay | 90s | `mempol.demo.consolidator_minimal` (write) |
| 7 | **Live GEPA optimization, +20pp** | replay | 2 min | already exists: `scripts/run_gepa_consolidator.py` + `scripts/gepa_live.py` dashboard |
| 8 | SOTA pareto plot, axis-coordinates | replay | 30s | `paper_leaderboard/render_pareto.py` (write) |

**Status:** demo 7 is already runnable (the smoke landed +20pp on conv-26). Demos 2-eval, 3-A exist as scripts (need light wrapping for video capture). Demos 2, 3-B, 6, 8 are new — 1–2 hours of code each. Demo 5 is design work in Excalidraw.

---

## Cited material from the wiki — episode 1

Direct references, in citation order. Every one links into `research/wiki/` for the viewer that wants the receipts.

**Papers (16):**
[[2402.17753-locomo]] · [[2410.10813-longmemeval]] · [[2309.02427-coala]] · [[2504.19413-mem0]] · [[2501.13956-zep]] · [[2601.02845-timem]] · [[2604.20943-scm-sleep]] · [[2605.20616-auto-dreamer]] · [[2510.16392-rgmem]] · [[2601.06282-amory]] · [[2512.12818-hindsight]] · [[2601.02163-evermemos]] · [[2507.07957-mirix]] · [[2507.19457-gepa]] · [[2508.19828-memory-r1]] · [[2510.23853-temporally-blind]]

**Concepts (7):**
[[write-time-vs-read-time]] · [[noreplay-vs-retrieval]] · [[memory-budget-curves]] · [[substrate-design-space]] · [[sleep-consolidation]] · [[gepa-vs-grpo]] · [[time-aware-memory]] · [[belief-revision]]

**Systems (5):**
[[mem0]] · [[zep]] · [[mastra-om]] · [[gitmem]] · [[evermemos]] · [[auto-dreamer]]

---

## Tone reference (what we are emulating; what we are not)

We are emulating:
- **Sasha Rush, "The Annotated Transformer"** — code interleaved with paper quotes, every concept defended against the paper.
- **Lilian Weng's blog notes** — dense, opinionated, well-cited, never hot-take.
- **Distill.pub posts** — interactive demos that drive the writing.
- **Karpathy's zero-to-hero series** — build-from-zero pacing, cite SOTA in one matter-of-fact sentence.

We are not emulating:
- "AI explained" YouTube channels (too breathless, too survey-y).
- "10 things wrong with X" Twitter-style essays (too polemical).
- Academic survey papers (too taxonomy-first, demo-deficient).

In voice terms: think *senior PhD student walking a junior PhD student through their lit review*. Knows the field cold, has opinions, but is teaching, not selling.

---

## What still needs to be decided before filling out

Three open calls — answer these and I'll write the body:

- **Q1 — Length target.** 55, 70, or 90 min final cut? Each shapes the per-chapter compression.
- **Q2 — Live or replayed demos?** Live is more engaging, replay is safer. I'd recommend live for Demos 0 and 7, replay for the rest.
- **Q3 — Reading list at end.** 8 items as proposed, or a longer 15-item version? Longer reading list is more researcher-y but eats screen time.

Tell me on those three and I start writing the body section by section, starting with Ch 0–2 (which I'll send for tone-check before continuing).
