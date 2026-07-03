---
title: "AI memory, from scratch: what memory means for LLM agents"
target_length: "45-60 min"
target_audience: "AI engineers building agentic systems; intermediate familiarity with LLMs"
status: "outline"
priority: 1
related_concepts: ["noreplay-vs-retrieval", "write-time-vs-read-time", "substrate-design-space", "sleep-consolidation"]
related_papers: ["2504.19413-mem0", "2410.10813-longmemeval", "2402.17753-locomo", "2605.20616-auto-dreamer", "2601.02163-evermemos", "2512.12818-hindsight"]
related_goals: ["03-public-research-wiki"]
---

# AI memory, from scratch: what memory means for LLM agents

## Thesis

"Long context" is not "memory." Vector search over a transcript is not "memory." Most published "memory benchmarks" don't measure memory. This video defines memory operationally, walks the design space, shows where current SOTA actually is, and gives the viewer a working mental model they can use to evaluate any new memory system they encounter.

## Outline

### 0:00 — Cold open (60s)
Open with a clip of ChatGPT or Claude failing to remember something across sessions. Then the receipts: Mem0's own paper reports a 40% extraction-failure rate (cite verbatim). EverMemOS hits 93.05% on LoCoMo — sounds great until you ask what LoCoMo actually measures.

The premise: every term in "long-term memory for LLM agents" is contested. Let's define them.

### 1:00 — What is memory, actually (5 min)
Quick definition contrast:
- Long-context reading: 1M-token Gemini → answer from one big prompt
- Retrieval-augmented generation: vector search on the transcript, stuff results in context
- Working memory: scratchpad inside the LLM's current call
- **External memory**: persistent state across sessions that selectively retains

The thing we mean when we say "memory" is the last one. Everything else is a different capability that sometimes substitutes.

- **Excalidraw diagram**: the four capabilities side by side, with arrows showing what each is good for
- **Reference**: Peter Yang's NoReplay framing (May 2026); link [[noreplay-vs-retrieval]]

### 6:00 — The NoReplay distinction (5 min)
Walk through Yang's discipline. Show why Mastra's 94.87% on LongMemEval is not measuring what people think it measures (their Reflector pre-computes structured state with full transcript access at write time).

- **Diagram**: Yang's NoReplay protocol — one-pass ingest, fixed scratchpad, freeze, answer
- **Table**: comparison of current systems vs NoReplay compliance (from our wiki page)

### 11:00 — The design space (8 min)
Show our 6-substrate diagram (Excalidraw, full screen). Walk each:
1. Flat vector store (Mem0)
2. Hierarchical summary tree (TiMem)
3. Typed KG (PIE, Zep)
4. Observation log + consolidator (Auto-Dreamer, Mastra)
5. Commit graph (GitMem, Mesa FS)
6. Filesystem (Letta FS)

For each: who's there, what they do well, what they fail at.

- **Asset**: substrate diagram (already rendered, in wiki)
- **Show**: real screenshots from each system's docs or paper Table 1

### 19:00 — Write-time vs read-time compression (4 min)
The big axis the field is splitting along. Mem0/Mastra/PIE pay LLM cost at write time. RLM, Search-R1 pay at read time. Sleep-consolidation is the synthesis (write cheap, consolidate offline, read cheap).

- **Animation idea**: clock + LLM-cost bar, showing where compute happens for each family
- **Cite**: [[2605.20616-auto-dreamer|Auto-Dreamer]] as the synthesis exemplar

### 23:00 — The current SOTA, honestly (8 min)
Pull the leaderboard data. EverMemOS 93.05%, Hindsight 89.61%, Amory 87.7% on LoCoMo. Mastra 94.87% on LongMemEval. OMEGA 95.4% (closed). But:

- Show the verification problem: most numbers are table-only, not abstract-verified
- Show Hindsight's awkward result: their Backboard baseline (90.0%) beats their full system (89.61%) — what does the architecture buy?
- Implication: at frontier-model scale, memory architecture sophistication shows ~1-3pp. The interesting question is no longer "what's the best architecture" but "what's the best architecture at low budget."

- **Asset**: leaderboard screenshot (your paper_leaderboard rendered)
- **Cite**: [[memory-budget-curves]] concept page

### 31:00 — The bifurcation (5 min)
Three real research directions emerging:
1. Sleep-consolidation with learned consolidator (Auto-Dreamer + ours)
2. Test-time recursion / read-time decompression (RLM, Search-R1)
3. Time-aware primitives (TML Interaction Models, Zep bi-temporal)

Each solves a different capability gap.

- **Diagram**: 3-way fork showing each direction's failure mode it addresses

### 36:00 — What you should build today (5 min)
Decision tree for the viewer:
- "High-QPS reads, single user, conversational" → flat vector + Anthropic contextual retrieval
- "Long-horizon project tracking" → filesystem with structured markdown + git
- "Multi-session personal AI" → observation log + consolidator (Mastra-OM-style)
- "Agent that takes actions in environments" → Auto-Dreamer-style typed memory bank
- "Research workflow" → research wiki like ours (gestures to the wiki)

- **Show**: actual code template for each decision

### 41:00 — Where research is going (3 min)
Three open problems:
1. Belief revision (no current system does it properly — see [[belief-revision]])
2. Continuous time perception (TML solved seconds-to-minutes; weeks-to-months unsolved)
3. Multi-agent shared memory state (almost nobody has built this)

These are the next-paper directions.

### 44:00 — End screen (1 min)
- Subscribe pointer (no beg, just "if this was useful")
- Link to wiki for deeper material
- Repo link for code

## Required production assets

| # | Asset | Format | Source |
|---|---|---|---|
| 1 | ChatGPT-forgets-cold-open clip | 15s screen rec | record a real session |
| 2 | 4-capabilities diagram | SVG | Excalidraw |
| 3 | NoReplay protocol diagram | SVG | Excalidraw |
| 4 | 6-substrate design space | SVG | already rendered in wiki |
| 5 | Write-time vs read-time clock animation | 30s video | Motion Canvas |
| 6 | Leaderboard screenshot | image | from your paper_leaderboard webapp |
| 7 | Backboard-vs-Hindsight diff illustration | SVG | Excalidraw |
| 8 | 3-direction fork diagram | SVG | Excalidraw |
| 9 | Decision tree for "what should you build" | SVG | Excalidraw |
| 10 | Title + end cards | image | Excalidraw |

## What's high-alpha here

- Most viewers will not know about NoReplay distinction (it's literally 3 weeks old)
- Most viewers will not have seen the leaderboard verification reality (most claim "SOTA" without table-grounding)
- Most viewers will not have a mental model that includes all 6 substrate families
- The "Backboard beats Hindsight" footnote is genuinely instructive and underdiscussed

If they only get one thing: the realization that "memory" benchmark scores are not directly comparable across systems with different write-time access patterns. That alone is worth 45 minutes.

## Production order

1. Lock script (3 days)
2. Make all 10 assets (4 days)
3. Record full voiceover in 3-4 sittings (1 day)
4. Edit with B-roll over the voice (3 days)
5. Review pass, tighten (1 day)
6. Thumbnail/title/description (1 day)
7. Schedule (immediate)

Total: ~2 weeks for a single 45-min video.
