---
title: "Sleep-phase consolidation: building an offline memory consolidator (live)"
target_length: "25-35 min"
target_audience: "AI engineers / researchers familiar with LLMs and basic RL concepts"
status: "outline"
priority: 1
related_concepts: ["sleep-consolidation", "gepa-vs-grpo", "write-time-vs-read-time"]
related_papers: ["2605.20616-auto-dreamer", "2507.19457-gepa", "2601.02163-evermemos"]
related_goals: ["01-gepa-consolidator-on-locomo"]
---

# Sleep-phase consolidation: building an offline memory consolidator

## Thesis

Every shipped LLM memory system synchronously decides what to remember at write time. That's the wrong place to make the decision — the right place is asynchronously, with hindsight, after multiple turns have arrived. We build the architecture from scratch in 30 minutes, train it with GEPA in another 10, and watch the consolidator prompt evolve in real time.

## Why this video matters

Three things converge here that make a really compelling explainer:
1. The neuroscience analog (CLS theory — hippocampus + cortex) is intuitive and visual
2. The Auto-Dreamer paper is fresh (May 2026) and well-documented
3. We have a live, working implementation that produces real numbers (60% → 80% in our smoke)

The viewer gets: the architectural pattern, the math, the code, and the actual result on a real benchmark. Nobody else has covered this.

## Outline (with timestamps)

### 0:00 — Cold open (45 sec)
**Hook**: Show terminal scrolling Auto-Dreamer-style consolidator output. Cut to the GEPA score graph going up. Voiceover: "I built a memory system that learns *what to remember* while you sleep. Score went from 60% to 80% with no fine-tuning. Here's how it works."

- **Asset**: 5-second terminal recording of our `python scripts/run_gepa_consolidator.py --smoke` output
- **Asset**: Simple line graph (Excalidraw) showing 0.6 → 0.8 over GEPA iterations

### 1:00 — The problem (3 min)
LLM agents have the same problem you do: you can't decide what's worth remembering while it's happening. You only know later, after you see how things played out.

- **Diagram**: Excalidraw timeline. Top: conversation arriving. Bottom: each turn forces an LLM call: "store / update / delete / ignore". Show the bandwidth issue.
- **Reference**: Mem0's self-reported 40% extraction-failure rate (need exact citation — verify from their paper before recording)
- **Reference**: Show Memory-R1 + DeltaMem screenshots from their abstracts — every published system synchronizes write decisions
- **Cite**: [[2605.20616-auto-dreamer|Auto-Dreamer]] paper, [[2504.19413-mem0|Mem0]]

### 4:00 — The neuroscience analog (4 min)
Complementary learning systems theory. Hippocampus = fast, single-shot, episodic. Cortex = slow, statistical, semantic. Sleep transfers memories between them, biased toward what's behaviorally relevant.

- **Animation idea (Manim or Motion Canvas)**: Two regions of a brain. Hippocampus accumulating colored dots during the day. At "night," dots flow to cortex with some merging, some discarded.
- **Reference**: McClelland, McNaughton, O'Reilly 1995 — the foundational CLS paper
- **Reference**: ML translation — [[2605.20616-auto-dreamer|Auto-Dreamer]] explicitly cites CLS as design inspiration

### 8:00 — The architecture (5 min)
Two layers + an async bridge:
- L1: raw observation log (append, no LLM)
- Consolidator: bounded tool-use over a "working region"
- L3: consolidated bank (read at QA time)

Walk through the data flow with Excalidraw. Show working_region selection: recent writes ∪ entries retrieved during last K sessions.

- **Diagram**: Excalidraw — three boxes (L1 raw log, consolidator process with tool icons, L3 bank). Arrows showing flow.
- **Reference**: Auto-Dreamer Figure 1 (rerender ours, cite theirs)
- **Code asset**: Show the `ConsolidatorSignature` from `mempol/dspy_consolidator/consolidator.py` — the DSPy module's input/output types

### 13:00 — Building it live in code (8 min)
Live screen recording in Cursor. Walk through:
- The DSPy signature (input: working_region, source_traces → output: consolidated_entries)
- The metric function (run consolidator → ingest into FlatBackend → answer LoCoMo questions → judge)
- The runner script

Run it on conv-26 chunk 1, narrate the output as the consolidated entries print to terminal.

- **Asset**: Cursor screen recording with ScreenStudio
- **Code**: live in repo at `mempol/dspy_consolidator/`
- **Show**: actual consolidator output for Caroline/Melanie chunk — Caroline's necklace gets correctly attributed

### 21:00 — Training it with GEPA (8 min)
Setup the GEPA loop. Explain: scalar reward = 1 bit; natural-language reflection = hundreds of bits. Show the actual reflection LM prompt template (DSPy GEPA internal).

Run `--smoke`. Watch the score progression live.

- **Diagram (Excalidraw)**: GEPA loop — sample trajectories → reflect on failures → propose new prompts → Pareto-frontier check
- **Show**: original prompt (7 lines) vs evolved prompt (11 sections) — actual file diff
- **Result**: 60% → 80% on our smoke

### 29:00 — What just happened, in 3 minutes (3 min)
The single most important thing GEPA added was the speaker-attribution rule: "the speaker field is the person whose life/knowledge the entry describes, NOT the turn's speaker." That single insight was what GEPA's reflection caught and rules-ified.

Show the diff. Speak it out loud.

- **Asset**: side-by-side text comparison of original vs evolved prompt, highlight the speaker-attribution paragraph

### 32:00 — What's next (2 min)
- Goal 02: reproduce Auto-Dreamer's full ScienceWorld result with GEPA instead of GRPO (35× less compute claim)
- Goal 03: this wiki + content (link to channel page)
- Cite: HORIZON benchmark as next target

End screen: pin to [[01-gepa-consolidator-on-locomo|Goal 01 page]] in wiki + GitHub repo.

## Required production assets

| # | Asset | Format | Source |
|---|---|---|---|
| 1 | Terminal recording of smoke run | mp4, 60s | ScreenStudio on Mac during run |
| 2 | Line graph: GEPA score over iterations | SVG (Excalidraw) | hand-drawn |
| 3 | Animation: hippocampus → cortex consolidation | 30s video | Motion Canvas or Manim |
| 4 | Architecture diagram: 3 boxes + tool icons | SVG (Excalidraw) | hand-drawn |
| 5 | Live code walkthrough (DSPy module) | 6 min recording | ScreenStudio in Cursor |
| 6 | Diff: original vs evolved prompt | side-by-side text | screenshot of git diff or VS Code |
| 7 | Title card + end card | static images | Excalidraw |

## Reference material to cite

- **Papers**: Auto-Dreamer (arXiv 2605.20616), GEPA (2507.19457), McClelland-McNaughton-O'Reilly 1995 CLS
- **Systems for comparison**: Mem0, Mastra OM, EverMemOS, TiMem
- **Wiki pages** (link in description): all related concepts + goals + papers

## Tweets / external content to reference (if relevant)

- DSPy team's GEPA release thread (Khattab — confirm exact URL before publishing)
- Karpathy's tweets about "agent memory is the hardest problem" (find via search)
- AI2 / Asta team's autoresearch work (cite their Nature Co-Scientist piece if relevant)

## What makes this video Karpathy-level vs cringe

Karpathy-level:
- Show actual code, no slides
- Run the experiment live; don't fake it
- Explain the *why* before the *what* (CLS first, then architecture)
- Honest about limits ("our smoke is 1 chunk, the real number needs the full run")
- Cite real numbers from real papers verbatim
- No "subscribe and like" tax; respect viewer time

Cringe to avoid:
- Clickbait title ("Why your AI has goldfish memory" — bad)
- Generic stock footage of brains, neurons, etc.
- AI-generated voiceover (use real voice; viewers can tell)
- Slide-deck with bullet points (use hand-drawn diagrams or live code)
- Recap-section padding ("So what did we learn?")

## Engagement plan (signal-to-noise)

Every minute should teach a specific thing or build toward one. Cut anything that doesn't.

- Cold open: gives the punchline (the result) so viewer knows what they're getting
- Each section: one main idea, one diagram, one cite, then move on
- Code section: keep it short — show structure, not every line
- End: actionable links (wiki, repo, paper) for viewer to go deeper

## Production timeline (if shipping in 2 weeks)

- Day 1-2: refine outline, lock script
- Day 3-4: create all diagrams in Excalidraw
- Day 5: record code walkthrough segments in Cursor
- Day 6: record voiceover separately for clarity (multiple takes per section)
- Day 7: animation segment (CLS hippocampus → cortex) in Motion Canvas
- Day 8-10: edit in DaVinci Resolve
- Day 11: review pass, tighten edits
- Day 12: thumbnail, title, description
- Day 13: upload, schedule
- Day 14: publish
