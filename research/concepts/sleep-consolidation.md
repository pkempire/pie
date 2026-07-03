---
title: "Sleep consolidation"
year: 2026
category: "architecture-pattern"
tags: ["consolidation", "CLS", "offline", "biological", "hippocampus", "cortex"]
---

# Sleep consolidation

The architectural split between **fast acquisition** (writes during a conversation/session) and **slow consolidation** (asynchronous compression of accumulated writes into stable long-term memory). Inspired by complementary learning systems theory (CLS) from neuroscience: the hippocampus encodes individual episodes; the neocortex gradually extracts shared structure across them during off-task periods, especially during sleep.

## Why this matters

Most LLM memory systems do *synchronous* writes. A turn arrives, an LLM decides what to store, an entry lands in the store. This couples acquisition and consolidation in one step.

Two problems with this:
1. **No hindsight.** Deciding what to keep from turn 12 is much easier after seeing turn 37. Synchronous writes can't use information that hasn't arrived yet.
2. **Wasted online compute.** Each write decision is an LLM call. Most turns don't contain durable knowledge. Synchronous systems pay for the LLM call anyway.

Sleep-consolidation fixes both. Writes during the session are cheap (raw log append). The expensive LLM reasoning happens later, in a batched offline pass over many turns at once.

## Who's done this

- [[2605.20616-auto-dreamer|Auto-Dreamer]] (May 2026) — the SOTA implementation. GRPO-trained consolidator with end-to-end task reward. Beats baselines by 7pp on ScienceWorld with 12× smaller memory bank.
- [[2604.20943-scm-sleep|SCM (Sleep-Consolidated Memory)]] (April 2026) — research preview with NREM + REM phases. All hand-coded, no learning.
- [[2601.02845-timem|TiMem]] (Jan 2026) — temporal-hierarchical consolidation, 5 levels (L1 fragment → L5 high-level). Prompt-only. 75.30% on LoCoMo.
- Anthropic's "Claude dreaming" feature (Sept 2025) — productisation of the same idea for Claude memory. Closed implementation.
- Mastra Observational Memory — Reflector pass is a form of consolidation but runs once per chunk, not as an explicit sleep phase.

## What the design choices are

When building a sleep-consolidator, you pick on four axes:

**1. Trigger.** What kicks off consolidation?
- Time-based (cron, every N hours) — TiMem
- Volume-based (when raw log exceeds K entries)
- Event-based (end of session, user logout)
- Query-driven (defer until next retrieval)

**2. Working region.** What does the consolidator see?
- Whole raw log since last consolidation — naive, scales badly
- Recent N turns + entries the agent retrieved during those turns — Auto-Dreamer
- A single hierarchical level → next level — TiMem
- Adaptive (depends on detected topic shifts)

**3. Output schema.** What does the consolidator emit?
- Typed entries (semantic facts + procedural steps) — Auto-Dreamer
- Hierarchical summaries (paragraph per level) — TiMem
- Flat facts — Mem0-style
- Graph updates — KG-shaped systems

**4. How is it learned (if at all)?**
- Not learned, prompt-only — TiMem, SCM, Mastra
- GRPO with end-to-end task reward — Auto-Dreamer
- [[gepa-vs-grpo|GEPA]] with reflective prompt evolution — *open opportunity, nobody has shipped this*
- Supervised on human-curated consolidations — possible but expensive

## The open opening

Combining GEPA + sleep-consolidation isn't published anywhere as of May 2026. Auto-Dreamer used GRPO (~25,600 rollouts on 8× H100). GEPA's claim is comparable or better outcomes at 35× fewer rollouts. If we GEPA-evolve a consolidator prompt against Auto-Dreamer's exact ScienceWorld setup, we test directly whether the natural-language-gradient training is sample-efficient enough to match RL on this problem.

See also: [[2507.19457-gepa|GEPA paper]] for the underlying optimizer, [[gepa-vs-grpo]] for the comparison.

## The biological inspiration, briefly

Hippocampus = fast learner, single-shot, episodic. Cortex = slow learner, statistical, semantic. Memories transfer hippocampus → cortex during slow-wave sleep, with the transfer biased toward emotionally-salient or behaviorally-relevant episodes. The same pattern shows up in LLM memory if you take it seriously: the raw log is the hippocampus, the consolidated bank is the cortex, the consolidator is the dialogue between them.

Most LLM-memory papers handwave this analogy. Auto-Dreamer is the first to operationalize it with a learned consolidator and end-to-end reward.
