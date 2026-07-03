---
title: "Memory budget curves"
year: 2026
category: "evaluation-axis"
tags: ["evaluation", "budget", "compression", "pareto", "reporting"]
---

# Memory budget curves

A reporting discipline. Don't report one memory-system number; report **accuracy as a function of memory budget**. The interesting question is "how much state do you need to answer well," not "can you answer."

## Why this matters

[[2512.12818-hindsight|Hindsight]] hits 89.61% on LoCoMo. [[2601.06282-amory|Amory]] hits 87.7%. [[2601.02163-evermemos|EverMemOS]] hits 93.05%. All three are different memory architectures using different base models with different compute budgets. The single accuracy number compresses away the question that actually decides product value: *given a fixed token/dollar budget per query, which system wins?*

Two systems can score identically on LoCoMo while one uses 500 stored tokens and the other 50,000. They are not the same product.

## What the curve looks like

X-axis: memory budget (tokens, or stored entries, or dollars per query).
Y-axis: judged accuracy on the held-out battery.

Each system traces out a curve as you vary its retention cap. The interesting comparison is the Pareto frontier: at any given budget, which system is on the upper-left edge.

## What to report

For any new memory system, report:
- Score @ budget = 1k tokens
- Score @ 4k tokens
- Score @ 10k tokens
- Score @ unlimited (≈ full-context oracle)

A system that hits 75% at 4k tokens is in a different league from a system that hits 80% at 50k tokens. The 5pp deficit at 12× lower cost is a Pareto win, but a single-number table hides it.

## Who reports this honestly

[[2605.20616-auto-dreamer|Auto-Dreamer]] is the cleanest example — Table 1 reports both `SR` and `active memory bank` (in tokens) side by side. Their +7pp ScienceWorld claim *and* their 12× smaller memory are both first-class.

[[2601.02845-timem|TiMem]] reports a single accuracy plus "52.20% reduction in recalled memory length" — directionally right but informal.

Most leaderboard-grade systems (EverMemOS, MemR3, Hindsight, Amory) report a single overall number and tag it with the methodology. The Pareto axis is not standardized.

## The honest implication for our work

Pushing past 90% LoCoMo with a complex architecture is one thing. Hitting 85% at 1/10th the per-query cost is a different and possibly more valuable result. The [[sleep-consolidation|sleep-consolidator]] direction we're pursuing should be evaluated at multiple budgets, not just at unbounded.

## See also

- [[noreplay-vs-retrieval]] — the orthogonal evaluation discipline that makes budget curves meaningful
- [[write-time-vs-read-time]] — write-time systems pay budget at ingestion; read-time at query
- [[2605.20616-auto-dreamer|Auto-Dreamer]] — the reference for reporting budget alongside accuracy
