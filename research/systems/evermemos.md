---
title: "EverMemOS"
year: 2026
category: "memory-system"
tags: ["SOTA", "LoCoMo", "memory-OS", "EverMind", "self-organizing"]
---

# EverMemOS

**Current LoCoMo SOTA at 93.05%** per paper_leaderboard (Jan 2026). Self-organizing memory operating system from EverMind.

## Paper

[[2601.02163-evermemos|EverMemOS: A Self-Organizing Memory Operating System for Structured Long-Horizon Reasoning]]

## What we know

- Agentic framework, not a fine-tuned model
- Self-organizing (graph topology adjusts over time)
- Semantic consolidation as named primitive
- Memory Operating System family (kin to MemOS, Letta, MemGPT)
- Closed implementation by EverMind

## Numbers

- LoCoMo: **93.05%** (#1 on the leaderboard as of May 2026)
- Same paper benchmarks Zep at 85.22% and MemOS at 80.76%

## What it means for our work

This is the LoCoMo number to engage with. Any contribution we ship has to either close the gap to 93%, beat it, or report on a different axis (cost, NoReplay-budget, generalization, etc.).

The closed nature limits what we can learn from it. The "self-organizing" framing is suggestive but unexamined without code.

## See also

- [[2601.02163-evermemos|EverMemOS paper]]
- [[sleep-consolidation]] — likely related to their semantic consolidation
- [[memory-budget-curves]] — 93% is the unbounded-budget number; budget-curve unknown
