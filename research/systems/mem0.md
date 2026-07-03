---
title: "Mem0"
year: 2024
category: "memory-system"
tags: ["production", "vector-store", "ADD-UPDATE-DELETE", "Mem0", "industry"]
---

# Mem0

The most-cited production memory system for LLM agents. Flat key-value vector store; per-turn LLM-as-judge prompt picks one of `ADD`, `UPDATE`, `DELETE`, `NONE`.

## Paper

[[2504.19413-mem0|Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory]]

## Architecture

- **Storage**: flat vector index over text-form facts.
- **Write controller**: per-turn LLM call with a hand-tuned prompt that classifies the turn into `ADD / UPDATE / DELETE / NONE` and returns the content to apply.
- **Read controller**: dense retrieval over the vector store, top-k passed to the answer model.

## Numbers we have

- LongMemEval: ~69% overall with gpt-4o (varies by config).
- LoCoMo: ~27% on a 50-question subset using their default settings (from our own re-run with gpt-4o judge).
- **Self-reported ~40% extraction-failure rate** in their own paper.

## Why it matters

Mem0 is the *reference industry implementation*. When papers say "we improve over hand-coded memory systems," Mem0 is what they mean. The 40% extraction-failure rate is the strongest single piece of public evidence that hand-coded write logic is the bottleneck.

## Where it sits in the design space

- [[write-time-vs-read-time|Write-time compression]] (synchronous LLM call per turn)
- Unbounded scratchpad (vector store grows)
- Not [[noreplay-vs-retrieval|NoReplay]]-compliant (vector search at QA time)
- Single-level structure (no [[sleep-consolidation|consolidation]])

## What systems beat it

- [[mastra-om]] — same shape but with an Observer+Reflector instead of the per-turn classifier
- [[2601.02845-timem|TiMem]] — hierarchical consolidation, 75.30% LoCoMo
- [[2605.20616-auto-dreamer|Auto-Dreamer]] — GRPO-trained consolidator, better agent-task generalization

## What we steal

- The 4-op vocabulary (ADD/UPDATE/DELETE/NONE) is small and clean
- The production packaging (their actual differentiator vs research systems)
- The 40% extraction-failure number — the hook for any "hand-coding is broken" pitch
