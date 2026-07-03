# Paper schema

Every file in `papers/` has the form:

```markdown
---
arxiv_id: "2508.19828"
title: "Memory-R1: Enhancing LLM Agents to Manage and Utilize Memories via RL"
authors: ["Yan, Sikuan", "Yang, Xiufeng", "Huang, Zuchao"]
year: 2025
date_published: "2025-08-28"
date_ingested: "2026-05-12"

# What family of approach does it belong to?
# Pick from: write-time-compression, read-time-decompression, RL-for-memory,
# RL-for-tool-use, agent-orchestration, temporal-reasoning, benchmark, substrate,
# theory, infrastructure, survey
approach_class: "RL-for-memory"

# One-sentence problem statement
problem: "Memory managers in LLM agents are hand-coded prompts; can they be trained with RL on QA outcome?"

# One-sentence approach statement
approach: "Trains Memory Manager (ADD/UPDATE/DELETE/NOOP) and Answer Agent jointly with PPO/GRPO on 152 QA pairs."

# Benchmarks used (list)
benchmarks: ["LoCoMo", "LongMemEval", "MSC"]

# Headline numerical result(s)
results:
  - "GPT-5.2 on LoCoMo: case-level acc"

# Reward shape (for RL papers) — pick from: trajectory-level, per-op-state-distance,
# per-op-outcome-attribution, verbal-reflection, supervised, none
reward_shape: "trajectory-level"

# Models / base
base_model: "Qwen2.5-7B + LoRA"
adapter_type: "LoRA"

# What we'd compare ourselves to / steal from
relevance: "high"          # high | medium | low
relevance_reason: "Closest published comparison to mempol's per-op approach."

# Concrete things to potentially steal
steal:
  - "Outcome-based reward as primary signal"
  - "PPO + GRPO comparison"

# Limitations they admit
limitations:
  - "Trajectory reward diffuses across long episodes"
  - "Only 152 QA training pairs"

# Citation
bibtex: |
  @article{yan2025memoryr1, ... }

# Tags (free-form, helpful for grep)
tags: ["RL", "memory-ops", "Qwen", "LoRA"]
---

# Memory-R1: Enhancing LLM Agents to Manage and Utilize Memories via RL

## Quick read

Two-sentence summary of what they actually do, written for someone who has
read 5 other papers in the area.

## Why it matters to us

Concrete connection to our project. What this paper does that we'd compete
with or build on. What number on what benchmark we'd have to beat.

## Method in one paragraph

Specifics. The op vocabulary, the env shape, how they get reward signal.

## Results in numbers

Verbatim numbers from the abstract / Table 1. Don't paraphrase.

## What they don't do

Honest read of the limitations.

## Open questions / followups

What we'd want to know that the paper doesn't answer.
```

## Required vs optional fields

- **Required**: `arxiv_id`, `title`, `year`, `approach_class`, `problem`, `approach`, `relevance`
- **Optional but strongly preferred**: `benchmarks`, `results`, `reward_shape` (for RL papers), `limitations`, `steal`
- **Auto-filled by ingest**: `date_ingested`, `bibtex`

A file is considered *complete* if all required + benchmarks + results are filled.
Files with missing fields appear in STATUS.md → "needs review" section.

## Approach class taxonomy

The taxonomy is fixed (so grouping queries work). Don't invent new classes
without updating the schema doc and re-running aggregate.

- `write-time-compression` — does work at ingestion to compress (Mem0, Mastra-OM, KGmem)
- `read-time-decompression` — does work at query time on raw store (RLM, Search-R1, naive RAG)
- `RL-for-memory` — RL on memory write/edit ops (Memory-R1, Mem-α, DeltaMem, mempol)
- `RL-for-tool-use` — RL on tool calls more broadly (Search-R1, ReAct-RL)
- `agent-orchestration` — multi-agent / planning (MultiAgentBench, DeepPlanning, ROMA)
- `temporal-reasoning` — time-aware capabilities (Time-R1, Test-of-Time, TicToc)
- `benchmark` — eval suite, not a method (LoCoMo, LongMemEval, τ²-Bench)
- `substrate` — storage/data structure for memory (Zep bi-temporal, Letta FS, Mesa FS)
- `theory` — formal frameworks (AGM, JTMS, COMA, Free Energy)
- `infrastructure` — runtime / serving / RL framework (Tinker, vLLM, Ramp KV cache)
- `survey` — review papers
