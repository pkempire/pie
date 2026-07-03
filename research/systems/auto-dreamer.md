---
title: "Auto-Dreamer"
year: 2026
category: "memory-system"
tags: ["consolidator", "GRPO", "CLS", "ScienceWorld", "ALFWorld", "WebArena"]
---

# Auto-Dreamer

GRPO-trained offline consolidator for language-agent memory. Inspired by complementary learning systems (CLS) theory — fast hippocampal acquisition + slow neocortical consolidation. May 2026, by Ye et al. (UIUC + Stanford).

## Paper

[[2605.20616-auto-dreamer|Auto-Dreamer: Learning Offline Memory Consolidation for Language Agents]]

## Architecture

- **Fast layer**: online writer emits typed entries into a memory bank during a session.
  - `INSERT_SEMANTIC{name, summary, details}` for factual knowledge
  - `INSERT_PROCEDURAL{name, type, summary, steps}` for how-to skills
- **Slow layer**: GRPO-trained consolidator runs offline on a "working region" of the bank.
  - Working region = recent writes ∪ entries the task agent retrieved during last k sessions
  - Bounded tool-use: `search_memory(query, k=5)`, `check_memory(ids)`, `get_source_trace(id)`, `synthesize(source_ids, type, name, summary, details|steps)`, `terminate()`
  - Up to 40 turns per consolidation pass
- **Replacement**: synthesized set S replaces the working region in the bank.

## Training

- Base: Qwen3-14B (consolidator), Qwen3.5-9B (task agent for reward)
- GRPO with 200 steps × group 8 × batch 16 = ~25,600 rollouts
- 8× H100 GPUs
- Reward: r = U_V(S) + α·r_cf(S; V)
  - U_V = mean task success on validation set V using only S
  - r_cf = counterfactual = U(S) − E[U(masked S)] for random masking
- LR 1e-6, KL coef 1e-3, no LoRA reported (full fine-tune implied)

## Numbers

- **ScienceWorld** (training): 41.1% SR / 6.9k tokens vs UMEM 34.1% / 80.9k tokens (+7pp, ~12× less memory)
- **ALFWorld** (held-out, no retraining): 60.2% / 11k tokens vs UMEM 58.4% / 63k (~6× less)
- **WebArena** (held-out): 52.3% / 927 tokens, tied lead

## What we steal

- Bounded tool-use during consolidation (the consolidator inspects its own bank)
- Typed entries: semantic (factual) vs procedural (how-to)
- Provenance-linked source trajectories
- Counterfactual reward term prevents bank bloat
- CLS framing operationalized

## Limitations they admit

- Text-only agent envs; no multimodal
- Writer/schema dependence (information missed by writer is unrecoverable)
- Retrieval-budget-sensitive (top-K=3 favors compact banks)
- Surrogate training objective (local-bank training)
- No seed/variance reporting

## What we'd do differently

Replace GRPO with [[2507.19457-gepa|GEPA]] — same architecture, prompt-only optimization. GEPA's claim: 35× fewer rollouts at comparable quality. If true, Auto-Dreamer-style consolidation becomes a thing you do on a laptop for $50 instead of 8× H100 for ~$2-4k.

That's the experiment we're going to run.

## Code

Not released as of 2026-05-26. Watch first-author Chongrui Ye's GitHub.

## See also

- [[sleep-consolidation]] — architectural family
- [[gepa-vs-grpo]] — the ablation we'd run on top of their architecture
- [[2605.20616-auto-dreamer|the paper]]
