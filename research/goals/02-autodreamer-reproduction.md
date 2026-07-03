---
title: "Goal 02 — Reproduce Auto-Dreamer with GEPA on ScienceWorld"
status: "planned"
priority: 2
started: null
owner: "us"
budget: "$300-500 (Tinker compute or equivalent)"
tags: ["GEPA", "Auto-Dreamer", "ScienceWorld", "reproduction", "planned"]
---

# Goal 02 — Reproduce Auto-Dreamer with GEPA on ScienceWorld

## What we're trying to prove

[[2605.20616-auto-dreamer|Auto-Dreamer]] trained a consolidator with GRPO and achieved +7pp on ScienceWorld at 12× smaller memory. Their compute: 200 GRPO steps × group 8 × batch 16 = ~25,600 rollouts on 8× H100.

GEPA paper claims 35× fewer rollouts for comparable/better quality on similar problems. Hypothesis: GEPA on the consolidator prompt matches or beats Auto-Dreamer's numbers at ~$300 instead of ~$2-4k. If it works, paper reproduction with substantially cheaper training is a strong artifact — paper-reproduction is the AI-for-science equivalent of unit testing.

## Status: planned (after Goal 01 lands)

Blocked on: Goal 01 working. If GEPA-on-consolidator doesn't beat hand-coded on LoCoMo, applying it to ScienceWorld is unlikely to work either.

## What we'd need to build

- ScienceWorld environment hooked into the DSPy program (their public release exists)
- Their task agent baseline (Qwen3-14B; we'd use closed model + DSPy for parity)
- Their counterfactual reward term: `r_cf = U(S) - E[U(masked S)]`
- Bounded-tool-use surface in the consolidator (search_memory, check_memory, get_source_trace, synthesize, terminate)
- LoCoMo + ALFWorld + WebArena held-out evals (Auto-Dreamer's setup)

## Success criterion

Match Auto-Dreamer's ScienceWorld 41.1% SR ± 3pp at substantially lower training cost. Held-out ALFWorld/WebArena numbers within their reported margins.

## What it means if it works

We publish: "GEPA-Dreamer: Auto-Dreamer-quality consolidator at 35× less training compute." Implication: the consolidator can be trained on a laptop budget, opening this technique to anyone without GPU access. This is the AI-for-science movement's reproduction-as-contribution angle and probably the cleanest paper-shaped story available to us.

## What it means if it doesn't

GEPA caps out below GRPO on this task. Equally informative — tells us operations-level decisions over long-horizon tool-use trajectories genuinely need weight-level fine-tuning. Still publishable as a methods paper.

## Related

- Goal: [[goal-01-gepa-consolidator-on-locomo]] (blocking dependency)
- Concept: [[gepa-vs-grpo]]
- System: [[auto-dreamer]]
- Paper: [[2605.20616-auto-dreamer]], [[2507.19457-gepa]]
