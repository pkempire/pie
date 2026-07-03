---
title: "Goal 01 — GEPA-optimised consolidator on LoCoMo"
status: "active"
priority: 1
started: "2026-05-26"
owner: "us"
budget: "$200 across initial experiments"
tags: ["GEPA", "DSPy", "consolidator", "LoCoMo", "active"]
---

# Goal 01 — GEPA-optimised consolidator on LoCoMo

## What we're trying to prove

The consolidator prompt that runs offline over a raw observation log is the right target for prompt-evolution optimization. Specifically: [[2507.19457-gepa|GEPA]] applied to a consolidator prompt should match or beat hand-coded consolidation on LoCoMo at a fraction of the compute cost of GRPO-based alternatives like [[2605.20616-auto-dreamer|Auto-Dreamer]].

If true, we can publish "GEPA on consolidator > hand-coded consolidator" with cheap reproducibility (no GPU access needed). If not true, we learn the actual ceiling of prompt-only optimization for this task.

## Current state (2026-05-26)

**Baseline loop works.** DSPy module is built, smoke-tested, producing real consolidated entries with correct speaker attribution.

| Run | Model | Chunks | Qs | Overall |
|---|---|---|---|---|
| Mini smoke | gpt-5-mini | 1 (30 turns) | 5 | **50%** |
| Scale | gpt-4o-mini | 3 (90 turns) | 10 | 20% |

Cost: ~$0.30 per full conv-26 baseline run with gpt-5-mini. ~$0.05 with gpt-4o-mini.

Files:
- `mempol/dspy_consolidator/consolidator.py` — the DSPy module
- `mempol/dspy_consolidator/run_baseline.py` — end-to-end runner
- `mempol/results/dspy_consolidator_baseline_conv-26.jsonl` — trace

## Next steps in order

1. **Full conv-26 baseline** with gpt-5-mini — establish the hand-coded number (~$0.30, 10 min).
2. **GEPA wrapper** — `scripts/run_gepa_consolidator.py` that wraps the DSPy module with the LoCoMo judge metric and runs GEPA optimization (~200 rollouts, $30-50, half a day).
3. **Compare**: GEPA-evolved consolidator vs hand-coded baseline on conv-26 (in-distribution) and convs 30/41 (held-out).
4. **Report numbers** — produce a clean Goal-01 result page with the curves.

## Success criterion

GEPA-evolved consolidator beats the hand-coded baseline by ≥5pp on conv-26 held-out and ≥3pp on cross-conv generalization. Anything less is interesting but not a result we'd lead with.

## Stretch (if Goal 01 succeeds)

Reproduce [[2605.20616-auto-dreamer|Auto-Dreamer]]'s ScienceWorld setup with GEPA instead of GRPO. Direct comparison: same architecture, 35× less compute (claim from GEPA paper). If it works, that's a strong paper-shaped result that anchors the publication.

## Related

- Concept: [[sleep-consolidation]] (architectural pattern)
- Concept: [[gepa-vs-grpo]] (training-method choice)
- Concept: [[memory-budget-curves]] (the reporting discipline)
- System: [[auto-dreamer]] (the SOTA we'd compete with on agent envs)
- Paper: [[2605.20616-auto-dreamer]], [[2507.19457-gepa]]
