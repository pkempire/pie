---
title: "GEPA vs GRPO"
year: 2026
category: "training-method"
tags: ["RL", "prompt-optimization", "GRPO", "GEPA", "DSPy"]
---

# GEPA vs GRPO

Two ways to optimize an LLM-based system against a verifiable reward. They differ in the *gradient signal*: GRPO uses scalar policy gradients computed from group-relative advantages; GEPA uses natural-language reflection on full trajectories.

## GRPO in one paragraph

Group Relative Policy Optimization, from [[2402.03300-deepseekmath-grpo|DeepSeekMath (Shao et al. 2024)]]. For each prompt, sample G rollouts from the current policy. Compute reward of each rollout. Advantage of rollout i = (r_i − mean_r) / std_r. Standard policy-gradient update with this advantage. No value/critic network. Works when reward is verifiable (exact match, judge, task success).

GRPO is what powers Search-R1, DeepSeek-R1, Memory-R1, [[2605.20616-auto-dreamer|Auto-Dreamer]], Mem-α. Default RL algorithm for tool-use training.

## GEPA in one paragraph

Genetic-Pareto reflective prompt evolution, from [[2507.19457-gepa|GEPA (Agrawal et al. 2025)]]. ICLR 2026 Oral. Given a system built from one or more LLM prompts: sample trajectories, reflect on failures in natural language ("the model conflated X with Y"), propose specific prompt edits, test them on a small batch, maintain a Pareto frontier of complementary prompt variants, combine winning lessons across the frontier.

The shift: a scalar reward is one bit per rollout; a natural-language critique is hundreds of bits per rollout. For the same compute budget, reflection-based search extracts much more signal per sample.

## The numbers

GEPA's paper reports vs GRPO across 6 tasks:
- +6% on average, +20% on best
- 35× fewer rollouts at matched/better quality
- vs MIPROv2 (prior prompt optimizer SOTA): +10% on average, +12% on AIME-2025

## When each wins

**GRPO wins when:**
- The capability needed isn't in the base model (must shift weights)
- Reward is dense and cheap to compute (e.g. token-level)
- Training compute is cheap; you can afford thousands of rollouts
- You want a portable artifact (the LoRA weights)

**GEPA wins when:**
- The capability IS in the base model, just badly invoked
- Reward is expensive (LLM-judge, end-to-end task)
- Training compute is the bottleneck
- You're using a closed model (can't fine-tune)
- The system is built from explicit prompts (DSPy-shaped)

## What GEPA can't do

- Change model knowledge or fundamental capability — only optimize how it's invoked
- Train models for non-verifiable open-ended generation
- Run during deployment (it's offline, like any optimizer)

## What this means for memory systems

The open opportunity: every paper in the [[sleep-consolidation]] cluster that uses GRPO (Memory-R1, Mem-α, [[2605.20616-auto-dreamer|Auto-Dreamer]]) is potentially replaceable by GEPA. The consolidator prompt is exactly the kind of stationary, low-dimensional target GEPA is built for. Same architecture, same reward signal, much cheaper training.

If GEPA can match Auto-Dreamer's numbers at 1/35th the compute, the practical implication is enormous: consolidation training becomes a thing you do on a laptop for $50, not on 8× H100 for $2k.

If GEPA can't match — equally informative. It tells us that operations-level decisions in long-horizon memory genuinely need weight-level fine-tuning, and prompt-only optimization caps out below the SOTA.

## Practical setup

```python
import dspy
from dspy.teleprompt import GEPA

# Define the program as DSPy modules.
class Consolidator(dspy.Module):
    def __init__(self):
        self.summarize = dspy.ChainOfThought(
            "working_region, source_traces -> consolidated_entries"
        )
    def forward(self, region, traces):
        return self.summarize(working_region=region, source_traces=traces)

# Metric is end-to-end task reward over the consolidated bank.
def task_reward(example, prediction, trace=None):
    bank = prediction.consolidated_entries
    return run_task_agent(example.tasks, bank).success_rate

# Train.
gepa = GEPA(metric=task_reward, num_threads=4)
optimized = gepa.compile(Consolidator(), trainset=scienceworld_trajectories)
```

The hard part is the metric (running the task agent inside the training loop). Same problem GRPO has.

## See also

- [[2402.03300-deepseekmath-grpo|GRPO paper]]
- [[2507.19457-gepa|GEPA paper]]
- [[2605.20616-auto-dreamer|Auto-Dreamer]] — uses GRPO; the natural ablation target
- [[sleep-consolidation]] — where this matters most
