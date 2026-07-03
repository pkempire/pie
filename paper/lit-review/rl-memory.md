# Literature Review: RL for Memory and Tool-Use in LLMs

## Current State of Memory-RL Landscape

The past 18 months have witnessed an explosion of work on reinforcement learning for memory management in language models, consolidating what was previously scattered heuristic-based approaches into a coherent research direction. At least three major concurrent efforts—Memory-R1, Mem-Alpha, and DeltaMem—are now pursuing similar objectives: training agents to autonomously decide what to store, update, delete, and retrieve from external memory. This convergence reflects a fundamental shift: memory is no longer fixed, hand-engineered, and static, but instead learned and adaptive through interaction and feedback. The field is exploring three broad reward shapes across these works. Memory-R1 uses outcome-driven RL with trajectory-level feedback (final QA accuracy on LoCoMo, LongMemEval, MSC benchmarks). Mem-Alpha trains on downstream answer accuracy pooled across a full dialogue trajectory, enabling agents to generalize from sequences of 30k tokens to 400k+ tokens. DeltaMem introduces the Memory-based Levenshtein Distance as a per-operation reward signal, directly measuring memory state evolution rather than only terminal outcomes. Search-R1 complements this space by focusing on tool-use (search queries) rather than explicit memory operations but uses the same outcome-based architecture: GRPO optimization over reasoning trajectories with final answer-EM reward.

The operational vocabulary across these systems varies. Memory-R1 defines four discrete operations: ADD (insert new entry), UPDATE (modify existing entry), DELETE (remove entry), and NOOP (skip). Mem-Alpha uses a richer memory ontology with core, episodic, and semantic components, each with specialized tools. DeltaMem formalizes "operation-level memory updating labels" with learned edit distance. This suggests the field has not yet settled on whether coarse-grained operations (ADD/UPDATE/DELETE) suffice or whether fine-grained in-place modifications are necessary. Your per-op counterfactual approach sits naturally between these: it treats each memory operation as a discrete action with a learned marginal value, avoiding trajectory-level credit diffusion but more granular than system-wide comparisons.

## Per-Op Counterfactual as a Novel Credit Assignment Mechanism

Your approach is genuinely novel in the RL-for-memory space, though it builds on a well-established baseline. The original COMA (Counterfactual Multi-Agent) framework (Foerster et al., 2018) applies leave-one-out counterfactual reasoning to multi-agent credit assignment: each agent's contribution is estimated by comparing the full trajectory reward against a trajectory where that agent's actions are replaced by a baseline policy. You apply this logic per-memory-operation rather than per-agent, creating a single-agent variant that decomposes trajectory reward into marginal operation values. The critical innovation: instead of a frozen baseline policy, you condition counterfactual rollouts on the exact memory state at that operation step, isolating the causal effect of each write/read choice.

Across the surveyed concurrent works, none explicitly adopt this framing. Memory-R1 clusters trajectory returns into groups by outcome (correct/incorrect) and applies GRPO's group advantage directly to all operations in a trajectory. Mem-Alpha uses sparse trajectory-level feedback. DeltaMem's Levenshtein distance is application-specific to memory state representation. The counterfactual marginal approach offers three advantages: (1) **lower variance per operation** by conditioning on realized history rather than averaging over trajectories, (2) **direct causality testing** via contrastive rollouts (what would happen if we had _not_ executed this operation?), and (3) **transferability** across different memory backends because the credit signal is operation-centric, not state-representation-specific. This is a concrete, implementable distinction from prior work.

## Memory-R1's Reward Architecture and Decomposition

Memory-R1 deserves close reading because it is the most directly comparable system. The paper does not expose an explicit per-operation reward formula in the abstract/intro, but the architecture clarifies the intent: a Memory Manager agent learns ADD/UPDATE/DELETE/NOOP policies, and an Answer Agent selects relevant entries. Both are trained with "outcome-driven RL (PPO and GRPO)." The outcome is final QA accuracy (EM on LoCoMo, binary correct/incorrect on LongMemEval). This is trajectory-level, not per-op. The training data is 152 QA pairs, suggesting a highly data-efficient regime where the RL signal must propagate backward through 10-50 operations per trajectory. Memory-R1 does not ablate memory retention (K) explicitly; you should verify if their published models use K=10 (typical in LoCoMo) and whether they attempt larger retention windows. If not, this is a concrete improvement vector: sweep K up to 20-30 and measure EM gains, then report the ablation.

The key insight from Memory-R1: **the reward is a binary or soft label on the final answer**, not a dense per-turn signal. This is pragmatic but coarse. Intermediate operations that fail to retrieve the necessary information get the same gradient signal as operations that succeed; only the final outcome matters. Your per-op counterfactual breaks this bottleneck by asking: "Did this specific operation improve the agent's ability to answer the question?" If the counterfactual rollout (without that operation) would still answer correctly, the operation gets zero credit. If removing it causes failure, the operation receives full credit. This is fundamentally different from trajectory averaging.

## Benchmark Coverage and Generalization

Memory-R1 evaluates on LoCoMo, MSC (multi-turn conversation), and LongMemEval. This is the consensus test suite. Your paper should adopt the same benchmarks to claim novelty relative to Memory-R1. Key differences: LoCoMo has ~10 turns per conversation with average context length 5-10k tokens; LongMemEval is longer (100+ turns, 50k+ token context). If your per-op counterfactual method shows gains on LongMemEval relative to Memory-R1's trajectory-level approach, that is a strong empirical claim: fine-grained credit assignment scales better to long horizons. Memory-R1 reports 3B-14B model scaling; verify that your LoRA trains on the same sizes and whether GRPO's group advantage scales comparably.

## Technical Directions and Low-Hanging Fruit

The field is moving toward three concrete improvements next quarter, all of which your work can subsume or extend:

1. **Joint training of read and write policies**: Memory-R1 trains two separate agents (Memory Manager + Answer Agent). Your per-op counterfactual framework naturally extends to joint training by factoring the reward into read-op advantages (e.g., "retrieve from entry X?") and write-op advantages (e.g., "add new entry?"), learning a unified policy that orchestrates both. Search-R1 hints at this by jointly learning search-query generation and reasoning steps.

2. **Longer episode horizons and recency bias**: Memory-R1's training conversations are 10-30 turns. LongMemEval scales to 100+. The next bottleneck is RL stability over 100-turn sequences. Your counterfactual baseline naturally inherits lower variance, which may reduce the training instability that plagued early long-horizon RL. Run smoke tests on 50-turn and 100-turn trajectories to validate.

3. **Off-policy replay and trajectory reuse**: Both Memory-R1 and Search-R1 use on-policy GRPO (like PPO). Mem-Alpha hints at data efficiency through "user-assistant dialogue dataset" pre-training. The gap is off-policy learning: storing successful memory trajectories and replaying them with importance weighting. Your counterfactual framework is amenable to Q-learning variants (learn Q(state, op) directly), which admit off-policy updates. This could 2-3x your sample efficiency.

## Critical Insights for Your Paper's Related Work

When writing related work, emphasize three points:

**First, position per-op counterfactual as a principled extension of COMA to the single-agent memory setting.** Cite Foerster et al. (2018) and note that prior memory-RL work has avoided explicit counterfactual reasoning, instead relying on trajectory averaging (Memory-R1) or state-level loss functions (DeltaMem). Your approach reintroduces causality testing at fine-grained timescales.

**Second, claim that outcome-driven RL (as in Memory-R1 and Mem-Alpha) is a necessary but not sufficient foundation.** Trajectory-level feedback diffuses credit across 20-50 operations; most are equally uninformative until the final operation determines the outcome. Counterfactual marginal utility directly tests the causal chain: "Without this operation, would we still succeed?" This is mechanistically stronger than group advantage.

**Third, distinguish between baseline (heuristic, fixed) and adaptive (learned) memory operations.** Your work assumes a fixed operation vocabulary (read, write, delete—or whatever your ops are) but learns when and where to apply each. Some concurrent work (e.g., DeltaMem's Levenshtein distance for edit detection) is more ambitious, learning the operations themselves. That is orthogonal; both approaches can co-exist.

## Concrete Ablations to Run Before Submission

1. **Counterfactual vs. trajectory-average reward**: Train two models—one with per-op counterfactual advantage, one with Memory-R1-style group EM reward—on the same dataset and model size. Compare EM on LoCoMo and LongMemEval. If you win by >2% on LongMemEval, that validates the hypothesis.

2. **Retention-window sweep (K=10, 15, 20, 30)**: Memory-R1 does not report this. Plot answer-EM vs. K. If your method plateaus at smaller K (e.g., K=15 instead of K=30), that is evidence your credit assignment is more efficient.

3. **Joint read-write vs. separate agents**: If you train two separate agents as in Memory-R1, also train a single unified policy that selects among all ops jointly. Measure inference speed and EM; if unified is faster with similar EM, that is a systems contribution.

## Summary

You sit at the intersection of two mature subfields: (1) counterfactual credit assignment in multi-agent RL (Foerster, 2018 onward) and (2) outcome-driven memory management in LLMs (Memory-R1, Mem-Alpha, 2025). Your innovation is the synthesis: applying counterfactual reasoning to single-agent memory operations, unlocking finer-grained credit assignment than trajectory averaging. The novelty is incremental but concrete. Memory-R1 is the closest competitor, and they have not explored per-operation counterfactuals or long-horizon (LongMemEval) performance carefully. If you ablate K and show wins there, and if your counterfactual method outperforms their group advantage on 100-turn sequences, you have a strong empirical story to tell. Write the related work to position COMA as inspiration, claim per-op counterfactual as your innovation, and present empirical results on the standard benchmarks.

---

## References (BibTeX stub)

```bibtex
@article{yan2025memoryr1,
  title={Memory-R1: Enhancing Large Language Model Agents to Manage and Utilize Memories via Reinforcement Learning},
  author={Yan, Sikuan and Yang, Xiufeng and Huang, Zuchao and others},
  journal={arXiv preprint arXiv:2508.19828},
  year={2025}
}

@article{jin2025searchr1,
  title={Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning},
  author={Jin, Bowen and Zeng, Hansi and Yue, Zhenrui and others},
  journal={arXiv preprint arXiv:2503.09516},
  year={2025}
}

@article{wang2025memalpha,
  title={Mem-$\alpha$: Learning Memory Construction via Reinforcement Learning},
  author={Wang, Yu and Takanobu, Ryuichi and Liang, Zhiqi and others},
  journal={arXiv preprint arXiv:2509.25911},
  year={2025}
}

@article{zhang2026deltamem,
  title={DeltaMem: Towards Agentic Memory Management via Reinforcement Learning},
  author={Zhang, Qi and Huang, Shen and Liu, Chu and others},
  journal={arXiv preprint arXiv:2604.01560},
  year={2026}
}

@article{shao2024deepseekmath,
  title={DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models},
  author={Shao, Zhihong and Wang, Peiyi and Zhu, Qihao and others},
  journal={arXiv preprint arXiv:2402.03300},
  year={2024}
}

@article{foerster2018counterfactual,
  title={Counterfactual Multi-Agent Policy Gradients},
  author={Foerster, Jakob N and others},
  journal={Proceedings of the 35th International Conference on Machine Learning},
  year={2018}
}

@article{luo2025graphr1,
  title={Graph-R1: Towards Agentic GraphRAG Framework via End-to-end Reinforcement Learning},
  author={Luo, Haoran and E, Haihong and Chen, Guanting and others},
  journal={arXiv preprint arXiv:2507.21892},
  year={2025}
}

@article{wang2025timer1,
  title={Time-R1: Post-Training Large Vision Language Model for Temporal Video Grounding},
  author={Wang, Ye and Wang, Ziheng and Xu, Boshen and others},
  journal={arXiv preprint arXiv:2503.13377},
  year={2025}
}
```
