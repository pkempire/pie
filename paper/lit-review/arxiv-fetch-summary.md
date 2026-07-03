# Arxiv Paper Summaries: Memory RL Landscape
## Compiled from workspace literature review | 2026-05-04

=============================================================================
1. Memory-R1 (arXiv:2508.19828)
=============================================================================
Authors: Yan, Yang, Huang et al. (Aug 2025)

CORE RESULTS:
- 4 discrete memory ops: ADD, UPDATE, DELETE, NOOP
- Two agents: Memory Manager (write) + Answer Agent (read), trained separately
- Trained with PPO and GRPO, outcome-driven RL
- Reward: trajectory-level final QA accuracy (EM LoCoMo, binary LongMemEval)
- 152 QA pairs; evaluated on LoCoMo, MSC, LongMemEval

LIMITATIONS:
- Trajectory-level reward diffuses credit across 10-50 ops per trajectory
- Two agents not jointly optimized
- No ablation on retention window size K
- On-policy only, no off-policy replay

OPEN PROBLEMS:
- Joint training of read+write policies (unified agent)
- Longer episode horizons (100+ turns for LongMemEval-grade)
- Off-policy replay for sample efficiency
- Fine-grained per-operation credit assignment


=============================================================================
2. Mem-Alpha (arXiv:2509.25911)
=============================================================================
Authors: Wang, Takanobu, Liang et al. (Sept 2025)

CORE RESULTS:
- Richer memory ontology: core, episodic, semantic components with specialized tools
- Trained on downstream answer accuracy pooled across full dialogue trajectory
- KEY CLAIM: generalizes from 30k tokens to 400k+ tokens at inference
- Sparse trajectory-level feedback; pre-training on user-assistant dialogue datasets

LIMITATIONS:
- Sparse trajectory-level reward, no per-operation credit signal
- Heavy reliance on pre-training data quality
- Complex ontology may not generalize across domains
- No explicit temporal reasoning or staleness handling

OPEN PROBLEMS:
- Long-horizon RL stability over 100+ turn sequences
- Scaling memory construction from 30k to 400k+ tokens without degradation
- Off-policy learning for data efficiency
- Whether coarse-grained ontology suffices for open-domain conversation


=============================================================================
3. Search-R1 (arXiv:2503.09516)
=============================================================================
Authors: Jin, Zeng, Yue et al. (Mar 2025, COLM 2025)

CORE RESULTS:
- Trains LLMs to interleave reasoning with search engine calls via GRPO
- Reward: final answer EM against ground truth (trajectory-level)
- Focus: tool-use (search), not memory ops, but same RL architecture
- Original (Qwen2.5-7B): NQ 42.9, TriviaQA 62.3, HotpotQA 38.6, 2Wiki 34.6
- Tinker replication IMPROVED: NQ 51.6, TriviaQA 67.3, HotpotQA 49.7, 2Wiki 42.8
- GRPO over multi-turn reasoning+search with group-relative advantage

LIMITATIONS:
- On-policy GRPO only, no experience replay
- Read-only (search/retrieval), no write-side or memory construction
- Domain-specific to search queries, not general memory management
- Group-relative advantage still diffuses credit across all steps
- Requires verifiable reward (EM), may not generalize to open-ended tasks

OPEN PROBLEMS:
- Joint learning of search-query generation AND reasoning steps
- Extension from search-tool RL to general memory-tool RL
- Off-policy learning for sample efficiency
- Scaling to longer reasoning horizons with many tool calls


=============================================================================
4. AgeMem / MemoryAgentBench (arXiv:2507.05257)
=============================================================================
NOTE: arxiv 2507.05257 = MemoryAgentBench (Hu,Wang,McAuley 2025 ICLR 2026)
A benchmark with 6 categories: Accurate Retrieval, Test-Time Learning,
Long-Range Understanding, Conflict Resolution, EventQA, FactConsolidation.
Repo: github.com/HUST-AI-HYZ/MemoryAgentBench

AgeMem (Agentic Memory, described separately in workspace, Jan 2026):
- Agent decides ALL memory ops (create,read,update,delete) via RL + step-wise GRPO
- Action space: {store_new, update_existing, delete, retrieve, do_nothing}
- After each turn, RL policy picks memory action; reward from downstream task
- Learns: when to store, update (contradictions), delete (stale), retrieve

LIMITATIONS (AgeMem):
- Requires training infrastructure and lots of data
- Reward signal is sparse and delayed
- Hard to apply to production without significant engineering

OPEN PROBLEMS:
- Dense per-step reward signals for faster learning
- Production-ready implementations
- Handling stability-plasticity dilemma (new learning overwriting old)


=============================================================================
5. DeltaMem (arXiv:2604.01560) -- from "DeltaMem memory levenshtein RL" search
=============================================================================
Authors: Zhang, Huang, Liu et al. (Apr 2026)

CORE RESULTS:
- Single-agent RL over learned operation vocabulary
- Ops: INSERT(entity,attr,val), DELETE(entity,attr), MODIFY(entity,attr,new_val), MERGE(a,b)
- KEY INNOVATION: Memory-based Levenshtein Distance as per-operation reward
- Semantic-aware: higher rewards for fixing critical information gaps
- Normalized: if op reduces Levenshtein by 50%, reward = +1.0 * 0.5 = +0.5
- Uses PPO; trained on synthetic dialogues with op-level labels from teacher
- HEADLINE: ~15% absolute improvement in memory coherence over Memory-R1
- Benchmarks: LoCoMo, HaluMem, PersonaMem; beats Mastra OM and manual rules
- Cost: ~4 GPU-hours on A100, under 50ms per op at inference
- Learned non-obvious strategies: selective entity merging, deferred deletes

LIMITATIONS:
- Requires reference memory state (teacher policy for op-level labels)
- State-distance signal (Levenshtein) does NOT measure downstream QA utility
- Synthetic training data; real-world transfer unproven
- PPO needs multiple model copies (policy, reference, reward model, critic)

OPEN PROBLEMS:
- Outcome-attribution rewards instead of state-distance (the mempol gap)
- Training without reference/teacher memory state
- Real-world dialogue training data
- Combining per-op credit with downstream task performance


=============================================================================
6. CoSearch (2026) -- from "CoSearch co-training reasoning ranking RL" search
=============================================================================
Status: Referenced in PAPER-SPEC.md and MEMORY-POLICY-RESEARCH-PLAN.md
No explicit arxiv ID found in workspace; referenced as 2026 paper

CORE RESULTS (from workspace references):
- Joint training of reasoning + ranking via reinforcement learning
- +6.6% F1 improvement over Search-R1
- Co-trains reasoner and ranker in alternating/joint RL loop
- Closest precedent for AlphaZero-style alternation in memory (write+read co-training)

LIMITATIONS (inferred):
- Domain-specific to search/QA (reasoning+ranking), not general memory ops
- Read-side only, no write/memory-construction component
- Co-training between reasoning+ranking; extending to write+read remains novel

OPEN PROBLEMS:
- Extending co-training from reasoning+ranking to memory-write+memory-read
- AlphaZero-style alternation for memory policy learning
- Joint optimization of complementary agent components


=============================================================================
CROSS-CUTTING SYNTHESIS
=============================================================================

REWARD SIGNAL TAXONOMY:
  Memory-R1    Trajectory-level    Outcome (QA accuracy)       PPO + GRPO
  Mem-Alpha    Trajectory-level    Outcome (pooled dialogue)    Not specified
  Search-R1    Trajectory-level    Outcome (answer EM)          GRPO
  AgeMem       Trajectory-level    Outcome (task performance)   Step-wise GRPO
  DeltaMem     Per-operation       State-distance (Levenshtein) PPO
  CoSearch     Trajectory-level    Outcome (F1)                 RL (joint)

KEY OPEN PROBLEM: CREDIT ASSIGNMENT
Trajectory-level rewards are too coarse. DeltaMem is the only published work
with per-operation rewards, but uses state-distance (Levenshtein to reference)
rather than outcome-attribution (did this op help downstream QA?). The mempol
project per-op counterfactual approach (COMA-style) occupies this gap.

KEY OPEN PROBLEM: JOINT READ+WRITE TRAINING
Memory-R1 trains two separate agents. No published system jointly trains read
and write policies with shared outcome rewards. CoSearch provides closest
precedent (co-training reasoning+ranking). AlphaZero-style alternation for
write+read remains unexplored.

CONVERGENCE POINTS:
1. All systems use GRPO or PPO (on-policy RL)
2. All evaluate on LoCoMo and/or LongMemEval
3. All define discrete operation vocabularies
4. All require verifiable outcome signals (QA accuracy, EM)
5. None have solved off-policy learning or experience replay for memory ops

