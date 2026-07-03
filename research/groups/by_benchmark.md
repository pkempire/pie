# Papers by benchmark

_(auto-generated. Edit papers/*.md to change.)_


## "Time-Bench (10y news  (1)

- **Time-R1: Towards Comprehensive Temporal Reasoning in LLMs** (2025) — A 3B model trained with the 3-stage RL curriculum beats 671B DeepSeek-R1 on future-event prediction
  → [Time-R1: Towards Comprehensive Temporal Reasoning in LLMs](papers/2505.13508-time-r1.md)

## 2WikiMultihopQA  (1)

- **Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning** (2025) — +41% over RAG baselines on NaturalQuestions with Qwen2.5-7B; Multi-hop QA results on HotpotQA and 2WikiMultihopQA
  → [Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning](papers/2503.09516-search-r1.md)

## 6-task suite total  (1)

- **GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning** (2025) — Outperforms GRPO by 6% on average, up to 20% across 6 tasks; 35x fewer rollouts than GRPO at matched/better quality; Beats MIPROv2 by >10% (+12% on AIME-2025); ICLR 2026 Oral acceptance
  → [GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning](papers/2507.19457-gepa.md)

## 8-test standardized benchmark suite (their own)  (1)

- **SCM: Sleep-Consolidated Memory with Algorithmic Forgetting for Large Language Models** (2026) — Perfect recall over 10-turn conversations; 90.9% reduction in memory noise via adaptive forgetting; <1ms memory search latency with hundreds of stored concepts
  → [SCM: Sleep-Consolidated Memory with Algorithmic Forgetting for Large Language Models](papers/2604.20943-scm-sleep.md)

## AIME-2025  (1)

- **GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning** (2025) — Outperforms GRPO by 6% on average, up to 20% across 6 tasks; 35x fewer rollouts than GRPO at matched/better quality; Beats MIPROv2 by >10% (+12% on AIME-2025); ICLR 2026 Oral acceptance
  → [GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning](papers/2507.19457-gepa.md)

## ALFWorld  (2)

- **Auto-Dreamer: Learning Offline Memory Consolidation for Language Agents** (2026) — ScienceWorld SR 41.1% vs UMEM 34.1% (+7pp) with 6.9k vs 80.9k tokens (~12x less memory); ALFWorld held-out 60.2% vs UMEM 58.4% with 10.9k vs 62.9k tokens (~6x less memory); WebArena held-out 52.3% (~tied lead) vs LightMem 52.0% / 370k tokens; Counterfactual reward ablation: without it, bank grows unbounded; with it, bank shrinks late in training while perf preserved
  → [Auto-Dreamer: Learning Offline Memory Consolidation for Language Agents](papers/2605.20616-auto-dreamer.md)
- **Retrospex: Language Agent Meets Offline Reinforcement Learning Critic** (2024) — Outperforms strong contemporary baselines on all 3 environments (EMNLP 2024); Specific numbers not extracted; see paper Table 1
  → [Retrospex: Language Agent Meets Offline Reinforcement Learning Critic](papers/2505.11807-retrospex.md)

## CueSpeak  (1)

- **Interaction Models: A Scalable Approach to Human-AI Collaboration** (2026) — TimeSpeak: 64.7 (vs GPT-Realtime-2 minimal 4.3); Direct sense of elapsed time built into architecture; Split design: interaction model + background model
  → [Interaction Models: A Scalable Approach to Human-AI Collaboration](papers/20260511-thinkingmachines-interaction-models.md)

## DMR  (1)

- **Zep: A Temporal Knowledge Graph Architecture for Agent Memory** (2024) — ~72% LongMemEval (varies by config); Reported 94.8% on DMR (Deep Memory Retrieval)
  → [Zep: A Temporal Knowledge Graph Architecture for Agent Memory](papers/2501.13956-zep.md)

## DeepPlanning  (1)

- **DeepPlanning: Benchmarking Long-Horizon Agentic Planning with Verifiable Constraints** (2026) — GPT-5.2-high: 44.6% avg case accuracy across both domains; GPT-5.2-high: 35.0% travel case accuracy; Claude-4.5-Opus reasoning: 22.7% travel case accuracy; Performance degrades monotonically as horizon (2-7 days) increases
  → [DeepPlanning: Benchmarking Long-Horizon Agentic Planning with Verifiable Constraints](papers/2601.18137-deepplanning.md)

## GSM8K  (1)

- **DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models** (2024) — Defined GRPO; now default RL algorithm for tool-use training
  → [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](papers/2402.03300-deepseekmath-grpo.md)

## HORIZON  (1)

- **The Long-Horizon Task Mirage? Diagnosing Where and Why Agentic Systems Break** (2026) — Evaluated GPT-5 variants and Claude models; Inter-annotator agreement κ=0.61; Human-judge agreement κ=0.84 (LLM-as-Judge validated); Public leaderboard: xwang2775.github.io/horizon-leaderboard
  → [The Long-Horizon Task Mirage? Diagnosing Where and Why Agentic Systems Break](papers/2604.11978-horizon.md)

## HotpotQA  (2)

- **Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning** (2025) — +41% over RAG baselines on NaturalQuestions with Qwen2.5-7B; Multi-hop QA results on HotpotQA and 2WikiMultihopQA
  → [Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning](papers/2503.09516-search-r1.md)
- **GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning** (2025) — Outperforms GRPO by 6% on average, up to 20% across 6 tasks; 35x fewer rollouts than GRPO at matched/better quality; Beats MIPROv2 by >10% (+12% on AIME-2025); ICLR 2026 Oral acceptance
  → [GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning](papers/2507.19457-gepa.md)

## LoCoMo  (10)

- **EverMemOS: A Self-Organizing Memory Operating System for Structured Long-Horizon Reasoning** (2026) — LoCoMo: 93.05% (leaderboard table-extracted, NOT abstract-verified); Per-category: 96.67 / 91.84 / 89.72 / 76.04 (4 LoCoMo categories); Same paper benchmarks Zep at 85.22%, MemOS at 80.76%
  → [EverMemOS: A Self-Organizing Memory Operating System for Structured Long-Horizon Reasoning](papers/2601.02163-evermemos.md)
- **TiMem: Temporal-Hierarchical Memory Consolidation for Long-Horizon Conversational Agents** (2026) — LoCoMo: 75.30% (state of the art); LongMemEval-S: 76.88%; 52.20% reduction in recalled memory length on LoCoMo; Outperforms all evaluated baselines under consistent eval setup
  → [TiMem: Temporal-Hierarchical Memory Consolidation for Long-Horizon Conversational Agents](papers/2601.02845-timem.md)
- **Amory: Building Coherent Narrative-Driven Agent Memory through Agentic Reasoning** (2026) — Amory (EM+SM): 87.7% on LoCoMo; Full-context baseline: 86.1%; Mem0: 59.9%
  → [Amory: Building Coherent Narrative-Driven Agent Memory through Agentic Reasoning](papers/2601.06282-amory.md)
- **FadeMem: Biologically-Inspired Forgetting for Efficient Agent Memory** (2026) — Specific accuracy not yet extracted
  → [FadeMem: Biologically-Inspired Forgetting for Efficient Agent Memory](papers/2601.18642-fadememem.md)
- **MIRIX: Multi-Agent Memory System for LLM-Based Agents** (2025) — MIRIX: 85.38% on LoCoMo; Zep: 79.09%; Mem0: 62.47%
  → [MIRIX: Multi-Agent Memory System for LLM-Based Agents](papers/2507.07957-mirix.md)
- **Memory-R1: Enhancing LLM Agents to Manage and Utilize Memories via Reinforcement Learning** (2025) — Closest published direct comparison to mempol's per-op approach
  → [Memory-R1: Enhancing LLM Agents to Manage and Utilize Memories via Reinforcement Learning](papers/2508.19828-memory-r1.md)
- **RGMem: Renormalization Group-inspired Memory Evolution for Language Agents** (2025) — RGMem: 86.17% on LoCoMo; Zep: 79.09%
  → [RGMem: Renormalization Group-inspired Memory Evolution for Language Agents](papers/2510.16392-rgmem.md)
- **Hindsight is 20/20: Building Agent Memory that Retains, Recalls, and Reflects** (2025) — Hindsight w/ Gemini-3: 89.61% on LoCoMo; Backboard (their RAG-style baseline): 90.0%; Memobase v0.0.37: 75.78%
  → [Hindsight is 20/20: Building Agent Memory that Retains, Recalls, and Reflects](papers/2512.12818-hindsight.md)
- **Evaluating Very Long-Term Conversational Memory of LLM Agents** (2024) — Per-question evidence labels (gold dia_ids) — enables coverage scoring
  → [Evaluating Very Long-Term Conversational Memory of LLM Agents](papers/2402.17753-locomo.md)
- **Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory** (2024) — ~69% on LongMemEval (varies by config); ~27% on LoCoMo (50 questions, gpt-4o judge); Self-reported 40% extraction-failure rate
  → [Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory](papers/2504.19413-mem0.md)

## LongMemEval  (4)

- **Memory-R1: Enhancing LLM Agents to Manage and Utilize Memories via Reinforcement Learning** (2025) — Closest published direct comparison to mempol's per-op approach
  → [Memory-R1: Enhancing LLM Agents to Manage and Utilize Memories via Reinforcement Learning](papers/2508.19828-memory-r1.md)
- **LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory** (2024) — GPT-4o judge protocol with 3-bucket scoring (1.0 / 0.5 / 0.0)
  → [LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory](papers/2410.10813-longmemeval.md)
- **Zep: A Temporal Knowledge Graph Architecture for Agent Memory** (2024) — ~72% LongMemEval (varies by config); Reported 94.8% on DMR (Deep Memory Retrieval)
  → [Zep: A Temporal Knowledge Graph Architecture for Agent Memory](papers/2501.13956-zep.md)
- **Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory** (2024) — ~69% on LongMemEval (varies by config); ~27% on LoCoMo (50 questions, gpt-4o judge); Self-reported 40% extraction-failure rate
  → [Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory](papers/2504.19413-mem0.md)

## LongMemEval-S  (1)

- **TiMem: Temporal-Hierarchical Memory Consolidation for Long-Horizon Conversational Agents** (2026) — LoCoMo: 75.30% (state of the art); LongMemEval-S: 76.88%; 52.20% reduction in recalled memory length on LoCoMo; Outperforms all evaluated baselines under consistent eval setup
  → [TiMem: Temporal-Hierarchical Memory Consolidation for Long-Horizon Conversational Agents](papers/2601.02845-timem.md)

## MATH  (1)

- **DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models** (2024) — Defined GRPO; now default RL algorithm for tool-use training
  → [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](papers/2402.03300-deepseekmath-grpo.md)

## MSC  (1)

- **Memory-R1: Enhancing LLM Agents to Manage and Utilize Memories via Reinforcement Learning** (2025) — Closest published direct comparison to mempol's per-op approach
  → [Memory-R1: Enhancing LLM Agents to Manage and Utilize Memories via Reinforcement Learning](papers/2508.19828-memory-r1.md)

## MultiAgentBench  (1)

- **MultiAgentBench: Evaluating the Collaboration and Competition of LLM Agents** (2025) — Canonical multi-agent benchmark; published at ACL 2025
  → [MultiAgentBench: Evaluating the Collaboration and Competition of LLM Agents](papers/2503.01935-multiagentbench.md)

## NaturalQuestions  (1)

- **Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning** (2025) — +41% over RAG baselines on NaturalQuestions with Qwen2.5-7B; Multi-hop QA results on HotpotQA and 2WikiMultihopQA
  → [Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning](papers/2503.09516-search-r1.md)

## ProactiveVideoQA  (1)

- **Interaction Models: A Scalable Approach to Human-AI Collaboration** (2026) — TimeSpeak: 64.7 (vs GPT-Realtime-2 minimal 4.3); Direct sense of elapsed time built into architecture; Split design: interaction model + background model
  → [Interaction Models: A Scalable Approach to Human-AI Collaboration](papers/20260511-thinkingmachines-interaction-models.md)

## RepCount-A  (1)

- **Interaction Models: A Scalable Approach to Human-AI Collaboration** (2026) — TimeSpeak: 64.7 (vs GPT-Realtime-2 minimal 4.3); Direct sense of elapsed time built into architecture; Split design: interaction model + background model
  → [Interaction Models: A Scalable Approach to Human-AI Collaboration](papers/20260511-thinkingmachines-interaction-models.md)

## Robotouille  (1)

- **Robotouille: An Asynchronous Planning Benchmark for LLM Agents** (2025) — gpt-4o ReAct: 47% sync → 11% async (-36pp drop); Agent failures cluster around interleaving parallel waits with continued reasoning
  → [Robotouille: An Asynchronous Planning Benchmark for LLM Agents](papers/2502.05227-robotouille.md)

## ScienceWorld  (2)

- **Auto-Dreamer: Learning Offline Memory Consolidation for Language Agents** (2026) — ScienceWorld SR 41.1% vs UMEM 34.1% (+7pp) with 6.9k vs 80.9k tokens (~12x less memory); ALFWorld held-out 60.2% vs UMEM 58.4% with 10.9k vs 62.9k tokens (~6x less memory); WebArena held-out 52.3% (~tied lead) vs LightMem 52.0% / 370k tokens; Counterfactual reward ablation: without it, bank grows unbounded; with it, bank shrinks late in training while perf preserved
  → [Auto-Dreamer: Learning Offline Memory Consolidation for Language Agents](papers/2605.20616-auto-dreamer.md)
- **Retrospex: Language Agent Meets Offline Reinforcement Learning Critic** (2024) — Outperforms strong contemporary baselines on all 3 environments (EMNLP 2024); Specific numbers not extracted; see paper Table 1
  → [Retrospex: Language Agent Meets Offline Reinforcement Learning Critic](papers/2505.11807-retrospex.md)

## TDBench (SQL-generated TSQA from temporal DB joins)  (1)

- **Harnessing Temporal Databases for Systematic Evaluation of Factual Time-Sensitive Question-Answering in LLMs** (2025) — Introduces a time-accuracy metric distinct from answer accuracy; scalable TSQA generation
  → [Harnessing Temporal Databases for Systematic Evaluation of Factual Time-Sensitive Question-Answering in LLMs](papers/2508.02045-tdbench.md)

## Test of Time (synthetic temporal-logic)  (1)

- **Test of Time: A Benchmark for Evaluating LLMs on Temporal Reasoning** (2024) — Used to isolate temporal reasoning from memorization; exposes brittleness on controlled structure
  → [Test of Time: A Benchmark for Evaluating LLMs on Temporal Reasoning](papers/2406.09170-test-of-time.md)

## TicToc  (1)

- **Your LLM Agents are Temporally Blind: The Misalignment Between Tool Use Decisions and Human Time Perception** (2025) — Best model alignment with human preferences: 65%; Prompt-engineering alone has limited effectiveness
  → [Your LLM Agents are Temporally Blind: The Misalignment Between Tool Use Decisions and Human Time Perception](papers/2510.23853-temporally-blind.md)

## TimeSpeak  (1)

- **Interaction Models: A Scalable Approach to Human-AI Collaboration** (2026) — TimeSpeak: 64.7 (vs GPT-Realtime-2 minimal 4.3); Direct sense of elapsed time built into architecture; Split design: interaction model + background model
  → [Interaction Models: A Scalable Approach to Human-AI Collaboration](papers/20260511-thinkingmachines-interaction-models.md)

## TriviaQA  (1)

- **Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning** (2025) — +41% over RAG baselines on NaturalQuestions with Qwen2.5-7B; Multi-hop QA results on HotpotQA and 2WikiMultihopQA
  → [Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning](papers/2503.09516-search-r1.md)

## ViDoRe (their own visual document retrieval benchmark)  (1)

- **ColPali: Efficient Document Retrieval with Vision Language Models** (2024) — Largely outperforms modern document retrieval pipelines on ViDoRe; Drastically simpler, faster, end-to-end trainable; ICLR 2025
  → [ColPali: Efficient Document Retrieval with Vision Language Models](papers/2407.01449-colpali.md)

## WebArena  (1)

- **Auto-Dreamer: Learning Offline Memory Consolidation for Language Agents** (2026) — ScienceWorld SR 41.1% vs UMEM 34.1% (+7pp) with 6.9k vs 80.9k tokens (~12x less memory); ALFWorld held-out 60.2% vs UMEM 58.4% with 10.9k vs 62.9k tokens (~6x less memory); WebArena held-out 52.3% (~tied lead) vs LightMem 52.0% / 370k tokens; Counterfactual reward ablation: without it, bank grows unbounded; with it, bank shrinks late in training while perf preserved
  → [Auto-Dreamer: Learning Offline Memory Consolidation for Language Agents](papers/2605.20616-auto-dreamer.md)

## Webshop  (1)

- **Retrospex: Language Agent Meets Offline Reinforcement Learning Critic** (2024) — Outperforms strong contemporary baselines on all 3 environments (EMNLP 2024); Specific numbers not extracted; see paper Table 1
  → [Retrospex: Language Agent Meets Offline Reinforcement Learning Critic](papers/2505.11807-retrospex.md)

## code optimization tasks  (1)

- **GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning** (2025) — Outperforms GRPO by 6% on average, up to 20% across 6 tasks; 35x fewer rollouts than GRPO at matched/better quality; Beats MIPROv2 by >10% (+12% on AIME-2025); ICLR 2026 Oral acceptance
  → [GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning](papers/2507.19457-gepa.md)

## math reasoning  (1)

- **DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models** (2024) — Defined GRPO; now default RL algorithm for tool-use training
  → [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](papers/2402.03300-deepseekmath-grpo.md)

## none (theory paper)  (1)

- **Cognitive Architectures for Language Agents (CoALA)** (2023) — TMLR camera-ready 2023; foundational citation for agent-memory papers; Companion awesome-language-agents repo organizes the literature by CoALA categories
  → [Cognitive Architectures for Language Agents (CoALA)](papers/2309.02427-coala.md)

## paired LLM negotiation (their own)  (1)

- **Real-Time Deadlines Reveal Temporal Awareness Failures in LLM Strategic Dialogues** (2026) — GPT-5.1 deal closure: 4% WITHOUT a clock-in-prompt vs 32% WITH periodic time updates (+28pp); The capability is latent in the model; it just doesn't condition on time on its own
  → [Real-Time Deadlines Reveal Temporal Awareness Failures in LLM Strategic Dialogues](papers/2601.13206-real-time-deadlines.md)

## standard spatial-temporal forecasting benchmarks  (1)

- **How Can Large Language Models Understand Spatial-Temporal Data?** (2024) — Matches dedicated ST-forecasting models using an LLM + tokeniser/adapter
  → [How Can Large Language Models Understand Spatial-Temporal Data?](papers/2401.14192-stg-llm.md)

## three task families)"  (1)

- **Time-R1: Towards Comprehensive Temporal Reasoning in LLMs** (2025) — A 3B model trained with the 3-stage RL curriculum beats 671B DeepSeek-R1 on future-event prediction
  → [Time-R1: Towards Comprehensive Temporal Reasoning in LLMs](papers/2505.13508-time-r1.md)

## τ²-Bench / τ²-Bench Telecom  (1)

- **τ²-Bench: Evaluating Conversational Agents in a Dual-Control Environment** (2025) — 38 model entries on the leaderboard (April 2026); Production-relevant agent eval used by Sierra (Bret Taylor)
  → [τ²-Bench: Evaluating Conversational Agents in a Dual-Control Environment](papers/2506.07982-tau2-bench.md)
