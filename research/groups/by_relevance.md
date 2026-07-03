# Papers by relevance to our project

_(auto-generated. Edit papers/*.md to change.)_


## high  (21)

- **Interaction Models: A Scalable Approach to Human-AI Collaboration** (2026, infrastructure) — Defines the continuous-time-perception problem at the model level. Validates background/interaction split.
  → [Interaction Models: A Scalable Approach to Human-AI Collaboration](papers/20260511-thinkingmachines-interaction-models.md)
- **EverMemOS: A Self-Organizing Memory Operating System for Structured Long-Horizon Reasoning** (2026, write-time-compression) — Current LoCoMo SOTA. 93% is the bar — anything we ship must move toward this or have a different value prop.
  → [EverMemOS: A Self-Organizing Memory Operating System for Structured Long-Horizon Reasoning](papers/2601.02163-evermemos.md)
- **Amory: Building Coherent Narrative-Driven Agent Memory through Agentic Reasoning** (2026, write-time-compression) — Top-5 LoCoMo result. Validates the [[2309.02427-coala|CoALA]] episodic/semantic split with a concrete architecture. Amazon production-grade.
  → [Amory: Building Coherent Narrative-Driven Agent Memory through Agentic Reasoning](papers/2601.06282-amory.md)
- **Real-Time Deadlines Reveal Temporal Awareness Failures in LLM Strategic Dialogues** (2026, temporal-reasoning) — Cleanest quantified evidence for the temporal-AWARENESS (behaving-in-time) thread, as distinct from temporal reasoning. The +28pp gap is the single best argument for putting a clock in every agent observation.
  → [Real-Time Deadlines Reveal Temporal Awareness Failures in LLM Strategic Dialogues](papers/2601.13206-real-time-deadlines.md)
- **DeepPlanning: Benchmarking Long-Horizon Agentic Planning with Verifiable Constraints** (2026, benchmark) — Hardest applied long-horizon benchmark for our contract/checkpoint orchestration thesis. GPT-5.2's 44.6% leaves enormous headroom.
  → [DeepPlanning: Benchmarking Long-Horizon Agentic Planning with Verifiable Constraints](papers/2601.18137-deepplanning.md)
- **The Long-Horizon Task Mirage? Diagnosing Where and Why Agentic Systems Break** (2026, benchmark) — Cross-domain long-horizon failure diagnosis is exactly the eval our consolidator work could land SOTA on. Failure attribution is a specific axis nobody else competes on cleanly.
  → [The Long-Horizon Task Mirage? Diagnosing Where and Why Agentic Systems Break](papers/2604.11978-horizon.md)
- **Auto-Dreamer: Learning Offline Memory Consolidation for Language Agents** (2026, RL-for-memory) — The state-of-the-art learned-consolidator with outcome reward. We initially proposed this exact thing; they shipped it 5 days ago. Direct competitor to anything we build in this space.
  → [Auto-Dreamer: Learning Offline Memory Consolidation for Language Agents](papers/2605.20616-auto-dreamer.md)
- **Robotouille: An Asynchronous Planning Benchmark for LLM Agents** (2025, benchmark) — Direct fit for contract/checkpoint orchestration thesis. The 36pp sync→async drop is the cleanest single-number target.
  → [Robotouille: An Asynchronous Planning Benchmark for LLM Agents](papers/2502.05227-robotouille.md)
- **MultiAgentBench: Evaluating the Collaboration and Competition of LLM Agents** (2025, benchmark) — Canonical academic benchmark for multi-agent claims. Needed for credibility of contract/orchestration paper.
  → [MultiAgentBench: Evaluating the Collaboration and Competition of LLM Agents](papers/2503.01935-multiagentbench.md)
- **Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning** (2025, RL-for-tool-use) — Structural template for our environment. We forked their recipe for memory_rl.
  → [Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning](papers/2503.09516-search-r1.md)
- **Time-R1: Towards Comprehensive Temporal Reasoning in LLMs** (2025, temporal-reasoning) — Strongest evidence that reasoning-over-time is fixable with data+RL curriculum, and that a small specialized model beats a giant generalist on it. Template for an mempol-adjacent 'train the temporal skill' result.
  → [Time-R1: Towards Comprehensive Temporal Reasoning in LLMs](papers/2505.13508-time-r1.md)
- **MIRIX: Multi-Agent Memory System for LLM-Based Agents** (2025, agent-orchestration) — Multi-agent memory framing — memory operations distributed across specialized agents. Connects directly to the multi-agent delegation problem.
  → [MIRIX: Multi-Agent Memory System for LLM-Based Agents](papers/2507.07957-mirix.md)
- **GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning** (2025, theory) — The optimizer we'd use to evolve the consolidator prompt. Core to Goal 01 and Goal 02.
  → [GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning](papers/2507.19457-gepa.md)
- **Memory-R1: Enhancing LLM Agents to Manage and Utilize Memories via Reinforcement Learning** (2025, RL-for-memory) — Closest direct competitor for the RL-on-memory-write thesis. Uses trajectory-level reward; we propose per-op.
  → [Memory-R1: Enhancing LLM Agents to Manage and Utilize Memories via Reinforcement Learning](papers/2508.19828-memory-r1.md)
- **Your LLM Agents are Temporally Blind: The Misalignment Between Tool Use Decisions and Human Time Perception** (2025, temporal-reasoning) — Direct evidence LLMs can't perceive time without architectural changes. Our TemporalBench positioning derives from this.
  → [Your LLM Agents are Temporally Blind: The Misalignment Between Tool Use Decisions and Human Time Perception](papers/2510.23853-temporally-blind.md)
- **Hindsight is 20/20: Building Agent Memory that Retains, Recalls, and Reflects** (2025, write-time-compression) — Second-place LoCoMo SOTA. Industrial, from Vectorize.io. Notable that their own baseline (Backboard, 90%) marginally beats their full system (89.61%) — a striking honesty signal.
  → [Hindsight is 20/20: Building Agent Memory that Retains, Recalls, and Reflects](papers/2512.12818-hindsight.md)
- **DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models** (2024, theory) — The RL algorithm we use everywhere. Every modern tool-use paper uses GRPO.
  → [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](papers/2402.03300-deepseekmath-grpo.md)
- **Evaluating Very Long-Term Conversational Memory of LLM Agents** (2024, benchmark) — Our primary training and eval set. Has gold evidence labels which most benchmarks lack.
  → [Evaluating Very Long-Term Conversational Memory of LLM Agents](papers/2402.17753-locomo.md)
- **LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory** (2024, benchmark) — Primary eval target. Mastra OM hit 94.87% with gpt-5-mini; OMEGA 95.4%. Our headline number lives here.
  → [LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory](papers/2410.10813-longmemeval.md)
- **Zep: A Temporal Knowledge Graph Architecture for Agent Memory** (2024, substrate) — Closest existing thing to time-aware memory primitives. Their bi-temporal model is what we'd extend with derivation propagation.
  → [Zep: A Temporal Knowledge Graph Architecture for Agent Memory](papers/2501.13956-zep.md)
- **Cognitive Architectures for Language Agents (CoALA)** (2023, theory) — Foundational theoretical framing for memory types (episodic, semantic, procedural). Every modern memory paper inherits this taxonomy implicitly.
  → [Cognitive Architectures for Language Agents (CoALA)](papers/2309.02427-coala.md)

## medium  (10)

- **TiMem: Temporal-Hierarchical Memory Consolidation for Long-Horizon Conversational Agents** (2026, write-time-compression) — NOT current LoCoMo SOTA — EverMemOS at 93.05%, Hindsight at 89.61%, Amory at 87.7% all beat TiMem's 75.30% by 12-18pp. Useful as a prompt-only reference implementation, but not the bar to beat. (Initial claim of 'SOTA' was wrong — corrected via paper_leaderboard verification 2026-05-26.)
  → [TiMem: Temporal-Hierarchical Memory Consolidation for Long-Horizon Conversational Agents](papers/2601.02845-timem.md)
- **FadeMem: Biologically-Inspired Forgetting for Efficient Agent Memory** (2026, write-time-compression) — Explicit forgetting is a primitive most systems handwave. Biologically-grounded decay is closer to right than ad-hoc TTL.
  → [FadeMem: Biologically-Inspired Forgetting for Efficient Agent Memory](papers/2601.18642-fadememem.md)
- **SCM: Sleep-Consolidated Memory with Algorithmic Forgetting for Large Language Models** (2026, write-time-compression) — Same sleep-consolidation framing as Auto-Dreamer but no learning; importance-tagging is heuristic. More of a research preview than a publishable system.
  → [SCM: Sleep-Consolidated Memory with Algorithmic Forgetting for Large Language Models](papers/2604.20943-scm-sleep.md)
- **τ²-Bench: Evaluating Conversational Agents in a Dual-Control Environment** (2025, benchmark) — Industry-applied multi-agent eval. Strong signal if we want enterprise customer pitch.
  → [τ²-Bench: Evaluating Conversational Agents in a Dual-Control Environment](papers/2506.07982-tau2-bench.md)
- **Harnessing Temporal Databases for Systematic Evaluation of Factual Time-Sensitive Question-Answering in LLMs** (2025, benchmark) — Closest thing to a 'what was true on date X' factual benchmark. The time-accuracy-vs-answer-accuracy split is the metric design any temporal-validity memory eval should copy.
  → [Harnessing Temporal Databases for Systematic Evaluation of Factual Time-Sensitive Question-Answering in LLMs](papers/2508.02045-tdbench.md)
- **RGMem: Renormalization Group-inspired Memory Evolution for Language Agents** (2025, write-time-compression) — Cross-domain inspiration (physics renormalization group). Hierarchical multi-scale memory is a real architectural pattern. Numbers strong but not SOTA.
  → [RGMem: Renormalization Group-inspired Memory Evolution for Language Agents](papers/2510.16392-rgmem.md)
- **Test of Time: A Benchmark for Evaluating LLMs on Temporal Reasoning** (2024, benchmark) — The right probe to cite when arguing that public temporal benchmarks are contaminated. Useful methodologically for any temporal eval we build.
  → [Test of Time: A Benchmark for Evaluating LLMs on Temporal Reasoning](papers/2406.09170-test-of-time.md)
- **ColPali: Efficient Document Retrieval with Vision Language Models** (2024, infrastructure) — If memory contains PDFs / screenshots / visual docs, ColPali is the SOTA retrieval. Relevant for the 'multimodal memory' direction the user asked about.
  → [ColPali: Efficient Document Retrieval with Vision Language Models](papers/2407.01449-colpali.md)
- **Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory** (2024, write-time-compression) — Reference industry implementation. Cited as the hand-coded baseline our paper attacks.
  → [Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory](papers/2504.19413-mem0.md)
- **Retrospex: Language Agent Meets Offline Reinforcement Learning Critic** (2024, RL-for-memory) — Uses experience replay via RL critic — different from consolidation but related. Validates the 'learn from past trajectories' direction.
  → [Retrospex: Language Agent Meets Offline Reinforcement Learning Critic](papers/2505.11807-retrospex.md)

## low  (1)

- **How Can Large Language Models Understand Spatial-Temporal Data?** (2024, temporal-reasoning) — Tangential to memory but completes the temporal taxonomy — the 'feed numeric time-series to an LLM' corner. Useful only if the work ever touches sensor/ST data.
  → [How Can Large Language Models Understand Spatial-Temporal Data?](papers/2401.14192-stg-llm.md)
