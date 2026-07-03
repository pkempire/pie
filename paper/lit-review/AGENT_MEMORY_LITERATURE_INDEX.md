# Agent Memory Literature Index

This is the scannable literature map for the project. It intentionally mixes peer-reviewed papers, arXiv preprints, major engineering posts, and product whitepapers because agent memory is moving through all four channels.

Confidence legend:

- `verified`: checked from the linked source during the May 2026 reset.
- `repo-note`: present in our repo notes, but not freshly re-verified.
- `user-link`: provided by user; needs deeper reading if it becomes central.

## Current Read Of The Field

The crowded part of the field is structured memory:

- Temporal KGs: Zep, AriadneMem, TSM.
- Hierarchical temporal memory: TiMem, EverMemOS, MemVerse.
- Multi-graph memory: MAGMA.
- Prompted/product memory: Mem0, Mastra, Letta, Honcho, OMEGA.

The more interesting "bitter lesson" direction is learned representation/control:

- Learned segmentation: H-Net.
- Learned offline consolidation: Auto-Dreamer.
- Learned read-time decompression: RLMs.
- Learned cache/latent representations: Cartridges, Titans.
- Joint training of retrieval/ranking/reasoning: CoSearch, Search-R1, Graph-R1.
- Unified memory tool-use policies: AgeMem.

Implication for mempol:

> A fixed ontology/write-tool KG is not enough. The strongest research direction is learned segmentation + learned consolidation + learned retrieval under a budgeted future-utility objective.

## Highest Relevance Papers

| Date | Work | Type | Core Idea | Evals / Results | What To Steal | Threat To Us | Status |
|---|---|---|---|---|---|---|---|
| 2026-05-20 | [Auto-Dreamer](https://arxiv.org/abs/2605.20616) | offline consolidation | Decouples fast per-session acquisition from slow learned consolidation over memory regions. | ScienceWorld +7 over strongest baseline, 12x smaller bank; transfer to ALFWorld/WebArena with 6x less memory on ALFWorld. | Region rewriting, provenance-linked evidence, counterfactual utility at consolidation level. | Makes per-turn writer RL look small and outdated. | verified |
| 2025-07-10 | [H-Net / Dynamic Chunking](https://arxiv.org/abs/2507.07955) | learned segmentation | Learns content/context-dependent chunks end-to-end; byte-level H-Net beats BPE transformer compute/data matched. | Better scaling; nearly 4x data efficiency in weaker-tokenization domains like Chinese/code/DNA. | Stop handcoding chunk size; train boundary detector and memory regions. | Shows fixed chunks/ontologies are anti-bitter-lesson. | verified |
| 2025-06-06 | [Cartridges](https://arxiv.org/abs/2506.06266) | learned latent memory | Trains small KV caches offline via self-study over a corpus. | Matches ICL while using 38.6x less memory and 26.4x higher throughput; extends 128k to 484k on MTOB. | Self-study synthetic conversations; amortized corpus-specific memory artifacts. | External text memory may be beaten by latent/cache memory for fixed corpora. | verified |
| 2025/2026 | [Recursive Language Models as Memory Systems](https://raw.works/recursive-language-models-as-memory-systems/?ref=footer), [code](https://github.com/rawwerks/longmemeval-rlm) | read-time decompression | Let the model recursively inspect chunks and aggregate structured results at query time. | Repo notes cite high LongMemEval performance with DSPy scaffolding; verify exact numbers before publication. | Learned/agentic read-side decompression; tool recursion instead of pre-extracting everything. | Raw store + smart reader may beat write-time extraction. | user-link |
| 2026-04-19 | [CoSearch](https://arxiv.org/abs/2604.17555) | joint RL retrieval | Jointly trains reasoning agent and generative document ranker with GRPO. | 7 QA benchmarks; oracle vs fixed retrieval gap up to +26.8% relative F1. | Train retriever/ranker too; do not freeze retrieval as a dumb tool. | Our fixed read/retrieval policy becomes bottleneck. | verified |
| 2026-01-05 | [AgeMem](https://arxiv.org/abs/2601.01885) | RL memory management | Integrates LTM/STM operations directly into agent policy with step-wise GRPO. | 5 long-horizon benchmarks; claims better task performance, memory quality, context efficiency. | Unified memory ops as tool actions; progressive RL curriculum. | Direct competitor to learned write/read policies. | verified |
| 2026-03-05 | [AriadneMem](https://arxiv.org/abs/2603.03290) | structured lifelong memory | Entropy-aware gating, conflict-aware coarsening, temporal edges, bridge discovery. | LoCoMo GPT-4o: Multi-Hop F1 +15.2%, Average F1 +9.0%, runtime -77.8%, 497 context tokens. | Entropy-aware gating, bridge discovery, coarsening while preserving transitions. | Already claims disconnected evidence + state update solution. | verified |
| 2026-01-06 | [TiMem](https://arxiv.org/abs/2601.02845) | temporal hierarchy | Temporal Memory Tree from raw observations to persona-level abstraction. | LoCoMo 75.30%, LongMemEval-S 76.88%, recalled memory length -52.20% on LoCoMo. | Hierarchical consolidation and complexity-aware recall. | Occupies "temporal hierarchical memory" framing. | verified |
| 2026-01-05 | [EverMemOS](https://arxiv.org/abs/2601.02163) | memory OS | MemCells, MemScenes, semantic consolidation, reconstructive recollection. | LoCoMo + LongMemEval SOTA claims; profile study on PersonaMem v2. | Lifecycle framing; memory scenes; reconstructive recall. | Similar "memory OS" language, crowded claim space. | verified |
| 2026-01-12 | [Temporal Semantic Memory](https://arxiv.org/abs/2601.07468) | temporal semantic memory | Separates semantic time from dialogue time; consolidates durative memory. | Needs deeper table read. | Semantic occurrence time, durative memory. | Directly weakens "temporal state" novelty. | verified |
| 2026-01-06 | [MAGMA](https://arxiv.org/abs/2601.03236) | multi-graph memory | Represents each memory across semantic, temporal, causal, and entity graphs; policy-guided traversal. | LoCoMo + LongMemEval; claims SOTA over agentic memory systems. | Orthogonal graph views + query-adaptive traversal. | Multi-graph is already claimed. | verified |

## Memory Benchmarks And Systems

| Date | Work | Type | Core Idea | Evals / Results | What To Steal | Status |
|---|---|---|---|---|---|---|
| 2024-02-27 | [LoCoMo](https://arxiv.org/abs/2402.17753) | benchmark | Long conversational memory with evidence labels and temporal/multi-hop questions. | Primary current repo dataset. | Evidence labels for reward/debugging; category breakdowns. | repo-note |
| 2024-10-14 | [LongMemEval](https://arxiv.org/abs/2410.10813) | benchmark | Long-term interactive memory for chat assistants. | 500 questions; common external leaderboard target. | Standard judge protocol and categories. | repo-note |
| 2025-07-07 | [MemoryAgentBench](https://arxiv.org/abs/2507.05257) | benchmark | Incremental multi-turn memory evaluation. | Needs deeper read. | More realistic interactive memory eval. | verified |
| 2026-?? | [BEAM](https://arxiv.org/abs/2510.27246) | benchmark | Large-scale memory stress benchmark. | Needs re-open/read. | Stress memory beyond context saturation. | repo-note |
| 2025-01-?? | [Zep / Graphiti paper](https://arxiv.org/abs/2501.13956), [GitHub](https://github.com/getzep/graphiti?tab=readme-ov-file) | temporal KG | Bi-temporal KG for agent memory. | DMR and LongMemEval claims vary by source. | Valid-time + transaction-time modeling. | verified/repo-note |
| 2025-04-?? | [Mem0](https://arxiv.org/abs/2504.19413) | product memory | Scalable long-term memory with extraction/update/delete style ops. | Product/research baseline. | API shape; pragmatic baselines. | repo-note |
| 2026-02 | [OMEGA benchmark](https://omegamax.co/benchmarks#whitepaper) | product/system | Local memory system benchmarked on LongMemEval. | Claims 95.4%, 466/500 on LongMemEval; category scores reported. | Local CPU/FTS/vector architecture; MemoryStress idea. | verified |
| 2026 | [Honcho](https://honcho.dev/) | product/system | User/peer representation and hosted memory layer. | Needs deeper docs read; prior repo notes cite strong claims. | Identity-model/product framing. | verified homepage only |
| 2025-09-29 | [Anthropic Context Engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) | engineering post | Context is finite; use compaction, structured notes, subagents, just-in-time context. | Not a benchmark paper. | Minimal high-signal context; note-taking; progressive disclosure. | verified |
| 2025-12-11 | [Letta Continual Learning in Token Space](https://www.letta.com/blog/continual-learning) | engineering/research post | Optimize learned context `C`, not just model weights; token-space memory is inspectable, portable, controllable. | Not benchmark paper. | Token-space continual learning formalism; sleep-time compute. | verified |
| 2026 | [Letta Context Repositories](https://www.letta.com/blog/context-repositories) | engineering post | Git-style context repositories for coding agents. | Needs deeper extraction. | Versioned memory/workspace as repo. | verified homepage only |

## RL, Search, And Tool-Use

| Date | Work | Type | Core Idea | Evals / Results | What To Steal | Status |
|---|---|---|---|---|---|---|
| 2024-02 | [DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300) | RL method | Group-relative policy optimization. | Math reasoning. | GRPO as trainer scaffold. | repo-note |
| 2025-03-12 | [Search-R1](https://arxiv.org/abs/2503.09516) | RL tool-use | Train LLMs to reason and use search with RL. | NQ, HotpotQA, 2Wiki, TriviaQA etc. | Search/action environment shape. | repo-note |
| 2025-07-25 / 2026-02-14 v2 | [GEPA](https://arxiv.org/abs/2507.19457), [code](https://github.com/gepa-ai/gepa) | reflective prompt evolution | Evolves prompts/program text from rollout traces using natural-language reflection plus Pareto candidate selection. | Six tasks; reports +6% avg vs GRPO and up to 35x fewer rollouts; accepted ICLR 2026 Oral. | Use trace reflection before/alongside RL: evolve memory-tool instructions, reward rubrics, and retrieval/write policies from failed LoCoMo trajectories. | If prompt/program evolution gets most of the gain cheaply, pure weight-space RL is not the first lever. | verified |
| 2025-07-29 | [Graph-R1](https://arxiv.org/abs/2507.21892) | RL GraphRAG | End-to-end RL for graph retrieval interactions. | Standard RAG datasets; claims better accuracy/retrieval efficiency. | Treat graph retrieval as multi-turn action process. | verified |
| 2025-08 | [Memory-R1](https://arxiv.org/abs/2508.19828) | RL memory | Memory Manager with ADD/UPDATE/DELETE/NOOP and Answer Agent via PPO/GRPO. | LoCoMo, LongMemEval, MSC. | Direct baseline; trajectory reward comparison. | repo-note |
| 2025-09 | Mem-alpha | RL memory | Learning memory construction via RL. | Needs re-verification. | Long-horizon memory construction. | repo-note |
| 2026-04 | DeltaMem | RL memory | Memory edit operations with state-distance reward. | Needs re-verification. | Alternative to outcome counterfactual. | repo-note |
| 2026-01-14 | [EvoFSM](https://arxiv.org/abs/2601.09465) | self-evolving agents | Evolves explicit FSM: macro flow + state-specific skill; self-evolving memory. | 5 multi-hop QA benchmarks; DeepSearch 58.0%. | Controlled policy evolution without free-form prompt drift. | verified |

## Temporal Awareness, Temporal Reasoning, And Async Planning

| Date | Work | Type | Core Idea | Evals / Results | What To Steal | Status |
|---|---|---|---|---|---|---|
| 2024-01 | [STG-LLM](https://arxiv.org/abs/2401.14192) | spatial-temporal data | Translator/adapters for spatial-temporal graph data. | ST forecasting benchmarks. | Temporal graph translation. | repo-note |
| 2024-06 | [Test of Time](https://arxiv.org/abs/2406.09170) | temporal reasoning benchmark | Synthetic temporal logic benchmark controlling leakage. | Probe, not product memory. | Temporal reasoning eval design. | verified |
| 2025-01 | [TemporalVQA](https://arxiv.org/abs/2501.10674) | visual temporal benchmark | Tests temporal order/time-lapse estimation in MLLMs. | GPT-4o 49.1% average consistent accuracy on temporal order, 70% time-lapse. | Multimodal temporal failure evidence. | verified |
| 2025-02 | [Robotouille](https://arxiv.org/abs/2502.05227) | async planning benchmark | LLM agents fail when tasks include asynchronous side effects. | Repo notes: gpt-4o ReAct 47% sync -> 11% async; re-read table before citing. | Async state/orchestration benchmark. | repo-note |
| 2025-03 | [Time-R1 grounding](https://arxiv.org/abs/2503.13377) | temporal video grounding | RL/post-training for temporal video grounding. | Needs deeper read. | Curriculum/reward design for temporal grounding. | user-link |
| 2025-05 | [Time-R1](https://arxiv.org/abs/2505.13508) | temporal reasoning RL | Comprehensive temporal reasoning via RL curriculum. | Needs table extraction. | Curriculum: understanding -> prediction -> generation. | verified |
| 2025-06 | [Discrete Minds / Time Passes](https://arxiv.org/abs/2506.05790) | temporal awareness | Token-Time Hypothesis; dialogue duration, urgency, BombRush. | Finds models have some awareness of time passage, varying by size/reasoning. | Contrast with "temporally blind"; design evals carefully. | verified |
| 2025-08 | [TDBench](https://arxiv.org/abs/2508.02045) | temporal DB QA | Generates time-sensitive QA from temporal DB joins. | Needs deeper read. | Systematic temporal factual QA generation. | verified |
| 2025-10 | [Temporally Blind / TicToc](https://arxiv.org/abs/2510.23853) | temporal tool-use | Tool-use decisions misalign with human time perception. | TicToc scenarios; repo notes cite best 65% human alignment. | Staleness/re-fetch behavior eval. | repo-note |
| 2026-01 | [Real-Time Deadlines](https://arxiv.org/abs/2601.13206) | temporal awareness | Models handle turn budgets better than wall-clock deadlines; periodic time helps. | Repo notes: GPT-5.1 closure 4% -> 32% with time updates. | Time injection as missing state variable. | repo-note |

## World Models, Latent Memory, And Foundation-Level Ideas

| Date | Work | Type | Core Idea | Evals / Results | What To Steal | Status |
|---|---|---|---|---|---|---|
| 2025-01 | [Titans](https://arxiv.org/abs/2501.00663) | neural memory architecture | Learns to memorize at test time with neural long-term memory. | Long-context benchmarks; needs table read. | Model-internal memory direction. | verified |
| 2025-06 | [General agents contain world models](https://arxiv.org/abs/2506.01622) | theory | General goal-directed agents require predictive world models. | Theoretical. | Justification for state/world-model layer. | verified |
| 2025-09 | [Dreamer 4](https://arxiv.org/abs/2509.24527) | world model RL | Trains agents inside scalable learned world models; Minecraft diamonds offline. | Minecraft; >20k actions from offline data. | Imagination/replay framing, not directly chat memory. | verified |
| 2025-12 | [MemVerse](https://arxiv.org/abs/2512.03627) | multimodal memory | Plug-and-play hierarchical retrieval + parametric/non-parametric memory. | Needs deeper read. | Multimodal lifelong memory framing. | verified |
| 2026-04 | [Memory Intelligence Agent](https://arxiv.org/abs/2604.04503) | memory + RL | Integrates non-parametric and parametric memory with RL. | Search result only; needs read. | Hybrid parametric/non-parametric training. | search-only |

## Surveys / Meta

| Date | Work | Type | Core Idea | Why It Matters | Status |
|---|---|---|---|---|---|
| 2026-03 | [Memory for Autonomous LLM Agents](https://arxiv.org/abs/2603.07670) | survey | Mechanisms, evaluation, and emerging frontiers. | Use for related-work taxonomy sanity. | search-only |
| 2026-04 | [Memory in the LLM Era](https://arxiv.org/abs/2604.01707) | survey | Modular architectures and strategies unified framework. | Use for broad taxonomy. | search-only |

## Unresolved User Links To Read Deeper

These were provided but are not yet summarized enough to cite:

- [arXiv 2604.17555](https://arxiv.org/abs/2604.17555) is CoSearch and already included.
- [arXiv 2601.09465](https://arxiv.org/abs/2601.09465) is EvoFSM and already included.
- [arXiv 2506.06266](https://arxiv.org/abs/2506.06266) is Cartridges and already included.
- [arXiv 2506.05790](https://arxiv.org/abs/2506.05790) is Discrete Minds and already included.
- [arXiv 2501.10674](https://arxiv.org/abs/2501.10674) is TemporalVQA and already included.
- [arXiv 2404.12353](https://arxiv.org/abs/2404.12353) needs title/result extraction.
- [OpenReview jWBZdlU5Xl](https://openreview.net/pdf?id=jWBZdlU5Xl) needs title/result extraction.
- [OpenReview moWiYJuSGF](https://openreview.net/forum?id=moWiYJuSGF) needs title/result extraction.
- [UMD scholarly paper](https://www.cs.umd.edu/sites/default/files/scholarly_papers/202501_Cavolowsky%2C_Mark_Scholarly_Paper.pdf) needs title/result extraction.
- [arXiv 2601.02845](https://arxiv.org/abs/2601.02845) is TiMem and already included.
- [arXiv 2601.07468](https://arxiv.org/abs/2601.07468) is TSM and already included.
- [arXiv 2601.01885](https://arxiv.org/abs/2601.01885) is AgeMem and already included.

## Research-Quality Direction After This Review

The repo should pivot from:

```text
LLM extracts typed memories into KG -> reader queries KG
```

to:

```text
raw histories
  -> learned boundary detector
  -> candidate regions
  -> learned consolidator / compressor
  -> learned read-time decompressor / retriever
  -> future utility reward under token budget
```

Concrete next experiments:

1. Fixed-chunk vs learned/LLM boundary segmentation on LoCoMo.
2. Writer-only vs region consolidation at matched token budgets.
3. Raw RAG + RLM-style recursive reader vs consolidated memory + simple reader.
4. Consolidator reward via future-query pairwise preference, not exhaustive leave-one-out.
5. Personal export: use actual future user turns as weak labels for memory usefulness.

The paper becomes high-quality only if it has a real Pareto frontier:

```text
x-axis: memory/read tokens
y-axis: future task accuracy
curves: raw RAG, prompted extraction, KG, writer-only, learned consolidation, RLM reader
```

Without that table, the project is mostly architecture taste.
