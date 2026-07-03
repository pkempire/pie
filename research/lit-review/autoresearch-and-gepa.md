# Autoresearch SOTA + GEPA fit

Compiled 2026-05-25. All claims web-sourced; URLs inline.

---

## 1. SOTA "autoresearch" systems (May 2026)

The field has bifurcated into **closed-loop optimization within a known design space** ("Karpathy-style") and **open-ended discovery agents** ("AI Scientist / Co-Scientist style").

**Most mature, published artifacts:**

- **Google DeepMind Co-Scientist** — graduated from demo to a *Nature* paper announced 2026-05-19. Multi-agent idea tournament (generate / critique / refine / prioritize). Wet-lab validated: helped Gary Peltz's lab find a drug-repurposing candidate that blocked 91% of a liver-fibrosis scarring response (published in *Advanced Science*). Now rolled out via Gemini for Science. https://deepmind.google/blog/co-scientist-a-multi-agent-ai-partner-to-accelerate-research/ ; https://labcritics.com/blog/2026/05/21/google-deepminds-co-scientist-graduates-from-research-demo-to-nature-paper/
- **Sakana AI Scientist-v2** (arXiv 2504.08066, Apr 2025; *Nature* paper early 2026) — agentic tree-search, no template. First fully AI-generated paper to pass workshop peer review at ICLR 2025 (one of three submissions, avg score 6.33, withdrawn per pre-agreement). Caveat: workshop track (~60–70% accept rate), not main conference. Code: https://github.com/SakanaAI/AI-Scientist-v2 . https://sakana.ai/ai-scientist-nature/
- **AI2 Asta / AstaBench / AutoDiscovery** — Asta is a literature-grounded assistant over 108M abstracts + 12M full-texts. AutoDiscovery (released Feb 2026, formerly AutoDS) autonomously generates hypotheses and uses **Bayesian surprise** to prioritize. AstaBench (arXiv 2510.21652) tests 57 agents across 22 classes on 2.4K problems. Leaderboard (spring 2026 update): Claude Opus 4.7 = 58.0% @ $3.54/problem, GPT-5.5 = 52.9% @ $1.61. Adopted by UK AISI, Elicit, SciSpace, EvoScientist. https://allenai.org/blog/autodiscovery ; https://allenai.org/blog/astabench-update-spring-2026
- **DeepMind AlphaEvolve** (May 2025) — FunSearch's successor. Hundreds-of-lines programs vs FunSearch's snippets. Shipped real optimizations inside Google (data-center scheduling, matrix-mul kernels). https://www.technologyreview.com/2025/05/14/1116438/
- **Karpathy autoresearch** (2026-03-07, github.com/karpathy/autoresearch) — 66K+ stars within a month. Single editable file + frozen evaluator + scalar metric + keep-or-revert loop. Karpathy's 2-day run: 700 experiments, 20 additive wins, GPT-2-time from 2.02h → 1.80h. Spreading into kernel tuning, build-time reduction. Limitation flagged: works inside a known design space; fails on novel architectures. Shopify's 53% speed claim flagged as overfit. https://github.com/karpathy/autoresearch ; https://www.techtimes.com/articles/316804/20260519/
- **EvoScientist** (arXiv 2603.08127, Mar 2026) — three-agent (Researcher / Engineer / EvolutionManager) with persistent ideation + experimentation memory. Beats 7 baselines on novelty/feasibility/relevance/clarity.
- **HF ml-intern** (2026-04-21) — open-source full post-training loop. Pushed Qwen3-1.7B from 10% → 32% GPQA in <10h. Self-diagnoses reward collapse. https://github.com/huggingface/ml-intern
- **ChemCrow, Virtual Lab, Agent Laboratory** — older (2024–2025), narrower, still cited. Virtual Lab produced novel SARS-CoV-2 nanobody binders (real wet-lab output).

**Maturity ranking (shipped artifacts that exist in the world):**
1. Co-Scientist (peer-reviewed drug-repurposing hit + Nature paper)
2. AlphaEvolve (deployed inside Google infra)
3. Sakana v2 (first AI-authored peer-reviewed workshop paper, Nature meta-paper)
4. Virtual Lab (wet-lab nanobodies)
5. AI2 Asta/AstaBench (best benchmark + dataset infra, lighter discovery claims)

---

## 2. GEPA + autoresearch — has anyone combined them?

**Thin answer.** GEPA itself (Agrawal et al., arXiv 2507.19457, ICLR 2026 Oral, https://openreview.net/forum?id=RQm2KQTM5r) is a reflective prompt optimizer: reads execution traces, diagnoses failures in NL, maintains a Pareto frontier of candidates. Beats GRPO by 6–20% with 35× fewer rollouts; beats MIPROv2 by 10–13%.

**Direct GEPA × autoresearch evidence is sparse and mostly blog-tier, not academic.** What exists:

- **Raja Patnaik, "LangGraph + DSPy + GEPA: Agentic Researcher with multi-stage prompt optimization"** (2025-10-23, https://www.rajapatnaik.com/blog/2025/10/23/langgraph-dspy-gepa-researcher) — production walkthrough, GEPA used to evolve prompts for a multi-stage research agent (search → read → synthesize → write). Blog, not paper.
- **Google ADK** — official Google Agent Development Kit ships built-in agent optimization powered by GEPA. Generic agent optimization, not autoresearch-specific.
- **SuperOptiX** — DSPy framework adding seven specialized GEPA feedback metrics (medical_accuracy, vulnerability_detection, multi_component_enterprise, etc.). General-purpose, no autoresearch-specific recipe.
- **ResearchPilot** (arXiv 2603.14629) uses DSPy modules for literature synthesis but does **not** use GEPA optimization.
- **EvoScientist's EvolutionManager** is conceptually GEPA-shaped (reflect → distill → reuse) but uses its own bespoke memory mechanism, not GEPA.
- No published paper combines GEPA with AstaBench, MLE-bench, or PaperBench as of 2026-05.

**Gap:** A paper "GEPA-optimized agent on AstaBench" or "GEPA on MLE-bench" does not exist. This is an open, fundable hole.

---

## 3. Right metric for autoresearch optimization

This is the load-bearing question. Survey of what's been tried:

| Reward signal | Used by | Strengths | Failure modes |
|---|---|---|---|
| **Code-execution correctness on held-out ML task** (val loss / accuracy / medal rate) | Karpathy autoresearch, MLE-bench (15 categories, 75 Kaggle comps), MLE-STAR (44% medal w/ Gemini-2.0-Flash vs AIDE 25.8%), HF ml-intern, ML-Agent (arXiv 2505.23723) | Cheap, scalar, ungameable, GEPA-compatible | Only rewards optimization within a known design space; novel architecture changes fail (Karpathy himself noted this) |
| **Paper-replication score** | PaperBench (OpenAI) — best agent 21.0% | Forces faithful end-to-end research | Expensive to evaluate; ceiling far below human |
| **Workshop peer-review score** | Sakana v2 (avg 6.33) | Closest to real scientific quality | Slow, expensive, not GEPA-iterable; one signal per several days of compute |
| **LLM-as-judge novelty/feasibility/relevance/clarity** | ResearchAgent, EvoScientist, BioVerge (arXiv 2511.08866), Co-Scientist tournament | Cheap, fast, multi-axis | Reward-hacking risk; correlates loosely with real value |
| **Bayesian surprise on observed data** | AI2 AutoDiscovery | Principled "interestingness" signal | Only works on data-driven discovery, not theory |
| **ELO from idea tournaments** | Co-Scientist; Bayes-Entropy agents (arXiv 2508.01746) — +116.3 ELO, +17.8 over real abstracts after 12 iters | Naturally calibrated, robust to absolute-score drift | Needs many comparable candidates |
| **Process reward models (PRMs)** | HF ml-intern, AgentPRM (Web Conf 2026) — 8× more compute-efficient | Dense, stepwise credit | PRM training cost; calibration drift |
| **Dynamic importance / time-aware citation prediction** | Dyport (arXiv 2312.03303) for biomedical | Grounded in real impact | Slow ground truth (years); not useful for inner loops |
| **Rediscovery benchmarks** | FIRE-Bench (arXiv 2602.02905); ReplicationBench (astro, arXiv 2510.24591) | Ground truth exists | Memorization risk |

**What actually works as a GEPA reward:**
GEPA needs *something it can call many times*. The only signals dense and cheap enough are (a) code-execution correctness, (b) LLM-as-judge tournament ELO, (c) PRMs. Peer-review and wet-lab signals are too sparse — they belong at the *outer* loop. The right architecture is **two-loop**: GEPA optimizes prompts on cheap signals (code-exec + LLM-judge), and the system gets occasional ground-truth checkpoints from peer review / replication.

---

## 4. Architecture sketch — GEPA-optimized autoresearch in DSPy

Treat the system as a DSPy program of typed modules. Each module is a `dspy.Signature`; GEPA optimizes the instruction string of each module against a module-local metric, with a system-level metric for joint Pareto.

```
LiteratureReview(question: str) -> (claims: list[Claim], gaps: list[str], citations: list[Paper])
    metric: recall@k against expert-curated reading list; semantic-scholar citation overlap.

HypothesisGeneration(claims, gaps) -> (hypotheses: list[H], rationales: list[str])
    metric: LLM-judge tournament ELO on (novelty, feasibility, testability); +ground-truth ELO vs real abstracts (cf. Bayes-Entropy method, +116 ELO over 12 iters).

ExperimentDesign(hypothesis, budget) -> (plan: ExperimentPlan, code_skeleton: str)
    metric: (a) execution-success of generated skeleton; (b) judge score on identifiability/control-quality.

Execute(code_skeleton, data) -> (results: Results, traces: ExecTraces)
    not GEPA-optimized — frozen evaluator (Karpathy invariant). Provides the scalar.

ResultAnalysis(results, hypothesis) -> (verdict, effect_size, caveats)
    metric: agreement with held-out statistical ground truth on synthetic datasets w/ known effects.

Writeup(hypothesis, results, analysis) -> (paper: str)
    metric: PaperBench-style rubric score from LLM grader; secondary: workshop-judge ELO.
```

**Outer loop (GEPA):** for each module, sample a minibatch of tasks, run, collect execution traces + scalar metric + NL feedback (errors, judge rationales), GEPA reflects, proposes new instruction, keeps Pareto frontier across the 6 modules jointly. Karpathy keep-or-revert is the inner-loop primitive *inside* `Execute`.

**System-level metric** (the one GEPA's Pareto is computed against): weighted combo of (i) end-to-end task-completion on AstaBench / MLE-bench-lite, (ii) judge-ELO of final writeup, (iii) cost-per-success. AstaBench is the most defensible eval because it already separates literature, code, dataset-analysis, and end-to-end discovery.

This is exactly the missing artifact in §2.

---

## 5. Educational content landscape (May 2026)

**On agent memory:**
- **Karpathy "LLM Wiki" gist (April 2026)** is the dominant idea. Treat knowledge as code you compile, not RAG you retrieve. Spawned: Nate Herk's live-build (Claude Code wiki-fies 36 YouTube transcripts in ~14 min), `agentmemory` by Rohit Ghumare (Apache-2.0), and a wave of "beyond RAG" posts on Level Up Coding / Gamgee / INovaBeing / Frank's World. Most-viewed content here is YouTube live-builds, not papers. https://akitaonrails.com/en/2026/05/18/ai-agent-memory-karpathy-llm-wiki-agentmemory/ ; https://levelup.gitconnected.com/beyond-rag-how-andrej-karpathys-llm-wiki-pattern-builds-knowledge-that-actually-compounds-31a08528665e
- **Lex Fridman #490 "State of AI in 2026"** (Feb 2026) w/ Nathan Lambert + Sebastian Raschka covered tool-use, continual learning, long context — broad but shallow on memory. https://lexfridman.com/ai-sota-2026/
- Karpathy has *not* done a 2026 Lex episode; last was the DeepSeek one in Feb 2025.
- Yannic Kilcher / AI Coffee Break / Jay Alammar: no significant 2026 coverage of autoresearch surfaced in searches. Yannic is doing org-prep-for-agents content; Jay is on RAG-with-Cohere. **This is the gap.**

**On research automation:**
- **DeepLearning.AI "DSPy: Build and Optimize Agentic Apps"** (Chen Qian, w/ Databricks) — the canonical course.
- **DataCamp & Verdent guides** on Karpathy autoresearch — most-trafficked tutorials, written for ML practitioners not researchers.
- **ICLR 2026 Oral GEPA talk** on YouTube (https://www.youtube.com/watch?v=HbGah-uP1fI) — primary source, niche audience.
- **HuggingFace cookbook for DSPy GEPA** (https://huggingface.co/learn/cookbook/en/dspy_gepa) — most accessible technical recipe.

**Underserved audience / content gap (your opening):**
- **"How to combine GEPA + autoresearch end-to-end"** — no flagship tutorial exists. The Patnaik blog is the closest, but it stops at a research agent, doesn't go to AstaBench-style eval, doesn't show the keep-or-revert inner loop.
- **"Reward signals for autoresearch, ranked"** — no one has written the table from §3.
- **Technical-but-accessible memory + autoresearch** crossover: Karpathy owns memory, Sakana/AI2 own discovery, no one bridges. A "smart-12-year-old-with-math" walkthrough of how memory feeds an autoresearch loop would have no competition. AI Coffee Break and Yannic are the natural venues but haven't covered it.
- **Video walk-through of building the §4 architecture in a notebook** — doesn't exist.

---

## Key URLs (high-signal)

- GEPA paper: https://arxiv.org/abs/2507.19457
- GEPA repo: https://github.com/gepa-ai/gepa
- DSPy GEPA docs: https://dspy.ai/api/optimizers/GEPA/overview/
- AstaBench paper: https://arxiv.org/abs/2510.21652
- AI2 AutoDiscovery: https://allenai.org/blog/autodiscovery
- Sakana v2 paper: https://arxiv.org/pdf/2504.08066
- Co-Scientist Nature announcement: https://labcritics.com/blog/2026/05/21/google-deepminds-co-scientist-graduates-from-research-demo-to-nature-paper/
- Karpathy autoresearch: https://github.com/karpathy/autoresearch
- HF ml-intern: https://github.com/huggingface/ml-intern
- EvoScientist: https://arxiv.org/abs/2603.08127
- Patnaik LangGraph+DSPy+GEPA researcher: https://www.rajapatnaik.com/blog/2025/10/23/langgraph-dspy-gepa-researcher
- MLE-bench: https://github.com/openai/mle-bench
- PaperBench: https://cdn.openai.com/papers/22265bac-3191-44e5-b057-7aaacd8e90cd/paperbench.pdf
