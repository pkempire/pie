# Application / Evaluation Map

Date: 2026-06-15

Purpose: choose which "AI memory" application is worth building and which public evaluation can make the work credible.

Core thesis:

```text
The system should improve agent continuity:

raw traces + current time + task + budget
-> select evidence
-> maintain current/valid state
-> choose answer / refresh / wait / interrupt / replan
-> log outcome
-> improve read/write/action policies offline
```

This is broad "AI memory," but the first release should not try to solve every memory problem. The first credible wedge is **long-running project continuity for agents**, because it combines memory, time, code, experiments, and measurable outcomes.

## Scope Decision

### Build Now

1. Long-running project continuity.
2. Long-term chat memory benchmark reporting.
3. Temporal tool-use / stale context.
4. Async planning and orchestration.
5. AI scientist / research-project memory.
6. Software architecture planner.
7. Literature/repo mining as support for the above.

### Defer

1. Enterprise/org memory.
2. Sales/CRM.
3. Full personal-life analytics.
4. Finance/subscriptions.
5. Visual/video memory except as a demo asset.

Reason: deferred areas have obvious product value but weaker public evals, heavier privacy/permissions risk, or expensive multimodal infra.

## Application Matrix

| Application | Best data sources | Public evals / benchmarks | Best SOTA wedge | First build |
|---|---|---|---|---|
| Long-running project continuity | Git repos, commits, run logs, docs, issues/PRs, agent traces | SWE-bench Verified, SWE-bench Pro, PaperBench, custom repo-continuation eval | Better next-session continuation and lower stale-claim rate than raw RAG/compaction under fixed budget | Repo ledger -> span index -> temporal project states -> continuation questions |
| Long-term chat / personal assistant memory | LongMemEval, LoCoMo, PersonaMem-style profiles, private chat exports | LongMemEval, LongMemEval-V2, LoCoMo, PersonaMem-v2 | Accuracy-per-token and NoReplay-budget, not only headline full-context accuracy | Finish LongMemEval matrix, budget curves, evidence audits |
| Temporal tool-use / stale context | TicToc scenarios, tool-call cache traces, API/web snapshots | TicToc, Real-Time Deadlines | Correct answer-vs-refresh-vs-wait decisions as elapsed wall-clock time changes | TicToc adapter with action-timing metrics |
| Async planning / orchestration | Robotouille traces, job queues, subagent logs, calendar/tasks | Robotouille, MultiAgentBench, AgentBench | Active-process state improves async planning and interruption handling | Robotouille active-process adapter |
| AI scientist / auto-research | PaperBench, AstaBench, PDFs, code repos, experiment logs | PaperBench, AstaBench, AI Scientist-style review loops | Preserve working theory, negative results, and evidence across research cycles | Research-object store and mini PaperBench continuation tasks |
| Software architecture planner | Local repos, GitHub issues/PRs, dependency graphs, reference repos | RepoBench, ExecRepoBench, SWE-bench Verified, SWE-bench Pro | Better repository context selection before implementation | Code span index with symbols/imports/tests/diffs |
| Universal context injection app | User-selected repos/docs/chats, browser tabs, calendar/tasks | GAIA, OSWorld, custom user-study/context-acceptance eval | Product metric: fewer repeated explanations, more accepted context packs, lower task time | Local app/extension: preview, edit, copy/inject context |
| Web agents with freshness | WebArena, VisualWebArena, Mind2Web, browser traces | WebArena, VisualWebArena, Mind2Web, OSWorld | Point-in-time cache freshness policy for web agents | Time-aware cache state over web observations |
| Sales / CRM cycle intelligence | CRM exports, emails, call transcripts, calendar, stage history | CRMArena-Pro, tau2-bench, SalesLLM | Temporal account state and next-best-action under CRM policy | Later: synthetic CRM eval first |
| Enterprise / org memory | Slack/Teams, Drive/SharePoint, GitHub/Jira, meetings, tickets | TheAgentCompany, OSWorld, CRMArena-Pro, tau2-bench | Permissions-aware organizational state with evidence | Later: requires permission/redaction/audit model |
| Planning fallacy reduction | Git commits, task history, calendar, time tracking, CI/build durations | Robotouille, Real-Time Deadlines, custom plan-vs-actual eval | Calibrated duration predictions from prior plans and outcomes | Repo plan dataset: estimate -> actual -> belief update |
| Proactive follow-up agent | Calendar/tasks, email threads, issue status, active_process logs | TicToc, tau2-bench, custom interruption-preference eval | Low false-positive wait/interrupt/refresh decisions | Start as daily briefing, not autonomous interruptions |
| Calendar/task agent | Calendar, tasks, email, notes, project ledger | Real-Time Deadlines, Robotouille, custom calendar assistant eval | Deadline-aware context/action policy | Daily briefing with due/blocked/waiting evidence |
| Literature review system | arXiv/Semantic Scholar PDFs, repos, tables/figures, citation graph | AstaBench, PaperBench, LitQA/PaperQA-style evals | Persistent working theory beats one-shot deep research on repeated novelty/claim checks | Use research wiki as corpus; citation-grounded claim eval |
| Open-source repo mining | GitHub repos, README/docs, dependency graphs, issues/PRs | RepoBench, ExecRepoBench, SWE-bench Verified, SWE-bench Pro | Architecture pattern retrieval improves implementation planning | Mine memory/agent repos into artifact/span index |
| Visual/video memory | Screen recordings, photos, OCR, frame embeddings | OSWorld, VisualWebArena, video QA evals | Multimodal temporal evidence retrieval | Defer unless tied to video/screen demo |
| Finance/subscription memory | Bank/Rocket Money CSV, email receipts, renewal dates | Mostly private/custom | Product utility only; weak public research path | Defer; deterministic privacy-first rules |
| Life simulation / personal analytics | Personal exports, calendar, photos, notes, health/location | Private eval; LoCoMo/LongMemEval are weak proxies | Longitudinal trajectory modeling | Defer; privacy-heavy and unclear first buyer |

## Most Valuable Public Benchmarks By Direction

### Memory / Personal Context

- [LongMemEval](https://arxiv.org/abs/2410.10813): best first public benchmark for long-term chat memory. Tests information extraction, multi-session reasoning, temporal reasoning, knowledge updates, and abstention.
- [LongMemEval-V2](https://arxiv.org/abs/2605.12493): more ambitious agent-memory setting with very large multimodal/web-agent histories. Harder and more relevant to "experienced operators."
- [LoCoMo](https://arxiv.org/abs/2402.17753): older but widely used long-term conversational memory benchmark with QA, summarization, and dialogue generation.
- [EverMemOS](https://arxiv.org/abs/2601.02163): not a benchmark, but a strong current system to compare against because it reports SOTA on LoCoMo/LongMemEval and uses memory lifecycle/recollection framing.

### Temporal Awareness / Timing

- [TicToc / Temporal Blindness](https://arxiv.org/html/2510.23853v2): closest direct eval for elapsed-time-sensitive tool-use decisions. The action space is basically "trust cached context or refresh."
- [Real-Time Deadlines](https://arxiv.org/abs/2601.13206): strongest clean argument that turn-based and wall-clock time are different. Good for video/paper motivation, less directly a product benchmark.
- [Robotouille](https://arxiv.org/abs/2502.05227): best public async-planning environment. ReAct GPT-4o reportedly drops from 47% sync to 11% async, leaving room for active-process memory.
- [Time-R1 / Time-Bench](https://arxiv.org/abs/2505.13508): useful for temporal reasoning/model training, but less directly about agent continuity.

### Software / Project Continuity

- [SWE-bench Verified](https://www.swebench.com/verified.html): industry-standard coding-agent eval, but increasingly saturated.
- [SWE-bench Pro](https://arxiv.org/abs/2509.16941): better for long-horizon tasks; public dataset and low scores make it more interesting for continuity/context systems.
- [RepoBench](https://openreview.net/forum?id=pPjZIOuQuF): repository-level retrieval/completion. Best fit for code span indexing and context selection.
- [PaperBench](https://arxiv.org/abs/2504.01848): best bridge from software agents to AI-science replication.

### Multi-Agent / Enterprise Agents

- [MultiAgentBench](https://arxiv.org/abs/2503.01935): best academic multi-agent benchmark, with milestone-based KPIs and topology comparisons.
- [tau2-bench](https://arxiv.org/abs/2506.07982): best production-like conversational agent benchmark with shared state and dual control.
- [CRMArena-Pro](https://arxiv.org/html/2505.18878v1): strongest public CRM/business-agent benchmark.
- [GAIA](https://arxiv.org/abs/2311.12983): broad assistant/tool-use benchmark, useful for context-injection product claims but not memory-specific.

### Web / Computer Use

- [WebArena](https://arxiv.org/abs/2307.13854): realistic self-hosted web-agent tasks.
- [VisualWebArena](https://jykoh.com/vwa): multimodal web-agent tasks.
- [Mind2Web](https://osu-nlp-group.github.io/Mind2Web/): broad website action prediction/execution data.
- [OSWorld](https://arxiv.org/abs/2404.07972): real desktop/computer-use tasks; high product relevance but heavy to run.

### AI Scientist / Research

- [PaperBench](https://arxiv.org/abs/2504.01848): most credible public benchmark for AI agents replicating ML papers.
- [AstaBench](https://allenai.org/asta/bench): science-agent benchmark suite and leaderboards.
- [AI Scientist-v2](https://arxiv.org/abs/2504.08066): architecture reference for agentic tree search and experiment-manager loops.
- [Kosmos / Edison Scientific](https://edisonscientific.com/news/accelerating-science-at-scale): product/reference signal for persistent world models in AI science; not a public benchmark.

## Recommended First Research Claim

Do not claim:

```text
We solved AI memory.
```

Claim:

```text
Time-conditioned context/action policies improve long-running agent continuation
under fixed context/tool budgets.
```

Minimum evidence package:

1. LongMemEval budget curve: raw RAG vs timeline synthesis vs consolidated memory.
2. TicToc: action timing alignment for refresh/trust decisions.
3. Robotouille mini: active-process state improves async tasks.
4. Repo-continuation custom eval: our own project history, hand-labeled, reproducible.

If all four move in the same direction, the project has a real spine.

