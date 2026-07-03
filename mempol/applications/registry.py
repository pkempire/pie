"""Canonical application/evaluation map.

This module is intentionally lightweight: it turns the sprawling set of product
ideas into a typed registry that scripts, docs, and planning agents can query.
It is not a final ontology for memory. It is a planning artifact.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ApplicationTarget:
    name: str
    problem_solved: str
    best_data_sources: list[str]
    public_evals: list[str]
    sota_wedge: str
    first_build: str
    in_scope_now: bool
    notes: str = ""
    links: list[str] = field(default_factory=list)


APPLICATION_TARGETS: tuple[ApplicationTarget, ...] = (
    ApplicationTarget(
        name="Long-running project continuity",
        problem_solved="Keeps an agent oriented across repo changes, experiment logs, decisions, stale claims, blockers, and next actions.",
        best_data_sources=["Git repos", "commit history", "run logs", "docs", "issue/PR threads", "agent traces"],
        public_evals=["SWE-bench Verified", "SWE-bench Pro", "PaperBench", "custom repo-continuation eval"],
        sota_wedge="Show better task continuation and lower stale-claim rate than raw RAG/context compaction under a fixed budget.",
        first_build="Repo ledger -> span index -> temporal project states -> context/action pack -> hand-labeled continuation tasks.",
        in_scope_now=True,
        links=[
            "https://www.swebench.com/verified.html",
            "https://arxiv.org/abs/2509.16941",
            "https://arxiv.org/abs/2504.01848",
        ],
    ),
    ApplicationTarget(
        name="Long-term chat / personal assistant memory",
        problem_solved="Answers questions over long personal histories with updates, temporal constraints, and abstention.",
        best_data_sources=["LongMemEval", "LoCoMo", "PersonaMem-style profiles", "personal chat exports for private eval"],
        public_evals=["LongMemEval", "LongMemEval-V2", "LoCoMo", "PersonaMem-v2"],
        sota_wedge="Beat strong memory systems on accuracy-per-token or NoReplay-budget, not just headline full-context accuracy.",
        first_build="Finish LongMemEval matrix, add budget curves, add exact evidence audits, compare against EverMemOS/Mem0-style baselines.",
        in_scope_now=True,
        links=[
            "https://arxiv.org/abs/2410.10813",
            "https://arxiv.org/abs/2605.12493",
            "https://arxiv.org/abs/2402.17753",
            "https://arxiv.org/abs/2601.02163",
        ],
    ),
    ApplicationTarget(
        name="Temporal tool-use / stale context",
        problem_solved="Decides when cached context/tool results are still valid vs when the agent must refresh.",
        best_data_sources=["TicToc scenarios", "web/API cache traces", "tool-call logs with elapsed time"],
        public_evals=["TicToc", "Real-Time Deadlines"],
        sota_wedge="Train action timing directly: answer vs refresh vs wait vs interrupt, conditioned on elapsed wall-clock time.",
        first_build="Implement TicToc adapter with our read/action policy and report alignment, unnecessary refreshes, and stale-use errors.",
        in_scope_now=True,
        links=[
            "https://arxiv.org/html/2510.23853v2",
            "https://arxiv.org/abs/2601.13206",
        ],
    ),
    ApplicationTarget(
        name="Async planning and orchestration",
        problem_solved="Tracks overlapping waits, resource locks, deadlines, interruptions, and partial progress.",
        best_data_sources=["Robotouille traces", "multi-agent task traces", "CI/job queues", "calendar/task logs"],
        public_evals=["Robotouille", "MultiAgentBench", "AgentBench"],
        sota_wedge="Use explicit active processes and time-conditioned replanning to close async-planning gaps.",
        first_build="Adapter for Robotouille: active_process state for each async action; policy decides wait/replan/act.",
        in_scope_now=True,
        links=[
            "https://arxiv.org/abs/2502.05227",
            "https://arxiv.org/abs/2503.01935",
            "https://openreview.net/forum?id=zAdUB0aCTQ",
        ],
    ),
    ApplicationTarget(
        name="AI scientist / auto-research memory",
        problem_solved="Preserves hypotheses, failed experiments, paper claims, code evidence, and open cruxes across research cycles.",
        best_data_sources=["PaperBench", "AstaBench", "research PDFs", "experiment logs", "code repos", "review rubrics"],
        public_evals=["PaperBench", "AstaBench", "AI Scientist-style generated-paper review loops"],
        sota_wedge="Improve replication/research-progress score by preserving evidence-backed working theory and negative results.",
        first_build="Research-object store over papers/repos/logs; eval on PaperBench mini tasks or our own paper-to-code continuation tasks.",
        in_scope_now=True,
        links=[
            "https://arxiv.org/abs/2504.01848",
            "https://allenai.org/asta/bench",
            "https://arxiv.org/abs/2504.08066",
        ],
    ),
    ApplicationTarget(
        name="Software architecture planner",
        problem_solved="Finds relevant repo patterns, APIs, constraints, tests, and design decisions before coding.",
        best_data_sources=["local repos", "GitHub issues/PRs", "dependency graphs", "open-source reference repos"],
        public_evals=["RepoBench", "SWE-bench Verified", "SWE-bench Pro", "ExecRepoBench"],
        sota_wedge="Better repository context selection before code generation or bug fixing.",
        first_build="Code span index: symbols, imports, tests, callsites, diffs; query-conditioned architecture context pack.",
        in_scope_now=True,
        links=[
            "https://openreview.net/forum?id=pPjZIOuQuF",
            "https://www.swebench.com/verified.html",
            "https://execrepobench.github.io/",
        ],
    ),
    ApplicationTarget(
        name="Universal context injection app",
        problem_solved="Lets a user inject the right context into Claude/Codex/ChatGPT from connected sources.",
        best_data_sources=["user-selected repos/docs/chats", "browser tabs", "calendar/tasks", "project ledger"],
        public_evals=["GAIA", "OSWorld", "custom user-study/context-acceptance eval"],
        sota_wedge="Product metric, not pure benchmark: fewer repeated explanations, more accepted context packs, lower task time.",
        first_build="Local app/extension: pick task -> preview context pack -> copy/inject -> log accepted/edited context.",
        in_scope_now=True,
        links=[
            "https://arxiv.org/abs/2311.12983",
            "https://arxiv.org/abs/2404.07972",
        ],
    ),
    ApplicationTarget(
        name="Web agents with freshness",
        problem_solved="Avoids stale cached web/tool observations and redundant refreshes during browser tasks.",
        best_data_sources=["WebArena", "VisualWebArena", "Mind2Web/Online-Mind2Web", "browser traces"],
        public_evals=["WebArena", "VisualWebArena", "Mind2Web", "OSWorld"],
        sota_wedge="Point-in-time freshness policy layered on top of web agents.",
        first_build="Add time-aware cache state to WebArena-style tool observations; score stale use vs wasted refresh.",
        in_scope_now=False,
        links=[
            "https://arxiv.org/abs/2307.13854",
            "https://jykoh.com/vwa",
            "https://osu-nlp-group.github.io/Mind2Web/",
        ],
    ),
    ApplicationTarget(
        name="Sales / CRM cycle intelligence",
        problem_solved="Tracks accounts, stakeholders, objections, stage changes, follow-up windows, and stale opportunities.",
        best_data_sources=["CRM exports", "sales emails", "call transcripts", "calendar", "deal stage history"],
        public_evals=["CRMArena-Pro", "tau2-bench", "SalesLLM / sales-specific roleplay evals"],
        sota_wedge="Temporal account-state tracking and next-best-action under CRM policies.",
        first_build="Do not start here unless using synthetic CRM eval; private data makes research hard.",
        in_scope_now=False,
        links=[
            "https://arxiv.org/html/2505.18878v1",
            "https://arxiv.org/abs/2506.07982",
            "https://arxiv.org/html/2604.07054v1",
        ],
    ),
    ApplicationTarget(
        name="Enterprise / org memory",
        problem_solved="Maintains project/account/team state across docs, meetings, chats, tickets, and repos.",
        best_data_sources=["Slack/Teams", "Drive/SharePoint", "GitHub/Jira", "calendar", "meeting transcripts"],
        public_evals=["TheAgentCompany", "OSWorld", "CRMArena-Pro", "tau2-bench"],
        sota_wedge="Permissions-aware continuity and evidence-backed organizational state.",
        first_build="Later: needs permission model, tenant isolation, redaction, and audit UI before benchmark claims.",
        in_scope_now=False,
        links=[
            "https://webarena.dev/",
            "https://arxiv.org/html/2505.18878v1",
            "https://arxiv.org/abs/2506.07982",
        ],
    ),
    ApplicationTarget(
        name="Planning fallacy reduction",
        problem_solved="Grounds plans in historical duration distributions, current workload, dependencies, and calendar reality.",
        best_data_sources=["Git commits", "task tracker history", "calendar", "time tracking", "CI/build durations"],
        public_evals=["Robotouille", "Real-Time Deadlines", "custom plan-vs-actual repo eval"],
        sota_wedge="Temporal calibration: predicted duration/error bars improve after observing past plans and outcomes.",
        first_build="Repo plan dataset: plan estimate -> actual commit/test completion time -> update duration model.",
        in_scope_now=True,
        links=[
            "https://arxiv.org/abs/2502.05227",
            "https://arxiv.org/abs/2601.13206",
        ],
    ),
    ApplicationTarget(
        name="Proactive follow-up agent",
        problem_solved="Knows when to remind, refresh, resume, or interrupt without a user prompt.",
        best_data_sources=["calendar/tasks", "email threads", "repo issue status", "active_process logs"],
        public_evals=["TicToc", "tau2-bench", "custom interruption preference eval"],
        sota_wedge="Correctly choose wait vs interrupt vs refresh with low false-positive interruption rate.",
        first_build="Start as a local daily/weekly briefing, not autonomous interruptions.",
        in_scope_now=False,
        links=[
            "https://arxiv.org/html/2510.23853v2",
            "https://arxiv.org/abs/2506.07982",
        ],
    ),
    ApplicationTarget(
        name="Calendar/task agent",
        problem_solved="Links obligations, deadlines, current context, and follow-up windows.",
        best_data_sources=["Google Calendar", "Google Tasks", "email", "notes", "project ledger"],
        public_evals=["Real-Time Deadlines", "Robotouille", "custom calendar assistant eval"],
        sota_wedge="Deadline-aware context/action policy; product value depends on trust and permissions.",
        first_build="Daily briefing with evidence and explicit due/blocked/waiting process list.",
        in_scope_now=False,
        links=[
            "https://arxiv.org/abs/2601.13206",
            "https://arxiv.org/abs/2502.05227",
        ],
    ),
    ApplicationTarget(
        name="Literature review system",
        problem_solved="Tracks claims, methods, assumptions, limitations, contradictions, and citations across papers.",
        best_data_sources=["Semantic Scholar/arXiv PDFs", "paper repos", "tables/figures", "citation graph"],
        public_evals=["LitQA2/PaperQA-style evals", "AstaBench", "PaperBench"],
        sota_wedge="Persistent working theory beats one-shot deep research on repeat queries and novelty checks.",
        first_build="Use current research wiki as corpus; create citation-grounded claim/evidence eval.",
        in_scope_now=True,
        links=[
            "https://allenai.org/asta/bench",
            "https://arxiv.org/abs/2504.01848",
        ],
    ),
    ApplicationTarget(
        name="Open-source repo mining",
        problem_solved="Learns reusable architecture patterns, APIs, and implementation strategies from many repos.",
        best_data_sources=["GitHub repos", "README/docs", "dependency graphs", "issues/PRs"],
        public_evals=["RepoBench", "ExecRepoBench", "SWE-bench Verified", "SWE-bench Pro"],
        sota_wedge="Architecture pattern retrieval that improves downstream implementation planning.",
        first_build="Mine 20 memory/agent repos into artifact/span index; ask architecture transfer questions with evidence.",
        in_scope_now=True,
        links=[
            "https://openreview.net/forum?id=pPjZIOuQuF",
            "https://execrepobench.github.io/",
            "https://www.swebench.com/verified.html",
        ],
    ),
    ApplicationTarget(
        name="Visual/video memory",
        problem_solved="Tracks screen/video/photo events over time and retrieves visual evidence.",
        best_data_sources=["screen recordings", "photos", "video transcripts", "OCR", "frame embeddings"],
        public_evals=["OSWorld", "VisualWebArena", "Video-MME-style QA"],
        sota_wedge="Multimodal temporal evidence retrieval; expensive and not the first wedge.",
        first_build="Only revive if tied to context-injection demo for screen recordings.",
        in_scope_now=False,
        links=[
            "https://arxiv.org/abs/2404.07972",
            "https://jykoh.com/vwa",
        ],
    ),
    ApplicationTarget(
        name="Finance/subscription memory",
        problem_solved="Detects recurring payments, budget drift, stale subscriptions, and changing spending patterns.",
        best_data_sources=["bank/Rocket Money CSV", "email receipts", "calendar renewal dates"],
        public_evals=["mostly private/custom; no strong public benchmark fit"],
        sota_wedge="Product utility only; not a first research artifact.",
        first_build="Later with strict privacy/redaction and deterministic rules first.",
        in_scope_now=False,
    ),
    ApplicationTarget(
        name="Life simulation / personal analytics",
        problem_solved="Builds longitudinal trajectories of habits, moods, projects, locations, and relationships.",
        best_data_sources=["personal exports", "calendar", "photos", "notes", "location/health data"],
        public_evals=["private longitudinal eval; LoCoMo/LongMemEval only weak proxies"],
        sota_wedge="Too privacy-heavy and hard to validate publicly.",
        first_build="Do not lead with this; use as long-term private product direction.",
        in_scope_now=False,
    ),
)


def application_targets(in_scope_only: bool = False) -> list[ApplicationTarget]:
    rows = list(APPLICATION_TARGETS)
    if in_scope_only:
        rows = [r for r in rows if r.in_scope_now]
    return rows


def to_markdown(rows: list[ApplicationTarget]) -> str:
    lines = [
        "| Application | Best data source | Public evals / benchmarks | SOTA wedge | First build | Scope |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            "| "
            + " | ".join([
                r.name,
                "<br>".join(r.best_data_sources),
                "<br>".join(r.public_evals),
                r.sota_wedge,
                r.first_build,
                "now" if r.in_scope_now else "later",
            ])
            + " |"
        )
    return "\n".join(lines)

