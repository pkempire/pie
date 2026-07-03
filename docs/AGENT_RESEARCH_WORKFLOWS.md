# Agent Research Workflows

This repo should use agents as workers that produce inspectable artifacts, not as chat sessions that disappear.

The core loop:

```text
objective
  -> source selection
  -> draft artifact
  -> specialist reviews
  -> meta-comparison
  -> human edit
  -> commit
  -> ledger ingest
```

## Built-In Artifact Workflow

Use this for drafts, scripts, design docs, and paper sections.

```bash
python3 -m mempol.ledger.artifact_workflow \
  --run-name temporal_video_revision \
  --artifact-id temporal-awareness-video \
  --prior research/content/temporal-awareness-video-2026-06.md \
  --objective "Write a rigorous but engaging technical video script for frontier AI researchers and agent builders. Open from transformer statelessness and agent runtimes, but cover the broader thesis: elapsed time and temporal world models are prerequisites for proactivity, async planning, long-term memory, and offline improvement from agent traces. Avoid hype, vague product language, and oversimplified creator-script tone." \
  --source research/content/temporal-awareness-video-2026-06.md \
  --source paper/lit-review/temporal.md \
  --source docs/REPO_OPERATING_SYSTEM.md \
  --source docs/AGENT_RESEARCH_WORKFLOWS.md \
  --source docs/AI_SCIENTIST_MEMORY_BACKEND.md
```

Dry-run without model calls:

```bash
python3 -m mempol.ledger.artifact_workflow \
  --run-name temporal_video_revision_dry \
  --artifact-id temporal-awareness-video \
  --objective "Generate prompts only." \
  --source research/content/temporal-awareness-video-2026-06.md \
  --dry-run
```

Outputs:

```text
mempol/results/<run>/artifact_workflows/<artifact>/
  draft.md
  review_taste.md
  review_science.md
  review_product.md
  comparison.md
  manifest.json
  prompts/
```

Commit only the final chosen artifact and compact analysis, not the raw generated run folder.

## Aide / Aider / Codex Prompt

Use this when running a coding agent against the whole repo.

```text
You are improving a research/code repository for long-horizon AI memory.

Goal:
Turn this repo into a professional, reproducible open-source project. Do not invent new side systems unless needed. Prefer finishing existing incomplete systems.

Primary tasks:
1. Read README.md, docs/REPO_OPERATING_SYSTEM.md, mempol/ledger, mempol/core, mempol/scripts/longmemeval_matrix.py, and research/content/temporal-awareness-video-2026-06.md.
2. Identify code that is not production/research-grade: unclear names, hidden constants, duplicate strategies, untracked assumptions, missing tests, missing metrics, or brittle data paths.
3. Make the smallest useful code changes to improve reproducibility.
4. Update docs only when they explain runnable code.
5. Run targeted tests.

Hard constraints:
- Do not run expensive benchmark jobs.
- Do not delete generated outputs unless asked.
- Do not touch personal exports or secrets.
- Preserve existing user changes.
- Every change must make a command, result, or artifact easier to reproduce.

Deliverable:
- A short markdown report listing changes, files touched, tests run, and remaining risks.
```

## Kosmos-Style Research Prompt

Use this for a local Kosmos implementation or any autonomous research loop.

```text
Research objective:
Design the temporal memory/control backend for long-running agents, using elapsed time, transition memory, and trace learning as core primitives.

Background:
The project studies memory for long-running agents across personal assistants, web agents, enterprise workflows, sales, life simulation, and AI research. The current thesis separates:
- temporal reasoning: logic over dates and event order
- temporal awareness: runtime control conditioned on elapsed/remaining real time
- temporal memory: world/project/account/user-state reconstruction from evidence-backed transitions

Local sources to read:
- research/content/temporal-awareness-video-2026-06.md
- paper/lit-review/temporal.md
- docs/REPO_OPERATING_SYSTEM.md
- docs/AI-LAKEHOUSE-HLD.md
- mempol/ledger/
- mempol/core/
- mempol/scripts/longmemeval_matrix.py

External sources to verify:
- Memory in the Age of AI Agents, arXiv:2512.13564
- Temporal Blindness in Multi-Turn LLM Agents, arXiv:2510.23853
- Real-Time Deadlines Reveal Temporal Awareness Failures, arXiv:2601.13206
- Robotouille, ICLR 2025
- Discrete Minds in a Continuous World, arXiv:2506.05790
- Cognition AI Productivity blog

Questions:
1. What exact memory backend is needed for agents that work across hours, days, and months?
2. Which state variables must be represented explicitly: elapsed time, validity, supersession, active processes, commitments, project/account/user status, evidence, open questions, and belief updates?
3. What parts can be deterministic ETL, what parts need LLM policy, and what parts should be learned from traces?
4. What demo can be built from this repo in one day that proves temporal state reconstruction or proactive timing?
5. What benchmark or experiment would produce publishable evidence that this backend improves long-running agents?

Output:
1. A concise research memo with citations.
2. A concrete backend design.
3. A list of exact claims that are safe to make.
4. A list of unsafe claims.
5. A proposed demo script and commands.

Quality bar:
- Prefer precise claims over dramatic claims.
- Cite every quantitative result.
- Separate evidence from hypothesis.
- Use this repo's actual code and commands where possible.
```

## Why Not Let Agents Edit Everything Freely?

Because the bottleneck is not generating more text. The bottleneck is retaining trustworthy progress.

Agents should create artifacts with:

- explicit objectives,
- source lists,
- versioned outputs,
- reviewer notes,
- meta-comparisons,
- test commands,
- and a commit after human approval.

This turns each agent run into memory the next agent can use.
