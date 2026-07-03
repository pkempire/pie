# Repo Operating System

This repo should behave like a memory system for the research itself. GitHub is the durable public ledger; local SQLite/result folders are working memory; tracked docs are consolidated memory.

## Source Of Truth

### Track in Git

- Source code, tests, CLIs, dashboards, prompts that are part of runnable systems.
- Canonical docs in `docs/`, paper notes in `paper/lit-review/`, public content in `research/content/`.
- Small result summaries: tables, analysis markdown, plots, and exact commands.
- Reproduction instructions for every result worth citing.

### Keep local only

- `.env`, API keys, secrets.
- Raw personal exports, private chats, browser history, email exports, photos, notes.
- SQLite run databases, embedding caches, benchmark result folders, full logs, videos.
- Heavy third-party datasets unless their license and size make tracking appropriate.

## Experiment Lifecycle

1. **Before run**

Record the hypothesis, command, dataset split, cells/strategies, models, and expected output directory.

2. **During run**

Write rows incrementally to JSONL so interrupted runs preserve partial work. Include errors as rows, not silent crashes.

3. **After run**

Write a compact `summary.json` and human-readable `side_by_side.md` or `analysis.md`.

4. **Promotion**

If the result changes what we believe, copy the compact conclusion into a tracked doc under `docs/` or `research/`.

5. **Commit**

Commit runnable code, tests, docs, and compact summaries together. Do not commit generated DBs/logs.

## Required Run Metadata

Every serious run should include:

- `run_name`
- `created_at`
- `git_sha`
- `command`
- `dataset`
- `split_or_rows`
- `models`
- `strategy_cells`
- `budget_knobs`
- `metrics`
- `output_paths`
- `known_failures`

## Research Ledger Commands

Ingest repo files and Git history:

```bash
python3 -m mempol.ledger.ingest_repo \
  --root . \
  --run-name research_ledger_repo_dev2 \
  --max-files 500 \
  --max-commits 80
```

List days seen by the ledger:

```bash
python3 -m mempol.ledger.day_report --run-name research_ledger_repo_dev2
```

Inspect one day:

```bash
python3 -m mempol.ledger.day_report \
  --run-name research_ledger_repo_dev2 \
  --day 2026-05-02 \
  --limit 100
```

Compile a context pack for an agent:

```bash
python3 -m mempol.ledger.compile_context \
  --run-name research_ledger_repo_dev2 \
  --task "Resume the memory benchmark work and identify the next trustworthy run." \
  --k 10 \
  --token-budget 4500
```

## How To Think About GitHub As Memory

A Git commit is not just backup. It is a timestamped state transition:

- `before`: repo state before the commit.
- `after`: repo state after the commit.
- `delta`: files changed.
- `intent`: commit message, PR description, issue link, run result.
- `evidence`: tests, benchmark summaries, screenshots, logs.

For agents, this matters because project memory is mostly not "facts." It is the trajectory of decisions, failed attempts, partial results, and artifacts that explain what to do next.

## Current Weak Spots To Fix Next

- Benchmark strategy names still use opaque labels like `flat_v0`; dashboards should display human names and exact mechanics.
- LongMemEval/LoCoMo runs need compact promoted summaries with model names, row counts, top-k settings, and costs.
- Result SQLite/JSONL outputs are local-only; we need tracked summary artifacts for any number used in a paper/video.
- Ledger retrieval is currently useful but still coarse; add semantic retrieval, reranking, and project/thread filters.
- PIE ingestion remains a valuable baseline but should be rerun from raw conversations when reporting PIE results.
