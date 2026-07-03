# Research Ledger Implementation Plan

Status: implementation started.

## Final Backend Idea

The system is a local-first context manager for long-running projects. It stores immutable raw evidence first, then builds a project/thread ledger above it.

```text
raw artifacts -> evidence spans -> project/thread memberships
              -> research objects and runs
              -> context packs
              -> trace/outcome data for learning
```

This is not a recursive language model. RLM-style reading is a possible future read policy. The current ledger is an evidence and project-state substrate.

## Why Offline Consolidation Comes Later

Offline consolidation is useful, but it should not be the first thing built. Auto-Dreamer-style consolidation and GEPA consolidators are valuable because they compress many episodes into reusable state, but they need a clean event log first.

First priority:

1. Ingest raw artifacts and Git history.
2. Show what happened by day.
3. Assign artifacts to projects/threads.
4. Compile context packs.
5. Log outcomes.

Then consolidation can learn from real traces instead of synthetic memory snippets.

## Current Implementation

Implemented package:

```text
mempol/ledger/
  schema.py
  store.py
  tagger.py
  ingest_repo.py
  day_report.py
  compile_context.py
```

Main commands:

```bash
python3 -m mempol.ledger.ingest_repo --root . --run-name research_ledger_repo --max-commits 250
python3 -m mempol.ledger.day_report --run-name research_ledger_repo
python3 -m mempol.ledger.day_report --run-name research_ledger_repo --day YYYY-MM-DD
python3 -m mempol.ledger.compile_context --run-name research_ledger_repo --task "What should I work on next for the memory project?"
```

Outputs:

```text
mempol/results/research_ledger_repo/core_memory.sqlite
mempol/results/research_ledger_repo/ledger.sqlite
mempol/results/research_ledger_repo/ledger_ingest_summary.json
mempol/results/research_ledger_repo/day_YYYY-MM-DD.md
mempol/results/research_ledger_repo/latest_context_pack.md
```

## Local File / Coding Project History

For tracked files, the system uses Git:

- commit hash
- authored time
- author
- subject/body
- changed files
- last commit touching each file

For uncommitted files, only filesystem metadata is available unless an agent session log captured the work. That means future agents should log command traces, tool calls, and outcomes into the ledger during work.

## Product Direction

The product should become a universal context shortcut:

```text
user types prompt anywhere
-> hits shortcut
-> app detects active app/file/repo/URL/selected text
-> guesses project/thread
-> compiles context pack
-> user previews sources
-> app pastes context into Claude/Codex/GPT
```

The first version can be manual clipboard/context-pack export. The polished version can be a Tauri/macOS app.

## Learning Plan

Policies to learn:

- thread router
- research writer
- context compiler
- sleep consolidator
- utility critic

Reward:

```text
task success
+ user accepts context
+ experiment progress
+ avoids repeated failed work
+ citation correctness
- context tokens
- unsupported claims
- stale context
- duplicate work
```

Exact per-op counterfactuals should only train/evaluate a critic on samples. They are too expensive as the main reward loop.

## Relationship To Existing Work

- GEPA consolidator is closest to Auto-Dreamer-style offline consolidation, but current local runs are tiny and not a final result.
- RLM temporal reconstruction demos are not true RLMs in this repo; they are read-time timeline reconstruction demos.
- The universal memory core is a SQLite prototype for `Artifact`, `Span`, `MemoryState`, and `TraceEvent`.
- PIE remains useful as a temporal-world-model baseline/view, not the final substrate.
- LongMemEval and LoCoMo remain benchmark harnesses, not the main product.

## Acceptance Criteria

The first usable system is done when:

- It ingests this repo and Git history.
- It can show raw artifacts and commits by day.
- It assigns evidence to project/thread partitions.
- It compiles a context pack with source evidence.
- It logs context-pack creation as a trace.
- It can be used before a Codex/Claude/GPT prompt to resume work.
