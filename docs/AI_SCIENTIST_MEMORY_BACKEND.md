# Temporal Agent Memory Backend

Scope document for the memory/control substrate this repo should build first.

## Goal

Build the memory backend for long-running agents that operate across real-world time.

The target system should support:

- proactive personal assistants,
- web agents with stale tools and changing pages,
- enterprise agents with projects, blockers, approvals, and owners,
- sales agents with account trajectories and follow-up windows,
- life simulations with persistent world state,
- AI scientist / autonomous research agents.

The backend should not assume the model is continuously alive. It should assume LLM calls are stateless and that persistent cognition must be represented by the runtime.

## Core Claim

For long-running agents, memory should be modeled as temporal world state, not as document retrieval alone.

Raw retrieval answers:

```text
What text exists?
```

Temporal memory answers:

```text
What changed?
What is still true?
What is stale?
What process is active?
What should happen next?
What evidence supports this state?
```

## Non-Goals

- Do not build a fully autonomous paper-writing system first.
- Do not rely on one hard-coded schema for every domain.
- Do not treat vector retrieval as the whole memory system.
- Do not optimize local write quality without downstream task outcomes.
- Do not create another one-off benchmark backend.

## System Objects

### Artifact

Immutable raw source.

Examples:

- paper PDF or extracted section,
- Git commit,
- source file,
- benchmark config,
- run log,
- metric summary,
- chat transcript,
- reviewer note,
- generated draft,
- human edit.

### Span

Addressable evidence inside an artifact.

Examples:

- paper paragraph,
- table row,
- code line range,
- diff hunk,
- benchmark row,
- exact quoted claim,
- transcript segment.

### State

Compressed current or historical belief about a user, project, account, process, world, or experiment, always backed by spans.

Examples:

- "LongMemEval turn-level dense retrieval currently beats the incomplete PIE cached baseline on sampled rows."
- "GEPA consolidator improved subsample score but validation remained weak; run was interrupted."
- "The artifact workflow failed on GPT-5 because `max_tokens` must be `max_completion_tokens`."
- "Account moved from discovery to procurement review; security docs are due within 24 hours."
- "The user was angry during the conversation, but the state likely decayed and should not be treated as durable."

### Transition

A change from one state to another.

Examples:

```text
old_state: artifact workflow is runnable
new_state: artifact workflow crashes on GPT-5 max_tokens parameter
evidence: stack trace, mempol.llm.py, artifact_workflow.py
operation: bug_discovered
```

```text
old_state: video target is broad accessible audience
new_state: video target is frontier AI researchers and AI-scientist memory backend
evidence: user correction, updated script
operation: scope_refined
```

### TraceEvent

Every meaningful runtime action.

Examples:

- retrieved spans,
- wrote memory state,
- launched experiment,
- generated draft,
- reviewed draft,
- accepted/rejected memory,
- human corrected claim,
- benchmark outcome changed belief.

## Required State Variables

The backend must represent these explicitly:

- `observed_at`: when evidence entered the system.
- `valid_from`: when a state became true or believed.
- `valid_until`: when a state stopped being true or should be treated as expired.
- `supersedes`: prior states replaced by this one.
- `source_span_ids`: exact evidence.
- `confidence`: calibrated belief, not just model vibes.
- `volatility`: how fast this state tends to change.
- `status`: active, blocked, superseded, stale, resolved.
- `open_questions`: uncertainty that should guide future work.
- `downstream_uses`: tasks or answers that used this state.
- `active_processes`: waits, jobs, commitments, external dependencies, follow-up windows.
- `trace_outcomes`: whether prior retrieval/write/action choices helped or hurt later tasks.

## Runtime Loop

```text
1. Ingest
   deterministic ETL creates artifacts and spans

2. Retrieve
   query retrieves relevant spans, states, transitions, and prior traces

3. Reconstruct
   system builds a current project state or state-at-time view

4. Act
   model reads context pack, answers, writes code, runs experiment, follows up,
   opens browser, drafts memo, or asks a question

5. Observe
   logs, metrics, diffs, and human edits become new artifacts

6. Update
   memory policy writes state transitions with provenance

7. Consolidate
   offline policy merges, compresses, archives, and updates active memory

8. Evaluate
   downstream task progress and human corrections score the memory decisions
```

## What Should Be Deterministic

Use deterministic code for:

- file discovery,
- Git commit extraction,
- artifact IDs,
- span IDs,
- checksums,
- timestamps,
- output paths,
- command recording,
- metric parsing where schema is known,
- privacy filters,
- large-file skipping.

This makes the base ledger reproducible.

## What Should Be LLM-Policy

Use strong models for:

- mapping messy evidence to states,
- identifying what changed,
- linking states across artifacts,
- extracting open questions,
- extracting commitments and active processes,
- writing context packs,
- reviewing artifacts,
- proposing next experiments.

The model should not silently mutate memory. It should emit proposed transitions with evidence.

## What Should Be Learned

Train policies for:

- which spans to retrieve,
- which memory states to write,
- when to update vs create,
- which states to consolidate,
- which states are stale,
- which context pack improves downstream task success.
- when to proactively interrupt,
- when to refresh tools,
- when to wait,
- when to resume a dormant thread,
- when to archive or compress traces.

Reward should be downstream:

```text
task_success
- storage_cost
- retrieval_cost
- latency_cost
- stale_context_penalty
- unsupported_claim_penalty
```

## First Publishable Experiment

Show that temporal state reconstruction beats raw retrieval for long-running agent tasks.

### Dataset

Use two datasets:

1. this repo's own history plus selected benchmark outputs,
2. synthetic or real task traces for proactive/async scenarios.

Repo evidence:

- commits,
- docs,
- benchmark summaries,
- failed command traces,
- generated artifacts,
- human corrections.

Task families:

- project continuation: resume research from prior traces,
- proactivity: decide whether to surface, wait, refresh, or interrupt,
- async planning: reason about active processes and elapsed time,
- stale retrieval: decide whether retrieved state is still valid.

Compare:

1. raw retrieval over files,
2. raw retrieval over commits/logs,
3. temporal ledger context pack,
4. temporal ledger plus LLM state reconstruction.

### Metrics

- factual accuracy about prior work,
- citation/evidence coverage,
- correct identification of superseded results,
- correct next action,
- token cost,
- unnecessary interruption rate,
- stale-context rate,
- human preference.

### Output

A reportable table:

```text
method | accuracy | evidence coverage | stale rate | action quality | interrupt precision | tokens
```

## One-Day Demo

1. Ingest repo:

```bash
python3 -m mempol.ledger.ingest_repo \
  --root . \
  --run-name ai_scientist_memory_demo \
  --max-files 800 \
  --max-commits 120
```

2. Compile context:

```bash
python3 -m mempol.ledger.compile_context \
  --run-name ai_scientist_memory_demo \
  --task "Resume the memory-policy research project. Identify the current trusted benchmark results, superseded claims, open implementation risks, and the next experiment." \
  --k 16 \
  --token-budget 8000
```

3. Generate artifact:

```bash
python3 -m mempol.ledger.artifact_workflow \
  --run-name ai_scientist_memory_demo_artifact \
  --artifact-id research-continuation-memo \
  --objective "Write a concise research-continuation memo from the current repo memory. Separate trusted results, superseded results, open risks, and next experiments. Cite source files and commands." \
  --source mempol/results/ai_scientist_memory_demo/latest_context_pack.md \
  --source docs/AI_SCIENTIST_MEMORY_BACKEND.md
```

## Application Views

Same substrate, different read-time views:

| Domain | State objects | Temporal decisions |
|---|---|---|
| Personal assistant | preferences, commitments, moods, projects, relationships | remind, refresh, follow up, suppress stale facts |
| Web agent | pages, sessions, prices, auth, API results | refresh, wait, retry, cache, escalate |
| Enterprise agent | projects, owners, blockers, approvals, docs | escalate, route, summarize, detect stale threads |
| Sales agent | accounts, stakeholders, stage, objections, next steps | follow up, update stage, revive dormant lead |
| Life sim | characters, goals, events, relationships, environment | advance world, resolve conflicts, maintain continuity |
| AI scientist | papers, hypotheses, code, runs, metrics, beliefs | resume, compare, supersede, design next experiment |

## Relation To Existing Systems

### Sakana AI Scientist / AI Scientist v2

These systems focus on the outer research loop: idea generation, code, experiments, paper drafting, and review. The backend here is complementary. It focuses on persistent state across many runs.

### Kosmos / Edison-style AI scientist systems

These systems motivate the need for a shared research ledger. Any multi-agent research system needs a durable substrate where agents can preserve evidence, hypotheses, failed attempts, and state changes.

### Aide / Aider / coding agents

Coding agents operate over a repo but often treat each session as local. This backend makes the repo history and experiment history queryable as state, not just files.

### LongMemEval / LoCoMo

These are useful memory benchmarks, but they mostly test recall over conversation history. The temporal backend needs additional tasks where the goal is not just answering a fact but choosing the next action from evidence and elapsed time.

## Current Repo Fit

Already useful:

- `mempol/core`: artifact/span/state/trace substrate.
- `mempol/ledger`: repo and Git ingestion.
- `mempol/scripts/longmemeval_matrix.py`: memory benchmark harness.
- `mempol/dspy_consolidator`: GEPA consolidation experiments.
- `mempol/recipes/memory_rl`: early RL policy environment.
- `research/content`: public artifact layer.

Needs work:

- semantic/reranked retrieval in ledger context packs,
- state-transition extraction from repo evidence,
- result supersession tracking,
- benchmark summary promotion,
- dashboard for project memory inspection,
- evaluation set for research-continuation tasks.
