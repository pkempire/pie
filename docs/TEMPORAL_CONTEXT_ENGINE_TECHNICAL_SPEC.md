# Temporal Context Engine: Technical Build Spec

Date: 2026-06-13

## One Bet

The highest-ROI build is a **Temporal Context Engine**:

```text
agent traces -> temporal state reconstruction -> context/action decision -> downstream outcome -> offline policy improvement
```

The product is not "an AI data lake." The product is a context/action compiler for agents that operate across real time.

The technical wedge is:

```text
Learn which evidence/state/processes belong in context, and when the agent should
answer, refresh, wait, interrupt, or replan.
```

Most memory systems optimize storage or retrieval. This system optimizes **next-session task performance**.

## Why This Can Beat Current Systems

As of June 2026, the strongest directions are specialized:

- Mem0/Zep/Graphiti: memory extraction, graph memory, temporal graph search.
- Letta Context Repositories: file/git-backed context for coding agents.
- RLM: recursive read-time inspection of large contexts.
- Auto-Dreamer: offline memory consolidation with downstream reward and counterfactual utility.
- Anthropic/Sourcegraph context engineering: context packing, compaction, tool-result management.

The step-function is to combine them into one objective:

```text
Given a trace prefix, task, clock time, and budget,
produce the context pack and action that maximize downstream outcome.
```

Formally:

```text
πθ(C, a | T≤t, q, now, B)
```

Where:

- `T≤t` = all prior traces available at time `t`
- `q` = current task/query/objective
- `now` = wall-clock time
- `B` = token/tool/latency budget
- `C` = selected context pack
- `a` = action: answer, refresh, wait, interrupt, replan

Reward:

```text
R =
  task_success
  + evidence_support
  + temporal_validity
  + correct_action_timing
  - token_cost
  - tool_cost
  - latency_cost
  - stale_context_penalty
  - unsupported_claim_penalty
  - unnecessary_interrupt_penalty
```

This is not top-k retrieval. It is policy learning over traces.

## Current Code Base Mapping

Already exists:

- `mempol/core/schema.py`: `Artifact`, `Span`, `MemoryState`, `TraceEvent`.
- `mempol/core/store.py`: SQLite persistence and lexical retrieval.
- `mempol/ledger`: repo/Git/project ingestion.
- `mempol/scripts/longmemeval_matrix.py`: benchmark harness.
- `mempol/dspy_consolidator`: GEPA prompt/consolidation experiments.
- `mempol/recipes/memory_rl`: early RL environments.

Added temporal layer:

- `mempol/temporal/schema.py`
- `mempol/temporal/store.py`

New temporal tables:

```text
temporal_states
state_transitions
active_processes
context_decisions
outcome_events
```

These are the minimum missing tables for the real system.

## Data Model

### Artifact

Immutable raw source.

Examples:

- chat turn,
- tool call,
- Git commit,
- code file,
- benchmark row,
- sales call transcript,
- email,
- calendar event,
- web page snapshot,
- experiment log.

### Span

Addressable evidence inside an artifact.

Examples:

- exact message,
- code line range,
- diff hunk,
- table row,
- transcript segment,
- quoted paper claim.

### TemporalState

Valid-time state backed by spans.

Schema:

```python
TemporalState(
    id: str,
    scope_id: str,
    key: str,
    content: str,
    state_type: str,
    valid_from: str,
    valid_until: str,
    observed_at: str,
    status: str,
    confidence: float,
    volatility_seconds: float | None,
    source_span_ids: list[str],
    supersedes_state_ids: list[str],
    metadata: dict,
)
```

`scope_id` is the partition:

```text
user:<id>
project:<id>
account:<id>
benchmark:<episode_id>
world:<sim_id>
repo:<path>
```

`key` is domain-light:

```text
user.diet
project.best_baseline
account.stage
web.cache.price_page
experiment.status
commitment.deadline
```

### StateTransition

State mutation record.

```python
StateTransition(
    transition_type="create|update|supersede|archive|refresh",
    old_state_ids=[...],
    new_state_id="...",
    reason="...",
    source_span_ids=[...],
    trace_event_ids=[...],
)
```

This is the unit that preserves trajectories.

### ActiveProcess

Any process whose truth changes with elapsed time.

Examples:

- waiting for a human reply,
- running benchmark,
- CI job,
- sales follow-up window,
- cached page freshness,
- deadline,
- subagent task,
- background research read.

Schema:

```python
ActiveProcess(
    kind: str,
    status: "active|waiting|blocked|done|stale|cancelled",
    started_at: str,
    expected_at: str,
    deadline_at: str,
    last_checked_at: str,
)
```

This is what turns cron jobs into temporal control.

### ContextDecision

Logged read-time decision.

```python
ContextDecision(
    task: str,
    now: str,
    action: "answer|refresh|wait|interrupt|replan",
    candidate_state_ids=[...],
    selected_state_ids=[...],
    selected_span_ids=[...],
    selected_process_ids=[...],
    token_budget=int,
    token_estimate=int,
)
```

This is the training example for the context/action policy.

### OutcomeEvent

Downstream supervision.

```python
OutcomeEvent(
    decision_id: str,
    score: float,
    outcome_type: "judge|human|benchmark|tool|product",
    feedback: str,
    metrics: dict,
)
```

No outcome, no learning.

## Runtime Algorithm

### Ingestion

Deterministic ETL:

```text
source -> Artifact -> Span -> TraceEvent
```

Rules:

- raw artifacts are immutable,
- IDs are content-stable where possible,
- all timestamps are normalized ISO strings,
- source locators must be reconstructable,
- personal/private data stays local.

### Transition Extraction

LLM policy input:

```text
new spans
current states for scope
recent transitions
active processes
task/project metadata
```

Output:

```json
{
  "proposed_transitions": [
    {
      "transition_type": "supersede",
      "key": "account.stage",
      "old_state_ids": ["..."],
      "new_state": {
        "content": "Account moved to procurement review...",
        "valid_from": "2026-06-13T14:00:00Z",
        "confidence": 0.86,
        "source_span_ids": ["..."]
      },
      "reason": "Call transcript states procurement is reviewing security docs."
    }
  ],
  "active_process_updates": [
    {
      "kind": "followup",
      "description": "Send security docs",
      "deadline_at": "2026-06-14T14:00:00Z"
    }
  ]
}
```

Validator:

- every new state must cite spans,
- superseded states must share scope/key or explain cross-key dependency,
- `valid_until` must not precede `valid_from`,
- sensitive/personal states require source support,
- no transition is committed silently without trace.

### Read-Time Reconstruction

Input:

```text
query/task q
scope_id
now
token_budget B
```

Steps:

1. retrieve candidate spans from lexical/dense/RLM search,
2. retrieve current temporal states where `valid_from <= now < valid_until OR valid_until=''`,
3. retrieve active processes due before `now` or relevant to `q`,
4. retrieve recent transitions and supersession chain,
5. rerank by predicted utility,
6. select context under budget,
7. decide action.

Pseudo-code:

```python
def compile_context(scope_id, task, now, budget):
    spans = span_retriever.search(task, scope_id)
    states = temporal.current_states(scope_id, at=now)
    processes = temporal.due_processes(scope_id, now=now)
    transitions = temporal.transitions(scope_id)

    candidates = featurize(spans, states, processes, transitions, task, now)
    scored = utility_critic.score(candidates)
    selected = knapsack(scored, budget, diversity_constraints=True)
    action = action_policy.predict(task, selected, processes, now)

    decision = log_context_decision(...)
    return ContextPack(selected), action, decision
```

### Context Pack Format

The pack should not be a blob of retrieved text.

```text
Task
Current time
Action recommendation
Current state
Expired/superseded state
Active processes
Relevant evidence
Open questions
Suggested next steps
Trace/debug appendix
```

## Offline Learning

After each session:

```text
trace -> judge/reviewer -> outcomes -> training rows
```

Offline jobs:

1. **Trace summarizer:** converts raw trace into objective, actions, failures, outcomes.
2. **Transition writer:** proposes state transitions from new evidence.
3. **Consolidator:** merges redundant states and archives low-utility states.
4. **Utility critic trainer:** predicts marginal usefulness of spans/states/processes.
5. **Action policy trainer:** predicts answer/refresh/wait/interrupt/replan.
6. **Eval generator:** turns failures into regression tests.

### Cheap Counterfactual Approximation

Do not run full counterfactual eval for every memory write.

Use amortized utility:

```text
For each decision:
  log candidate set
  log selected set
  log outcome

During offline eval:
  mask random subsets of selected items
  rerun answer/action on small held-out tasks
  compute delta
  train critic(item_features, task_features) -> predicted_delta
```

Features:

```text
item kind
scope/key
age
validity window
confidence
volatility
retrieval score
citation use
supersession depth
process deadline distance
token cost
source type
historical outcome stats
```

This gives most of Auto-Dreamer-style utility without brute-force per-op counterfactuals.

## Benchmark Adapters

### LongMemEval

Mapping:

```text
Artifact: session transcript
Span: turn or learned chunk
TemporalState: facts/preferences/updates with valid time
ContextDecision: question-specific context pack
Outcome: judge score
```

Cells:

```text
raw full context
turn RAG
session RAG
temporal state reconstruction
temporal state + RLM reader
temporal state + offline consolidated active memory
```

Metrics:

```text
accuracy
category accuracy
tokens/query
stale-state errors
evidence coverage
latency
```

### LoCoMo

Mapping:

```text
Artifact: conversation turn
Span: turn
TemporalState: person/event/preference/project state
Transition: updates across sessions
Outcome: QA score
```

Add temporal-specific metrics:

```text
validity error rate
supersession error rate
multi-hop temporal chain success
```

### TicToc / Temporal Tool Use

Mapping:

```text
Artifact: prior observation/tool call
TemporalState: cached world fact
ActiveProcess: freshness/expiration window
Action: reuse | refresh | ask | defer
Outcome: human preference / benchmark label
```

This directly trains the action policy.

### Robotouille / Async Planning

Mapping:

```text
Artifact: action/observation log
ActiveProcess: cooking/waiting/background task
TemporalState: world state
Action: start | wait | check | parallelize | replan
Outcome: task success
```

This trains active-process handling.

### Repo Continuation Benchmark

Create a local benchmark from this repo:

```text
input: repo trace up to date T
task: "resume project X"
gold: trusted current result, superseded result, next action, risk
```

Methods:

```text
raw file retrieval
git commit retrieval
ledger context
temporal reconstruction
temporal reconstruction + critic
```

This is the most product-relevant eval.

## Infra Plan

### Local Prototype

```text
SQLite WAL
FTS5 lexical index
OpenAI embeddings or local embeddings
Reranker optional
Streamlit/Next dashboard
JSONL trace logs
```

### Production

```text
Object store: S3/R2/GCS for artifacts
Transactional DB: Postgres
Vector: pgvector or Qdrant
Search: Postgres FTS / OpenSearch
Queue: Temporal.io / Hatchet / Celery / BullMQ
Trace/event bus: Kafka/Redpanda optional
Analytics: DuckDB/Iceberg for offline training data
Policy training: Tinker/Prime/HF/TRL depending model
Serving: MCP server + SDK + HTTP API
```

### Why Not "AI Data Lake"

The lake is a component, not the product.

The product is:

```text
context/action decisions with measurable downstream improvement
```

If packaging for developers:

```text
Temporal Context Engine for AI agents
```

If packaging for enterprises:

```text
Agent Memory Control Plane
```

If packaging for consumers:

```text
Context Manager
```

Same backend. Different skin.

## Implementation Milestones

### M0: Temporal Substrate

Done/started:

- temporal states,
- transitions,
- active processes,
- context decisions,
- outcomes.

Next:

- integrate with repo ingestion,
- dashboard inspection,
- context compiler reads temporal states.

### M1: Repo Continuation Demo

Build:

```text
ingest repo -> extract temporal project states -> compile resume context -> evaluate against hand-labeled questions
```

Deliverable:

```text
10-30 repo-continuation tasks with side-by-side outputs
```

### M2: Benchmark Integration

Add temporal state cells to:

- LongMemEval,
- LoCoMo,
- TicToc-style tasks if dataset available,
- synthetic async tasks.

### M3: Offline Critic

Build:

```text
decision_training_rows -> feature table -> small critic model
```

Start with:

- logistic/GBM/scikit model,
- then small transformer/reranker,
- then GRPO or GEPA for policy optimization.

### M4: Product Surface

Build:

- MCP server,
- SDK,
- dashboard,
- browser/repo/email connectors,
- context-pack preview and edit UI.

## What Would Be SOTA

A publishable claim would be:

```text
Temporal Context Engine reduces stale-state errors and improves answer/action
quality per token across conversation QA, async planning, and project-continuation
tasks by learning context/action policies from agent traces.
```

The important part is cross-benchmark generality:

- recall benchmarks: LongMemEval/LoCoMo,
- timing benchmarks: TicToc/Real-Time/Robotouille,
- product benchmark: repo-continuation tasks.

If one substrate improves all three, that is the story.
