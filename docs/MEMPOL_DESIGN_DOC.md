# MemPol Design Document: Budgeted Temporal State Memory

## 0. Thesis, Without Marketing

The project should not claim that "temporal memory" alone is new. It is not. Zep/Graphiti already uses temporal KGs, LongMemEval and LoCoMo already test temporal reasoning, and recent papers explicitly test temporal awareness.

The sharper claim is:

> Long-horizon agents need a learned policy that turns an unbounded event stream into a compact, temporally valid state model that maximizes future task utility under storage and retrieval budgets.

The word "temporal" matters only if the system represents more than the latest snapshot. If the memory says `Caroline wants to adopt`, that is just state. If it says:

```json
{
  "subject": "Caroline",
  "state": "researching adoption agencies",
  "observed_at": "2023-05-25",
  "valid_from": "2023-05-25",
  "valid_until": null,
  "previous_state": "interested in family/adoption",
  "transition": "goal became active",
  "evidence": ["D2:8", "D2:9"]
}
```

that is temporal state. It can answer:

- What is true now?
- What used to be true?
- When did it become true?
- Is it likely stale?
- What changed?
- Which future reminders/actions should fire because time passed?

If we do not evaluate those questions, we should stop saying temporal and just say state.

## 1. First Principles

Memory is not storage. Memory is a policy for allocating future context.

The primitive problem:

```text
Input:  unbounded stream of turns/events/documents/tool results
Output: bounded memory state used by future readers/agents
Goal:   maximize future task success per token, call, dollar, and latency
```

The irreducible uncertainty is future utility. At write time, we do not know which future query will matter. So the system must learn compression that preserves information likely to be useful later.

Core objective:

```text
maximize   E_future_tasks[ task_score(reader(query, memory)) ]
minimize   storage_tokens + retrieval_tokens + write_calls + read_calls + stale_wrong_answers
```

This yields the real metrics:

- `task_score`: judged QA/task success.
- `memory_tokens`: durable memory size.
- `compression_ratio`: raw history tokens divided by durable memory tokens.
- `retrieval_tokens`: context tokens read per query.
- `retrieval_precision`: fraction of retrieved tokens actually useful.
- `evidence_recall`: whether necessary source evidence survived.
- `staleness_error`: answers that use expired facts as current facts.
- `update_rate`: create vs update vs merge ratio.
- `counterfactual_utility`: how much answer quality drops if a memory item or consolidated region is removed.

## 2. Why Temporal State Instead Of Plain State?

Plain state is a snapshot:

```text
User is vegetarian.
Project status is blocked.
Agent A is working on file X.
```

Temporal state is a trajectory:

```text
User was vegetarian until March, then pescatarian after a trip.
Project was blocked on API quota, quota was fixed, now blocked on eval reliability.
Agent A started file X at 10:32, touched files Y/Z, timed out at 10:49.
```

The value is not date arithmetic. The value is validity and change.

Temporal state is needed when:

- Facts expire: "I am angry", "I am busy", "the server is down".
- Facts evolve: "I'm vegetarian" -> "I'm pescatarian".
- Plans span time: "follow up Friday", "train after benchmark finishes".
- Multi-agent work overlaps: agent states, locks, partial outputs, elapsed time.
- Retrieval must know whether to prefer latest, historical, or full trajectory.
- Proactivity requires elapsed-time state outside the LLM context window.

The novelty cannot be "we store timestamps." The novelty must be:

> We learn which state transitions to preserve/consolidate because they improve future behavior under budget.

If experiments do not show gains on stale/current, before/after, multi-hop temporal, or proactive timing tasks, the temporal part is not carrying the paper.

## 3. Current Repo: What Exists

There are two major code eras.

### PIE era

Useful pieces:

- `pie/core/world_model.py`: entity, relationship, state transition model.
- `pie/core/models.py`: typed entities, relationships, transition types.
- `pie/core/temporal.py`: temporal features and time-aware helpers.
- `pie/retrieval/*`: temporal/hybrid retrieval and context compilation.
- `pie/ingestion/*`: prompted extraction baseline.
- `pie/eval/*`: older temporal/extraction quality eval ideas.

What to steal:

- The world-model schema.
- Transition history as first-class data.
- Temporal context compilation ideas.
- Existing cached KGs as baselines.

What not to worship:

- The old extraction pipeline is a baseline, not the final system.
- Entity resolution heuristics are fragile.
- Graph structure by itself has not proven better than strong raw/observational retrieval.

### MemPol era

Useful pieces:

- `mempol/backends/base.py`: backend abstraction.
- `mempol/backends/flat.py`: simple raw-turn BM25+dense baseline.
- `mempol/backends/pie_kg.py`: PIE world model wrapped as a backend with hybrid retrieval and write ops.
- `mempol/policies/v1_write.py`: current LLM teacher writer.
- `mempol/policies/v1_heuristic.py`: read policy with reformulate/retrieve/expand/rerank.
- `mempol/recipes/memory_rl/write_tools.py`: write op vocabulary and logging.
- `mempol/eval/counterfactual.py`: per-op leave-one-out utility reward.
- `mempol/scripts/simulate_on_locomo.py`: live writer trace.
- `mempol/scripts/dashboard.py`: Streamlit dashboard for live decisions/training traces.
- `mempol/scripts/eval_locomo_writer.py`: chunked writer -> memory -> QA eval path.
- `mempol/scripts/random_baseline.py`: random/budget baseline scaffold.
- `mempol/data/necessity_miner.py`, `future_eval.py`, `headtohead_chatgpt.py`: pieces for personal-export future-query eval.

What is broken or not final:

- `mempol/recipes/memory_rl/write_env.py` still creates a fresh `PIEBackend()` for each rollout.
- `existing_entities_summary` is still empty in the GRPO dataset builder.
- Current GRPO write training is per-turn, not true continual memory.
- The live writer can update a persistent world model, but observed runs still over-create entities and rarely update/merge.
- There is no offline consolidator yet.
- There is no real budgeted consolidation benchmark yet.

## 4. Existing Results We Can Trust As Repo-State, Not Paper Claims

From local result files:

```text
LoCoMo conv-26, 199 questions:
  full_context: 75.1 overall
  naive_rag:    66.1 overall
  pie_temporal: 61.3 overall

Category note:
  pie_temporal beats naive_rag on temporal questions in that run:
    pie_temporal temporal: 78.4
    naive_rag temporal:    64.9
  but pie_temporal loses overall, especially single-hop/adversarial.
```

Interpretation:

- Temporal structure may help temporal QA.
- Current PIE extraction/retrieval is not competitive overall.
- This is exactly why the paper should be about budgeted consolidation and category-specific gains, not "KG beats RAG."

Live writer traces show:

```text
locomo_live_fresh:
  processed turns: 52
  entities: 56
  transitions: 56
  relationships: 0
  creates: 56
  updates: 0
  merges: 0
```

Interpretation:

- The current writer is still basically extracting many entities.
- It is not yet learning relationships, updates, or long-horizon structure.
- Chunking helps, but the write policy still needs consolidation.

## 5. Proposed Architecture

### Layer 0: Raw Source Log

Always keep raw turns/documents/tool results as immutable source of truth.

Purpose:

- Auditability.
- Re-answering when extraction was wrong.
- Training future consolidators.
- Building future-query evals.

Storage:

```json
{
  "source_id": "conv-26:D2:9",
  "text": "...",
  "speaker": "Caroline",
  "observed_at": "2023-05-25T13:14:00Z",
  "session_id": "D2",
  "modality": "chat",
  "metadata": {}
}
```

### Layer 1: Fast Online Writer

This should be simple. It does not need to solve global memory.

Input:

- Current chunk/session.
- Recent local context.
- Lookup results from existing memory.
- Current wall-clock/observation time.

Output:

- Candidate observations.
- Candidate entities.
- Candidate transitions.
- Provenance links.

Important change:

The online writer should process chunks/sessions, not isolated single turns by default.

### Layer 2: Learned Memory Bank, Not A Fixed Ontology

The durable memory store should not be built around a brittle hand-written ontology like:

```json
{"kind": "entity|event|goal|decision|belief|procedure|relationship"}
```

That taxonomy is useful for debugging and baselines, but it should not be the final representation. It is exactly the kind of janky human bottleneck the bitter lesson warns against: the true boundary between "goal", "project", "belief", "event", and "procedure" is often task-dependent and should emerge from data.

The production/research version should treat memories as learned objects with minimal required invariants:

```json
{
  "memory_id": "...",
  "content": "...",
  "embedding": "...",
  "latent_type": "...",
  "routing_hints": [],
  "time": {
    "observed_at": "...",
    "valid_from": null,
    "valid_until": null,
    "last_confirmed_at": null
  },
  "provenance": ["source ids"],
  "supersedes": ["memory ids"],
  "utility": {
    "estimated_future_use": null,
    "last_used_at": null,
    "tasks_helped": []
  }
}
```

The fixed labels become optional supervision, not architecture:

- Use labels as SFT scaffolding so models learn tool grammar.
- Let the consolidator invent compressed entries that do not fit the ontology.
- Learn routing/index keys from future-query utility.
- Evaluate by task utility per token, not by whether entries fit a human type.

The key is `valid_until` should not imply deletion. Historical facts remain searchable, but readers need to know whether the fact is current.

### Layer 2.5: Learned Segmentation

The repo currently chunks by turn, fixed window, or session. That is not the endgame.

H-Net-style dynamic chunking is the right inspiration: learn content- and context-dependent boundaries from data instead of pretending "12 turns" is a natural unit. For this project, the deployable approximation is:

```text
raw stream -> learned / LLM-proposed boundaries -> candidate episodes -> consolidation regions
```

Initial implementation can be a cheap teacher:

- Ask a frontier model to mark segment boundaries where topic, goal, speaker intent, or temporal state changes.
- Train a small boundary model on those labels.
- Evaluate boundary quality only through downstream memory utility.

The research direction is stronger if the unit of memory is learned. Hardcoded entity/event types are scaffolding; learned boundaries and learned compressed state are the actual bet.

### Layer 3: Offline Consolidator

This is the missing piece.

Input:

- A region of memory: recent writes plus retrieved related older memories.
- Provenance raw chunks.
- Budget target.
- Future/eval task set.

Output:

- Replacement memory set.
- Supersession links.
- Compression report.
- Counterfactual utility estimates.

The consolidator does what the online writer cannot:

- Merge duplicates.
- Abstract repeated patterns.
- Convert facts into slots/procedures.
- Preserve temporal trajectories.
- Remove/relegate low-utility memories.
- Fix contradictions.

This follows the bitter-lesson direction: let the online path collect signal; use more compute offline to compress and reorganize. The consolidator should be allowed to produce arbitrary high-utility memory text, not only CRUD ops into a fixed KG.

### Layer 4: Read Planner

Read should be tool-using, not one vector search.

Tools:

- `search_raw(query, filters)`
- `search_memory(query, filters)`
- `lookup_entity(name)`
- `expand(memory_id, relation_type, depth)`
- `timeline(entity_id)`
- `current_state(entity_id, as_of_time)`
- `source(memory_id)`

Reader policy should choose:

- Raw vs consolidated memory.
- Current vs historical view.
- Whether to expand relations.
- Whether to inspect provenance.
- When to stop.

### Layer 5: Temporal Runtime

This is how elapsed time connects back.

The runtime computes state between LLM calls:

```json
{
  "now": "...",
  "elapsed_since_last_session": "...",
  "active_threads": [],
  "stale_claims": [],
  "upcoming_deadlines": [],
  "unresolved_commitments": [],
  "recommended_proactive_checks": []
}
```

This is not a separate paper unless evaluated. It is a product layer and a future benchmark axis.

## 6. Training Loop

### Do not train the current fresh-KG-per-turn loop as the final method

That loop teaches:

- create from isolated evidence,
- maybe noop,
- weak dedup,
- little update behavior,
- no consolidation,
- no relationship learning over time.

It is useful for smoke testing tool calls and reward code. It is not the real training setup.

### Real training unit

The training episode should be:

```text
history prefix H_0:t
existing memory M_t
new chunk/session C_t
writer/consolidator emits ops
memory becomes M_t+1
future task set Q_t+future scores M_t+1
reward = task utility - budget/cost/staleness penalties
```

### SFT first

Build a teacher dataset from:

- current `v1_write.py` heuristic writer,
- PIE prompted extraction,
- LLM consolidation prompts,
- successful traces from eval runs,
- human-edited gold examples for 20-50 hard cases.

SFT teaches format and tool grammar. It is not the final intelligence.

### RL second

Use GRPO or similar only after the eval harness is stable.

Recommended reward:

```text
R = task_score(memory_after)
    - λ_storage * memory_tokens
    - λ_read * retrieved_tokens
    - λ_write * write_calls
    - λ_stale * stale_wrong_answers
    - λ_unsupported * unsupported_claims
```

Counterfactual utility should move up a level:

```text
entry_utility = score(bank) - score(bank without entry)
region_utility = score(bank) - score(bank before consolidation)
```

Use single-write counterfactuals only as diagnostics. The better reward is region-level consolidation utility, because memory quality is emergent.

Scalable reward alternatives:

- Future-query likelihood: does memory improve prediction/answering of actual later user turns?
- Retrieval-use reward: when a reader succeeds, credit the memory items it actually used, not every possible write.
- Pairwise preference: judge compares answer with memory A vs memory B at the same token budget.
- Compression frontier: reward dominated Pareto points, not individual entries.
- Self-study synthetic QA: generate many future-looking queries from raw history, train memory to support them.
- Off-policy replay: cache `(raw region, memory bank, queries, scores)` so consolidation policies train without re-running every judge.
- Distilled reward model: train a small evaluator to approximate expensive LLM judge/counterfactual scores.

The counterfactual idea is still conceptually clean, but full leave-one-out over every write will not scale. Treat it as a gold diagnostic on small samples, then train cheaper proxy rewards.

### Random-K baseline

Random-K should not be the main reward if it confuses the objective.

Use it as an evaluation floor:

```text
At the same memory budget, does learned consolidation beat keeping K random raw chunks?
```

Also compare against:

- no memory,
- full raw RAG,
- BM25 raw,
- dense raw,
- prompted extraction,
- writer-only no consolidation.

## 7. Experiments To Run

### Experiment A: Sanity Pareto On LoCoMo Conv-26

Question:

Does any structured memory beat raw retrieval under budget?

Conditions:

- Full context.
- Raw BM25/dense RAG over turns.
- Raw BM25/dense RAG over sessions.
- Current PIE temporal baseline.
- Current writer-only memory.
- Writer + simple consolidation prompt.

Metrics:

- Overall score.
- Category scores.
- Memory tokens.
- Retrieved tokens per question.
- Cost and latency.

Pass condition:

At a strict budget, consolidated memory should beat raw random/session retention and writer-only.

### Experiment B: Temporal State Ablation

Question:

Does temporal transition structure actually help?

Conditions:

- Current-state only.
- Current-state plus observed timestamps.
- Full transition timeline.
- Full transition timeline plus validity fields.

Measure:

- Temporal questions.
- Knowledge-update/stale questions.
- Before/after questions.
- Questions where the latest state is not the answer.

Pass condition:

Transition/validity versions win specifically on temporal/update categories.

### Experiment C: Chunking Ablation

Question:

Should writes happen per turn, fixed chunk, or session?

Conditions:

- Turn.
- 6-turn chunk.
- 12-turn chunk.
- Session.
- Adaptive chunking by topic/session boundary.

Measure:

- Entity duplication rate.
- Create/update ratio.
- Relationship count.
- QA score.
- Write calls.

Pass condition:

Chunk/session writer should reduce duplicate creates and increase updates/relations without losing answer quality.

### Experiment D: Consolidator MVP

Question:

Can offline consolidation compress writer spam without hurting QA?

Implementation:

- Run writer on one conversation.
- Select all memories for that conversation as one region.
- Prompt an LLM to rewrite them into a budgeted memory bank.
- Preserve provenance.
- Evaluate QA before/after.

Budgets:

- 500 tokens.
- 1k tokens.
- 2k tokens.
- 5k tokens.

Pass condition:

Same or better QA at fewer tokens than writer-only.

### Experiment E: Personal Export Future-Query Eval

Question:

Does memory from earlier ChatGPT turns help answer actual future questions from the user?

Better than synthetic-only:

- Use each future user message as the query.
- Ask whether prior memory would have helped respond.
- Use actual future behavior as weak supervision.

Conditions:

- Raw previous conversations.
- PIE prompted extraction.
- Current writer.
- Consolidated memory.

Measure:

- Pairwise judge preference.
- Human spot-check on 50 examples.
- Token budget and retrieval cost.

Pass condition:

Consolidated memory wins pairwise over prompted extraction at lower token budget.

### Experiment F: Elapsed-Time Behavior

Question:

Does temporal runtime/state injection change agent behavior when time passes?

Conditions:

- No time context.
- Raw timestamps.
- Computed temporal briefing.
- Temporal briefing plus memory of commitments/deadlines.

Tasks:

- stale fact detection,
- deadline surfacing,
- gap-aware resumption,
- async multi-agent status,
- commitment follow-up.

Pass condition:

Computed temporal state beats raw timestamps. If it does not, the elapsed-time thesis is not adding enough.

## 8. Commands To Run Next

Use these in order. Stop after each one and inspect outputs.

### 1. Watch live writer decisions

```bash
python3 -m mempol.scripts.simulate_on_locomo \
  --conv-idx 0 \
  --max-turns 80 \
  --context-turns 12 \
  --checkpoint-every 1 \
  --sleep-sec 0.2 \
  --out-dir mempol/results/locomo_live_design_check
```

In another terminal:

```bash
streamlit run mempol/scripts/dashboard.py -- \
  --log_path mempol/results/locomo_live_design_check
```

What to inspect:

- Are lookup matches non-empty after memory exists?
- Is create/update ratio still insane?
- Are relations actually applied?
- Does `world_model.md` show duplicate entities?

### 2. Run chunked writer with no QA first

```bash
python3 -m mempol.scripts.eval_locomo_writer \
  --n-convs 1 \
  --chunk-size 12 \
  --max-chunks-per-conv 8 \
  --skip-qa \
  --run-name locomo_writer_chunks_noqa
```

Purpose:

- Verify ingestion completes.
- Inspect memory size and duplicates before spending judge calls.

### 3. Run small QA eval

```bash
python3 -m mempol.scripts.eval_locomo_writer \
  --n-convs 1 \
  --chunk-size 12 \
  --max-chunks-per-conv 8 \
  --max-qs-per-conv 20 \
  --read-policy v1_fast \
  --reader-k 8 \
  --judge-mode llm \
  --progress-every 5 \
  --run-name locomo_writer_chunks_20q
```

Purpose:

- First writer-memory answer score.
- Compare against existing conv-26 numbers: full context 75.1, naive RAG 66.1, PIE temporal 61.3.

### 4. Run chunk-size ablation

```bash
for C in 4 8 12 24; do
  python3 -m mempol.scripts.eval_locomo_writer \
    --n-convs 1 \
    --chunk-size $C \
    --max-chunks-per-conv 8 \
    --max-qs-per-conv 20 \
    --read-policy v1_fast \
    --reader-k 8 \
    --judge-mode llm \
    --run-name locomo_writer_chunk_${C}_20q
done
```

Purpose:

- Answer the per-turn vs chunk question empirically.

### 5. Re-run stable baselines on the same slice

```bash
python3 -m benchmarks.eval_harness \
  --benchmarks locomo \
  --baseline naive_rag \
  --subset 20
```

If the harness supports full context:

```bash
python3 -m benchmarks.eval_harness \
  --benchmarks locomo \
  --baseline full_context \
  --subset 20
```

Purpose:

- Do not compare writer results to a different question subset.

## 9. Missing Implementation For The New System

Add a new module:

```text
mempol/consolidation/
  types.py
  region_selector.py
  prompt_consolidator.py
  budget.py
  evaluator.py
```

Minimum objects:

```python
ConsolidationRegion:
    region_id
    memory_ids
    source_ids
    reason_selected
    token_budget

ConsolidatedMemory:
    memory_id
    text
    kind
    temporal_fields
    provenance
    supersedes
    utility_estimate
```

Minimum script:

```text
mempol/scripts/eval_consolidation.py
```

Flow:

```text
load writer-produced world model
select region
load provenance/raw chunks
produce consolidated memory bank at budget B
evaluate QA using same reader
write pareto row:
  budget, memory_tokens, score, category_scores, retrieved_tokens
```

## 10. Product Outputs

### Most realistic product

High-density memory compiler for long-running projects/users.

Input:

- ChatGPT export,
- Slack/Discord/Linear/GitHub history,
- agent logs,
- project docs.

Output:

- budgeted memory cards,
- temporal project state,
- unresolved commitments,
- stale assumptions,
- "what changed since last time",
- retrieval/debug dashboard.

Value:

- lower briefing cost,
- fewer repeated explanations,
- better project continuity,
- safer multi-agent orchestration,
- user-owned memory artifact instead of provider black box.

### Best demo

Show the same long history compressed to 500, 1k, 2k, 5k tokens, then ask future questions. Make the Pareto frontier visible.

### Multi-agent orchestration angle

Temporal state is especially useful for agents because agent work is stateful and concurrent:

- who owns what,
- when they started,
- what files/resources they touched,
- whether they are stale,
- whether outputs conflict,
- what changed since last coordinator step.

Mesa-style persistent filesystems solve durable workspace state. This project sits above that: semantic/temporal memory over what the agents did, decided, blocked on, and learned.

## 11. Paper Shape

Title direction:

```text
Budgeted Temporal Memory Consolidation for Long-Horizon Agents
```

Core contribution:

- A two-timescale memory architecture: fast writer + offline consolidator.
- A temporal state schema with validity/provenance/supersession.
- A budgeted evaluation protocol comparing memory quality per token.
- Empirical results on LoCoMo/LongMemEval/personal future-query eval.

Do not overclaim:

- Not "first temporal memory."
- Not "graph beats RAG."
- Not "single-write reward solves memory."

Claim:

> Offline consolidation of temporally valid state transitions improves the memory-quality/cost frontier over raw retrieval and writer-only extraction.

## 12. Immediate Decision

The next code work should not be more prompt tweaking inside the online writer.

The next code work should be:

1. Run current writer eval on a fixed LoCoMo slice.
2. Build the simplest offline consolidator.
3. Produce a budget/score table.
4. Only then decide whether RL is worth training.

## 13. Bitter-Lesson Rewrite Of The Research Bet

Less promising:

- More hand-authored memory types.
- More regex/threshold entity resolution.
- Bigger prompt taxonomies.
- Per-turn isolated write decisions.
- Per-op counterfactual as the only reward.

More promising:

- Learned segmentation over raw streams.
- Learned region consolidation.
- Learned retrieval/decompression at read time.
- Budgeted Pareto optimization.
- Self-study data generation from raw histories.
- Joint training of reader + retriever/ranker + memory compressor.
- Keeping raw source logs immutable so learned representations can improve later.

The deep-learning-shaped version of this project:

```text
Raw histories
  -> learned boundary detector
  -> candidate episodes / regions
  -> consolidator produces compressed memory bank
  -> reader recursively retrieves/decompresses when needed
  -> reward from future task utility under token budget
```

The old KG can remain as one view and one baseline. It should not be the center of the invention.

Until there is a Pareto table, there is no paper and no product signal.
