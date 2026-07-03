# mempol — Code Review (2026-06-06)

Scope: full-depth read of the `mempol/` RL framework. Verified against the actual
source (not docstrings/claims). Environment note: in this checkout `tinker`,
`tinker_cookbook`, and `dspy` are **not installed**; `pie` and `mempol` import
fine with `PYTHONPATH=<repo root>`. All file:line refs are against the tree as of
HEAD `f0c90e3` ("Coverage floor: non-zero gradient when counterfactual collapses").

---

## 0. TL;DR

- **Two independent, runnable pipelines exist today.** (a) The **DSPy/GEPA
  consolidator** (`dspy_consolidator/` + repo-root `scripts/run_gepa_consolidator.py`)
  — genuinely ran, produced real artifacts (`results/gepa_consolidator/`,
  baseline 0.6 -> GEPA 0.8 on conv-26/5Q). (b) The **write-policy GRPO loop**
  (`recipes/memory_rl/train_write.py` -> `write_env` -> `write_tools` ->
  `write_reward` -> `eval/counterfactual`) — fully wired, imports clean, but
  **requires a tinker-cookbook clone to actually train** and has not been run at
  scale here (only smoke dirs under `results/`).
- **The expensive thing** is the per-op counterfactual reward
  (`eval/counterfactual.py`). Per write trajectory it costs
  **`(K_mut + 1) x Q` reader-runs and the same number of judge calls**, where
  `K_mut` = #mutating ops and `Q` = battery size. Each reader-run additionally
  triggers entity **re-embedding** inside a freshly-rebuilt `PIEBackend`. With
  GRPO group size `G` and batch `B`, one training step costs `B x G x (K_mut+1) x Q`
  of each. Full quote + derivation in section 2.
- **Genuinely never-run / broken:** Phase A read-policy LoRA as a *frozen R* is a
  hard `NotImplementedError` (`write_reward.py:558-580`); the heuristic R is always
  used. `cotrain.py` orchestrates A<->B but has never produced checkpoints here and
  its eval/stopping step is a `TODO` (`cotrain.py:240`). The `counterfactual._smoke`
  patches the wrong judge symbol so it silently scores 0 (see section 5).

---

## 1. The ACTUAL end-to-end training pipeline (file-by-file)

There is **no single "the" pipeline**; there are three entry points. Ranked by
how real/wired they are.

### 1A. Write-policy GRPO (the wired-in RL hot path) — `train_write`

Entry command (from `train_write.py:18-22` docstring, must run inside a
tinker-cookbook clone where `mempol` is importable as
`tinker_cookbook.recipes.memory_rl`):

```
python -m tinker_cookbook.recipes.memory_rl.train_write \
    n_convs=8 train_frac=0.8 batch_size=4 group_size=8 max_turns=4 \
    learning_rate=4e-5 lora_rank=32 log_path=/tmp/mempol/phaseB_v1
```

Data flow, function by function:

| Step | File:symbol | What it does |
|---|---|---|
| CLI/chz config -> `train.Config` | `train_write.py:105 cli_main` | builds `WriteRLDatasetBuilder`, hands to `tinker_cookbook.rl.train.main` |
| Dataset build | `write_env.py:336 WriteRLDatasetBuilder.__call__` | `load_locomo(n_convs)`; per conv builds `evidence_index: dia_id->[QA]`; one `WriteDatum` per turn that >=1 QA's `evidence` references. Builds per-conv `FlatBackend` over chunked units (`data._conv_to_serializable_units`, W=6/S=3). |
| Group build (GRPO) | `write_env.py:197 WriteEnvGroupBuilder.make_envs` | for each of `group_size` members: fresh `PIEBackend()` + `WriteTool`; reads reward-mix from env vars (`MEMPOL_W_CF=0.7`, `W_QA=0.3`, `W_GAIN=0`, `W_OVERLAP=0`, `W_COV_FLOOR=0.05`, `K_MAX=12`); wraps tools via `build_agent_tool_env`. |
| Rollout tools | `write_tools.py WriteTool` | 9 tools. Each public tool appends `(op_name, args)` to `ops_log` **before** dispatching to a `_*_impl` (`write_tools.py:181-263`). Counters (`n_creates`, ...) are ground truth for cost. |
| Reward | `write_reward.py:228 WriteReward.__call__` | (1) `enforce_budget` prunes KG to `k_max`; (2) `battery_coverage` (free); (3) **`per_op_counterfactual`** (primary, section 2); (4) QA anchor reused from `cf_result.full_state_score`; (5) cost from `WriteTool` counters. |
| Reward composition | `write_reward.py:387-394` | `reward = w_cf*cf + w_qa*mean_qa + w_gain*gain + w_overlap*overlap + w_cov_floor*cov - cost`. With defaults: `0.7*cf + 0.3*qa + 0.05*cov - cost`. |

The frozen reader R is **always** `HeuristicPolicy(first_k=8, final_k=4, ...)`
(`write_reward.py:548`). `resolve_r_runner_from_env()` is called
(`write_env.py:217`) but returns `None` unless `MEMPOL_R_CHECKPOINT` is set, and
even then `make_tinker_r_runner` **raises `NotImplementedError`** -> falls back to
heuristic. So Phase-B-v2 ("trained R as judge") does not exist in runnable form.

### 1B. Read-policy GRPO (Phase A) — `train.py`

Separate, parallel loop. `train.py -> memory_env.MemoryRLDatasetBuilder ->
MemoryTool (read tools: memory_search/expand/filter/top_n/rerank) ->
reward.JudgeReward`. `JudgeReward` (`reward.py:32`) is a simple
`format_coef*(fmt-1) + judge(answer)` over the final `Answer:` line. Self-contained
and runnable in a cookbook clone. It is the loop `cotrain` shells out to for Phase A.
Nothing here is stubbed, but **no Phase A run output exists** in `results/` (all
dirs are write/eval/baseline/universal smoke).

### 1C. Co-training outer loop — `cotrain.py`

`cotrain.py:201 cli_main` alternates `_train_R_phase` (shells `-m ...train`) and
`_train_W_phase` (shells `-m ...train_write`, passing `r_checkpoint=` from the
previous R via `_extract_checkpoint_path`, a regex scrape of the log —
`cotrain.py:125`). **Status: orchestration complete but unproven.** Real-cost
guardrails in the docstring (~$1.5-2.5k for T=5). Per-iter eval + stopping is an
explicit `TODO` (`cotrain.py:240`), so it just runs T fixed iterations. Since the
R-as-judge swap is not implemented (section 5), the W phase in every iteration
silently uses the heuristic R regardless of `r_ckpt`.

### 1D. GEPA consolidator (the thing that actually produced numbers)

`scripts/run_gepa_consolidator.py` (repo root, **untracked**) optimizes
`dspy_consolidator.Consolidator` (one `dspy.ChainOfThought`) with
`dspy.teleprompt.GEPA`. Metric (`run_gepa_consolidator.py:130 make_metric`):
run consolidator -> flatten entries to Units -> `FlatBackend` -> retrieve top-k ->
`answer_question` -> `judge`. Real run recorded in
`results/gepa_consolidator/summary.json`: baseline 0.6 -> GEPA 0.8 (+0.2),
30 metric calls, 150 answer + 150 judge calls, conv-26, 5 questions. Only pipeline
with a real result artifact and a real `gepa_state.bin`. Requires `dspy` + OpenAI.

---

## 2. Per-op counterfactual reward — cost (the optimization target)

File: `eval/counterfactual.py`. Called from `write_reward.py:270`.

### The cost formula

Let `K_mut` = number of **mutating** ops in the trajectory (lookups/noops are
excluded — `_classify_ops` filters by `WriteTool.MUTATING_OPS`,
`counterfactual.py:81-84`), and `Q` = `len(battery)`.

**Per trajectory** (one env / one group member):

```
reader.run calls    = (K_mut + 1) x Q     # 1 full state + K_mut leave-one-out, each over Q questions
judge calls         = (K_mut + 1) x Q     # one judge per reader.run
PIEBackend replays  = K_mut + 1           # each replay rebuilds the KG from scratch and re-embeds entities
```

**Per GRPO step**, multiply by group size `G` and batch size `B`:
`B x G x (K_mut+1) x Q` reader-runs and the same number of judge calls.

The "+1" full-state pass is **reused** as the QA anchor (`mean_qa =
cf_result.full_state_score`, `write_reward.py:331-332`), so the anchor term is free
given the counterfactual. But there is **no caching across leave-one-out variants**:
every variant re-runs the reader (which itself does reformulate + route LLM calls +
retrieval) and a judge on all `Q` questions, even for questions whose evidence
cannot possibly be affected by the ablated op.

### The exact code

Full-state pass and per-op fan-out (`counterfactual.py:189-205`):

```python
    # 1. Score the full trajectory once.
    full_backend = _replay(ops_log, current_dia_id, current_timestamp)
    full_scores = await _score_battery(reader, full_backend, battery)
    full_mean = sum(full_scores) / max(len(full_scores), 1)

    # 2. Score each leave-one-out variant in parallel.
    async def _delta_for(idx: int) -> tuple[str, float]:
        leave_out_ops = [op for j, op in enumerate(ops_log) if j != idx]
        b = _replay(leave_out_ops, current_dia_id, current_timestamp)
        scores_minus = await _score_battery(reader, b, battery)
        deltas = [s_full - s_minus for s_full, s_minus
                  in zip(full_scores, scores_minus)]
        op_name = ops_log[idx][0]
        return op_name, sum(deltas) / max(len(deltas), 1)

    per_op = await asyncio.gather(*[_delta_for(i) for i in mut_indices])
```

`_score_battery` is where the reader+judge fan-out over `Q` happens
(`counterfactual.py:131-154`):

```python
async def _score_battery(reader, backend, battery, judge_fn=_judge_sync) -> list[float]:
    loop = asyncio.get_running_loop()
    async def _one(q: str, gold: str) -> float:
        trace = await loop.run_in_executor(None, reader.run, q, backend)   # HeuristicPolicy: reformulate + route + retrieve LLM calls
        ans = trace.answer or "not in context"
        score, _ = await loop.run_in_executor(None, judge_fn, q, gold, ans)  # 1 judge LLM call
        return float(score)
    return await asyncio.gather(*[_one(q, g) for q, g, _ev in battery])
```

### Hidden multiplier: re-embedding inside `_replay`

Each `_replay` builds a **fresh `PIEBackend()`** (`counterfactual.py:96`) and replays
the ops. The reader's first `retrieve`/`lookup` then calls
`PIEBackend._ensure_embeddings` (`pie_kg.py:263-273`) which embeds entities lacking
vectors via `llm.embed(...)`. So every one of the `(K_mut+1)` replays pays an
embedding pass the first time the reader touches it (disk-cached by `llm.embed`,
so amortized across identical entity texts, cold on first appearance). The
docstring's "~16 R+judge calls for K=4, Q=4" (`counterfactual.py:39-42`)
**undercounts**: it is `(4+1)x4 = 20` reader-runs and 20 judge calls, not 16 — the
"+1" full pass is omitted in the docstring arithmetic. With the heuristic reader
doing reformulate+route (2 extra LLM calls) per question, the real LLM-call count
per trajectory is closer to `(K_mut+1) x Q x 3` (reformulate+route+answer) plus
`(K_mut+1) x Q` judge.

### Cheap wins (no design change)

- Filter the battery per ablated op to only questions whose `evidence` overlaps the
  ablated op's `source_dia_id`; for the rest, `delta = 0` by construction. This is
  exactly the argument used to skip lookups/noops but is **not** applied at the
  question level.
- Memoize `reader.run(q, backend)` keyed by the backend's stored-dia-id set: many
  leave-one-out states are identical from the reader's retrieval POV.
- Disable `do_reformulate`/`do_route` on the frozen R used *inside* the
  counterfactual; each costs an LLM call x `(K_mut+1) x Q`.

---

## 3. Backends — interface conformance & status

Interface: `backends/base.py` `Backend` ABC = `ingest`, `retrieve` (abstract),
`expand`, `filter_by_time` (default impls). `Unit`/`Hit` dataclasses.

| Backend | File | Status | Notes |
|---|---|---|---|
| `FlatBackend` | `flat.py` | **Works, core dependency** | BM25 + dense (OpenAI embed) + RRF hybrid. Substrate for full-text backend, GEPA, eval runner. Solid. |
| `PIEBackend` | `pie_kg.py` | **Works, RL hot path** | Wraps external `pie.core.world_model.WorldModel`. All write ops implemented (create/update/merge/add_relation/mark_contradiction/forget). `_ensure_embeddings` does live OpenAI embeds. **Hard dep on the `pie` package** (present in repo root). |
| `ProviderBackend` | `providers.py` | Shim, works if providers do | Adapts `memory_providers.*` (mem0/zep/supermemory/honcho/pie) to the ABC. Each `make_*` lazily imports a provider needing its own keys/SDK. |
| `MastraBackend` | `mastra.py` | Standalone, LLM-heavy | Observer/Reflector pattern over LoCoMo; no external HTTP — runs via `llm.chat`. Self-contained but expensive. **Modified (uncommitted).** |
| `GitMemBackend` | `gitmem.py` | **Untracked, orphaned** | 22 KB git-log-style memory. Not imported by any RL/eval entry point; `__main__` smoke only. Not wired in. |

`base.filter_by_time` silently drops hits whose `metadata['timestamp']` is `None`
(`base.py:46-47`) — a correctness footgun if metadata lacks timestamps (provider
hits often do).

---

## 4. Eval signals — live vs deprecated

| Module | Reward role | Cost | Status |
|---|---|---|---|
| `eval/counterfactual.py` | **PRIMARY** (`w_cf=0.7`) | `(K_mut+1)xQ` reader+judge | Live. Section 2. |
| `eval/judge.py` | Used everywhere | 1 LLM call | Live. Buckets to {0,0.5,1.0}. Now logs parse failures (`judge.py:42-49`). |
| `eval/evidence_coverage.py` | Coverage **floor** (`w_cov_floor=0.05`) + diagnostics | **Free** (no LLM) | Live, cheap, deterministic. The chicken-and-egg breaker when cf collapses. |
| `eval/answer_gain.py` | `w_gain=0.0` -> **off by default** | `Q` reader+judge (baseline cached) | Deprecated ("v2"), kept for ablation. |
| `eval/reader_overlap.py` | `w_overlap=0.0` -> **off by default** | ~`Q` reader-runs + reformulate | Deprecated ("v1", confirmed structurally biased low, `write_reward.py:117-121`). `enforce_budget` (used live) lives here. |
| `eval/runner.py` | non-RL `(backend,policy)` eval harness | `Q` reader+judge per conv | Live, standalone (`python -m mempol.eval.runner --backend flat --policy v1_heuristic`). |
| `eval/metrics.py`, `qa_generator.py` | helpers | — | Present; `qa_generator` only used by some scripts. |

Of the four downstream reward signals, **two are live** (counterfactual +
coverage-floor) and **two are dead-by-default** (answer_gain, reader_overlap) but
still importable and still executed if their weight env var is set > 0.

---

## 5. Genuinely broken / never-run

1. **Trained R-as-frozen-judge: hard `NotImplementedError`.**
   `write_reward.py:558-580 make_tinker_r_runner` raises unconditionally;
   `resolve_r_runner_from_env` catches it and falls back. Net effect: every W /
   cotrain run uses `HeuristicPolicy` as R, regardless of `r_checkpoint=`. The
   entire "Phase B v2" / "co-trained R" story is non-functional in code.

2. **`cotrain.py` never produced output here and has a `TODO` eval.**
   Orchestration is real but (a) depends on #1 being implemented to be meaningful,
   (b) extracts checkpoints by regex-scraping subprocess logs
   (`cotrain.py:125-134`) — fragile, (c) eval/stop is a `TODO` (`cotrain.py:240`).

3. **`counterfactual._smoke` is mis-patched.** It patches
   `mempol.eval.counterfactual._judge_sync` (`counterfactual.py:252`) but
   `_score_battery`'s default arg `judge_fn=_judge_sync` is **bound at function
   definition time**, so the patch never takes effect; the smoke hits the real
   `gpt-4o-mini` judge with `"Boston"` as input and logs three JSON-parse failures,
   scoring 0.0. Confirmed by running it. The hot path is fine — only the smoke's
   claim of correctness is invalid.

4. **`WriteRLDataset.__len__` floor-divide guard** (`write_env.py:305-311`) returns
   `max(1, ...)` to avoid silent zero-batch no-ops — good defensive fix, but signals
   this class of bug has bitten before.

5. **Untracked production code.** `dspy_consolidator/`, `gitmem.py`, `core/`,
   `recipes/memory_rl/universal_*` (`train_universal`, `universal_env/reward/tools`),
   and several `scripts/` are **git-untracked** (`??`). The GEPA path that produced
   the headline result lives partly in untracked files
   (`scripts/run_gepa_consolidator.py`). Risk: newest, most-cited work is uncommitted.

---

## 6. External-service dependencies (what fails offline)

| Dependency | Used by | Offline behavior |
|---|---|---|
| **OpenAI** (`llm.chat`, `llm.embed`) | everything: judge, reader, embeddings, GEPA, consolidator | Hard fail at first call. No mock path in production code. `llm.embed` has a disk cache so re-runs on identical text are free, but cold text needs the API. |
| **tinker / tinker_cookbook** | `train.py`, `train_write.py`, `cotrain.py`, `write_env`, `memory_env` (guarded) | Modules import via the `HAS_TINKER`/`tinker_compat` guard, but `build_agent_tool_env` **raises** (`tinker_compat.py:86-91`) and the `train_*` CLIs import `tinker_cookbook.rl.train` at top level -> **ImportError on launch**. Not installed here. |
| **dspy** | `dspy_consolidator/*`, `scripts/run_gepa_consolidator.py` | ImportError on launch. Not installed here. |
| **`pie` package** | `PIEBackend` (and thus the entire W reward via `_replay`) | Present in repo root (`pie/core/...`); imports fine. If missing, `eval/counterfactual.py` and `write_reward.py` would fail to import. |
| `memory_providers.*` SDKs | `backends/providers.py` `make_*` | Lazy — only fail when that specific backend is constructed. |

Practical consequence: to train W you need OpenAI **and** a tinker-cookbook clone
**and** tinker compute. To run the GEPA consolidator you need OpenAI **and** dspy.
The cheapest fully-runnable thing offline is *nothing* — even `eval/runner.py` needs
OpenAI for embeddings + judge.

---

## 7. Biggest code-quality / correctness risks (ranked)

1. **Counterfactual cost is super-linear in ops and unbounded by relevance**
   (section 2). No per-question evidence filtering, no reader-result memoization
   across near-identical leave-one-out states. Dominant training cost and the
   clearest optimization target.

2. **The "primary" signal is degenerate exactly when you most need it.** When the
   heuristic R can't answer any battery question against `M_full`,
   `full_state_score=0` -> all leave-one-out scores 0 -> all per-op deltas 0 -> `cf=0`.
   The code knows this (the entire `w_cov_floor` mechanism, `write_reward.py:140-158`,
   exists to paper over it). Early W training is therefore driven almost entirely by
   the 0.05 coverage floor minus cost, i.e. "preserve evidence dia_ids" — a
   retrieval-recall objective, not the advertised downstream-QA objective. Until a
   competent R exists (blocked by #5/section 5.1), the per-op counterfactual mostly
   contributes noise + cost.

3. **Frozen-R quality ceiling.** R is a hand-tuned heuristic doing
   reformulate+route+rerank with OpenAI calls. Every reward eval's fidelity is
   bounded by it, and section 5.1 means the "train a better R and swap it in" escape
   hatch is unimplemented. The co-training thesis hinges on a function body that
   raises `NotImplementedError`.

4. **Replay fidelity / ordering.** `_replay` swallows per-op exceptions
   (`counterfactual.py:122-127`) on the theory a failed op "wouldn't have applied in
   the counterfactual either." But op **ordering matters**: ablating an early
   `create_entity` makes a later `update_state`/`merge_entities` reference a
   now-missing uid, so its `_impl` early-returns `{"ok": False}` and is silently a
   no-op. The leave-one-out state for op i therefore also loses the *downstream*
   effects of every op that depended on i — the per-op delta conflates op i's
   marginal value with its dependents'. Defensible as "true marginal contribution
   including dependents," but it is **not** the independent per-op credit the
   docstring claims (`counterfactual.py:26-29`).

5. **Op-count source-of-truth split.** Cost uses `WriteTool` counters when present
   else scrapes history (`write_reward.py:358-372`); `ops_log` (used by the
   counterfactual) is a *third* record. Three parallel accounts of "what the policy
   did" that can drift (e.g. a tool that raises after appending to `ops_log` but
   before incrementing its counter).

6. **Untracked critical code** (section 5.5) — the GEPA win and the
   `core/`/`universal_*` work are not committed; trivial to lose.

7. **`config.py` defaults to paid reasoning models everywhere**
   (`ANSWER/REFORMULATE/OBSERVER/REFLECTOR = gpt-5-mini`, judge `gpt-4o-mini`,
   embed `text-embedding-3-large`). The counterfactual's `(K_mut+1)xQx~3` LLM calls
   at `gpt-5-mini` per *trajectory* x `G` x `B` x steps is the real dollar driver,
   not Tinker compute, for small smoke runs.

---

## Appendix: import-sanity matrix (this checkout)

Ran `importlib.import_module` with `PYTHONPATH=<repo root>`:

```
OK   mempol.eval.judge
OK   mempol.eval.evidence_coverage
OK   mempol.eval.counterfactual
OK   mempol.backends.pie_kg
OK   mempol.recipes.memory_rl.write_reward
OK   mempol.recipes.memory_rl.write_tools
OK   mempol.policies.v1_heuristic
```

`train.py` / `train_write.py` / `cotrain.py` are **not** import-clean here (top-level
`from tinker_cookbook.rl import train`). `dspy_consolidator` needs `dspy`.
`eval/counterfactual.py` `_smoke` runs end-to-end but mis-patches the judge
(section 5.3).
