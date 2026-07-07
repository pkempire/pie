Combined context pack — single source of truth for precise codebase answers.

Top quick facts (plain & prominent)
- Default GRPO / write env group_size = 4 (explicit): WriteEnvGroupBuilder default group_size = 4. WriteRLDatasetBuilder default group_size = 4. This is the default used when constructing write-training groups (GRPO). [src: mempol/recipes/memory_rl/write_env.py] [src: mempol/recipes/memory_rl/write_env.py (WriteRLDatasetBuilder)]
- LLM judge bucketing → 1.0 threshold: The LLM judge's numeric score is bucketed to 1.0 when score >= 0.75. (Buckets: >=0.75 → 1.0; >=0.25 → 0.5; else 0.0.) [src: mempol/eval/judge.py] 
- Per-op counterfactual credit assignment (plain): per_op_counterfactual replays the full ops_log to compute full_state_score. For each mutating op index it replays a leave-one-out variant (skips that mutating op), re-scores the QA battery, and records the mean score delta relative to full_state. Each op's marginal utility = that mean delta. Trajectory reward = sum(per-op deltas) − cost_per_mut * n_mutating_ops. WriteReward normally passes cost_per_op (default 0.001). [src: mempol/eval/counterfactual.py] [src: mempol/recipes/memory_rl/write_reward.py]
- WriteReward default per-op cost: DEFAULT_COST_PER_OP = 0.001. [src: mempol/recipes/memory_rl/write_reward.py]

Key write-environment clarifications (concise & explicit)
- WriteEnvGroupBuilder defaults (explicit): group_size = 4 (default), max_turns = 32 (default), format_coef = 0.0 (default), max_trajectory_tokens = 4096 (default), context_overflow_reward = -0.1 (default). Each env built gets a fresh PIEBackend and WriteTool and shares a frozen R runner (if provided via MEMPOL_R_CHECKPOINT). Tools exposed to agents: lookup_entity, lookup_relation, create_entity, update_state, merge_entities, add_relation, mark_contradiction, forget, noop. [src: mempol/recipes/memory_rl/write_env.py]
- WriteRLDatasetBuilder defaults (explicit): model_name_for_tokenizer (required), n_convs=8, train_frac=0.8, batch_size=8, group_size = 4 (default), max_turns=4, max_battery_per_turn=0, min_battery_per_turn=1, n_prior_turns_in_context=2, seed=0, renderer_name optional. Builder produces WriteDatum items per (conv,turn) and yields (train_ds, eval_ds). [src: mempol/recipes/memory_rl/write_env.py]

Concise supporting write-tool & reward facts (for precise answers)
- WriteTool ops & non-mutating list: NON_MUTATING_OPS = ("lookup_entity","lookup_relation","noop"). MUTATING_OPS = ("create_entity","update_state","merge_entities","add_relation","mark_contradiction","forget"). WriteTool appends (op_name,args) to ops_log BEFORE executing impls; it also maintains counters n_lookups, n_creates, n_updates, n_merges, n_relations, n_contradictions, n_forgets, n_noops. [src: mempol/recipes/memory_rl/write_tools.py]
- WriteReward mix & env var defaults: env vars parsed into floats/ints with defaults: MEMPOL_W_CF "0.7" → w_cf=0.7; MEMPOL_W_QA "0.3" → w_qa=0.3; MEMPOL_W_GAIN "0.0" → w_gain=0.0; MEMPOL_W_OVERLAP "0.0" → w_overlap=0.0; MEMPOL_W_COV_FLOOR "0.05" → w_cov_floor=0.05; MEMPOL_K_MAX "12" → k_max=12. WriteReward default cost coefficients: DEFAULT_COST_PER_OP = 0.001, DEFAULT_COST_PER_LOOKUP = 0.0, DEFAULT_COST_PER_ENTITY = 0.0. [src: mempol/recipes/memory_rl/write_env.py] [src: mempol/recipes/memory_rl/write_reward.py]
- WriteReward behavior (concise): Computes battery_coverage, optionally per_op_counterfactual (if w_cf>0 and ops_log present), optionally answer_gain and reader_overlap; mean_qa comes from cf_result.full_state_score (if cf used) or by running r_runner and judge; counts ops using WriteTool counters (preferred) or by scraping assistant tool_calls; cost applied = cost_per_op * n_mutations + cost_per_lookup * n_lookups + cost_per_entity * n_entities; final reward = weighted sum (w_cf*cf_reward + w_qa*mean_qa + w_gain*mean_gain + w_overlap*mean_overlap) + coverage_floor − costs. May dump trajectory JSON if MEMPOL_TRAJECTORY_DUMP_DIR set. [src: mempol/recipes/memory_rl/write_reward.py]

Counterfactual essentials (concise)
- per_op_counterfactual behavior: identifies mutating op indices via _classify_ops (based on WriteTool.MUTATING_OPS), replays full ops to compute full_mean, then in parallel replays leave-one-out variants for each mutating op to compute per-op mean deltas. Returns PerOpReward with fields: trajectory_reward = sum(deltas) − cost_per_mut*len(mut_indices), per_op_deltas list, n_ablated (mutating ops count), n_battery, full_state_score (full_mean). Caller (WriteReward) typically supplies cost_per_op=0.001. [src: mempol/eval/counterfactual.py]

Related evaluation primitives (concise)
- Judge behavior: mempol.eval.judge returns 0.0 for empty/"not in context"/error outputs; otherwise calls llm.chat with config.JUDGE_MODEL and expects JSON {"score":..., "reason":...}. Numeric score parsed then bucketed to {1.0 if >=0.75, 0.5 if >=0.25, else 0.0}. [src: mempol/eval/judge.py]
- Evidence coverage: battery_coverage computes fraction of evidence dia_ids that are present in backend stored dia ids. [src: mempol/eval/evidence_coverage.py]
- Answer gain: battery_answer_gain compares to deterministic random-K baseline (cached per conv/question/K) to get judge-margin gain. [src: mempol/eval/answer_gain.py]
- Reader overlap: enforce_budget prunes PIEBackend.wm.entities to at most k_max and battery_reader_overlap computes fraction of reference dia_ids recovered from stored provenance. [src: mempol/eval/reader_overlap.py]

Where to look for authoritative lines (short pointers)
- WriteEnvBuilder & defaults, env var parsing, how WriteRLDatasetBuilder constructs groups: mempol/recipes/memory_rl/write_env.py [src: mempol/recipes/memory_rl/write_env.py]
- WriteTool op logging, op classification constants, and counters: mempol/recipes/memory_rl/write_tools.py [src: mempol/recipes/memory_rl/write_tools.py]
- WriteReward dense mix, per-op CF integration, default costs, and trajectory dump: mempol/recipes/memory_rl/write_reward.py [src: mempol/recipes/memory_rl/write_reward.py]
- Per-op leave-one-out implementation and exact function names/returns: mempol/eval/counterfactual.py [src: mempol/eval/counterfactual.py]
- Judge bucketing & behavior: mempol/eval/judge.py [src: mempol/eval/judge.py]

If you need more granular code lines or another specific default not explicitly listed here, mark it MISSING-FROM-PACK with the source path to add.