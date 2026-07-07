Combined context pack — single source of truth for precise codebase answers.

Top quick facts (plain & prominent)
- LLM judge bucketing → 1.0 threshold: The LLM judge's raw numeric score is bucketed to 1.0 when score >= 0.75. (Buckets: >=0.75 → 1.0; >=0.25 → 0.5; else 0.0.) [src: mempol/eval/judge.py] 
- How per-op counterfactual assigns credit (plain): per_op_counterfactual replays the full ops_log to compute full_state_score. Then for each mutating op index it replays a leave-one-out variant (skips that mutating op), re-scores the QA battery, and records the mean score delta relative to full_state. Each op's marginal utility = that mean delta. Trajectory reward = sum(per-op deltas) − cost_per_mut * n_mutating_ops. WriteReward normally passes cost_per_op (default 0.001) so effective per-op penalty in WriteReward is typically 0.001 per mutating op. [src: mempol/eval/counterfactual.py] [src: mempol/recipes/memory_rl/write_reward.py]
- Default GRPO group size used for write training: default group_size = 4 (WriteRLDatasetBuilder default and typical WriteEnvGroupBuilder construction uses group_size=4). [src: mempol/recipes/memory_rl/write_env.py] [src: mempol/recipes/memory_rl/write_env.py (WriteRLDatasetBuilder)]

Core quick facts / clarifications (concise)
- Embedding cache key & path: embeddings cached on-disk per-model keyed by sha1(text) hex digest. Cache path = config.CACHE_DIR / f"emb_{model.replace('/', '_')}.jsonl". [src: mempol/llm.py]
- Write-tool non-mutating ops: NON_MUTATING_OPS = ("lookup_entity","lookup_relation","noop"); mutating ops = all others in op vocabulary (create_entity, update_state, merge_entities, add_relation, mark_contradiction, forget). [src: mempol/recipes/memory_rl/write_tools.py]
- Default LLMs in config: ANSWER_MODEL="gpt-5-mini", REFORMULATE/OBSERVER/REFLECTOR default "gpt-5-mini", JUDGE_MODEL="gpt-4o-mini", EMBED_MODEL="text-embedding-3-large". [src: mempol/config.py]

Orchestration / CoTrain
- CoTrainConfig defaults: n_outer=5; r_steps_per_iter=200; w_steps_per_iter=100; model_name="Qwen/Qwen3-4B-Instruct-2507"; lora_rank=32; learning_rate=4e-5; r_batch_size=4; r_group_size=8; r_max_turns=6; w_batch_size=4; w_group_size=8; w_max_turns=4; w_max_battery_per_turn=6; n_convs=8; train_frac=0.8; seed=2; log_path=None; wandb_project=None; behavior_if_log_dir_exists="delete". Mechanism alternates Phase A (train R) and Phase B (train W) launching subprocess CLIs and storing history.json. [src: mempol/recipes/memory_rl/cotrain.py]
- Subprocess helpers: _run_subprocess runs CLI async teeing stdout/stderr to log_file and returns exit code; _extract_checkpoint_path extracts "tinker://..." sampler_path from logs. [src: mempol/recipes/memory_rl/cotrain.py]
- Phase helpers: _train_R_phase launches train; _train_W_phase launches train_write and sets MEMPOL_R_CHECKPOINT when provided. [src: mempol/recipes/memory_rl/cotrain.py]

Data conversion & MemoryDatum
- MemoryDatum TypedDict keys: question, gold_answer, category, data_source, sample_id, qid, conversation_units (list of chunk dicts). _conv_to_serializable_units creates sliding-window chunks with metadata (session, session_date, first/last dia_id, dia_ids, speaker, n_turns, chunk_idx_in_session, timestamp, dia_id). [src: mempol/recipes/memory_rl/data.py]

Memory-read RL environment (read-policy)
- MEMORY_TASK_INSTRUCTIONS: tools (memory_search, memory_expand, memory_filter, memory_rerank, memory_top_n). Reward = correctness − tool-use cost; format_coef default 0.1. [src: mempol/recipes/memory_rl/memory_env.py]
- MemoryEnvGroupBuilder ctor: datum, model_name, renderer_name, max_turns, group_size, format_coef=0.1, max_trajectory_tokens=32*1024, max_generation_tokens=None, context_overflow_reward=-0.1. make_envs builds independent FlatBackend per env, MemoryTool, JudgeReward, and agent env with the memory tools. logging_tags → [data_source, category]. [src: mempol/recipes/memory_rl/memory_env.py]
- MemoryRLDataset and MemoryRLDatasetBuilder: builder defaults include model_name_for_tokenizer required, dataset="locomo", n_convs=8, batch_size=16, group_size=8, max_turns=6, format_coef=0.1. __call__ returns (train_ds, eval_ds). [src: mempol/recipes/memory_rl/memory_env.py]

Read-side reward (JudgeReward)
- JudgeReward fields: gold_answer, question, format_coef=0.1, correct_reward=1.0, partial_reward=0.5, wrong_reward=0.0. __call__ extracts last assistant Answer line; format_ok = 1.0 if Answer present else 0.0; if Answer present, runs mempol.eval.judge synchronously in thread; judge_score bucketed to {1.0 if >=0.75, 0.5 if >=0.25, else 0.0}. Reward = format_coef*(format_ok-1) + correctness_value. Returns (reward, {"format":..., "correct":...}). [src: mempol/recipes/memory_rl/reward.py] [src: mempol/eval/judge.py]

Tinker compatibility shim
- Exports tool, build_agent_tool_env, simple_tool_result, ToolResult, HAS_TINKER. If tinker_cookbook absent, wrappers provide .to_spec() and simple fallbacks; build_agent_tool_env raises RuntimeError instructing to install cookbook in fallback. [src: mempol/recipes/memory_rl/tinker_compat.py]

Memory tools (MemoryTool)
- MemoryTool fields: backend, last_hits, n_searches, max_searches=8. Methods/tools: memory_search(query,k=10,source="hybrid") enforces max_searches and clamps k; memory_expand(seed_uids,k_per=2) limits seeds to 5 and merges avoiding duplicates; memory_filter(predicate,value) supports several predicates (session_*/speaker_eq/date_*/type_eq/keyword_in/not_in) and updates last_hits with robust parsing; memory_rerank(strategy,query) supports dense reordering or session sort; memory_top_n(n) truncates last_hits. All return simple_tool_result observations. [src: mempol/recipes/memory_rl/tools.py]

Universal memory (mixed write/read)
- UNIVERSAL_MEMORY_INSTRUCTIONS require freezing raw access before final answer; tools: search_raw_spans, write_memory_state, freeze_raw_access, retrieve_memory_states. [src: mempol/recipes/memory_rl/universal_env.py]
- UniversalMemoryTool: store (SQLiteMemoryStore), counters raw_searches, memory_searches, writes, token_cost, written_state_ids; enforces max_raw_searches=8, max_memory_searches=8, max_writes=24, raw_enabled flag. Methods: search_raw_spans_impl, write_memory_state_impl (stable id sid via stable_id("rl_state",...), upserts MemoryState), retrieve_memory_states_impl, freeze_raw_access_impl, stats(). Token costs updated with estimate. [src: mempol/recipes/memory_rl/universal_tools.py]
- UniversalMemoryReward fields: question,gold_answer,tool,format_coef=0.1,write_bonus=0.05,token_cost_coef=0.0005,raw_left_open_penalty=0.15. Reward = judge_score + format_coef*(format_ok-1) + write_bonus(if writes>0) − token_cost_coef*token_cost − raw_penalty(if raw_enabled). [src: mempol/recipes/memory_rl/universal_reward.py]

Write environment & tooling (write-policy / GRPO)
- WRITE_TASK_INSTRUCTIONS: system prompt instructing write policy to use tools (lookup_entity, create_entity, update_state, merge_entities, add_relation, mark_contradiction, forget, noop), prefer noop default, output expected as <tool_call> JSON blocks; reward computed after episode vs held-out QA battery. [src: mempol/recipes/memory_rl/write_env.py]
- WriteDatum includes conv_id, turn_idx, turn_text, turn_dia_id, session_date, prior_turns_text, existing_entities_summary, query_battery (list of (question,gold,evidence_dia_ids)), full_text_backend, full_text_cache, baseline_cache. [src: mempol/recipes/memory_rl/write_env.py]
- WriteEnvGroupBuilder builds group_size envs with fresh PIEBackend and WriteTool per env and a shared frozen R runner resolved via resolve_r_runner_from_env (env var MEMPOL_R_CHECKPOINT). Builder params include group_size (default commonly 4), max_turns (default 32), format_coef default 0.0, max_trajectory_tokens default 4096, context_overflow_reward default -0.1. Tools list passed into agent env includes lookup_entity, lookup_relation, create_entity, update_state, merge_entities, add_relation, mark_contradiction, forget, noop. [src: mempol/recipes/memory_rl/write_env.py]
- WriteReward env var weights (strings parsed to floats/ints) with defaults: MEMPOL_W_CF "0.7", MEMPOL_W_QA "0.3", MEMPOL_W_GAIN "0.0", MEMPOL_W_OVERLAP "0.0", MEMPOL_W_COV_FLOOR "0.05", MEMPOL_K_MAX "12". [src: mempol/recipes/memory_rl/write_env.py]

Write tooling internals (WriteTool & ops)
- WriteTool wraps PIEBackend; tracks current_turn_text/dia_id/timestamp, counters n_lookups, n_creates, n_updates, n_merges, n_relations, n_contradictions, n_forgets, n_noops; ops_log is append-only list of (op_name,args) recorded BEFORE executing impl. Tool wrappers correspond to implementations: lookup_entity, lookup_relation, create_entity, update_state, merge_entities, add_relation, mark_contradiction, forget, noop. NON_MUTATING_OPS = ("lookup_entity","lookup_relation","noop"). [src: mempol/recipes/memory_rl/write_tools.py]

Write reward (WriteReward) — dense combination & per-op counterfactual
- Purpose: combine evidence coverage (deterministic), QA judge (deferred LLM judge via frozen R), per-op counterfactual marginal utility (per_op_counterfactual), and small cost regulariser. Default cost per op = 0.001. (DEFAULT_COST_PER_OP = 0.001). WriteReward fields include backend, query_battery, full_text_backend, reader, r_runner, write_tool, conv_id, w_cf/w_qa/w_gain/w_overlap/w_cov_floor, k_max, cost_per_op/cost_per_lookup/cost_per_entity, caches, etc. __call__ enforces budget, computes battery_coverage, runs per_op_counterfactual when w_cf>0, optionally battery_answer_gain and battery_reader_overlap, forms mean_qa, counts ops using write_tool counters or scraping, computes costs, and final reward = weighted sum plus coverage floor minus costs; dumps trajectory JSON if MEMPOL_TRAJECTORY_DUMP_DIR set. _kg_snapshot serializes backend.wm.entities for diagnostics. [src: mempol/recipes/memory_rl/write_reward.py] [src: mempol/eval/counterfactual.py]

Counterfactual details (implementation)
- per_op_counterfactual main flow: classify mutating op indices via _classify_ops; _replay builds fresh PIEBackend and transient WriteTool then replays ops (private impls) tolerant of failures; _score_battery runs reader.run per question and mempol.eval.judge in threadpool; per-op leave-one-out replayed in parallel; returns PerOpReward with per_op_deltas, n_ablated, n_battery, full_state_score; trajectory_reward = sum(deltas) − cost_per_mut*len(mut_indices). Default cost_per_mut used by caller; WriteReward typically supplies cost_per_op (default 0.001). [src: mempol/eval/counterfactual.py]

Supporting eval modules (coverage, gain, overlap, judge)
- evidence_coverage.coverage/battery_coverage: compute fraction of evidence dia_ids present in backend stored dia ids. [src: mempol/eval/evidence_coverage.py]
- answer_gain.battery_answer_gain: computes judge-margin over deterministic random-K baseline backend sampled per (conv_id,question,K) and cached in baseline_cache. [src: mempol/eval/answer_gain.py]
- reader_overlap.enforce_budget: prunes backend.wm.entities to at most k_max by (importance asc,last_seen asc) and rebuilds embeddings; battery_reader_overlap computes fraction of reference dia_ids recovered from stored provenance. [src: mempol/eval/reader_overlap.py]
- judge(question,gold,pred) returns 0.0 for empty/"not in context"/error strings else calls llm.chat with JUDGE_MODEL and expects JSON {"score":..., "reason":...}; parsed numeric score is bucketed to {1.0 if >=0.75, 0.5 if >=0.25, else 0.0}. [src: mempol/eval/judge.py]

Backends highlights
- FlatBackend: BM25 + dense via llm.embed; ingest computes embeddings and stores normalized matrix; retrieve supports dense/bm25/hybrid (hybrid fuses top-2k lists via RRF). [src: mempol/backends/flat.py]
- PIEBackend (pie_kg): wraps pie.core.world_model.WorldModel for a KG read/write backend with hybrid retrieval including NER/BM25/dense rankers; exposes lookup_entity, lookup_relation, create_entity, update_state, add_relation, merge_entities, mark_contradiction, forget. _entity_to_hit includes provenance/n_transitions/importance. [src: mempol/backends/pie_kg.py]
- GitMemBackend: content-addressable commit model for ops, supports commit/checkout/state_at/diff/log. [src: mempol/backends/gitmem.py]
- MastraBackend: Observer+Reflector abstraction producing observations & reflections, get_full_context returns stable context for answers; retrieves reflections + semantic search over observations + recent raw turns. [src: mempol/backends/mastra.py]
- ProviderBackend: wrapper for external MemoryProvider results. [src: mempol/backends/providers.py]

Policies (read & write)
- HeuristicPolicy (v1_heuristic): reformulate → retrieve first_k (12) → optional expand → rerank → answer_with_context; default first_k=12 final_k=6. [src: mempol/policies/v1_heuristic.py]
- HeuristicWritePolicy (v1_write): LLM-based write teacher that gates & emits ops (noop/create/update/add_relation/mark_contradiction/forget), resolves create targets with LLM prompt, then applies via WriteTool impls. [src: mempol/policies/v1_write.py]
- NaivePolicy, TemporalRLMPolicy, TemporalGroundPolicy, ContinuityTeacherPolicy exist with documented flows in policies/*. [src: mempol/policies/*]

LLM layer & token/embedding caching
- llm.embed uses per-model on-disk JSONL cache at config.CACHE_DIR / f"emb_{model.replace('/', '_')}.jsonl", keys by sha1(text) hex digest. Chat retries up to 3 times and maps max_tokens→max_completion_tokens for reasoning models. [src: mempol/llm.py]
- config sets default model names and creates CACHE_DIR / RESULTS_DIR / TRACES_DIR on import. [src: mempol/config.py]

Rollout utilities
- rollout.collect_rollouts samples G policies, runs traces, computes judge-based reward minus simple cost (cost_of uses lambda_step & lambda_retrieval). compute_advantages group-relative advantage. [src: mempol/rollout.py]

Notes about where to look for details
- For per-op LOO implementation and exact function names/returns: mempol/eval/counterfactual.py. [src: mempol/eval/counterfactual.py]
- For WriteReward weight parsing, KG snapshot & trajectory dump behavior: mempol/recipes/memory_rl/write_reward.py. [src: mempol/recipes/memory_rl/write_reward.py]
- For WriteTool op logging and classification constants: mempol/recipes/memory_rl/write_tools.py. [src: mempol/recipes/memory_rl/write_tools.py]
- For judge behavior & bucketing: mempol/eval/judge.py. [src: mempol/eval/judge.py]

(Keep this file as the single truth for codebase Q&A. If you need more granular code lines or another specific default that isn't explicitly listed here, flag it as MISSING-FROM-PACK with the source path to add.)