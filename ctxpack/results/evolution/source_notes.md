[src: mempol/recipes/memory_rl/__init__.py] (file empty)  

[src: mempol/recipes/memory_rl/cotrain.py] CoTrainConfig class — defaults: n_outer=5, r_steps_per_iter=200, w_steps_per_iter=100, model_name="Qwen/Qwen3-4B-Instruct-2507", lora_rank=32, learning_rate=4e-5, r_batch_size=4, r_group_size=8, r_max_turns=6, w_batch_size=4, w_group_size=8, w_max_turns=4, w_max_battery_per_turn=6, n_convs=8, train_frac=0.8, seed=2, log_path=None, wandb_project=None, behavior_if_log_dir_exists="delete".  
[src: mempol/recipes/memory_rl/cotrain.py] _run_subprocess(cmd, log_file) — async runner that tees subprocess stdout/stderr to log_file, returns exit code, inherits env (e.g., TINKER_API_KEY).  
[src: mempol/recipes/memory_rl/cotrain.py] _extract_checkpoint_path(log_path) — regex parses cookbook log for "sampler_path" → returns "tinker://..." checkpoint path or None.  
[src: mempol/recipes/memory_rl/cotrain.py] _train_R_phase(cfg,t,w_ckpt,out_dir) — launches train (Phase A) subprocess with config mapped to train.py args, returns extracted R checkpoint path.  
[src: mempol/recipes/memory_rl/cotrain.py] _train_W_phase(cfg,t,r_ckpt,out_dir) — launches train_write (Phase B) subprocess; if r_ckpt provided it's forwarded as r_checkpoint arg; returns extracted W checkpoint path.  
[src: mempol/recipes/memory_rl/cotrain.py] cli_main(cfg) — orchestration loop: for t in 1..n_outer run Phase A then Phase B, persist history.json with r_ckpt/w_ckpt; Phase B v1 uses HeuristicPolicy when r_ckpt is None.  

[src: mempol/recipes/memory_rl/data.py] MemoryDatum TypedDict keys: question, gold_answer, category, data_source, sample_id, qid, conversation_units (list[dict]).  
[src: mempol/recipes/memory_rl/data.py] _conv_to_serializable_units(conv, window=6, stride=3) — chunk sessions into overlapping windows producing units with uid, text, metadata including session, session_date, first/last dia_id, timestamp, dia_id, speaker, n_turns.  
[src: mempol/recipes/memory_rl/data.py] locomo_to_memory_data(n_convs=None, train_frac=0.8, seed=0) — loads LoCoMo, shuffles, splits by conversation into train/eval, converts to MemoryDatum using chunking (defaults produce ~120 chunks for conv-30).  
[src: mempol/recipes/memory_rl/data.py] longmemeval_jsonl_to_memory_data(jsonl_path) — compatibility parser for raw LongMemEval JSONL → MemoryDatum list.  
[src: mempol/recipes/memory_rl/data.py] longmemeval_to_memory_data(variant="longmemeval_s", n_rows=None, train_frac=0.8, seed=0, per_category=0, download=True) — loads LongMemEval via loader, optional balanced prefix per_category, returns (train, eval) MemoryDatum lists.  
[src: mempol/recipes/memory_rl/data.py] mix_sources(sources, weights=None, seed=0) — interleaves multiple MemoryDatum sources with optional weights (default uniform per dataset), oversamples smaller ones.  
[src: mempol/recipes/memory_rl/data.py] _balanced_prefix(conv_qas, per_category) — keeps at most per_category convs per category for balanced sampling.  

[src: mempol/recipes/memory_rl/memory_env.py] MEMORY_TASK_INSTRUCTIONS — system prompt describing memory tools, scoring: reward = correctness − tool-use cost, and guidance on tool usage/format.  
[src: mempol/recipes/memory_rl/memory_env.py] _backend_from_units(units_dicts) — builds FlatBackend, ingests Unit(uid,text,metadata) list.  
[src: mempol/recipes/memory_rl/memory_env.py] _initial_messages(datum, renderer, memory_tool) — builds tool schemas from memory_tool methods .to_spec(), creates conversation prefix with tools + user question.  
[src: mempol/recipes/memory_rl/memory_env.py] MemoryEnvGroupBuilder — constructor params: datum, model_name, renderer_name, max_turns, group_size, format_coef=0.1, max_trajectory_tokens=32*1024, max_generation_tokens=None, context_overflow_reward=-0.1; make_envs() creates group_size envs each with its own FlatBackend, MemoryTool, JudgeReward; tools passed: memory_search, memory_expand, memory_filter, memory_rerank, memory_top_n.  
[src: mempol/recipes/memory_rl/memory_env.py] MemoryRLDataset — holds env_group_builders and batch_size; get_batch(index) returns slice index*batch_size : +batch_size; __len__ returns max(1, len(builder)//batch_size) but 0 if no builders.  
[src: mempol/recipes/memory_rl/memory_env.py] MemoryRLDatasetBuilder (chz) fields/defaults: model_name_for_tokenizer (required), dataset="locomo", n_convs=8, lme_rows=120, lme_per_category=0, train_frac=0.8, batch_size=16, group_size=8, renderer_name=None, max_turns=6, format_coef=0.1, max_trajectory_tokens=16*1024, seed=0; __call__ loads dataset variant (locomo,longmemeval,mixed), shuffles train_data, converts to MemoryEnvGroupBuilder list and returns (train_ds, eval_ds).  

[src: mempol/recipes/memory_rl/reward.py] _extract_answer(text) — returns final "Answer:" line after last occurrence or None if not present.  
[src: mempol/recipes/memory_rl/reward.py] JudgeReward dataclass — fields: gold_answer, question, format_coef=0.1, correct_reward=1.0, partial_reward=0.5, wrong_reward=0.0; __call__(history) finds last assistant message, extracts text, runs mempol.eval.judge in thread, maps judge_score ≥0.75→1.0, ≥0.25→0.5, else 0.0; reward = format_coef*(correct_format-1)+correct_answer; returns (reward, {"format":..., "correct":...}).  

[src: mempol/recipes/memory_rl/tinker_compat.py] HAS_TINKER flag — True when tinker_cookbook.tool_use imports succeed; otherwise False.  
[src: mempol/recipes/memory_rl/tinker_compat.py] tool(...) — if HAS_TINKER wraps real decorator; fallback: no-op decorator that attaches .to_spec() metadata and ._mempol_tool marker to wrapped function.  
[src: mempol/recipes/memory_rl/tinker_compat.py] build_agent_tool_env(...) — pass-through to tinker_cookbook real factory; fallback raises RuntimeError telling to install cookbook.  
[src: mempol/recipes/memory_rl/tinker_compat.py] simple_tool_result / ToolResult — real or fallback ToolResult(dict) with content field; simple_tool_result builds appropriate ToolResult.  
[src: mempol/recipes/memory_rl/tinker_compat.py] Purpose — provide identical import path for tools and env builders usable both standalone (tests) and inside tinker-cookbook (training).  

[src: mempol/recipes/memory_rl/tools.py] _format_hit(h, max_chars=200) — maps Hit→dict with uid, rounded score, source, speaker, session, session_date, truncated text.  
[src: mempol/recipes/memory_rl/tools.py] _format_observation(hits,note="") — JSON with note, hits (first 10 formatted), n_hits; used as tool observation.  
[src: mempol/recipes/memory_rl/tools.py] MemoryTool dataclass — backend: Backend, last_hits:list[Hit]=[], n_searches=0, max_searches=8; methods exposed as tools: memory_search, memory_expand, memory_filter, memory_rerank, memory_top_n.  
[src: mempol/recipes/memory_rl/tools.py] memory_search(query,k=10,source="hybrid") — enforces max_searches, clamps k 1..20, source ∈ {bm25,dense,hybrid}, calls backend.retrieve(...), updates last_hits, increments n_searches, returns formatted observation.  
[src: mempol/recipes/memory_rl/tools.py] memory_expand(seed_uids,k_per=2) — limits seeds to 5, calls backend.expand(seed_uids,k_per), merges with last_hits avoiding duplicates, updates last_hits.  
[src: mempol/recipes/memory_rl/tools.py] memory_filter(predicate,value) — predicates: session_lt|gt|eq (int), speaker_eq (str), date_lt|date_gt|date_between (ISO date or range), type_eq, keyword_in, keyword_not_in; operates on last_hits, updates last_hits, robust date parsing with best-effort, returns observation or error JSON.  
[src: mempol/recipes/memory_rl/tools.py] memory_top_n(n) — clamps n 1..50, truncates last_hits to top-n.  
[src: mempol/recipes/memory_rl/tools.py] memory_rerank(strategy="dense", query=None) — if dense+query re-retrieves and orders by fresh dense retrieval; else session_desc/session_asc sort by metadata.session; updates last_hits.  
[src: mempol/recipes/memory_rl/tools.py] Note — in cookbook real @tool decorator should be added to methods when integrated.  

[src: mempol/recipes/memory_rl/train.py] CLIConfig defaults: model_name="Qwen/Qwen3-4B-Instruct-2507", lora_rank=32, renderer_name=None, learning_rate=4e-5, batch_size=8, seed=2, max_tokens=1024, eval_every=0, max_steps=None, dataset="locomo", n_convs=8, lme_rows=120, lme_per_category=0, train_frac=0.8, group_size=8, max_turns=6, format_coef=0.1, max_trajectory_tokens=16*1024, context_overflow_reward=-0.1, log_path=None, wandb_project=None, wandb_name=None, behavior_if_log_dir_exists="delete".  
[src: mempol/recipes/memory_rl/train.py] cli_main(cfg) — builds MemoryRLDatasetBuilder with cfg mapped, constructs train.Config for tinker_cookbook.rl.train with model_name, renderer_name, log_path, dataset_builder, learning_rate, max_tokens, eval_every, wandb fields, lora_rank, max_steps and calls train.main(config).  

[src: mempol/recipes/memory_rl/train_universal.py] CLIConfig defaults: model_name="Qwen/Qwen3-4B-Instruct-2507", lora_rank=32, renderer_name=None, learning_rate=4e-5, batch_size=2, seed=2, max_tokens=2048, eval_every=0, max_steps=None, temperature=1.1, kl_penalty_coef=0.02, n_convs=2, train_frac=0.8, group_size=4, max_turns=10, max_trajectory_tokens=12*1024, log_path=None, wandb_project=None, wandb_name=None, num_groups_to_log=8, behavior_if_log_dir_exists="delete".  
[src: mempol/recipes/memory_rl/train_universal.py] cli_main(cfg) — builds UniversalMemoryRLDatasetBuilder, optionally constructs KLReferenceConfig if kl_penalty_coef>0, maps to train.Config including temperature and kl_penalty_coef, calls train.main.  

[src: mempol/recipes/memory_rl/train_write.py] CLIConfig defaults for write phase: model_name="Qwen/Qwen3-4B-Instruct-2507", lora_rank=32, renderer_name=None, learning_rate=4e-5, batch_size=4, seed=2, max_tokens=2048, eval_every=0, max_steps=None, temperature=1.0, kl_penalty_coef=0.0, kl_reference_base_model=None, n_convs=8, train_frac=0.8, group_size=8, max_turns=32, max_battery_per_turn=0, n_prior_turns_in_context=12, r_checkpoint="" (v1 heuristic), log_path=None, wandb_project=None, wandb_name=None, num_groups_to_log=8, behavior_if_log_dir_exists="delete".  
[src: mempol/recipes/memory_rl/train_write.py] cli_main(cfg) — if cfg.r_checkpoint set, sets env var MEMPOL_R_CHECKPOINT for WriteRLDatasetBuilder to pick up (Phase B v2); builds WriteRLDatasetBuilder, optional KLReferenceConfig for KL penalty, constructs train.Config with temperature/kl_penalty_coef and calls train.main.  

[src: mempol/recipes/memory_rl/universal_env.py] UNIVERSAL_MEMORY_INSTRUCTIONS — system prompt describing tools: search_raw_spans, write_memory_state, freeze_raw_access, retrieve_memory_states and constraints (must freeze before answering or penalized).  
[src: mempol/recipes/memory_rl/universal_env.py] UniversalDatum TypedDict keys: question, gold_answer, qid, sample_id, source, raw_spans.  
[src: mempol/recipes/memory_rl/universal_env.py] _locomo_raw_span_data(n_convs=None, train_frac=0.8, seed=0) — converts LoCoMo turns to raw spans/artifacts with artifact ids and span ids, returns (train, eval) lists of UniversalDatum.  
[src: mempol/recipes/memory_rl/universal_env.py] _build_store_for_datum(datum) — creates a temporary SQLiteMemoryStore file, upserts artifacts and spans from datum["raw_spans"], commits, returns store.  
[src: mempol/recipes/memory_rl/universal_env.py] _initial_messages(datum, renderer, tool) — builds tool specs for UniversalMemoryTool methods and creates conversation prefix + user question.  
[src: mempol/recipes/memory_rl/universal_env.py] UniversalMemoryEnvGroupBuilder — params: datum, model_name, renderer_name, group_size, max_turns=10, max_trajectory_tokens=12*1024, max_generation_tokens=None, context_overflow_reward=-0.2; make_envs() requires HAS_TINKER True, builds per-env SQLiteMemoryStore, UniversalMemoryTool, UniversalMemoryReward, registers tools: search_raw_spans, write_memory_state, freeze_raw_access, retrieve_memory_states.  
[src: mempol/recipes/memory_rl/universal_env.py] UniversalMemoryRLDatasetBuilder (chz) defaults: model_name_for_tokenizer (required), n_convs=2, train_frac=0.8, batch_size=2, group_size=4, renderer_name=None, max_turns=10, max_trajectory_tokens=12*1024, seed=0; __call__ uses _locomo_raw_span_data and returns (train_ds, eval_ds).  

[src: mempol/recipes/memory_rl/universal_reward.py] UniversalMemoryReward dataclass — fields: question, gold_answer, tool(UniversalMemoryTool), format_coef=0.1, write_bonus=0.05, token_cost_coef=0.0005, raw_left_open_penalty=0.15; __call__(history) extracts final assistant text, uses _extract_answer, judges via mempol.eval.judge in thread to get judge_score (float), load tool.stats() and computes write_bonus if writes>0, cost = token_cost_coef * token_cost, raw_penalty = raw_left_open_penalty if raw_enabled, reward = judge_score + format_coef*(format_ok-1) + write_bonus - cost - raw_penalty; returns (reward, dict of stats including judge_score, writes, raw_searches, memory_searches, token_cost, raw_left_open).  

[src: mempol/recipes/memory_rl/universal_tools.py] UniversalMemoryTool dataclass — store: SQLiteMemoryStore, counters raw_searches=0, memory_searches=0, writes=0, token_cost=0, written_state_ids=[], max_raw_searches=8, max_memory_searches=8, max_writes=24, raw_enabled=True; provides tool methods: search_raw_spans, write_memory_state, freeze_raw_access, retrieve_memory_states and internal impls that update counters/token_cost/written_state_ids and enforce limits.  
[src: mempol/recipes/memory_rl/universal_tools.py] search_raw_spans_impl(query,k=8) — errors if raw_disabled or max_raw_searches reached, retrieves spans (k*2 then filter kind==span) up to k, increments raw_searches, increments token_cost by token_estimate of hits, returns JSON hits with span_id, artifact_id, source, score, text truncated, locator.  
[src: mempol/recipes/memory_rl/universal_tools.py] write_memory_state_impl(content, source_span_ids) — enforces max_writes, validates span ids exist in store, rejects empty content, creates stable_id("rl_state", content, source_span_ids), upserts MemoryState to store, commits, increments writes, appends id to written_state_ids, increments token_cost by estimate_tokens(content), returns written_state_id and tokens_est.  
[src: mempol/recipes/memory_rl/universal_tools.py] retrieve_memory_states_impl(query,k=8) — enforces max_memory_searches, clamps k 1..20, retrieves memory_state kind hits up to k, increments memory_searches, increments token_cost by token_estimate, returns hits with memory_state_id, source, score, truncated content, source_span_ids.  
[src: mempol/recipes/memory_rl/universal_tools.py] freeze_raw_access_impl(reason="") — sets raw_enabled False and returns status; freeze_raw_access is tool exposing this impl.  
[src: mempol/recipes/memory_rl/universal_tools.py] stats() — returns dict with raw_searches, memory_searches, writes, token_cost, written_state_ids, raw_enabled.

---

[src: mempol/recipes/memory_rl/write_env.py] WRITE_TASK_INSTRUCTIONS: system prompt for W policy describing tools (lookup_entity, create_entity, update_state, merge_entities, add_relation, mark_contradiction, forget, noop), strategy (default noop, call lookup before create, prefer update over create, merge aggressively), output XML-wrapped JSON tool_call format and that reward is computed after episode against held-out QA battery.

[src: mempol/recipes/memory_rl/write_env.py] WriteDatum keys and meanings: conv_id, turn_idx, turn_text, turn_dia_id, session_date, prior_turns_text, existing_entities_summary, query_battery (list of (question, gold_answer, evidence_dia_ids)), full_text_backend (FlatBackend), full_text_cache (per-conv question→dia_id set), baseline_cache (per-conv random-K baseline cache).

[src: mempol/recipes/memory_rl/write_env.py] WriteEnvGroupBuilder role: build a GRPO env group per WriteDatum; creates per-env fresh PIEBackend and WriteTool, shares frozen R runner, exposes make_envs and logging_tags.

[src: mempol/recipes/memory_rl/write_env.py] WriteEnvGroupBuilder defaults/params: group_size, max_turns default 32, format_coef default 0.0, max_trajectory_tokens default 4096, context_overflow_reward default -0.1; max_generation_tokens optional.

[src: mempol/recipes/memory_rl/write_env.py] Tokenizer/renderer selection: uses tokenizer_utils.get_tokenizer(model_name) and model_info.get_recommended_renderer_name(model_name) if renderer_name not provided.

[src: mempol/recipes/memory_rl/write_env.py] Per-env components assembled: backend = PIEBackend(), wtool = WriteTool(backend), wtool.current_turn_text/dia_id/timestamp set, initial_messages from _initial_messages, reward_fn = WriteReward(...), env built via build_agent_tool_env with tools list [lookup_entity, lookup_relation, create_entity, update_state, merge_entities, add_relation, mark_contradiction, forget, noop].

[src: mempol/recipes/memory_rl/write_env.py] Env-level reward mix read from env vars (defaults provided in code): MEMPOL_W_CF (default "0.7"), MEMPOL_W_QA ("0.3"), MEMPOL_W_GAIN ("0.0"), MEMPOL_W_OVERLAP ("0.0"), MEMPOL_W_COV_FLOOR ("0.05"), MEMPOL_K_MAX ("12") -> parsed as floats/ints and passed to WriteReward as w_cf, w_qa, w_gain, w_overlap, w_cov_floor, k_max.

[src: mempol/recipes/memory_rl/write_env.py] resolve_r_runner_from_env used to pick frozen R if MEMPOL_R_CHECKPOINT set; otherwise heuristic R default inside WriteReward.__post_init__.

[src: mempol/recipes/memory_rl/write_env.py] _initial_messages: composes renderer.create_conversation_prefix_with_tools(tools, WRITE_TASK_INSTRUCTIONS) + user prompt containing session date, prior turns, focal turn, existing_entities_summary and question "What write ops...".

[src: mempol/recipes/memory_rl/write_env.py] WriteRLDatasetBuilder role: iterates LoCoMo conversations to build WriteDatum per (conv, turn) when at least min_battery_per_turn QAs reference that turn's session; builder hyperparams: model_name_for_tokenizer, n_convs=8, train_frac=0.8, batch_size=8, group_size=4, max_turns=4, max_battery_per_turn=0 (0 means use all), min_battery_per_turn=1, n_prior_turns_in_context=2, seed=0, renderer_name optional.

[src: mempol/recipes/memory_rl/write_reward.py] Module purpose: deferred, dense reward combining evidence coverage (deterministic), QA judge (deferred LLM judge via a frozen R), per-op counterfactual marginal utility (primary v3 signal), and small cost regulariser; optionally dumps trajectory JSON if MEMPOL_TRAJECTORY_DUMP_DIR set.

[src: mempol/recipes/memory_rl/write_reward.py] Trajectory dump behavior: if MEMPOL_TRAJECTORY_DUMP_DIR set, _maybe_dump_trajectory writes JSON with ts, reward, flat metrics, per_question_coverage, kg_snapshot, messages into that dir; failures are logged and swallowed.

[src: mempol/recipes/memory_rl/write_reward.py] Default cost coefficients: DEFAULT_COST_PER_OP = 0.001, DEFAULT_COST_PER_LOOKUP = 0.0, DEFAULT_COST_PER_ENTITY = 0.0.

[src: mempol/recipes/memory_rl/write_reward.py] Default reward mix constants: DEFAULT_W_CF=0.7, DEFAULT_W_QA=0.3, DEFAULT_W_GAIN=0.0, DEFAULT_W_OVERLAP=0.0, DEFAULT_W_COV_FLOOR=0.05, DEFAULT_K_MAX=12.

[src: mempol/recipes/memory_rl/write_reward.py] WriteReward dataclass fields (one-line each): backend (PIEBackend), query_battery, full_text_backend, reader, r_runner (callable(question, backend) -> answer_str), write_tool, conv_id, w_cf/w_qa/w_gain/w_overlap/w_cov_floor, k_max, cost_per_op/cost_per_lookup/cost_per_entity, full_text_cache, baseline_cache, _last_metrics.

[src: mempol/recipes/memory_rl/write_reward.py] WriteReward.__post_init__: if r_runner is None and w_qa>0 then r_runner = _default_r_runner; if reader None then reader = _HEURISTIC_R (HeuristicPolicy).

[src: mempol/recipes/memory_rl/write_reward.py] WriteReward.__call__ flow: returns (reward, metrics); steps: early exit if empty battery (-0.01), enforce budget enforce_budget(self.backend, k_max), compute battery_coverage, run per_op_counterfactual if w_cf>0 and write_tool.ops_log exists and reader set (async), optionally battery_answer_gain if w_gain>0, optionally battery_reader_overlap if w_overlap>0, compute mean_qa via cf_result.full_state_score if cf_result else by running r_runner + judge per question, count ops from write_tool or by scraping history, compute cost = cost_per_op*n_mutations + cost_per_lookup*n_lookups + cost_per_entity*n_entities, coverage floor cov_floor = w_cov_floor * cov_result.mean_coverage, final reward = w_cf*cf_reward + w_qa*mean_qa + w_gain*mean_gain + w_overlap*mean_overlap + cov_floor - cost, fill _last_metrics with many diagnostics, attach per_question_coverage and kg_snapshot into _last_metrics_full and possibly dump trajectory.

[src: mempol/recipes/memory_rl/write_reward.py] _kg_snapshot: serializes up to max_entities of backend.wm.entities into compact dict with uid, name, type, current_state, n_transitions, source_dia_id; returns dict with n_entities, stored_dia_ids, entities.

[src: mempol/recipes/memory_rl/write_reward.py] _collect_dia_ids: collects entity.created_from and transitions.trigger_conversation_id from backend to build set of stored dia_ids.

[src: mempol/recipes/memory_rl/write_reward.py] _assistant_tool_calls: extracts structured tool_calls from assistant message (handles newer structured form and older substring fallback of <tool_call> JSON blocks); used by _count_ops/_count_lookups.

[src: mempol/recipes/memory_rl/write_reward.py] Cost & op counting: prefer write_tool counters if present (n_lookups, n_creates, n_updates, n_merges, n_relations, n_contradictions, n_forgets, n_noops); fallback to history scraping.

[src: mempol/recipes/memory_rl/write_reward.py] Frozen R: _HEURISTIC_R = HeuristicPolicy(first_k=8, final_k=4, do_reformulate=True, do_expand=True); _default_r_runner runs that heuristic and returns trace.answer or "not in context".

[src: mempol/recipes/memory_rl/write_reward.py] resolve_r_runner_from_env: if env MEMPOL_R_CHECKPOINT set tries to build tinker.SamplingClient(ckpt) and make_tinker_r_runner; on failure returns None and logs; make_tinker_r_runner currently raises NotImplementedError (stub).

[src: mempol/recipes/memory_rl/write_tools.py] WriteTool dataclass: wraps PIEBackend per-env; tracks current_turn_text/dia_id/timestamp/observation text and counters n_lookups, n_creates, n_updates, n_merges, n_relations, n_contradictions, n_forgets, n_noops; ops_log append-only list of (op_name, args_dict) captured pre-execution.

[src: mempol/recipes/memory_rl/write_tools.py] WriteTool metadata helpers: _metadata() returns source_dia_id and observed_at_timestamp and optional observed_at, _with_write_metadata inserts provenance into state dict.

[src: mempol/recipes/memory_rl/write_tools.py] Tool impls (one-line roles): _lookup_entity_impl -> backend.lookup_entity(...) and increments n_lookups and returns JSON matches via simple_tool_result; _lookup_relation_impl -> backend.lookup_relation; _create_entity_impl -> backend.create_entity with metadata, increments n_creates, returns uid; _update_state_impl -> validates uid exists, backend.update_state with metadata, increments n_updates and n_contradictions on type 'contradiction'; _merge_entities_impl -> validates uids exist, backend.merge_entities, increments n_merges; _add_relation_impl -> validates uids exist, backend.add_relation, increments n_relations; _mark_contradiction_impl -> validates uid exists, backend.mark_contradiction, increments n_contradictions; _forget_impl -> backend.forget, increments n_forgets; _noop_impl -> increments n_noops.

[src: mempol/recipes/memory_rl/write_tools.py] Tinker @tool wrappers: each wrapper appends (op_name, args) to ops_log BEFORE delegating to corresponding _*_impl; wrappers are: lookup_entity, lookup_relation, create_entity, update_state, merge_entities, add_relation, mark_contradiction, forget, noop.

[src: mempol/recipes/memory_rl/write_tools.py] Op classification constants: NON_MUTATING_OPS = ("lookup_entity","lookup_relation","noop"); MUTATING_OPS = ("create_entity","update_state","merge_entities","add_relation","mark_contradiction","forget").

[src: mempol/recipes/memory_rl/write_tools.py] write_stats(): returns dict of counters for shaping/logging.

[src: mempol/recipes/memory_rl/write_tools.py] smoke(): end-to-end example showing typical sequence: lookup→create→lookup→create(no dedupe)→update_state→mark_contradiction→noop and prints backend stats.

[src: mempol/eval/answer_gain.py] battery_answer_gain: computes judge-margin over a random-K baseline; builds random backend by sampling K units from full_text_backend deterministically; caches baseline per (conv_id, question, K) in baseline_cache; returns GainResult(mean_gain, per_question list, n_random_baseline_calls).

[src: mempol/eval/answer_gain.py] _build_random_K_backend: deterministic sampling with Random(seed) and fallback to full backend if units <= K.

[src: mempol/eval/counterfactual.py] Purpose: per-op leave-one-out counterfactual to compute per-op marginal utility; main function per_op_counterfactual(ops_log, battery, reader, current_dia_id, current_timestamp, cost_per_mut=0.005) returns PerOpReward with trajectory_reward=sum per-op deltas - cost_per_mut * n_mutating_ops.

[src: mempol/eval/counterfactual.py] _classify_ops: returns indices of mutating ops in ops_log by comparing names to WriteTool.MUTATING_OPS.

[src: mempol/eval/counterfactual.py] _replay: builds fresh PIEBackend, constructs transient WriteTool with given provenance, dispatches each op to corresponding private impl method to recreate state; failures are logged (replay tolerant).

[src: mempol/eval/counterfactual.py] _score_battery: runs reader.run(q, backend) in threadpool and judge_fn per question in parallel using asyncio.gather, returns per-question scores.

[src: mempol/eval/counterfactual.py] per_op_counterfactual flow: if no ops or battery -> zero; get mutating indices; replay full ops to get full_scores and full_mean; for each mutating index, replay leave-one-out and compute mean delta across battery; per-op deltas gathered in parallel; traj_reward = sum(deltas) - cost_per_mut*len(mut_indices); returns PerOpReward(trajectory_reward, per_op_deltas, n_ablated, n_battery, full_state_score=full_mean).

[src: mempol/eval/evidence_coverage.py] stored_dia_ids: collects every dia_id from Entity.created_from and StateTransition.trigger_conversation_id in backend.wm.

[src: mempol/eval/evidence_coverage.py] coverage(evidence, stored): returns fraction of evidence dia_ids present in stored; returns 0.0 for empty evidence.

[src: mempol/eval/evidence_coverage.py] battery_coverage: computes per-question coverage, mean_coverage, and counts n_stored_dia_ids, n_evidence_dia_ids_total, n_evidence_dia_ids_hit; returns CoverageResult.

[src: mempol/eval/judge.py] judge(question,gold,pred) behavior: returns 0.0 if pred empty or starts with "not in context" or "error"; otherwise calls llm.chat with JUDGE_MODEL (from config) using system + user prompts expecting JSON {"score":..., "reason":...}, parses score and buckets into {1.0 if >=0.75, 0.5 if >=0.25, else 0.0}; returns (score, reason) and logs parse failures returning 0.0 with reason 'judge_err:...'.

[src: mempol/eval/metrics.py] Result dataclass fields and summarise(results): aggregates overall_acc, avg_steps, avg_retrievals, by_category accuracies, and avg_evidence_recall if present.

---

[src: mempol/eval/qa_generator.py] _GEN_SYSTEM: system prompt instructing LLM to generate evaluation questions covering mixed categories and return ONLY a JSON array.  
[src: mempol/eval/qa_generator.py] _GEN_USER_TEMPLATE: user prompt template that asks for n QAs with fields question, gold_answer, category ∈ {single-hop,multi-hop,temporal,open-domain,adversarial}, evidence_text (verbatim) and target LoCoMo proportions (50/25/15/7/3).  
[src: mempol/eval/qa_generator.py] GeneratedQA dataclass: fields question, gold_answer, category, evidence_text, evidence_dia_ids (filled by link_evidence_to_dia_ids).  
[src: mempol/eval/qa_generator.py] generate(transcript,n=10,model=None,cache_dir=None) -> list[GeneratedQA]: uses model=config.OBSERVER_MODEL or "gpt-4o-mini"; caches results keyed by sha256(transcript+"::n=..::m=..")[:16] into qa_{key}.json if cache_dir given; truncates transcript to 120_000 chars before sending; calls llm.chat(..., json_mode=True) and parses JSON array (supports fallback "questions" key).  
[src: mempol/eval/qa_generator.py] link_evidence_to_dia_ids(qas, turns): maps evidence_text→dia_ids by computing overlap via shared 5-grams vs each turn.text lowercased and picks top 2 dia_ids with score>0.2; populates GeneratedQA.evidence_dia_ids.  
[src: mempol/eval/reader_overlap.py] enforce_budget(backend: PIEBackend, k_max: int) -> int: prunes backend.wm.entities to at most k_max by sorting by (importance asc, last_seen asc), removing lowest entries and their transitions/relationships, calls wm.rebuild_embedding_matrix() after bulk drop; returns number removed.  
[src: mempol/eval/reader_overlap.py] OverlapResult dataclass: mean_overlap, per_question list[(question,score)], n_full_text_dia_ids_total, n_full_text_dia_ids_recovered.  
[src: mempol/eval/reader_overlap.py] _hits_to_dia_ids(hits) -> set[str]: extracts dia_ids from Hit.unit.metadata['dia_ids'] or 'dia_id' (str or list) and fallback metadata['source_dia_id'] for KG hits.  
[src: mempol/eval/reader_overlap.py] _stored_dia_ids(backend: PIEBackend) -> set[str]: collects provenance dia_ids from backend.wm.entities' created_from and from transitions' trigger_conversation_id in wm.transitions referenced by wm._entity_transitions.  
[src: mempol/eval/reader_overlap.py] battery_reader_overlap(backend, battery, full_text_backend, reader, full_text_cache=None) -> OverlapResult: for each q in battery, obtains ref dia_ids by running reader.run(q, full_text_backend) (cached if full_text_cache provided), computes fraction of ref recovered in stored_dia_ids, mean reward = mean per-question recall; uses reader.final_hits as traces output.  
[src: mempol/eval/runner.py] conv_to_units(conv: Conversation) -> list[Unit]: converts LoCoMo Conversation.turns into Backend Unit objects with uid="{sample_id}::{dia_id}", text="{speaker}: {text}", metadata includes session, speaker, dia_id, session_date, timestamp=float(session).  
[src: mempol/eval/runner.py] evidence_recall(retrieved_uids, gold_dia_ids) -> float: computes fraction of gold_dia_ids present in retrieved_uids by stripping UID prefix before ::.  
[src: mempol/eval/runner.py] run(backend_factory, policy, n_convs=1, max_qs_per_conv=None, run_name="smoke", categories=None) -> dict: loads conversations via load(n_convs), writes traces JSONL to config.TRACES_DIR/run_name.jsonl and summary to config.RESULTS_DIR/run_name/summary.json; filters qas by categories set if provided; for each QA runs policy.run, judges via judge(), computes evidence_recall on trace.final_hits uids, logs Result entries and trace JSON lines.  
[src: mempol/eval/runner.py] _BACKENDS/_POLICIES maps: backends {"flat": FlatBackend}; policies {"v0_naive": NaivePolicy, "v1_heuristic": HeuristicPolicy, "rlm_temporal": TemporalRLMPolicy, "temporal_ground": TemporalGroundPolicy}.  
[src: mempol/policies/base.py] Step dataclass: op (string), args (dict), obs_summary (short text).  
[src: mempol/policies/base.py] Trace dataclass: qid, question, backend, policy, steps list[Step], final_hits list[Hit], answer str, cost_tokens int, n_retrievals int.  
[src: mempol/policies/base.py] ReadPolicy abstract class: attribute name and abstract run(question, backend) -> Trace.  
[src: mempol/policies/continuity.py] ContinuityTeacherPolicy.name = "continuity_teacher": multi-stage teacher policy that routes question, retrieves turn spans, expands, builds session backend, reconstructs temporary states and timeline via LLM, chooses action, answers via LLM; logs each step into Trace and returns ContinuityRun with trace, route, temporary_states, timeline, missing_evidence, action, session_hits.  
[src: mempol/policies/continuity.py] ContinuityTeacherPolicy default params: turn_k=18, session_k=2, expand_seed_k=8, final_turn_k=10, max_session_chars=4500.  
[src: mempol/policies/continuity.py] _route/_reconstruct_state/_reconstruct_timeline/_answer: call llm.chat with prompts (_ROUTE_SYS, _STATE_SYS, _TIMELINE_SYS, _ANSWER_SYS) using model=config.REFORMULATE_MODEL for routing/state/timeline and config.ANSWER_MODEL for final answer; use json_mode=True and max_tokens limits; fallback heuristics if LLM fails (keyword-based routing, state error text).  
[src: mempol/policies/continuity.py] _build_session_backend(backend, max_session_chars): groups units by session and session_date, concatenates formatted units up to max_session_chars, creates FlatBackend named "session_flat" with metadata dia_ids aggregated.  
[src: mempol/policies/rlm_temporal.py] TemporalRLMPolicy.name = "rlm_temporal": routes via LLM whether needs_timeline, retrieves first_k=24 hits, optionally expands expand_seed_k=8, dedupes, if needs_timeline reconstructs timeline via _timeline_for_evidence (LLM over grouped session blocks) and answers via _answer_from_timeline using config.ANSWER_MODEL; if not needs_timeline just answers using top final_k via answer_with_context.  
[src: mempol/policies/rlm_temporal.py] TemporalRLMPolicy params: first_k=24, final_k=12, expand_seed_k=8, force_timeline=False; uses _timeline_cache keyed by stable hash of hits.  
[src: mempol/policies/temporal_ground.py] TemporalGroundPolicy.name="temporal_ground": compiles deterministic worldline and optional belief ledger and arithmetic_card, classifies question into operator and frame, answers via LLM with structured worldline+ledger+card and operator-specific rule; falls back to answer_with_context if undated.  
[src: mempol/policies/temporal_ground.py] TemporalGroundPolicy params: first_k=24, final_k=12, expand_seed_k=8, use_belief_ledger=True, force_temporal=False; caches ledger and now_ts.  
[src: mempol/policies/temporal_ground.py] _now_ts: computes NOW as max parsed session_date across backend.units else from retrieved events; caches per backend id.  
[src: mempol/policies/v0_naive.py] NaivePolicy.name="v0_naive": retrieve k (default 10) hybrid hits, set final_hits, answer via answer_with_context which uses _ANSWER_SYS and model=config.ANSWER_MODEL.  
[src: mempol/policies/v1_heuristic.py] HeuristicPolicy.name="v1_heuristic": multi-step teacher: reformulate (LLM with REFORMULATE_MODEL), optional route (LLM with ROUTE_PROMPT), first retrieval first_k (default 12), optional 1-hop expand, rerank to final_k (default 6) via dense retrieval, answer via answer_with_context; logs steps and increments n_retrievals for retrieve+rerank.  
[src: mempol/policies/v1_heuristic.py] HeuristicPolicy params: first_k=12, final_k=6, do_reformulate=True, do_expand=True, do_route=True.  
[src: mempol/policies/v1_write.py] HeuristicWritePolicy.name="v1_write": per-turn LLM-based write teacher that gates and emits ops (noop, create_entity, update_state, add_relation, mark_contradiction, forget), resolves create_entity into existing entities by calling _resolve_create_target which asks LLM (_RESOLVE_SYS) to decide update_existing vs create_new; applies ops via WriteTool implementation methods.  
[src: mempol/policies/v1_write.py] HeuristicWritePolicy params: lookup_top_k=5, model=config.REFORMULATE_MODEL by default, gate_with_llm=True.  
[src: mempol/policies/v1_write.py] write op vocabulary and semantics documented in _WRITE_SYS (types for create_entity, transition_type for update_state, rel_type for add_relation, etc.) and resolution rules: do not create duplicate entities, prefer update_existing when LLM resolution returns that.  
[src: mempol/policies/v1_write.py] step(...) -> WriteDecision: looks up nearby entities via write_tool._lookup_entity_impl or backend.lookup_entity, prompts LLM with _WRITE_SYS and lookup context (json_mode), parses raw_ops, expands short uid prefixes via uid_map derived from lookup matches, calls write_tool._create_entity_impl/_update_state_impl/_add_relation_impl/_mark_contradiction_impl/_forget_impl accordingly while collecting applied_ops and errors.  
[src: mempol/backends/base.py] Unit dataclass: uid, text, metadata dict (session, speaker, dia_id, timestamp, etc.).  
[src: mempol/backends/base.py] Hit dataclass: unit: Unit, score: float, source: str ∈ {"dense","bm25","expand",...}.  
[src: mempol/backends/base.py] Backend abstract class: name attribute, abstract ingest(units), retrieve(query,k=10,source="hybrid")->list[Hit]; default expand(seed_uids,k_per=3) returns []; filter_by_time(hits, (lo,hi)) filters hits by unit.metadata['timestamp'] against window.

---

[src: mempol/backends/flat.py] _TOK regex = r"[a-zA-Z0-9']+"; _tokens(text) lowercases and tokenizes by that regex.
[src: mempol/backends/flat.py] BM25Index class: compact BM25 with defaults k1=1.2, b=0.75, stores docs, df, tf, dl, lazily builds idf and avgdl on _dirty flag.
[src: mempol/backends/flat.py] BM25Index.add(tokens) updates docs, tf, dl, df and marks _dirty=True.
[src: mempol/backends/flat.py] BM25Index._build() computes _avgdl and _idf = log((n - df + 0.5)/(df + 0.5) + 1).
[src: mempol/backends/flat.py] BM25Index.score(q_tokens, i) computes BM25 score per document i using stored tf/dl,idf,k1,b.
[src: mempol/backends/flat.py] BM25Index.topk(q,k) tokenizes q and returns top-k (index,score) with score>0 sorted desc.
[src: mempol/backends/flat.py] _rrf(rank_lists, k=60): Reciprocal Rank Fusion: accumulates 1/(k+rank+1) per list and sorts descending.
[src: mempol/backends/flat.py] FlatBackend.name = "flat"; in-memory list of Unit, uid->idx map, BM25Index and optional normalized embedding matrix _emb.
[src: mempol/backends/flat.py] FlatBackend.ingest(units): appends Units, updates _uid_to_idx, bm25 (tokens of u.text), then computes embeddings via llm.embed(texts) and stores L2-normalized _emb.
[src: mempol/backends/flat.py] FlatBackend._dense_topk(query,k): returns top-k indices and cosine similarities against normalized _emb using llm.embed(query).
[src: mempol/backends/flat.py] FlatBackend.retrieve(query,k=10,source="hybrid"): supports "dense","bm25","hybrid"; hybrid fuses dense and bm25 lists (each top 2k) via _rrf and returns Hits with source tags.
[src: mempol/backends/flat.py] FlatBackend.expand(seed_uids,k_per=3): returns adjacent-turn expansion within same session metadata (±1 index) as Hits score=0.5, source="expand", limited to k_per*len(seed_uids).

[src: mempol/backends/gitmem.py] OpRecord dataclass fields: kind, args, resulting_state, target_uid (optional).
[src: mempol/backends/gitmem.py] Commit dataclass fields: sha, parent_shas, timestamp, dia_id, ops, message; Commit.to_text() makes one-line summary for indexing.
[src: mempol/backends/gitmem.py] Branch dataclass: name, head_sha, entity_uid (None => global).
[src: mempol/backends/gitmem.py] _sha(body): content-addressable SHA-1 over sorted JSON dump, returns first 16 hex chars.
[src: mempol/backends/gitmem.py] GitMemBackend.name = "gitmem"; in-memory commits dict, branches dict, indexes _commits_by_entity and _commits_by_time, lazy BM25 over commit text cached in _commit_texts with _bm25_dirty flag.
[src: mempol/backends/gitmem.py] ingest(units): converts each Unit to single-op OpRecord(kind="ingest_chunk", args={"text": u.text[:200]}) and commits to branch "default" with dia_id from metadata.
[src: mempol/backends/gitmem.py] retrieve(query,k=10,source="hybrid"): uses rank_bm25.BM25Okapi over commit to_text() corpus (lazy rebuild) to score and returns top-k Hits (source="bm25").
[src: mempol/backends/gitmem.py] expand(seed_uids,k_per=3): expands each commit sha to its parents and children (children found by scanning commits whose parents include sha), returns Hits with source="expand".
[src: mempol/backends/gitmem.py] commit(ops,timestamp,dia_id,message="",branch="default",parent_shas=None): computes parent_shas default to branch HEAD, builds body, sha=_sha(body), stores Commit, updates/creates Branch (entity_uid if all ops share same target_uid), updates _commits_by_entity, inserts into _commits_by_time sorted by timestamp, sets _bm25_dirty=True and returns sha.
[src: mempol/backends/gitmem.py] merge(target_branch,source_branch,...,reconciliation_ops,message="merge"): creates a commit on target with two parents (target HEAD, source HEAD) using commit().
[src: mempol/backends/gitmem.py] checkout(sha): reconstructs full state at commit by topological walk of parents and applying ops in order — returns mapping entity_uid -> resulting_state.
[src: mempol/backends/gitmem.py] state_at(timestamp,entity_uid): iterates _commits_by_time ascending until ts>timestamp, tracks latest resulting_state for target_uid and returns it (O(n) linear scan across time-sorted index but early exit on ts>timestamp).
[src: mempol/backends/gitmem.py] diff(sha_a,sha_b): walks back from sha_b through first parents collecting ops until sha_a (exclusive) — linear-only path diff.
[src: mempol/backends/gitmem.py] log(branch,limit=20): follows head_sha through first parents returning most recent commits.
[src: mempol/backends/gitmem.py] dump(): returns JSON-able snapshot: commits (ops serialized) and branches.
[src: mempol/backends/gitmem.py] load(blob): reconstructs backend from dump, repopulates indexes and marks _bm25_dirty=True.
[src: mempol/backends/gitmem.py] _rebuild_bm25(): builds BM25Okapi over lowercased commit.to_text().split() corpus, sets _bm25_dirty=False.

[src: mempol/backends/mastra.py] MastraBackend.name = "mastra_om"; architecture: raw turns → Observer → dated bullet observations → Reflector → condensed reflections; context for queries = reflections + observations + recent raw turns.
[src: mempol/backends/mastra.py] _estimate_tokens(text) = max(1, len(text)//4) — cheap token proxy (~4 chars/token).
[src: mempol/backends/mastra.py] Observer system prompt constant _OBSERVER_SYS instructs concise dated prioritized observations and expects "Current task" and "Suggested response" lines.
[src: mempol/backends/mastra.py] Reflector system prompt constant _REFLECTOR_SYS asks to condense observations preserving durable content.
[src: mempol/backends/mastra.py] Data classes: _RawTurn(uid,text,metadata), ObservationBlock(date_label,markdown,source_uids,current_task,suggested_response,n_input_chars,n_output_chars), ReflectionBlock(markdown,n_observations_consumed,n_input_chars,n_output_chars).
[src: mempol/backends/mastra.py] MastraBackend.__init__ defaults: observer_token_threshold=3000, reflector_token_threshold=8000, keep_recent_n=20; state: _raw_buffer, _all_turns, observations, reflections, _obs_embeddings, _obs_index_dirty, _stats counters.
[src: mempol/backends/mastra.py] retrieve(query,k=10,source="hybrid"): always includes all reflections (score=1.0), semantic search over observations via llm.embed and normalized dot product (reindexes if dirty), then appends recent raw turns (last keep_recent_n) with score=0.4.
[src: mempol/backends/mastra.py] _reindex_observations(): builds embeddings via llm.embed(texts) and L2-normalizes to _obs_embeddings, sets _obs_index_dirty False; exceptions printed.
[src: mempol/backends/mastra.py] expand(seed_uids,k_per=2): expands observation UIDs of form "observ::<idx>" to adjacent observation blocks with score=0.5, source="expand_om".
[src: mempol/backends/mastra.py] get_full_context(): returns concatenated markdown: reflections, observations, recent raw turns (with speaker and dia_id) — intended stable context for answer LLM.
[src: mempol/backends/mastra.py] memory_log_md(): human-readable dump summarizing counts and compression stats, includes reflections, observations, and recent raw turns.
[src: mempol/backends/mastra.py] stats() returns _stats dict counters.
[src: mempol/backends/mastra.py] save(path)/load(path): pickle-based persistence for in-memory state; embeddings cached separately.
[src: mempol/backends/mastra.py] ingestion loop: ingest(units) calls _ingest_one for each Unit then _maybe_run_observer(force=True) to flush.
[src: mempol/backends/mastra.py] _ingest_one(unit): appends _RawTurn to _raw_buffer and _all_turns, increments stats, calls _maybe_run_observer and _maybe_run_reflector.
[src: mempol/backends/mastra.py] _buffer_tokens() sums _estimate_tokens over _raw_buffer; _observation_tokens() sums tokens over observations.
[src: mempol/backends/mastra.py] _maybe_run_observer(force=False): runs observer when buffer tokens ≥ observer_token_threshold or force=True; appends ObservationBlock from _call_observer and clears buffer; increments counters and marks _obs_index_dirty.
[src: mempol/backends/mastra.py] _maybe_run_reflector(): runs reflector when observation tokens ≥ reflector_token_threshold; appends ReflectionBlock, increments counters, computes reflection_chars and drops consumed observations leaving keep_tail = max(2, len(observations)//4).
[src: mempol/backends/mastra.py] _format_buffer_for_observer(buffer) formats lines "[dia | date | speaker] text".
[src: mempol/backends/mastra.py] _call_observer(buffer): calls llm.chat with _OBSERVER_SYS and user content, model=config.OBSERVER_MODEL, extracts Current task, Suggested response, Date via regex and returns ObservationBlock or None on failure.
[src: mempol/backends/mastra.py] _call_reflector(observations): concatenates observation markdown with separators and calls llm.chat with _REFLECTOR_SYS and model=config.REFLECTOR_MODEL, returns ReflectionBlock or None.

[src: mempol/backends/pie_kg.py] PIEBackend.name = "pie_kg"; wraps pie.core.world_model.WorldModel for KG read/write with hybrid retrieval.
[src: mempol/backends/pie_kg.py] Helper conversions: _as_entity_type, _as_transition_type, _as_relationship_type map strings to PIE enums with fallbacks (CONCEPT, UPDATE, RELATED_TO).
[src: mempol/backends/pie_kg.py] _entity_to_text builds short textual rep of entity: "name (type) — state_str" truncated to 300 chars.
[src: mempol/backends/pie_kg.py] _entity_to_hit(entity,score,source,n_transitions=0): builds Hit with Unit uid=entity.id, text from _entity_to_text, metadata including name,type,current_state,first_seen,last_seen,timestamp,observed_at,valid_from,valid_until,n_transitions,importance,aliases.
[src: mempol/backends/pie_kg.py] ingest(units): cheap fallback that creates low-importance EVENT entities via WorldModel.create_entity with u.text[:500].
[src: mempol/backends/pie_kg.py] _build_bm25_index(): builds BM25Index over per-entity text = name + aliases + primitive current_state values lowercased; returns (index,uids).
[src: mempol/backends/pie_kg.py] retrieve(query,k=10,source="hybrid"): supports sources "ner","bm25","dense","hybrid"; backfills missing embeddings via llm.embed and WorldModel.set_entity_embedding/rebuild_embedding_matrix; primitive rankers: _ner_rank uses find_by_string_match(threshold=0.6), _bm25_rank uses BM25Index.score on _tokens(query), _dense_rank uses llm.embed(query) and wm.find_by_embedding(top_k=k*4).
[src: mempol/backends/pie_kg.py] retrieve hybrid: gathers ner_uids, bm25_uids, dense_uids, unions them to all_uids, maps to indices and fuses with _rrf(k=60) then returns Hits via _entity_to_hit including n_transitions from wm.get_transitions(uid).
[src: mempol/backends/pie_kg.py] expand(seed_uids,k_per=2): uses wm.get_neighbors(uid) and returns neighbors as _entity_to_hit(score=0.5, source="expand_kg").
[src: mempol/backends/pie_kg.py] _ensure_embeddings(): backfills missing entity embeddings via llm.embed and rebuilds embedding matrix.
[src: mempol/backends/pie_kg.py] lookup_entity(query,type=None,top_k=5): recall-oriented candidate gathering via BM25 over entity text and embedding similarity; returns top_k JSONable dicts with match_score, state, importance, aliases, first_seen,last_seen.
[src: mempol/backends/pie_kg.py] lookup_relation(uid_a,uid_b=None): returns list of relationship dicts from wm.get_relationships(uid_a); filters by uid_b if provided.
[src: mempol/backends/pie_kg.py] create_entity(name,type,state,source="",timestamp=0.0) -> returns new entity id via wm.create_entity with _as_entity_type.
[src: mempol/backends/pie_kg.py] update_state(uid,new_state,transition_type="update",...,is_contradiction if transition_type=="contradiction") calls wm.update_entity_state and returns bool success.
[src: mempol/backends/pie_kg.py] add_alias(uid,alias) returns bool(wm.add_alias).
[src: mempol/backends/pie_kg.py] merge_entities(canonical_uid,alias_uid): reassigns transitions and relationships from alias to canonical, archives alias by update_entity_state({"merged_into": canonical_uid}), returns True on success.
[src: mempol/backends/pie_kg.py] add_relation(source_uid,target_uid,rel_type,...) maps rel_type via _as_relationship_type and calls wm.add_relationship returning bool.
[src: mempol/backends/pie_kg.py] mark_contradiction(uid,contradicting_state,...) delegates to update_state with transition_type="contradiction" and trigger_summary.
[src: mempol/backends/pie_kg.py] forget(uid,reason="") marks entity archived via update_state(new_state={"archived":True,"reason":reason}) and returns success.
[src: mempol/backends/pie_kg.py] stats() returns self.wm.stats(); save(path) sets wm.persist_path and calls wm.save().

[src: mempol/backends/providers.py] _units_to_sessions(units): groups Units by metadata['session'] into sessions list of dicts [{"role","content","dia_id"}], returns (sessions,dates_per_session) and uses session_date from metadata.
[src: mempol/backends/providers.py] ProviderBackend wraps MemoryProvider with name default "provider:{provider.name}", caches units list and uid->index mapping for expand().
[src: mempol/backends/providers.py] ProviderBackend.ingest(units): appends units to local cache and calls provider.ingest(sessions,dates) where sessions computed by _units_to_sessions.
[src: mempol/backends/providers.py] ProviderBackend.retrieve(query,k,source): calls provider.search(query,top_k=k) and maps results to Hits with uid from result.metadata.uid or dia_id or generated id, score=result.score, source=self.name.
[src: mempol/backends/providers.py] ProviderBackend.expand(seed_uids,k_per=2): approximates graph via adjacent-turn fallback over cached _units returning Hits score=0.4 source="adjacent".
[src: mempol/backends/providers.py] make_* factory functions produce ProviderBackend instances for Mem0, Zep, Supermemory, Honcho, and a pie_provider adapter.

[src: mempol/llm.py] OpenAI client wrapper: client() lazily constructs OpenAI(api_key=config.OPENAI_API_KEY).
[src: mempol/llm.py] embed(texts, model=None, batch=64): uses model=config.EMBED_MODEL default, caches embeddings on-disk per-model at config.CACHE_DIR/emb_{model}.jsonl keyed by sha1(text); fetches missing via OpenAI embeddings API in batches and appends to cache file; returns numpy float32 array (N,D).
[src: mempol/llm.py] _embed_cache_path(model) = config.CACHE_DIR / f"emb_{model.replace('/', '_')}.jsonl".
[src: mempol/llm.py] _key(text) = sha1(text) hex digest used for cache keys.
[src: mempol/llm.py] _is_reasoning_model(model) considers models starting with "gpt-5","o1","o3","o4" as reasoning models which require special handling of sampling params.
[src: mempol/llm.py] chat(messages,model=None,json_mode=False,**kw): defaults model=config.ANSWER_MODEL, sets temperature=0.0 for non-reasoning models, strips sampler params for reasoning models and maps max_tokens → max_completion_tokens, retries up to 3 times on errors with exponential sleep, returns choice content string.

[src: mempol/config.py] ROOT path = repo root; LOCOMO_PATH, RESULTS_DIR, TRACES_DIR, CACHE_DIR built relative to ROOT; RESULTS_DIR/TRACES_DIR/CACHE_DIR are created on import.
[src: mempol/config.py] .env loader _load_dotenv(path) sets os.environ keys not already set, called for ROOT/.env and cwd/.env.
[src: mempol/config.py] Model defaults (overridable via env MEMPOL_*):
  ANSWER_MODEL default "gpt-5-mini"
  REFORMULATE_MODEL "gpt-5-mini"
  OBSERVER_MODEL "gpt-5-mini"
  REFLECTOR_MODEL "gpt-5-mini"
  JUDGE_MODEL default "gpt-4o-mini"
  EMBED_MODEL default "text-embedding-3-large"
[src: mempol/config.py] OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

[src: mempol/rollout.py] StepRecord dataclass fields: step_index,state_text,op,args,obs_summary,sampled_tokens,logprobs,prompt_tokens.
[src: mempol/rollout.py] Trajectory dataclass fields: qid,question,gold,answer,judge_score,cost,reward,step_records.
[src: mempol/rollout.py] cost_of(trace,lambda_step=0.005,lambda_retrieval=0.01): cost = lambda_step * len(trace.steps) + lambda_retrieval * trace.n_retrievals.
[src: mempol/rollout.py] trace_to_records(trace,question): converts Trace.steps into StepRecord list and accumulates state_so_far string with <op>/<obs> tags.
[src: mempol/rollout.py] collect_rollouts(question,gold,backend,policy_sampler,G=8,cost_lambda_step=0.005,cost_lambda_retrieve=0.01): samples G policies via policy_sampler(), runs each to get Trace, scores via judge(), computes cost and reward = judge_score - cost, returns list of Trajectory.
[src: mempol/rollout.py] compute_advantages(rewards,normalise=True): group-relative advantage = rewards - mean; if normalise and len>1 divides by population std (sqrt(sum(a*a)/N)).