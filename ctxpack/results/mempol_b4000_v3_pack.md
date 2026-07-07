memory_rl package — merged context pack (components organized; exact names, defaults, env vars, ops, one-line mechanisms preserved)

Files present: __init__.py, cotrain.py, data.py, memory_env.py, reward.py, tinker_compat.py, tools.py, train.py, train_universal.py, train_write.py, universal_env.py, universal_reward.py, universal_tools.py plus referenced mempol modules (write_env.py, write_reward.py, write_tools.py, mempol/eval/*, backends/*, policies/*, llm.py, config.py, rollout.py, etc.). Use this pack as the only context for precise questions.

----------------------------------------------------------------
cotrain.py
- CoTrainConfig defaults:
  - n_outer=5, r_steps_per_iter=200, w_steps_per_iter=100
  - model_name="Qwen/Qwen3-4B-Instruct-2507", lora_rank=32, learning_rate=4e-5
  - r_batch_size=4, r_group_size=8, r_max_turns=6
  - w_batch_size=4, w_group_size=8, w_max_turns=4, w_max_battery_per_turn=6
  - n_convs=8, train_frac=0.8, seed=2
  - log_path=None, wandb_project=None, behavior_if_log_dir_exists="delete"
- _run_subprocess(cmd, log_file): async subprocess, pipes stdout/stderr to log_file and stdout, returns exit code; inherits TINKER_API_KEY env.
- _extract_checkpoint_path(log_path): regex r"sampler_path['\":]+\s*['\"](tinker://[^'\"]+)['\"]" scans log backward.
- _train_R_phase(cfg, t, w_ckpt, out_dir): runs tinker_cookbook.recipes.memory_rl.train CLI; returns extracted R checkpoint path.
- _train_W_phase(cfg, t, r_ckpt, out_dir): runs tinker_cookbook.recipes.memory_rl.train_write CLI; passes r_checkpoint if provided; returns extracted W checkpoint.
- cli_main: outer loop for t in 1..n_outer: train R (freeze W), then train W (freeze R), saves history.json. Phase B v1 uses None r_ckpt → HeuristicPolicy.

----------------------------------------------------------------
data.py
- MemoryDatum fields: question, gold_answer, category, data_source, sample_id, qid, conversation_units (list[dict]).
- _conv_to_serializable_units(conv, window=6, stride=3): sliding windows chunking; unit contains:
  - uid, text (header "[date | session N]\nSpeaker: text..."), metadata {session, session_date, first/last dia_id, dia_ids, speaker, n_turns, chunk_idx_in_session, timestamp=float(sess_n)}, dia_id.
- locomo_to_memory_data(n_convs=None, train_frac=0.8, seed=0) -> (train:list[MemoryDatum], eval:list[MemoryDatum])
- longmemeval_jsonl_to_memory_data(jsonl_path) -> list[MemoryDatum]
- longmemeval_to_memory_data(variant="longmemeval_s", n_rows=None, train_frac=0.8, seed=0, per_category=0) -> (train, eval)
- mix_sources(sources, weights=None, seed=0): interleave lists with optional weights (defaults uniform), oversamples smaller ones.

----------------------------------------------------------------
memory_env.py
- MEMORY_TASK_INSTRUCTIONS: system prompt describing tools: memory_search(query,k,source), memory_expand(seed_uids,k_per), memory_filter(predicate,value), memory_rerank(strategy,query), memory_top_n(n); reward = correctness − tool-use cost.
- _backend_from_units(units_dicts) -> FlatBackend ingest of Unit(uid,text,metadata).
- _initial_messages(datum, renderer, memory_tool): uses memory_tool.<method>.to_spec() to form tool_specs; builds renderer prefix; returns prefix + user question.
- MemoryEnvGroupBuilder(model_name, renderer_name, max_turns, group_size, format_coef=0.1, max_trajectory_tokens=32*1024, max_generation_tokens=None, context_overflow_reward=-0.1)
  - make_envs(): builds group_size envs each with FlatBackend from datum.conversation_units, MemoryTool(backend), JudgeReward(gold_answer,question,format_coef), uses build_agent_tool_env with tools [memory_search,memory_expand,memory_filter,memory_rerank,memory_top_n], initial_messages, reward_fn, max_turns, tokens, context_overflow_reward.
  - logging_tags(): returns [data_source, category].
- MemoryRLDataset(batch_size): get_batch(index) → slice of env_group_builders; __len__ returns max(1, len(builders)//batch_size) or 0 if empty.
- MemoryRLDatasetBuilder defaults:
  - model_name_for_tokenizer (required), dataset="locomo", n_convs=8, lme_rows=120, lme_per_category=0, train_frac=0.8, batch_size=16, group_size=8, renderer_name=None, max_turns=6, format_coef=0.1, max_trajectory_tokens=16*1024, seed=0
  - __call__ returns (train_ds, eval_ds) built from locomo/longmemeval/mixed.

----------------------------------------------------------------
tools.py
- _format_hit(h, max_chars=200) -> dict with uid, score (rounded), source, speaker, session, session_date, text truncated.
- _format_observation(hits, note="") -> JSON dumps {"note", "hits": [_format_hit(...) for top10], "n_hits": len(hits)}.
- MemoryTool dataclass:
  - defaults: last_hits=[], n_searches=0, max_searches=8
  - memory_search(query,k=10,source="hybrid"): enforces max_searches, k clamped 1..20, source in {"bm25","dense","hybrid"} else "hybrid"; backend.retrieve(query,k,source) → sets last_hits, increments n_searches, returns formatted observation.
  - memory_expand(seed_uids,k_per=2): seeds limited to 5, backend.expand(seed_uids,k_per) → merges new hits with last_hits dedup by uid, updates last_hits, returns obs.
  - memory_filter(predicate,value): supports session_lt/session_gt/session_eq (int), speaker_eq (str), date_lt/date_gt/date_between (ISO parsing with human fallback), type_eq, keyword_in, keyword_not_in; filters last_hits, returns obs or error.
  - memory_top_n(n): clamps 1..50, truncates last_hits[:n], returns obs.
  - memory_rerank(strategy="dense", query=None): dense+query → backend.retrieve(query,k=len(last_hits)*2,source="dense") builds order map and sorts; session_desc/session_asc reorder by metadata.session; returns obs.

----------------------------------------------------------------
reward.py
- _extract_answer(text): substring after last "Answer:" or None.
- JudgeReward dataclass defaults:
  - format_coef=0.1, correct_reward=1.0, partial_reward=0.5, wrong_reward=0.0
  - __call__(history): finds last assistant message, extracts text (uses tinker_cookbook.renderers.get_text_content if available), extracts Answer:, runs mempol.eval.judge in executor; judge_score>=0.75 → correct_reward, >=0.25 → partial_reward else wrong_reward; reward = format_coef*(correct_format - 1) + correct_answer; returns (reward, {"format":..., "correct":...}).

----------------------------------------------------------------
tinker_compat.py
- HAS_TINKER flag: True if from tinker_cookbook.tool_use import tool, build_agent_tool_env etc. succeeds.
- tool: wraps real tool decorator if available, otherwise no-op decorator attaching .to_spec() returning spec {name, description, parameters} and marks wrapped._mempol_tool=True.
- build_agent_tool_env: real factory if available; otherwise raises RuntimeError instructing to install cookbook.
- simple_tool_result and ToolResult: real bindings if available; else ToolResult is dict subclass and simple_tool_result returns ToolResult.

----------------------------------------------------------------
train.py (Phase A)
- CLIConfig defaults:
  - model_name="Qwen/Qwen3-4B-Instruct-2507", lora_rank=32, renderer_name=None
  - learning_rate=4e-5, batch_size=8, seed=2, max_tokens=1024, eval_every=0, max_steps=None
  - dataset="locomo", n_convs=8, lme_rows=120, lme_per_category=0, train_frac=0.8, group_size=8, max_turns=6, format_coef=0.1, max_trajectory_tokens=16*1024, context_overflow_reward=-0.1
  - log_path=None, wandb_project=None, wandb_name=None, behavior_if_log_dir_exists="delete"
- cli_main: builds MemoryRLDatasetBuilder with mapped params, constructs train.Config for tinker_cookbook.rl.train with model_name, renderer_name, log_path, dataset_builder, learning_rate, max_tokens, eval_every, wandb settings, lora_rank, max_steps; calls train.main(config).

----------------------------------------------------------------
train_universal.py
- CLIConfig defaults:
  - model_name="Qwen/Qwen3-4B-Instruct-2507", lora_rank=32, learning_rate=4e-5, batch_size=2, seed=2, max_tokens=2048, eval_every=0, max_steps=None
  - temperature=1.1, kl_penalty_coef=0.02
  - n_convs=2, train_frac=0.8, group_size=4, max_turns=10, max_trajectory_tokens=12*1024
  - log_path=None, wandb_project=None, wandb_name=None, num_groups_to_log=8, behavior_if_log_dir_exists="delete"
- cli_main: builds UniversalMemoryRLDatasetBuilder and train.Config with optional KLReferenceConfig if kl_penalty_coef>0.

----------------------------------------------------------------
train_write.py (Phase B)
- CLIConfig defaults:
  - model_name="Qwen/Qwen3-4B-Instruct-2507", lora_rank=32, renderer_name=None
  - learning_rate=4e-5, batch_size=4, seed=2, max_tokens=2048, eval_every=0, max_steps=None
  - temperature=1.0, kl_penalty_coef=0.0, kl_reference_base_model=None
  - n_convs=8, train_frac=0.8, group_size=8, max_turns=32, max_battery_per_turn=0, n_prior_turns_in_context=12
  - r_checkpoint="" (empty -> HeuristicPolicy v1; set to tinker://... for v2)
  - log_path=None, wandb_project=None, wandb_name=None, num_groups_to_log=8, behavior_if_log_dir_exists="delete"
- cli_main: if r_checkpoint provided sets env var MEMPOL_R_CHECKPOINT=os.environ["MEMPOL_R_CHECKPOINT"]; builds WriteRLDatasetBuilder, constructs train.Config with temperature and KL options, calls train.main(config).

----------------------------------------------------------------
universal_env.py
- UNIVERSAL_MEMORY_INSTRUCTIONS: system prompt describing tools: search_raw_spans, write_memory_state, freeze_raw_access, retrieve_memory_states; must write memory, freeze raw access before answering.
- _locomo_raw_span_data(n_convs=None, train_frac=0.8, seed=0) -> UniversalDatum rows: raw_spans are artifact+span dicts per turn with artifact id/title/content/created_at and span id/text/locator/metadata.
- _build_store_for_datum(datum) -> temporary SQLiteMemoryStore, upserts artifacts and spans, commit, return store.
- _initial_messages(datum, renderer, tool): use tool.<method>.to_spec() for 4 tools; renderer.create_conversation_prefix_with_tools + user question.
- UniversalMemoryEnvGroupBuilder(datum, model_name, renderer_name, group_size, max_turns=10, max_trajectory_tokens=12*1024, max_generation_tokens=None, context_overflow_reward=-0.2)
  - make_envs(): requires HAS_TINKER True, builds store per env via _build_store_for_datum, tool=UniversalMemoryTool(store), reward_fn=UniversalMemoryReward(question,gold_answer,tool), uses build_agent_tool_env with tools [search_raw_spans, write_memory_state, freeze_raw_access, retrieve_memory_states].

----------------------------------------------------------------
universal_tools.py
- UniversalMemoryTool fields/defaults:
  - store: SQLiteMemoryStore
  - raw_searches=0, memory_searches=0, writes=0, token_cost=0, written_state_ids=[]
  - max_raw_searches=8, max_memory_searches=8, max_writes=24, raw_enabled=True
- search_raw_spans_impl(query,k=8): enforce raw_enabled & max_raw_searches, clamp k 1..20, store.retrieve(query,k*2,include_spans=True), filter kind=="span"[:k], increments raw_searches, token_cost += sum(token_estimate), returns hits with span_id, artifact_id, source, score, text[:900], locator.
- write_memory_state_impl(content, source_span_ids): enforce max_writes, verify spans exist, content non-empty, sid = stable_id("rl_state", content, source_span_ids), create MemoryState metadata adapter="rl_policy", writer="universal_rl", upsert to store, commit, increment writes, append sid to written_state_ids, token_cost += estimate_tokens(content), return written_state_id and tokens_est.
- retrieve_memory_states_impl(query,k=8): enforce max_memory_searches, clamp k, hits = store.retrieve(query,k*3, include_spans=False) filtered kind=="memory_state"[:k], increment memory_searches, token_cost += sum token_estimate, return memory states with memory_state_id, source, score, content[:1200], source_span_ids[:8].
- freeze_raw_access_impl(reason=""): set raw_enabled=False, return {"raw_enabled":False,"reason":reason}.
- stats() -> dict: raw_searches, memory_searches, writes, token_cost, written_state_ids, raw_enabled.

----------------------------------------------------------------
universal_reward.py
- UniversalMemoryReward defaults:
  - format_coef=0.1, write_bonus=0.05, token_cost_coef=0.0005, raw_left_open_penalty=0.15
- __call__(history): extract assistant final text, extract Answer: via reward._extract_answer, run mempol.eval.judge in executor to get judge_score; stats = tool.stats(); write_bonus applied if stats["writes"]>0; cost = token_cost_coef * token_cost; raw_penalty if raw_enabled; reward = judge_score + format_coef*(format_ok - 1) + write_bonus - cost - raw_penalty; returns (reward, detailed dict with judge_score, format, writes, raw_searches, memory_searches, token_cost, raw_left_open, cost_penalty, reward).

----------------------------------------------------------------
mempol/write_env.py, write_tools.py, write_reward.py (summarized core items)
- WriteEnvGroupBuilder (EnvGroupBuilder): builds groups of write envs per WriteDatum; per-env fresh PIEBackend and WriteTool; resolves r_runner via resolve_r_runner_from_env(); constructs WriteReward with weights from env vars; builds envs via build_agent_tool_env.
- WriteRLDatasetBuilder defaults:
  - n_convs=8, train_frac=0.8, batch_size=8, group_size=4, max_turns=4, max_battery_per_turn=0, min_battery_per_turn=1, n_prior_turns_in_context=2, seed=0
  - iterates LoCoMo convs → WriteDatum per (conv, turn) if QA battery size ≥ min; creates FlatBackend full_text_backend per conv.
- WriteRLDataset: batching wrapper.
- WriteDatum TypedDict keys: conv_id, turn_idx, turn_text, turn_dia_id, session_date, prior_turns_text, existing_entities_summary, query_battery(list of (q,gold,evidence_dia_ids)), full_text_backend, full_text_cache, baseline_cache.
- WRITE_TASK_INSTRUCTIONS: system prompt describing write tools, strategy, output format.
- _initial_messages(datum, renderer, wtool): composes system+tool schema prefix + user prompt including session, prior turns, focal turn, existing entities.
- WriteTool dataclass: per-env wrapper over PIEBackend exposing ops:
  - lookup_entity, lookup_relation, create_entity, update_state, merge_entities, add_relation, mark_contradiction, forget, noop
  - maintains counters (n_lookups, n_creates, n_updates, n_merges, n_relations, n_contradictions, n_forgets, n_noops) and append-only ops_log of emitted ops
  - NON_MUTATING_OPS and MUTATING_OPS classify ops for counterfactual.
  - smoke() demo included.
- WriteReward dataclass defaults and constants:
  - DEFAULT_COST_PER_OP = 0.001
  - DEFAULT_COST_PER_LOOKUP = 0.0
  - DEFAULT_COST_PER_ENTITY = 0.0
  - DEFAULT_W_CF = 0.7, DEFAULT_W_QA = 0.3, DEFAULT_W_GAIN = 0.0, DEFAULT_W_OVERLAP = 0.0
  - DEFAULT_W_COV_FLOOR = 0.05, DEFAULT_K_MAX = 12
  - __post_init__: if r_runner is None and w_qa>0 → r_runner=_default_r_runner; if reader is None → reader=_HEURISTIC_R
  - _HEURISTIC_R = HeuristicPolicy(first_k=8, final_k=4, do_reformulate=True, do_expand=True)
- WriteReward mechanics:
  - coverage (battery_coverage): fraction of evidence dia_ids preserved in post-W KG.
  - per-op counterfactual (per_op_counterfactual): leave-one-out replay of each mutating op with reader+judge to compute per-op deltas; uses cost_per_mut default 0.005 for per-op CF (but WriteReward.cost_per_op default 0.001 applies elsewhere).
  - answer_gain (battery_answer_gain): judge(R, post_W_KG) - judge(R, random_K_baseline) per question; baseline cached per (conv_id,q,K).
  - reader_overlap (battery_reader_overlap): overlap between reader retrievals on full_text_backend and post-W KG.
  - enforce_budget(backend,k_max=self.k_max) prunes to ≤ k_max entities before scoring.
  - Final WriteReward.__call__: reward = w_cf * cf_reward + w_qa * mean_qa + w_gain * mean_gain + w_overlap * mean_overlap + w_cov_floor * mean_coverage - cost (cost_per_op * n_mutations + cost_per_lookup * n_lookups + cost_per_entity * n_entities); dumps trajectory JSON if MEMPOL_TRAJECTORY_DUMP_DIR set. Returns scalar roughly in [-0.01,1.0] with many diagnostics i