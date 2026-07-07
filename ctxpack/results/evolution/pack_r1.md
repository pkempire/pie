Combined context pack — organized by component. Keep this as the single source of truth for answering precise questions.

Quick facts / Clarifications (plainly stated)
- Embedding cache key: embeddings are cached on-disk per-model keyed by sha1(text) hex digest of the full text. Cache file path: config.CACHE_DIR / f"emb_{model.replace('/', '_')}.jsonl". [src: mempol/llm.py] 
- Write-tool non-mutating operations (explicit): NON_MUTATING_OPS = ("lookup_entity", "lookup_relation", "noop"). Mutating ops are everything else in the op vocabulary (create_entity, update_state, merge_entities, add_relation, mark_contradiction, forget). [src: mempol/recipes/memory_rl/write_tools.py]
- LLM judge score → 1.0 threshold: raw judge score is bucketed to 1.0 when score >= 0.75. (Buckets: >=0.75 → 1.0; >=0.25 → 0.5; else 0.0.) [src: mempol/eval/judge.py] [src: mempol/recipes/memory_rl/reward.py]
- How per-op counterfactual assigns credit (plain): per_op_counterfactual replays the full ops_log to compute the baseline (full_state_score). Then, for each mutating op index, it replays a leave-one-out variant (skip that mutating op), re-scores the QA battery, and records the mean score delta relative to full_state. The per-op marginal utility is that delta. The trajectory reward is sum(per-op deltas) − cost_per_mut * n_mutating_ops. WriteReward normally passes cost_per_op (default 0.001) as the cost_per_mut parameter, so the effective per-op penalty applied in WriteReward is typically 0.001 per mutating op. [src: mempol/eval/counterfactual.py] [src: mempol/recipes/memory_rl/write_reward.py]
- Default GRPO group size used for write training: default group_size = 4. This comes from WriteRLDatasetBuilder default group_size=4 and is the typical value used to build WriteEnvGroupBuilder groups for training. (WriteEnvGroupBuilder accepts group_size but is usually constructed via the dataset/builder which defaults to 4.) [src: mempol/recipes/memory_rl/write_env.py] [src: mempol/recipes/memory_rl/write_env.py (WriteRLDatasetBuilder)]

CoTrain / orchestration
- CoTrainConfig (co-training outer loop)
  - Defaults: n_outer=5; r_steps_per_iter=200; w_steps_per_iter=100
  - model_name="Qwen/Qwen3-4B-Instruct-2507"; lora_rank=32; learning_rate=4e-5
  - r_batch_size=4; r_group_size=8; r_max_turns=6
  - w_batch_size=4; w_group_size=8; w_max_turns=4; w_max_battery_per_turn=6
  - n_convs=8; train_frac=0.8; seed=2
  - log_path: None; wandb_project: None; behavior_if_log_dir_exists default "delete"
  - Mechanism: alternates Phase A (train R using frozen W writes) and Phase B (train W rewarded by R). Each phase launched as subprocess running provided CLIs: tinker_cookbook.recipes.memory_rl.train (Phase A), train_write (Phase B). Checkpoints passed as tinker:// paths extracted from run logs; cotrain stores history.json with iter, r_ckpt, w_ckpt. [src: mempol/recipes/memory_rl/cotrain.py]

Subprocess & checkpoint helpers
- _run_subprocess(cmd, log_file)
  - Uses asyncio.create_subprocess_exec; tees stdout+stderr to log_file; returns exit code. [src: mempol/recipes/memory_rl/cotrain.py]
- _extract_checkpoint_path(log_path)
  - Regex searches backward for sampler_path ... "tinker://..." in log file lines; returns the tinker:// checkpoint string or None. [src: mempol/recipes/memory_rl/cotrain.py]

Phase helpers (train/train_write)
- _train_R_phase: invokes module tinker_cookbook.recipes.memory_rl.train with CLI args from CoTrainConfig; seed=cfg.seed + t; returns extracted R checkpoint path. [src: mempol/recipes/memory_rl/cotrain.py]
- _train_W_phase: invokes tinker_cookbook.recipes.memory_rl.train_write; seed=cfg.seed + 1000 + t; can pass r_checkpoint via env var MEMPOL_R_CHECKPOINT; returns extracted W checkpoint path. [src: mempol/recipes/memory_rl/cotrain.py]

Data types & conversion
- data.MemoryDatum (TypedDict)
  - keys: question, gold_answer, category, data_source, sample_id, qid, conversation_units
  - conversation_units: list[dict] — chunked windows with uid, text, metadata [src: mempol/recipes/memory_rl/data.py]
- _conv_to_serializable_units(conv, window=6, stride=3)
  - Produces sliding-window chunks per session; header "[date | session N]" + concatenated "speaker: text" lines; metadata: session, session_date, first/last dia_id, dia_ids, speaker, n_turns, chunk_idx_in_session, timestamp, dia_id. [src: mempol/recipes/memory_rl/data.py]
- locomo_to_memory_data(n_convs=None, train_frac=0.8, seed=0)
  - Loads LoCoMo convs; shuffles; splits by conversation-level train_frac; returns (train_list, eval_list) of MemoryDatum. [src: mempol/recipes/memory_rl/data.py]
- longmemeval_jsonl_to_memory_data(jsonl_path)
  - Parses LongMemEval JSONL into MemoryDatum; handles session field variants. [src: mempol/recipes/memory_rl/data.py]
- longmemeval_to_memory_data(variant="longmemeval_s", n_rows=None, train_frac=0.8, seed=0, per_category=0, download=True)
  - Loads LongMemEval via loader; supports balanced prefix per_category; returns (train, eval). [src: mempol/recipes/memory_rl/data.py]
- mix_sources(sources: dict[str,list], weights=None, seed=0)
  - Interleaves multiple sources with optional per-source weights (defaults uniform); shuffles pools and samples until exhausted. [src: mempol/recipes/memory_rl/data.py]

Memory-read RL environment
- MEMORY_TASK_INSTRUCTIONS (read-policy)
  - Tools: memory_search(query,k,source), memory_expand(seed_uids,k_per), memory_filter(predicate,value), memory_rerank(strategy,query), memory_top_n(n)
  - Reward = correctness − tool-use cost; format_coef default 0.1 [src: mempol/recipes/memory_rl/memory_env.py]
- _backend_from_units(units): builds FlatBackend, ingests Unit(uid,text,metadata) [src: mempol/recipes/memory_rl/memory_env.py]
- _initial_messages(datum, renderer, memory_tool): registers tool specs via .to_spec(); creates system prefix + user question [src: mempol/recipes/memory_rl/memory_env.py]
- MemoryEnvGroupBuilder
  - Args: datum, model_name, renderer_name, max_turns, group_size, format_coef=0.1, max_trajectory_tokens=32*1024, max_generation_tokens=None, context_overflow_reward=-0.1
  - make_envs: builds independent FlatBackend per env, MemoryTool(backend), JudgeReward(gold_answer, question, format_coef), calls build_agent_tool_env with tools [memory_search,memory_expand,memory_filter,memory_rerank,memory_top_n], initial_messages, reward_fn, max_turns. [src: mempol/recipes/memory_rl/memory_env.py]
  - logging_tags: returns [data_source, category] [src: mempol/recipes/memory_rl/memory_env.py]
- MemoryRLDataset(batch_size)
  - get_batch returns slice; __len__ defensive: if empty returns 0 else max(1, len//batch_size) [src: mempol/recipes/memory_rl/memory_env.py]
- MemoryRLDatasetBuilder (chz schema)
  - model_name_for_tokenizer: required; dataset="locomo"; n_convs=8; lme_rows=120; lme_per_category=0; train_frac=0.8; batch_size=16; group_size=8; renderer_name=None; max_turns=6; format_coef=0.1; max_trajectory_tokens=16*1024; seed=0
  - __call__: builds train/eval via locomo/longmemeval/mixed options; shuffles train_data; returns MemoryRLDataset instances. [src: mempol/recipes/memory_rl/memory_env.py]

Read-side reward
- reward.JudgeReward
  - Fields: gold_answer, question, format_coef=0.1, correct_reward=1.0, partial_reward=0.5, wrong_reward=0.0 [src: mempol/recipes/memory_rl/reward.py]
  - _extract_answer(text): returns last substring after final "Answer:" or None [src: mempol/recipes/memory_rl/reward.py]
  - __call__(history): finds last assistant message, extracts text (uses tinker_cookbook.renderers.get_text_content if available), extracts Answer line; correct_format = 1.0 if Answer present else 0.0; if Answer present, runs mempol.eval.judge.judge synchronously in thread; judge_score >=0.75 → correct_reward; >=0.25 → partial_reward; else wrong_reward.
  - Return: (reward, {"format":..., "correct":...})
  - Note: format_coef default 0.1. [src: mempol/recipes/memory_rl/reward.py] [src: mempol/eval/judge.py]

tinker_compat shim
- Exports: tool, build_agent_tool_env, simple_tool_result, ToolResult, HAS_TINKER
- If tinker_cookbook available: wrapper passthrough.
- If not: tool is no-op decorator that attaches .to_spec() returning {name,description,parameters}; wrapped function gets attribute _mempol_tool=True; build_agent_tool_env raises RuntimeError; ToolResult is dict subclass; simple_tool_result returns ToolResult. [src: mempol/recipes/memory_rl/tinker_compat.py]

Memory tools (read tool)
- tools.MemoryTool (per-env)
  - Fields: backend: Backend, last_hits: list[Hit]=[], n_searches=0, max_searches=8
  - Helpers: _format_hit, _format_observation (caps hits in obs at 10)
  - memory_search(query, k=10, source="hybrid")
    - Enforces max_searches; k bounded 1..20; source ∈ {"bm25","dense","hybrid"}; calls backend.retrieve(query,k,source); updates last_hits, n_searches; returns simple_tool_result(JSON observation) [src: mempol/recipes/memory_rl/tools.py]
  - memory_expand(seed_uids, k_per=2)
    - seed_uids limited to 5; new_hits = backend.expand(seed_uids,k_per); merges avoiding duplicates; updates last_hits; returns obs [src: mempol/recipes/memory_rl/tools.py]
  - memory_filter(predicate, value)
    - Works on last_hits; predicates: session_lt/session_gt/session_eq (int), speaker_eq (str), date_lt/date_gt/date_between (ISO or human parsed), type_eq (str), keyword_in/keyword_not_in (substring); on parse error or unknown predicate returns error obs; updates last_hits to kept subset and returns obs [src: mempol/recipes/memory_rl/tools.py]
  - memory_top_n(n)
    - Truncates last_hits to top-n bounded 1..50; returns obs [src: mempol/recipes/memory_rl/tools.py]
  - memory_rerank(strategy="dense", query=None)
    - If strategy=="dense" and query provided: fresh = backend.retrieve(query,k=len(last_hits)*2,source="dense"); orders last_hits by fresh order
    - Strategies "session_desc"/"session_asc" sort by metadata session [src: mempol/recipes/memory_rl/tools.py]

Universal memory (write/read mixed)
- UNIVERSAL_MEMORY_INSTRUCTIONS: tools: search_raw_spans, write_memory_state, freeze_raw_access, retrieve_memory_states; must freeze before answering; final answer must match "Answer: <short answer>" [src: mempol/recipes/memory_rl/universal_env.py]
- universal_env.UniversalDatum fields: question, gold_answer, qid, sample_id, source, raw_spans (list of dict with artifact/span) [src: mempol/recipes/memory_rl/universal_env.py]
- _locomo_raw_span_data(n_convs=..., train_frac=0.8, seed=0)
  - Converts LoCoMo turns into raw_spans with artifact/span ids prefixed locomo_artifact_/locomo_span_; returns train/eval UniversalDatum lists [src: mempol/recipes/memory_rl/universal_env.py]
- _build_store_for_datum(datum)
  - Creates temporary SQLiteMemoryStore; upserts artifacts and spans; commits; returns SQLiteMemoryStore [src: mempol/recipes/memory_rl/universal_env.py]
- UniversalMemoryEnvGroupBuilder
  - Args: datum, model_name, renderer_name, group_size, max_turns=10, max_trajectory_tokens=12*1024, max_generation_tokens=None, context_overflow_reward=-0.2
  - make_envs: requires HAS_TINKER True; builds per-env SQLiteMemoryStore via _build_store_for_datum, UniversalMemoryTool(store), UniversalMemoryReward(question,gold_answer,tool); registers tools [search_raw_spans,write_memory_state,freeze_raw_access,retrieve_memory_states]. [src: mempol/recipes/memory_rl/universal_env.py]

UniversalMemoryReward & tools
- UniversalMemoryReward
  - Fields: question, gold_answer, tool: UniversalMemoryTool, format_coef=0.1, write_bonus=0.05, token_cost_coef=0.0005, raw_left_open_penalty=0.15
  - __call__(history): extracts final assistant text, Answer line via _extract_answer; judge_score via mempol.eval.judge in thread; stats = tool.stats(); write_bonus applied if writes>0; cost = token_cost_coef * token_cost; raw_penalty applied if raw_enabled True; reward = judge_score + format_coef*(format_ok - 1.0) + write_bonus - cost - raw_penalty; returns (reward, diagnostics dict) [src: mempol/recipes/memory_rl/universal_reward.py]
- UniversalMemoryTool (per-env)
  - Fields: store: SQLiteMemoryStore; raw_searches=0; memory_searches=0; writes=0; token_cost=0; written_state_ids=[]; max_raw_searches=8; max_memory_searches=8; max_writes=24; raw_enabled=True
  - search_raw_spans_impl(query,k=8)
    - Error if raw_enabled False or max_raw_searches reached; k bounded 1..20; hits = store.retrieve(include_spans=True) filtered kind=="span" limited; increments raw_searches; token_estimate added to token_cost; returns JSON obs with span_id, artifact_id, source, score, text[:900], locator [src: mempol/recipes/memory_rl/universal_tools.py]
  - write_memory_state_impl(content, source_span_ids)
    - Enforces max_writes; validates source_span_ids exist; content trimmed non-empty; sid = stable_id("rl_state",content,source_span_ids); upserts MemoryState(id=sid, content, source_span_ids, timestamps); increments writes, appended sid to written_state_ids, token_estimate(content) added to token_cost; returns obs with written_state_id and tokens_est [src: mempol/recipes/memory_rl/universal_tools.py]
  - retrieve_memory_states_impl(query,k=8)
    - Enforces max_memory_searches; k bounded 1..20; hits = store.retrieve(include_spans=False) filtered kind=="memory_state"; increments memory_searches; token_cost += sum token_estimate; returns obs with memory_state_id, source, score, content[:1200], source_span_ids[:8] [src: mempol/recipes/memory_rl/universal_tools.py]
  - freeze_raw_access_impl(reason=""): sets raw_enabled=False; returns obs [src: mempol/recipes/memory_rl/universal_tools.py]
  - stats(): returns dict with raw_searches, memory_searches, writes, token_cost, written_state_ids, raw_enabled [src: mempol/recipes/memory_rl/universal_tools.py]

Write environment & tooling
- write_env.WriteDatum (TypedDict)
  - fields: conv_id, turn_idx, turn_text, turn_dia_id, session_date, prior_turns_text, existing_entities_summary, query_battery (list of (q,gold,evidence_dia_ids)), full_text_backend, full_text_cache, baseline_cache [src: mempol/recipes/memory_rl/write_env.py]
- WriteEnvGroupBuilder
  - Builds group_size envs per WriteDatum; per-env fresh PIEBackend and WriteTool; shared frozen R runner resolved via resolve_r_runner_from_env (falls back to heuristic) [src: mempol/recipes/memory_rl/write_env.py]
  - Key params: model_name, renderer_name, group_size, max_turns, format_coef, max_trajectory_tokens (default 4096), max_generation_tokens, context_overflow_reward=-0.1
  - _initial_messages(datum, renderer, wtool) → system prompt with tool schemas + user block containing session_date, prior_turns_text, focal turn, existing_entities_summary; instruction to default to noop. [src: mempol/recipes/memory_rl/write_env.py]
  - WRITE_TASK_INSTRUCTIONS: tools listed: lookup_entity, create_entity, update_state, merge_entities, add_relation, mark_contradiction, forget, noop; output expects <tool_call> JSON blocks. [src: mempol/recipes/memory_rl/write_env.py]
  - r_runner: if env var MEMPOL_R_CHECKPOINT set resolves to tinker runner; Phase B v2: train_write sets MEMPOL_R_CHECKPOINT so write_env consumes it. [src: mempol/recipes/memory_rl/train_write.py] [src: mempol/recipes/memory_rl/write_env.py]
  - Reward weights read from env vars (strings):
    - MEMPOL_W_CF default "0.7" → w_cf
    - MEMPOL_W_QA default "0.3" → w_qa
    - MEMPOL_W_GAIN default "0.0" → w_gain
    - MEMPOL_W_OVERLAP default "0.0" → w_overlap
    - MEMPOL_W_COV_FLOOR default "0.05" → w_cov_floor
    - MEMPOL_K_MAX default "12" → k_max (int) [src: mempol/recipes/memory_rl/write_env.py]
  - Tools passed to agent env: lookup_entity, lookup_relation, create_entity, update_state, merge_entities, add_relation, mark_contradiction, forget, noop. [src: mempol/recipes/memory_rl/write_env.py]
- WriteRLDataset & WriteRLDatasetBuilder
  - WriteRLDataset wraps list of WriteEnvGroupBuilder with batch_size; get_batch returns slice; __len__ returns max(1, floor(len(builders)/batch_size)) if builders non-empty [src: mempol/recipes/memory_rl/wri