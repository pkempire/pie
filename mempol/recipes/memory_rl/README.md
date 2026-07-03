# memory_rl — Tinker recipe for memory-policy GRPO

Forked from `tinker_cookbook/recipes/search_tool/`. Same structure, swapped
domain: instead of searching Wikipedia to answer multi-hop QA, the policy
operates over a *conversational memory store* (a LoCoMo conversation) using
4 memory tools.

## Files

| file | role | corresponds to |
|---|---|---|
| `data.py`        | LoCoMo / LongMemEval → `MemoryDatum` (`question, gold, conv_units, …`) | search_env.py:download_search_r1_dataset |
| `tools.py`       | `MemoryTool.{memory_search,expand,filter,rerank}` over a `Backend`     | tools.py:ChromaTool |
| `reward.py`      | `JudgeReward` — LLM-judge over the agent's "Answer: …" line          | tools.py:TextAnswerReward |
| `memory_env.py`  | `MemoryEnvGroupBuilder`, `MemoryRLDataset`, `MemoryRLDatasetBuilder`   | search_env.py |
| `train.py`       | Entrypoint — calls Tinker's RL train main with our dataset builder    | search_tool/train.py |
| `cotrain.py`     | Outer loop alternating R-phase and W-phase (the paper's novelty)      | (new) |

## Setup

**You do NOT fork the cookbook on GitHub.** Just clone it and symlink our
recipe in. We stay an external, drop-in directory; their upgrades never cause
merge conflicts for us.

```bash
# 1. Clone the cookbook (vanilla, no fork)
git clone https://github.com/thinking-machines-lab/tinker-cookbook ~/tinker-cookbook
cd ~/tinker-cookbook && uv pip install -e .

# 2. Symlink our recipe into theirs (relative path; adjust as needed)
ln -s ~/personal-intelligence-system/mempol/recipes/memory_rl \
      tinker_cookbook/recipes/memory_rl

# 3. Make mempol/ importable
export PYTHONPATH=~/personal-intelligence-system:$PYTHONPATH

# 4. Set creds
export TINKER_API_KEY=...
export OPENAI_API_KEY=...   # for our judge / embeddings
```

The two edits needed in our recipe files (one-time):

1. In `tools.py` and `write_tools.py`, add at the top:
   ```python
   from tinker_cookbook.tool_use import tool
   ```
   then add `@tool` above each public method (the ones the policy can call).
2. In `memory_env.py`, confirm the cookbook still exposes
   `build_agent_tool_env` from `tinker_cookbook.tool_use` (it does as of
   commit c53f137, last edited Mar 2026).

## First run (read-policy only — no co-training)

```bash
python -m tinker_cookbook.recipes.memory_rl.train \\
    model_name_for_tokenizer=Qwen/Qwen3-4B-Instruct-2507 \\
    n_convs=8 batch_size=4 group_size=8 max_turns=6 \\
    learning_rate=4e-5 lora_rank=32 save_every=20
```

LongMemEval Phase-A read-policy training:

```bash
python -m tinker_cookbook.recipes.memory_rl.train \\
    dataset=longmemeval_s lme_per_category=20 \\
    batch_size=4 group_size=8 max_turns=6 \\
    learning_rate=4e-5 lora_rank=32 \\
    log_path=/tmp/mempol/phaseA_lme_s \\
    wandb_project=mempol wandb_name=phaseA_lme_s
```

Mixed LoCoMo + LongMemEval training:

```bash
python -m tinker_cookbook.recipes.memory_rl.train \\
    dataset=mixed n_convs=8 lme_per_category=20 \\
    batch_size=4 group_size=8 max_turns=6 \\
    learning_rate=4e-5 lora_rank=32 \\
    log_path=/tmp/mempol/phaseA_mixed
```

Expected: ~10–25 GRPO steps before the policy reliably emits valid tool calls
and the answer format. Watch `env/all/turns_per_episode` — if it stays at 1
the model is just guessing without searching. If it grows past 2, the model is
learning multi-step retrieval.

## What to expect (numbers)

The Search-R1 Tinker replication on `Qwen/Qwen2.5-7B-Instruct` reports:

| Benchmark        | Original Search-R1 paper | Tinker recipe |
|------------------|--------------------------|---------------|
| Natural Questions| 42.9                     | **51.6**      |
| TriviaQA         | 62.3                     | **67.3**      |
| HotpotQA         | 38.6                     | **49.7**      |
| 2WikiMultihopQA  | 34.6                     | **42.8**      |

For LoCoMo, our heuristic teacher (v1) currently gets ~50% on the smoke
sample. Goal: GRPO-trained R (single policy, no co-training) clears 75% on a
single conv → publishable Section 5.1 number.

## Co-training (Phase B is in progress)

`cotrain.py` sketches the alternating loop:

```python
for outer_iter in range(N):
    # Phase A: freeze W, train R via GRPO on memory built by W
    R = train_R_phase(W, train_data)
    # Phase B: freeze R, train W via GRPO with R-built rewards
    W = train_W_phase(R, train_data)   # TODO
```

Phase A reuses the standard memory_rl trainer end-to-end (drop-in).
Phase B requires a `WriteEnvGroupBuilder` (sketched in `cotrain.py`) — a
different env where the action vocab is *write* ops on a Backend, and the
reward is a deferred call to the read policy on a held-out QA battery.

## Multi-source training data

Pass multiple datasets via `mix_sources` in `data.py`:

```python
from mempol.recipes.memory_rl.data import (
    locomo_to_memory_data, longmemeval_to_memory_data, mix_sources
)

train, _ = locomo_to_memory_data(n_convs=8, train_frac=0.8)
lme = longmemeval_to_memory_data(Path("data/longmemeval_train.jsonl"))
mixed = mix_sources({"locomo": train, "lme": lme}, weights={"locomo": 1.0, "lme": 0.5})
```

This is critical for the transferability claim — training only on LoCoMo
overfits to its conversation style; mixing with LongMemEval + (later) MSC +
WildChat-synth gives multi-domain coverage.

## What changed vs the search_tool recipe

1. **Tools**: 4 memory ops instead of 1 search tool. All operate on the
   same `Backend` instance (per env, not shared like Chroma).
2. **Initial messages**: a memory-task system prompt + the question. No
   external knowledge base — everything's in the per-env backend.
3. **Reward**: `JudgeReward` (calls our `mempol.eval.judge`) instead of
   `TextAnswerReward` (string-match Wikipedia answer).
4. **Dataset**: `MemoryDatum` carries a serialized list of conversation units
   in addition to (question, gold). Each env rehydrates a backend from those
   units.
5. **No shared resource**: no Chroma server. Backends are cheap (in-memory
   FlatBackend with on-disk embedding cache).

## TODO list (pre-first-Tinker-run)

- [x] Add `@tool` decorator to MemoryTool methods in `tools.py`
- [ ] Verify `tinker_cookbook.rl.train.main` import path matches the installed Tinker cookbook version
- [ ] Add a Tree (FTS5) backend variant for the Backend/transfer ablation
- [ ] Write `tests/test_memory_env.py` mirroring `tests/test_search_env.py`
- [x] Implement `WriteEnvGroupBuilder` scaffold for cotrain.py Phase B
- [ ] Replace write-policy fresh-KG-per-turn training with chronological accumulated-state episodes
- [ ] Add trained read-policy checkpoint as a `longmemeval_matrix` cell
