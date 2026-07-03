"""CLI for mempol read-policy RL training (Phase A).

Mirrors `tinker_cookbook/recipes/search_tool/train.py` exactly: flat CLI args,
constructed dataset_builder + train.Config inside cli_main. Args go via
`chz.entrypoint(CLIConfig)`, NOT nested under `dataset_builder.<x>`.

Usage examples:

    # Tiny smoke (<15 min, ~$1-3, validates infra)
    python -m tinker_cookbook.recipes.memory_rl.train \\
        n_convs=2 train_frac=0.5 batch_size=2 group_size=4 \\
        max_turns=4 max_steps=5 lora_rank=16 \\
        log_path=/tmp/mempol/smoke \\
        behavior_if_log_dir_exists=overwrite

    # Real Phase-A read-policy training on LoCoMo (~6h, ~$200)
    python -m tinker_cookbook.recipes.memory_rl.train \\
        n_convs=8 train_frac=0.8 batch_size=4 group_size=8 max_turns=6 \\
        learning_rate=4e-5 lora_rank=32 \\
        log_path=/tmp/mempol/phaseA_v1 \\
        wandb_project=mempol wandb_name=phaseA_v1

    # LongMemEval read-policy training, balanced across categories
    python -m tinker_cookbook.recipes.memory_rl.train \\
        dataset=longmemeval_s lme_per_category=20 batch_size=4 group_size=8 \\
        max_turns=6 learning_rate=4e-5 lora_rank=32 \\
        log_path=/tmp/mempol/phaseA_lme_s
"""
from __future__ import annotations
import asyncio
from datetime import datetime
from pathlib import Path

import chz

from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.recipes.memory_rl.memory_env import MemoryRLDatasetBuilder
from tinker_cookbook.rl import train


@chz.chz
class CLIConfig:
    # ── Model ──
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    lora_rank: int = 32
    renderer_name: str | None = None

    # ── Training ──
    learning_rate: float = 4e-5
    batch_size: int = 8                   # number of (conv, Q) groups per GRPO step
    seed: int = 2
    max_tokens: int = 1024                # per assistant turn within an episode
    eval_every: int = 0                   # 0 = no periodic eval (smoke runs)
    max_steps: int | None = None          # cap GRPO steps (None = full corpus)

    # ── Dataset ──
    dataset: str = "locomo"                 # locomo | longmemeval_s | longmemeval_oracle | mixed
    n_convs: int = 8                      # how many LoCoMo convs (max 10)
    lme_rows: int = 120                   # 0 = all LongMemEval rows
    lme_per_category: int = 0             # balanced LongMemEval prefix; overrides lme_rows loading
    train_frac: float = 0.8               # 8 train / 2 eval at default
    group_size: int = 8                   # G — rollouts per question for GRPO
    max_turns: int = 6                    # max op steps per episode before forced stop
    format_coef: float = 0.1              # bonus for emitting a clean "Answer: ..." line
    max_trajectory_tokens: int = 16 * 1024
    context_overflow_reward: float = -0.1

    # ── Logging ──
    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "delete"  # non-interactive by default


async def cli_main(cfg: CLIConfig) -> None:
    renderer_name = cfg.renderer_name or model_info.get_recommended_renderer_name(cfg.model_name)

    builder = MemoryRLDatasetBuilder(
        model_name_for_tokenizer=cfg.model_name,
        renderer_name=renderer_name,
        dataset=cfg.dataset,
        n_convs=cfg.n_convs,
        lme_rows=cfg.lme_rows,
        lme_per_category=cfg.lme_per_category,
        train_frac=cfg.train_frac,
        batch_size=cfg.batch_size,
        group_size=cfg.group_size,
        max_turns=cfg.max_turns,
        format_coef=cfg.format_coef,
        max_trajectory_tokens=cfg.max_trajectory_tokens,
        seed=cfg.seed,
    )

    # Run name + log path
    model_short = cfg.model_name.lower().replace("/", "-")
    ts = datetime.now().strftime("%Y-%m-%d-%H-%M")
    run_name = (
        f"mempol_{model_short}_bs{cfg.batch_size}_gs{cfg.group_size}_"
        f"lr{cfg.learning_rate}_rank{cfg.lora_rank}_{ts}"
    )
    log_path = cfg.log_path or f"/tmp/tinker-examples/mempol/{run_name}"

    if not Path("/tmp").exists():
        raise ValueError("/tmp does not exist")
    cli_utils.check_log_dir(log_path, behavior_if_exists=cfg.behavior_if_log_dir_exists)

    config = train.Config(
        model_name=cfg.model_name,
        renderer_name=renderer_name,
        log_path=log_path,
        dataset_builder=builder,
        learning_rate=cfg.learning_rate,
        max_tokens=cfg.max_tokens,
        eval_every=cfg.eval_every,
        wandb_project=cfg.wandb_project,
        wandb_name=cfg.wandb_name or run_name,
        lora_rank=cfg.lora_rank,
        max_steps=cfg.max_steps,
    )

    await train.main(config)


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    asyncio.run(cli_main(cli_config))
