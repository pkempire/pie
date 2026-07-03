"""CLI for full RL over the universal memory substrate.

This is not SFT. It trains the policy with GRPO over an environment where the
model must search raw spans, write compressed MemoryStates, freeze raw access,
retrieve memory, and answer.

Usage inside a tinker-cookbook clone after symlinking this recipe:

  python -m tinker_cookbook.recipes.memory_rl.train_universal \\
    n_convs=2 train_frac=0.8 batch_size=2 group_size=4 max_turns=10 \\
    max_steps=5 lora_rank=16 log_path=/tmp/mempol/universal_wiring_check
"""
from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path

import chz

from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.recipes.memory_rl.universal_env import UniversalMemoryRLDatasetBuilder
from tinker_cookbook.rl import train


@chz.chz
class CLIConfig:
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    lora_rank: int = 32
    renderer_name: str | None = None
    learning_rate: float = 4e-5
    batch_size: int = 2
    seed: int = 2
    max_tokens: int = 2048
    eval_every: int = 0
    max_steps: int | None = None
    temperature: float = 1.1
    kl_penalty_coef: float = 0.02
    n_convs: int = 2
    train_frac: float = 0.8
    group_size: int = 4
    max_turns: int = 10
    max_trajectory_tokens: int = 12 * 1024
    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None
    num_groups_to_log: int = 8
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "delete"


async def cli_main(cfg: CLIConfig) -> None:
    renderer_name = cfg.renderer_name or model_info.get_recommended_renderer_name(cfg.model_name)
    builder = UniversalMemoryRLDatasetBuilder(
        model_name_for_tokenizer=cfg.model_name,
        renderer_name=renderer_name,
        n_convs=cfg.n_convs,
        train_frac=cfg.train_frac,
        batch_size=cfg.batch_size,
        group_size=cfg.group_size,
        max_turns=cfg.max_turns,
        max_trajectory_tokens=cfg.max_trajectory_tokens,
        seed=cfg.seed,
    )

    model_short = cfg.model_name.lower().replace("/", "-")
    ts = datetime.now().strftime("%Y-%m-%d-%H-%M")
    run_name = f"mempol_universal_{model_short}_bs{cfg.batch_size}_gs{cfg.group_size}_{ts}"
    log_path = cfg.log_path or f"/tmp/tinker-examples/mempol_universal/{run_name}"
    if not Path("/tmp").exists():
        raise ValueError("/tmp does not exist")
    cli_utils.check_log_dir(log_path, behavior_if_exists=cfg.behavior_if_log_dir_exists)

    kl_ref = None
    if cfg.kl_penalty_coef > 0:
        try:
            from tinker_cookbook.rl.train import KLReferenceConfig  # type: ignore
            kl_ref = KLReferenceConfig(base_model=cfg.model_name)
        except Exception:
            kl_ref = None

    train_config = train.Config(
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
        temperature=cfg.temperature,
        kl_penalty_coef=cfg.kl_penalty_coef,
        kl_reference_config=kl_ref,
        num_groups_to_log=cfg.num_groups_to_log,
    )
    await train.main(train_config)


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    asyncio.run(cli_main(cli_config))
