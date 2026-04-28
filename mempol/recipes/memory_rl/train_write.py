"""CLI for mempol Phase B (write-policy RL training).

Mirrors `train.py` (Phase A read-policy training) using the same flat-CLI chz
pattern, but builds a `WriteRLDatasetBuilder` and trains over write episodes
(one per conversation turn whose evidence is needed by ≥1 LoCoMo question).

The reward is *deferred*: after each write trajectory ends, a frozen R policy
is run against a held-out battery of LoCoMo questions whose evidence depends
on the focal turn. Reward = mean-judge-score − cost. See `write_reward.py`.

Phase B v1 default: R is `mempol.policies.v1_heuristic.HeuristicPolicy`
(deterministic). Phase B v2: pass `r_checkpoint=tinker://...` to load a
Phase-A-trained LoRA as the frozen R.

Usage:

    # Smoke (5 GRPO steps, ~$5-10, validates Phase B infra)
    python -m tinker_cookbook.recipes.memory_rl.train_write \\
        n_convs=2 train_frac=0.5 \\
        batch_size=2 group_size=4 max_turns=4 \\
        max_steps=5 lora_rank=16 \\
        log_path=/tmp/mempol/write_smoke

    # Real Phase B v1 (heuristic R as judge)
    python -m tinker_cookbook.recipes.memory_rl.train_write \\
        n_convs=8 train_frac=0.8 \\
        batch_size=4 group_size=8 max_turns=4 \\
        learning_rate=4e-5 lora_rank=32 \\
        log_path=/tmp/mempol/phaseB_v1

    # Phase B v2 (frozen Tinker-trained R as judge — needs a Phase A ckpt)
    python -m tinker_cookbook.recipes.memory_rl.train_write \\
        n_convs=8 train_frac=0.8 \\
        batch_size=4 group_size=8 max_turns=4 \\
        learning_rate=4e-5 lora_rank=32 \\
        r_checkpoint=tinker://<phase_a_run>:train:0/sampler_weights/final \\
        log_path=/tmp/mempol/phaseB_v2
"""
from __future__ import annotations
import asyncio
import logging
from datetime import datetime
from pathlib import Path

import chz

from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.recipes.memory_rl.write_env import WriteRLDatasetBuilder
from tinker_cookbook.rl import train

logger = logging.getLogger(__name__)


@chz.chz
class CLIConfig:
    # ── Model ──
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    lora_rank: int = 32
    renderer_name: str | None = None

    # ── Training ──
    learning_rate: float = 4e-5
    batch_size: int = 4               # write-side prompts per GRPO step
    seed: int = 2
    max_tokens: int = 512             # write episodes are short — small per-turn budget
    eval_every: int = 0
    max_steps: int | None = None

    # ── Exploration / regularization ──
    # The smoke runs showed entropy collapsing fast (0.18 → 0.10 → 0.07 → 0.04
    # over 4 steps). Two cookbook knobs to fight that:
    #
    # temperature: sampling temperature for rollouts. Default 1.0 in the
    #   cookbook; raising to 1.1–1.3 widens the on-policy distribution and
    #   keeps lookups/noops in the action mix at smoke scale. Too high (>1.5)
    #   degrades signal quality.
    # kl_penalty_coef: weight on KL(π_θ || π_ref). Default 0.0 means no
    #   regularizer — the LoRA can drift arbitrarily far from base. 0.02–0.1
    #   is the working range for a 4B+rank-32 LoRA. The cookbook warns
    #   temperature and KL interact, so prefer pulling on KL first.
    temperature: float = 1.0
    kl_penalty_coef: float = 0.0
    kl_reference_base_model: str | None = None    # if None, uses model_name

    # ── Dataset (LoCoMo write episodes) ──
    n_convs: int = 8
    train_frac: float = 0.8
    group_size: int = 8               # G — rollouts per write episode for GRPO
    max_turns: int = 4                # max ops per write episode
    max_battery_per_turn: int = 6     # held-out QA battery cap (cost control)
    n_prior_turns_in_context: int = 2

    # ── Frozen R (the judge that evaluates W's memory state) ──
    # v1: empty string → use HeuristicPolicy. v2: "tinker://..." path.
    r_checkpoint: str = ""

    # ── Logging ──
    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None
    num_groups_to_log: int = 8         # rollouts dumped per step for inspect
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "delete"


async def cli_main(cfg: CLIConfig) -> None:
    if cfg.r_checkpoint:
        # Phase B v2 hook — pass a Tinker R checkpoint to write_env via env var
        # (avoids polluting WriteRLDatasetBuilder's chz schema).
        import os
        os.environ["MEMPOL_R_CHECKPOINT"] = cfg.r_checkpoint
        logger.info("Phase B v2: using R checkpoint %s", cfg.r_checkpoint)
    else:
        logger.info("Phase B v1: using HeuristicPolicy as frozen R")

    renderer_name = (
        cfg.renderer_name
        or model_info.get_recommended_renderer_name(cfg.model_name)
    )

    builder = WriteRLDatasetBuilder(
        model_name_for_tokenizer=cfg.model_name,
        renderer_name=renderer_name,
        n_convs=cfg.n_convs,
        train_frac=cfg.train_frac,
        batch_size=cfg.batch_size,
        group_size=cfg.group_size,
        max_turns=cfg.max_turns,
        max_battery_per_turn=cfg.max_battery_per_turn,
        n_prior_turns_in_context=cfg.n_prior_turns_in_context,
        seed=cfg.seed,
    )

    # Run name + log path
    model_short = cfg.model_name.lower().replace("/", "-")
    ts = datetime.now().strftime("%Y-%m-%d-%H-%M")
    run_name = (
        f"mempol_write_{model_short}_bs{cfg.batch_size}_gs{cfg.group_size}_"
        f"lr{cfg.learning_rate}_rank{cfg.lora_rank}_{ts}"
    )
    log_path = cfg.log_path or f"/tmp/tinker-examples/mempol_write/{run_name}"

    if not Path("/tmp").exists():
        raise ValueError("/tmp does not exist")
    cli_utils.check_log_dir(log_path, behavior_if_exists=cfg.behavior_if_log_dir_exists)

    # Reference model for KL — defaults to the same base. Importing lazily
    # because KLReferenceConfig isn't always exported in older cookbook
    # versions; we fall back to no KL if it's unavailable.
    kl_ref = None
    if cfg.kl_penalty_coef > 0:
        try:
            from tinker_cookbook.rl.train import KLReferenceConfig  # type: ignore
            kl_ref = KLReferenceConfig(
                base_model=cfg.kl_reference_base_model or cfg.model_name,
            )
        except Exception as e:
            logger.warning(
                "kl_penalty_coef=%s requested but KLReferenceConfig unavailable "
                "(%s); proceeding with no KL.", cfg.kl_penalty_coef, e,
            )

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
        temperature=cfg.temperature,
        kl_penalty_coef=cfg.kl_penalty_coef,
        kl_reference_config=kl_ref,
        num_groups_to_log=cfg.num_groups_to_log,
    )

    await train.main(config)


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    asyncio.run(cli_main(cli_config))
