"""Co-training outer loop — alternating R-phase and W-phase GRPO.

This is the actual contribution of the paper. Production version of what
was previously a sketch.

Algorithm:

    Initialize:
      W₀ = LoRA on Qwen3 (initialized from random or SFT'd)
      R₀ = LoRA on Qwen3 (initialized from random or SFT'd)

    For outer_iter t in 1..T:
      # Phase A_t: train R, freeze W
      R_t = train_read_policy(
          base R = R_{t-1},
          memory built by frozen W_{t-1}'s writes,
          steps = S_R,
      )

      # Phase B_t: train W, freeze R
      W_t = train_write_policy(
          base W = W_{t-1},
          reward = R_t's accuracy on held-out QA battery per write trajectory,
          steps = S_W,
      )

      # Eval
      log eval_metrics(R_t, W_t)
      if not_improving: break

Each phase is a full Tinker training run launched as a subprocess. The
checkpoints handed off between phases are Tinker artifact paths
(`tinker://<run>:train:0/sampler_weights/<step>`).

Phase B v1 simplification: W's reward uses HeuristicPolicy as the frozen R
during the FIRST outer iteration (we don't have R₀ yet). Subsequent
iterations use the actual R_{t-1} Tinker checkpoint.

Usage:

    # Co-training run (5 outer iterations)
    python -m tinker_cookbook.recipes.memory_rl.cotrain \\
        n_outer=5 \\
        r_steps_per_iter=200 \\
        w_steps_per_iter=100 \\
        n_convs=8 train_frac=0.8 \\
        log_path=/tmp/mempol/cotrain_v1

NOTE: each outer iteration is ~$300-500 of Tinker compute. Full T=5 run is
~$1500-2500. Don't fire blind — start with T=1 to validate.
"""
from __future__ import annotations
import asyncio
import logging
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import chz

from tinker_cookbook import cli_utils

logger = logging.getLogger(__name__)


@chz.chz
class CoTrainConfig:
    # ── Outer loop ──
    n_outer: int = 5
    r_steps_per_iter: int = 200       # GRPO steps per R phase
    w_steps_per_iter: int = 100       # GRPO steps per W phase

    # ── Shared Tinker training params ──
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    lora_rank: int = 32
    learning_rate: float = 4e-5

    # ── Read-side (Phase A) ──
    r_batch_size: int = 4
    r_group_size: int = 8
    r_max_turns: int = 6

    # ── Write-side (Phase B) ──
    w_batch_size: int = 4
    w_group_size: int = 8
    w_max_turns: int = 4
    w_max_battery_per_turn: int = 6

    # ── Dataset ──
    n_convs: int = 8
    train_frac: float = 0.8
    seed: int = 2

    # ── Logging ──
    log_path: str | None = None
    wandb_project: str | None = None
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "delete"


async def _run_subprocess(cmd: list[str], log_file: Path) -> int:
    """Run a Tinker training command as a subprocess, tee output to a log file.
    Returns the exit code. The subprocess inherits TINKER_API_KEY etc."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Running: %s", " ".join(cmd))
    logger.info("Tee logs to: %s", log_file)
    with log_file.open("w") as f:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        async for line in proc.stdout:                    # type: ignore[union-attr]
            decoded = line.decode("utf-8", errors="replace")
            sys.stdout.write(decoded)
            sys.stdout.flush()
            f.write(decoded)
            f.flush()
        await proc.wait()
        return proc.returncode or 0


def _extract_checkpoint_path(log_path: Path) -> str | None:
    """Parse the cookbook's checkpoint-save log line to recover the
    `tinker://...` path of the final checkpoint."""
    pat = re.compile(r"sampler_path['\":]+\s*['\"](tinker://[^'\"]+)['\"]")
    for line in log_path.read_text().splitlines()[::-1]:  # search backwards
        m = pat.search(line)
        if m:
            return m.group(1)
    return None


async def _train_R_phase(cfg: CoTrainConfig, t: int, w_ckpt: str | None,
                         out_dir: Path) -> str | None:
    """Launch Phase A (read) training as a subprocess. Returns the path to
    the saved R checkpoint, or None if extraction failed."""
    log_path = out_dir / f"iter{t}_R" / "log.txt"
    cmd = [
        sys.executable, "-m", "tinker_cookbook.recipes.memory_rl.train",
        f"model_name={cfg.model_name}",
        f"lora_rank={cfg.lora_rank}",
        f"learning_rate={cfg.learning_rate}",
        f"n_convs={cfg.n_convs}",
        f"train_frac={cfg.train_frac}",
        f"batch_size={cfg.r_batch_size}",
        f"group_size={cfg.r_group_size}",
        f"max_turns={cfg.r_max_turns}",
        f"max_steps={cfg.r_steps_per_iter}",
        f"seed={cfg.seed + t}",
        f"log_path={log_path.parent}",
        "behavior_if_log_dir_exists=delete",
    ]
    if cfg.wandb_project:
        cmd += [
            f"wandb_project={cfg.wandb_project}",
            f"wandb_name=cotrain_iter{t}_R",
        ]
    rc = await _run_subprocess(cmd, log_path)
    if rc != 0:
        raise RuntimeError(f"Phase A iter {t} failed with exit code {rc}")
    return _extract_checkpoint_path(log_path)


async def _train_W_phase(cfg: CoTrainConfig, t: int, r_ckpt: str | None,
                         out_dir: Path) -> str | None:
    """Launch Phase B (write) training as a subprocess. Returns the path to
    the saved W checkpoint."""
    log_path = out_dir / f"iter{t}_W" / "log.txt"
    cmd = [
        sys.executable, "-m", "tinker_cookbook.recipes.memory_rl.train_write",
        f"model_name={cfg.model_name}",
        f"lora_rank={cfg.lora_rank}",
        f"learning_rate={cfg.learning_rate}",
        f"n_convs={cfg.n_convs}",
        f"train_frac={cfg.train_frac}",
        f"batch_size={cfg.w_batch_size}",
        f"group_size={cfg.w_group_size}",
        f"max_turns={cfg.w_max_turns}",
        f"max_battery_per_turn={cfg.w_max_battery_per_turn}",
        f"max_steps={cfg.w_steps_per_iter}",
        f"seed={cfg.seed + 1000 + t}",
        f"log_path={log_path.parent}",
        "behavior_if_log_dir_exists=delete",
    ]
    if r_ckpt:
        cmd += [f"r_checkpoint={r_ckpt}"]
    if cfg.wandb_project:
        cmd += [
            f"wandb_project={cfg.wandb_project}",
            f"wandb_name=cotrain_iter{t}_W",
        ]
    rc = await _run_subprocess(cmd, log_path)
    if rc != 0:
        raise RuntimeError(f"Phase B iter {t} failed with exit code {rc}")
    return _extract_checkpoint_path(log_path)


async def cli_main(cfg: CoTrainConfig) -> None:
    ts = datetime.now().strftime("%Y-%m-%d-%H-%M")
    run_root = cfg.log_path or f"/tmp/tinker-examples/mempol_cotrain/{ts}"
    out_dir = Path(run_root)
    cli_utils.check_log_dir(str(out_dir),
                             behavior_if_exists=cfg.behavior_if_log_dir_exists)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Co-training run rooted at %s", out_dir)

    r_ckpt: str | None = None     # Phase B v1: None → HeuristicPolicy as R
    w_ckpt: str | None = None
    history: list[dict] = []

    for t in range(1, cfg.n_outer + 1):
        logger.info("=" * 60)
        logger.info("Outer iter %d / %d", t, cfg.n_outer)
        logger.info("=" * 60)

        # ── Phase A_t: train R, freeze W ──
        logger.info("Phase A: training read policy (steps=%d)", cfg.r_steps_per_iter)
        r_ckpt_new = await _train_R_phase(cfg, t, w_ckpt, out_dir)
        logger.info("  → R checkpoint: %s", r_ckpt_new)
        if r_ckpt_new:
            r_ckpt = r_ckpt_new

        # ── Phase B_t: train W, freeze R ──
        logger.info("Phase B: training write policy (steps=%d)",
                    cfg.w_steps_per_iter)
        w_ckpt_new = await _train_W_phase(cfg, t, r_ckpt, out_dir)
        logger.info("  → W checkpoint: %s", w_ckpt_new)
        if w_ckpt_new:
            w_ckpt = w_ckpt_new

        history.append({
            "iter": t,
            "r_ckpt": r_ckpt,
            "w_ckpt": w_ckpt,
        })

        # TODO: eval R_t + W_t on held-out LoCoMo / LongMemEval / TemporalBench
        # and check stopping criterion. For now, we just run the full T outer
        # iterations.

    logger.info("=" * 60)
    logger.info("Co-training complete. Final checkpoints:")
    logger.info("  R: %s", r_ckpt)
    logger.info("  W: %s", w_ckpt)
    (out_dir / "history.json").write_text(
        __import__("json").dumps(history, indent=2)
    )


if __name__ == "__main__":
    cfg = chz.entrypoint(CoTrainConfig)
    asyncio.run(cli_main(cfg))
