"""WriteReward — deferred + dense reward for the write policy.

The write policy W emits a sequence of memory-store mutations during a single
turn. The reward is composed of two terms (plus a cost regulariser):

  1. **Coverage** (dense, deterministic): for each held-out QA whose evidence
     mentions a dia_id from this turn or its session, what fraction of the
     evidence dia_ids did W actually preserve in the resulting KG?
     Computed by `mempol.eval.evidence_coverage.battery_coverage` from
     `Entity.created_from` and `StateTransition.trigger_conversation_id`.
     This signal is free (no LLM calls) and dense (per-question fractional
     score), so it gives clean GRPO advantages even when the judge is noisy.

  2. **QA judge** (deferred, expensive): a frozen R policy answers the same
     battery against the post-W memory state and an LLM judge scores the
     answers. We keep this as a 30% blend term so the W policy is not
     rewarded for storing source dia_ids that are unanswerable in practice.

  reward = w_cov * mean_coverage + w_qa * mean_qa - cost(τ)
  cost(τ) = α n_mutations + β n_lookups + γ n_entities

For Phase B v1 the frozen R is the deterministic heuristic policy
`mempol.policies.v1_heuristic.HeuristicPolicy`. After Phase A trains a real
read-policy LoRA, swap R via `r_runner` (set the env var
`MEMPOL_R_CHECKPOINT` to a Tinker sampler path; resolved by
`make_tinker_r_runner`).

The signal is a single scalar at the end of the trajectory, broadcast over
all generated tokens during GRPO advantage computation. Per-op credit
(counterfactual ablation) lives in `mempol.rewards.credit.credit_counterfactual`.
"""
from __future__ import annotations
import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable

from mempol.backends.pie_kg import PIEBackend
from mempol.eval.evidence_coverage import battery_coverage
from mempol.eval.judge import judge as _judge_sync
from mempol.policies.v1_heuristic import HeuristicPolicy

logger = logging.getLogger(__name__)


# Default cost coefficients. Tunable, reported as ablation in the paper.
DEFAULT_COST_PER_OP = 0.005
DEFAULT_COST_PER_LOOKUP = 0.002    # lookup is cheaper than create / update / merge
DEFAULT_COST_PER_ENTITY = 0.001    # storage cost — penalises bloat

# Reward mix. Coverage is the primary signal because it's dense and free;
# the QA judge term keeps the policy honest against the actual downstream
# task. Both ablation values are reported in the paper.
DEFAULT_W_COVERAGE = 0.6
DEFAULT_W_QA = 0.4


@dataclass
class WriteReward:
    """Deferred + dense reward for one write trajectory.

    Lifetime: one instance per env (per group member). Bound to a specific
    PIEBackend (the one being mutated by the W tools in that env) and a
    pre-computed QA battery — each entry is `(question, gold_answer,
    evidence_dia_ids)`. The evidence list is what enables the dense
    `coverage` term; without it the reward collapses to the legacy
    judge-only signal.

    Attributes:
        backend: the PIEBackend mutated in-place by W's tool calls during the
            episode. We do NOT make a fresh copy — the backend's final state
            is exactly what we score.
        query_battery: list of (question, gold, evidence_dia_ids) tuples
            drawn from LoCoMo's QA labels for THIS turn.
        r_runner: callable(question: str, backend: PIEBackend) -> answer_str.
            For Phase B v1, defaults to a HeuristicPolicy run. Set to None
            to skip the QA term entirely (coverage-only training).
        w_coverage / w_qa: blend weights for the two reward terms.
        cost_per_op / cost_per_lookup / cost_per_entity: cost coefficients.
    """
    backend: PIEBackend
    query_battery: list[tuple[str, str, list[str]]]
    r_runner: Callable[[str, PIEBackend], str] | None = None
    w_coverage: float = DEFAULT_W_COVERAGE
    w_qa: float = DEFAULT_W_QA
    cost_per_op: float = DEFAULT_COST_PER_OP
    cost_per_lookup: float = DEFAULT_COST_PER_LOOKUP
    cost_per_entity: float = DEFAULT_COST_PER_ENTITY
    # Internal: track per-evaluation metrics for logging
    _last_metrics: dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        # Coverage-only mode: caller can disable judge by passing r_runner=None
        # AND w_qa=0. If r_runner is None but w_qa>0, fall back to the
        # heuristic so we never silently train on stale signal.
        if self.r_runner is None and self.w_qa > 0:
            self.r_runner = _default_r_runner

    async def __call__(self, history: list[dict]) -> tuple[float, dict[str, float]]:
        """Tinker-compatible reward signature.

        Args:
            history: the W policy's trajectory — list of message dicts with
                roles {system, user, assistant, tool}. We don't actually need
                the history itself (the backend has been mutated by tool
                dispatches), but we use it to count ops for cost.

        Returns:
            (reward, metrics_dict) — reward is a single scalar; metrics_dict
            is logged per-step by the trainer.
        """
        # 1. Empty battery → no signal. Return small negative so the W
        #    policy doesn't learn that empty batteries are good.
        if not self.query_battery:
            self._last_metrics = {"battery_size": 0.0}
            return -0.01, dict(self._last_metrics)

        # 2. Coverage — deterministic, free.
        cov_result = battery_coverage(self.backend, self.query_battery)
        mean_cov = cov_result.mean_coverage

        # 3. QA judge — only run if we want a non-zero qa weight.
        mean_qa = 0.0
        if self.w_qa > 0 and self.r_runner is not None:
            loop = asyncio.get_running_loop()

            async def _score_one(question: str, gold: str) -> float:
                try:
                    answer = await loop.run_in_executor(
                        None, self.r_runner, question, self.backend
                    )
                    judge_score, _ = await loop.run_in_executor(
                        None, _judge_sync, question, gold, answer
                    )
                    return float(judge_score)
                except Exception as e:
                    logger.warning(
                        "WriteReward: R-runner / judge failed for q=%s: %s",
                        question[:60], e,
                    )
                    return 0.0

            scores = await asyncio.gather(
                *[_score_one(q, g) for q, g, _ev in self.query_battery]
            )
            mean_qa = sum(scores) / max(len(scores), 1)

        # 4. Cost from the trajectory. Lookups are cheaper than mutations
        # (intent: encourage exploration before commits). We subtract lookups
        # from n_ops_total before pricing mutations to avoid double-charging.
        n_ops_total = self._count_ops(history)
        n_lookups = self._count_lookups(history)
        n_mutations = max(0, n_ops_total - n_lookups)
        n_entities = len(self.backend.wm.entities)
        cost = (
            self.cost_per_op * n_mutations
            + self.cost_per_lookup * n_lookups
            + self.cost_per_entity * n_entities
        )

        reward = self.w_coverage * mean_cov + self.w_qa * mean_qa - cost
        self._last_metrics = {
            "coverage_mean": mean_cov,
            "qa_mean": mean_qa,
            "cost_total": cost,
            "n_ops": float(n_ops_total),
            "n_lookups": float(n_lookups),
            "n_entities": float(n_entities),
            "battery_size": float(len(self.query_battery)),
            "stored_dia_ids": float(cov_result.n_stored_dia_ids),
            "evidence_hit_frac": (
                cov_result.n_evidence_dia_ids_hit
                / max(cov_result.n_evidence_dia_ids_total, 1)
            ),
        }
        return reward, dict(self._last_metrics)

    @staticmethod
    def _count_ops(history: list[dict]) -> int:
        """Count assistant messages that contain a tool_call (each ≈ one op).
        Lookups are counted separately for cost shaping."""
        n = 0
        for msg in history:
            try:
                role = msg.get("role")
            except AttributeError:
                role = getattr(msg, "role", None)
            if role != "assistant":
                continue
            content = (msg.get("content") if isinstance(msg, dict)
                       else getattr(msg, "content", "")) or ""
            if isinstance(content, str) and "<tool_call>" in content:
                n += content.count("<tool_call>")
        return max(n, 0)

    @staticmethod
    def _count_lookups(history: list[dict]) -> int:
        """Count tool calls that are lookup ops (cheaper than mutations)."""
        n = 0
        for msg in history:
            content = (msg.get("content") if isinstance(msg, dict)
                       else getattr(msg, "content", "")) or ""
            if isinstance(content, str):
                n += content.count('"lookup_entity"') + content.count('"lookup_relation"')
        return n


# ── Frozen R runners ────────────────────────────────────────────────────────
# Phase B v1 uses the deterministic heuristic. Phase B v2 (after Phase A
# trains) swaps in a Tinker-backed R LoRA via `make_tinker_r_runner`.

_HEURISTIC_R = HeuristicPolicy(first_k=8, final_k=4, do_reformulate=True, do_expand=True)


def _default_r_runner(question: str, backend: PIEBackend) -> str:
    """Default frozen R: deterministic heuristic from mempol/policies/v1_heuristic.py.
    Returns the answer string only; the trace is discarded."""
    trace = _HEURISTIC_R.run(question, backend)
    return trace.answer or "not in context"


def make_tinker_r_runner(sampling_client: Any, renderer: Any) -> Callable:
    """Construct an R runner backed by a Tinker LoRA checkpoint.

    Phase B v2: pass a `tinker.SamplingClient` already loaded with a
    Phase-A-trained LoRA. Returns a callable matching `_default_r_runner`'s
    signature so WriteReward can swap drop-in.

    Status: stubbed. The Tinker SamplingClient API for inline rollouts
    inside a reward callback is not yet stable enough to depend on. Until
    Phase A produces a checkpoint AND we benchmark wall-clock against the
    heuristic, the recommended path is:
      1. Train Phase A on raw chunks (HeuristicPolicy is fine as the
         frozen-R during W training; coverage carries the dense signal
         anyway).
      2. After Phase A, evaluate the trained R against the heuristic R on
         a fixed held-out battery; if R wins by >5% qa_mean and runs in
         <2x wall, plumb it in here.
    """
    raise NotImplementedError(
        "Tinker R-runner not wired yet. Coverage reward (w_coverage=0.6) is "
        "the dense signal — heuristic R only contributes the 0.4 judge term. "
        "See WriteReward docstring for the upgrade path."
    )


def resolve_r_runner_from_env() -> Callable | None:
    """If MEMPOL_R_CHECKPOINT is set, attempt to build a Tinker-backed R
    runner from it. Returns None on any failure (caller falls back to the
    heuristic). Used by `WriteEnvGroupBuilder` so the train_write CLI's
    `r_checkpoint=...` arg actually does something."""
    ckpt = os.environ.get("MEMPOL_R_CHECKPOINT", "").strip()
    if not ckpt:
        return None
    try:
        # Lazy import — Tinker client lives outside this module's import
        # graph so users without it can still run Phase B v1.
        import tinker                                      # type: ignore
        client = tinker.SamplingClient(ckpt)               # type: ignore[attr-defined]
        renderer = None                                    # built lazily inside runner
        return make_tinker_r_runner(client, renderer)
    except NotImplementedError as e:
        logger.warning(
            "MEMPOL_R_CHECKPOINT=%s set but Tinker R runner not implemented "
            "(%s) — falling back to HeuristicPolicy.",
            ckpt, e,
        )
        return None
    except Exception as e:
        logger.warning(
            "MEMPOL_R_CHECKPOINT=%s could not be resolved (%s) — falling "
            "back to HeuristicPolicy.",
            ckpt, e,
        )
        return None
