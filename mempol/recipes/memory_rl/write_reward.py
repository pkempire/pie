"""WriteReward — deferred reward for the write policy.

The write policy W emits a sequence of memory-store mutations during a single
turn (or short window of turns). The reward is **deferred**: after W's
trajectory ends, the resulting memory store is queried by a frozen read
policy R against a held-out battery of future questions. The reward is the
mean of R's QA accuracy minus a storage / op-count cost.

Mechanics:
  reward = mean( judge(R(q_i, M_τ), gold_i) for i ) - λ_w * cost(τ)

  cost(τ) = α * n_write_ops + β * n_lookup_ops + γ * n_entities_in_M_τ

The signal is a single scalar at the end of the trajectory, broadcast over
all generated tokens during GRPO advantage computation. Per-op credit
assignment (counterfactual ablation) is supported separately by
`mempol.rewards.credit.credit_counterfactual`.

For Phase B v1 the frozen R is the deterministic heuristic policy
`mempol.policies.v1_heuristic.HeuristicPolicy`. After Phase A trains a real
read-policy LoRA, swap R to a Tinker sampling client backed by that LoRA.
"""
from __future__ import annotations
import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from mempol.backends.pie_kg import PIEBackend
from mempol.eval.judge import judge as _judge_sync
from mempol.policies.v1_heuristic import HeuristicPolicy

logger = logging.getLogger(__name__)


# Default cost coefficients. Tunable, reported as ablation in the paper.
DEFAULT_COST_PER_OP = 0.005
DEFAULT_COST_PER_LOOKUP = 0.002    # lookup is cheaper than create / update / merge
DEFAULT_COST_PER_ENTITY = 0.001    # storage cost — penalises bloat


@dataclass
class WriteReward:
    """Deferred reward for one write trajectory.

    Lifetime: one instance per env (per group member). Bound to a specific
    PIEBackend (the one being mutated by the W tools in that env) and a
    pre-computed query battery for the conversation context the W trajectory
    is operating on.

    Attributes:
        backend: the PIEBackend mutated in-place by W's tool calls during the
            episode. We do NOT make a fresh copy — the backend's final state
            is exactly what we score.
        query_battery: list of (question, gold_answer) pairs that the
            held-out R policy will be evaluated on. For LoCoMo these come
            from the conv's QA list, filtered to questions whose evidence
            depends on the turn(s) this episode covered.
        r_runner: callable(question: str, backend: PIEBackend) -> answer_str.
            For Phase B v1, defaults to a HeuristicPolicy run.
        cost_per_op / cost_per_lookup / cost_per_entity: cost coefficients.
    """
    backend: PIEBackend
    query_battery: list[tuple[str, str]]                          # (question, gold)
    r_runner: Callable[[str, PIEBackend], str] | None = None
    cost_per_op: float = DEFAULT_COST_PER_OP
    cost_per_lookup: float = DEFAULT_COST_PER_LOOKUP
    cost_per_entity: float = DEFAULT_COST_PER_ENTITY
    # Internal: track per-evaluation metrics for logging
    _last_metrics: dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        if self.r_runner is None:
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
        # 1. Empty battery → no signal. Return small negative to prevent
        #    the W policy from learning that empty batteries are good.
        if not self.query_battery:
            self._last_metrics = {"battery_size": 0.0}
            return -0.01, dict(self._last_metrics)

        # 2. Score the held-out battery using the (frozen) R runner.
        scores: list[float] = []
        loop = asyncio.get_running_loop()
        for question, gold in self.query_battery:
            try:
                # Run R synchronously inside an executor to avoid blocking
                # the asyncio event loop. The R runner is sync (heuristic
                # uses blocking OpenAI calls); a future Tinker-backed R
                # would be async-native and we'd just `await` it.
                answer = await loop.run_in_executor(
                    None, self.r_runner, question, self.backend
                )
                judge_score, _ = await loop.run_in_executor(
                    None, _judge_sync, question, gold, answer
                )
                scores.append(float(judge_score))
            except Exception as e:
                logger.warning(
                    "WriteReward: R-runner / judge failed for q=%s: %s",
                    question[:60], e,
                )
                scores.append(0.0)

        mean_qa = sum(scores) / len(scores)

        # 3. Cost from the trajectory.
        n_ops = self._count_ops(history)
        n_lookups = self._count_lookups(history)
        n_entities = len(self.backend.wm.entities)
        cost = (
            self.cost_per_op * n_ops
            + self.cost_per_lookup * n_lookups
            + self.cost_per_entity * n_entities
        )

        reward = mean_qa - cost
        self._last_metrics = {
            "qa_mean": mean_qa,
            "cost_total": cost,
            "n_ops": float(n_ops),
            "n_lookups": float(n_lookups),
            "n_entities": float(n_entities),
            "battery_size": float(len(self.query_battery)),
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

    NOTE: not yet implemented — wire after Phase A produces a checkpoint.
    """
    raise NotImplementedError(
        "Tinker R-runner not implemented yet. Use the heuristic default until "
        "Phase A produces a trained read-policy LoRA."
    )
