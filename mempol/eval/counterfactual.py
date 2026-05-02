"""Per-op counterfactual marginal utility — the dense write reward.

The mempol write policy emits a sequence of tool-call ops per turn. The
deferred-terminal-reward signal we previously used (`answer_gain` over a
random-K baseline) gives one scalar per *trajectory*, then GRPO diffuses
that scalar uniformly over every emitted token. Per-op resolution is lost.

This module computes per-op marginal utility via leave-one-out replay.
For each mutating op a_i in the trajectory:

    M_full      = replay(ops)                  # the actual post-W KG
    M_minus_i   = replay(ops without a_i)      # counterfactual
    For each held-out question q in the battery:
        score_with(a_i, q)    = judge(R(q, M_full),    gold_q)
        score_without(a_i, q) = judge(R(q, M_minus_i), gold_q)
        delta(a_i, q)         = score_with - score_without
    R_op(a_i) = mean_q delta(a_i, q) - λ · cost(a_i)

The trajectory-level reward GRPO sees is the SUM over ops:

    R_traj = Σ_i R_op(a_i)

Why this is the right shape
===========================

  • Per-op signal lets GRPO attribute reward to specific decisions
    rather than diffusing over all 4-12 token positions of a tool call.
    Sample efficiency improves because high-variance group advantages
    are now per-op, not per-trajectory.
  • Per-question evaluation preserves task-level resolution. Pooling
    questions before differencing washes out per-op signal entirely.
  • Lookups and noops are skipped because their effect on M_full is
    zero by definition; their leave-one-out delta is trivially 0.
    This cuts the cost ~2× in practice.

Cost
====

For K=4 mutating ops × |Q|=4 battery questions, this adds ~16 R+judge
calls per rollout on top of the existing answer_gain signal. With
parallel asyncio.gather over the leave-one-out variants we observe ~5-10
seconds added to env_step:mean.

References
==========
  • Memory-R1 (Aug 2025) trains a Memory Manager via RL with downstream
    QA reward; we strengthen this with per-op leave-one-out instead of
    trajectory-level rollups.
  • DeltaMem proposes state-distance reward; we sidestep needing target
    states by using future-task downstream judge instead.
  • The technique itself is the analogue of "credit assignment via
    counterfactual reasoning" from policy-gradient literature
    (Foerster et al. 2018, COMA).
"""
from __future__ import annotations
import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from mempol.backends.base import Backend
from mempol.backends.pie_kg import PIEBackend
from mempol.eval.judge import judge as _judge_sync
from mempol.recipes.memory_rl.write_tools import WriteTool

logger = logging.getLogger(__name__)


@dataclass
class PerOpReward:
    """Reward + diagnostics for one trajectory, broken out per op."""
    trajectory_reward: float                 # sum of per-op rewards
    per_op_deltas: list[tuple[str, float]] = field(default_factory=list)
    # ↑ list of (op_name, delta) for the mutating ops that were ablated.
    # Lookups/noops are not included (their delta is 0 by construction).
    n_ablated: int = 0
    n_battery: int = 0
    full_state_score: float = 0.0


def _classify_ops(ops_log: list[tuple[str, dict]]) -> list[int]:
    """Return the indices of mutating ops in ops_log (which we ablate)."""
    return [i for i, (name, _) in enumerate(ops_log)
            if name in WriteTool.MUTATING_OPS]


def _replay(ops_log: list[tuple[str, dict]], current_dia_id: str = "",
              current_timestamp: float = 0.0) -> PIEBackend:
    """Build a fresh PIEBackend and replay the given op sequence on it.

    We construct a transient WriteTool to dispatch each op via the same
    private impl methods the original execution used. This guarantees
    that "with op_i" and "without op_i" states are bit-identical except
    for op_i's effect.
    """
    backend = PIEBackend()
    tool = WriteTool(backend=backend,
                      current_dia_id=current_dia_id,
                      current_timestamp=current_timestamp)
    for name, args in ops_log:
        try:
            if name == "lookup_entity":
                tool._lookup_entity_impl(**args)
            elif name == "lookup_relation":
                tool._lookup_relation_impl(**args)
            elif name == "create_entity":
                tool._create_entity_impl(**args)
            elif name == "update_state":
                tool._update_state_impl(**args)
            elif name == "merge_entities":
                tool._merge_entities_impl(**args)
            elif name == "add_relation":
                tool._add_relation_impl(**args)
            elif name == "mark_contradiction":
                tool._mark_contradiction_impl(**args)
            elif name == "forget":
                tool._forget_impl(**args)
            elif name == "noop":
                tool._noop_impl(**args)
            else:
                logger.warning("counterfactual replay: unknown op %r", name)
        except Exception as e:
            # Replay failures don't block — they just mean this op
            # wouldn't have applied cleanly in the counterfactual world
            # either, which is information we want to preserve.
            logger.debug("replay of %s failed (likely missing prior op): %s",
                         name, e)
    return backend


async def _score_battery(
    reader: Any,
    backend: PIEBackend,
    battery: list[tuple[str, str, list[str]]],
    judge_fn: Callable = _judge_sync,
) -> list[float]:
    """Run the reader on each question against `backend`, return per-Q
    judge scores. Parallelised via asyncio.gather over the battery."""
    loop = asyncio.get_running_loop()

    async def _one(q: str, gold: str) -> float:
        try:
            trace = await loop.run_in_executor(None, reader.run, q, backend)
            ans = trace.answer or "not in context"
            score, _ = await loop.run_in_executor(
                None, judge_fn, q, gold, ans
            )
            return float(score)
        except Exception as e:
            logger.warning("counterfactual: score_battery failed q=%s: %s",
                           q[:60], e)
            return 0.0

    return await asyncio.gather(*[_one(q, g) for q, g, _ev in battery])


async def per_op_counterfactual(
    ops_log: list[tuple[str, dict]],
    battery: list[tuple[str, str, list[str]]],
    reader: Any,
    current_dia_id: str = "",
    current_timestamp: float = 0.0,
    cost_per_mut: float = 0.005,
) -> PerOpReward:
    """Compute per-op marginal utility for a write trajectory.

    Args:
        ops_log: the (op_name, args) sequence the policy emitted, as
            recorded by WriteTool.ops_log.
        battery: held-out (question, gold, evidence) triples scored against
            both the full and leave-one-out memory states.
        reader: frozen R policy (HeuristicPolicy or trained LoRA).
        current_dia_id, current_timestamp: provenance for the replay.
        cost_per_mut: per-mutation cost subtracted from each op's reward.

    Returns:
        PerOpReward — trajectory_reward is what gets fed to GRPO; the
        per-op deltas are logged for analysis and (later) PRM training.
    """
    if not ops_log or not battery:
        return PerOpReward(trajectory_reward=0.0)

    mut_indices = _classify_ops(ops_log)
    if not mut_indices:
        # All ops were lookups/noops — no counterfactual signal possible.
        return PerOpReward(trajectory_reward=0.0,
                            n_ablated=0, n_battery=len(battery))

    # 1. Score the full trajectory once.
    full_backend = _replay(ops_log, current_dia_id, current_timestamp)
    full_scores = await _score_battery(reader, full_backend, battery)
    full_mean = sum(full_scores) / max(len(full_scores), 1)

    # 2. Score each leave-one-out variant in parallel.
    async def _delta_for(idx: int) -> tuple[str, float]:
        leave_out_ops = [op for j, op in enumerate(ops_log) if j != idx]
        b = _replay(leave_out_ops, current_dia_id, current_timestamp)
        scores_minus = await _score_battery(reader, b, battery)
        # Per-question delta, then mean.
        deltas = [s_full - s_minus for s_full, s_minus
                  in zip(full_scores, scores_minus)]
        op_name = ops_log[idx][0]
        return op_name, sum(deltas) / max(len(deltas), 1)

    per_op = await asyncio.gather(*[_delta_for(i) for i in mut_indices])

    # 3. Sum to trajectory-level reward, minus per-mutation cost.
    traj_reward = sum(d for _, d in per_op) - cost_per_mut * len(mut_indices)

    return PerOpReward(
        trajectory_reward=traj_reward,
        per_op_deltas=list(per_op),
        n_ablated=len(mut_indices),
        n_battery=len(battery),
        full_state_score=full_mean,
    )


# ─── Smoke ───────────────────────────────────────────────────────────────────
def _smoke():
    """Synthetic end-to-end check using a hand-built ops_log."""
    import asyncio
    from unittest.mock import patch
    import numpy as np
    from mempol.policies.v1_heuristic import HeuristicPolicy

    def fake_embed(texts):
        return np.random.RandomState(0).randn(len(texts), 8).astype("float32")

    def fake_judge(q, gold, pred):
        return (1.0 if gold.lower() in (pred or "").lower() else 0.0,
                "ok" if gold.lower() in (pred or "").lower() else "wrong")

    def fake_chat(*a, **kw):
        return "Boston"

    ops_log = [
        ("lookup_entity",  {"query": "caroline"}),
        ("create_entity",  {"name": "Caroline lives in Boston",
                              "type": "person", "state": {"city": "Boston"}}),
        ("noop",           {"reason": "filler"}),
        ("create_entity",  {"name": "Random other entity",
                              "type": "concept", "state": {}}),
    ]
    battery = [("Where does Caroline live?", "Boston", ["D1:5"])]
    reader = HeuristicPolicy(first_k=4, final_k=2,
                              do_reformulate=False, do_expand=False)

    with patch("mempol.backends.pie_kg.llm.embed", side_effect=fake_embed), \
         patch("mempol.policies.v0_naive.llm.chat", side_effect=fake_chat), \
         patch("mempol.policies.v1_heuristic.llm.chat", side_effect=fake_chat), \
         patch("mempol.eval.counterfactual._judge_sync", side_effect=fake_judge):
        result = asyncio.run(per_op_counterfactual(
            ops_log=ops_log, battery=battery, reader=reader,
            current_dia_id="D1:5", current_timestamp=1.0,
        ))
    print(f"trajectory_reward = {result.trajectory_reward:+.3f}")
    print(f"full_state_score  = {result.full_state_score:.3f}")
    print("per-op deltas (mutating ops only):")
    for op_name, delta in result.per_op_deltas:
        print(f"  {op_name:20s}  Δ = {delta:+.3f}")


if __name__ == "__main__":
    _smoke()
