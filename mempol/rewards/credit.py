"""Credit assignment for write-policy training.

The hardest problem in training a memory write policy is that most write ops
have no measurable downstream effect, and the ones that do don't pay off until
hours/days later. We support three reward sources and let the trainer compose
them:

  1. EPISODE — uniform reward across every write op in a trajectory.
     Cheap. Noisy.

  2. COUNTERFACTUAL — for each write op a_i in trajectory τ, build a memory
     state with vs without a_i applied, run the read policy R on a query
     battery, and credit a_i with the accuracy delta.
     Expensive but gives clean per-op causal credit.

  3. RETRIEVAL-TRACE — track which write-produced units are retrieved by R
     during downstream queries; credit those positively. Units stored but
     never retrieved get negative credit. Cheaper than counterfactual; noisier
     because "retrieved" doesn't mean "useful in the answer."

In practice, mix: episode for the bulk, counterfactual on a small high-value
subset for calibration, retrieval-trace as a regulariser.

Core abstraction:
    @dataclass
    class WriteOpRecord:
        op: str            # "create_entity" | "update_state" | ...
        args: dict
        produced_uids: list[str]   # the uids the op created or modified

    credit_episode(τ, total_reward) -> list[float]
    credit_counterfactual(τ, conv, queries, gold, R, backend_factory) -> list[float]
    credit_retrieval_trace(τ, queries, R) -> list[float]

Each returns a list of per-op credits aligned with τ's op order.
"""
from __future__ import annotations
import copy
from dataclasses import dataclass, field
from typing import Callable, Iterable

from ..backends.base import Backend, Hit


# ----------------------------------------------------------------------
# Data classes
# ----------------------------------------------------------------------
@dataclass
class WriteOpRecord:
    """One write op. Captures the call so it can be re-applied or skipped
    when reconstructing a memory state (counterfactual ablation)."""
    op: str
    args: dict
    produced_uids: list[str] = field(default_factory=list)
    # Filled in by ablation/trace machinery; not used during the rollout itself.
    counterfactual_credit: float | None = None
    retrieval_trace_credit: float | None = None
    episode_credit: float | None = None

    def total_credit(
        self,
        w_episode: float = 0.5,
        w_counterfactual: float = 1.0,
        w_trace: float = 0.3,
    ) -> float:
        """Composed credit. Defaults weight counterfactual most."""
        s = 0.0
        n = 0.0
        if self.episode_credit is not None:
            s += w_episode * self.episode_credit; n += w_episode
        if self.counterfactual_credit is not None:
            s += w_counterfactual * self.counterfactual_credit; n += w_counterfactual
        if self.retrieval_trace_credit is not None:
            s += w_trace * self.retrieval_trace_credit; n += w_trace
        return s / n if n > 0 else 0.0


@dataclass
class WriteTrajectory:
    """A full conversation's worth of write ops."""
    conv_id: str
    ops: list[WriteOpRecord] = field(default_factory=list)
    final_backend: Backend | None = None    # the memory state after applying every op


# ----------------------------------------------------------------------
# 1. Episode-level credit (cheap)
# ----------------------------------------------------------------------
def credit_episode(traj: WriteTrajectory, total_reward: float) -> list[float]:
    """Uniform per-op reward. Simplest possible. Used for the bulk of training.

    `total_reward` is typically `mean_QA_score(R, final_backend) − cost(traj)`.
    """
    if not traj.ops:
        return []
    per_op = total_reward / len(traj.ops)
    for r in traj.ops:
        r.episode_credit = per_op
    return [per_op] * len(traj.ops)


# ----------------------------------------------------------------------
# 2. Counterfactual ablation (expensive, clean)
# ----------------------------------------------------------------------
def credit_counterfactual(
    traj: WriteTrajectory,
    queries: list[dict],                    # [{"question", "gold"}, ...]
    answer_with_memory: Callable[[str, Backend], str],
    judge: Callable[[str, str, str], tuple[float, str]],
    backend_factory: Callable[[], Backend],
    apply_op: Callable[[WriteOpRecord, Backend], None],
    n_ablate: int | None = None,
    seed: int = 0,
) -> list[float]:
    """For each op a_i in the trajectory:

       memory_full = apply(traj, ∅)
       memory_minus_i = apply(traj.ops \\ {a_i}, ∅)
       credit_i = mean(judge(R(q, memory_full)) for q in queries)
                  - mean(judge(R(q, memory_minus_i)) for q in queries)

    `apply_op` is a backend-specific callback (e.g. `PIEBackend.create_entity`,
    `MastraBackend.store`). Caller provides it because op semantics differ
    across backends — that's by design.

    To control cost, set `n_ablate` to test only a random subset of ops; the
    rest get credit None and fall back to episode-level credit.

    Returns: list of per-op credits aligned with traj.ops (entries may be NaN
    for unablated ops).
    """
    import math
    import random

    rng = random.Random(seed)

    # Build memory_full once and score the QA battery on it
    memory_full = backend_factory()
    for op in traj.ops:
        apply_op(op, memory_full)
    full_scores = []
    for q in queries:
        ans = answer_with_memory(q["question"], memory_full)
        s, _ = judge(q["question"], q["gold"], ans)
        full_scores.append(s)
    full_mean = sum(full_scores) / max(1, len(full_scores))

    indices = list(range(len(traj.ops)))
    if n_ablate is not None and n_ablate < len(indices):
        indices = rng.sample(indices, n_ablate)
    indices = set(indices)

    credits: list[float] = [math.nan] * len(traj.ops)
    for i, op in enumerate(traj.ops):
        if i not in indices:
            continue
        memory_minus = backend_factory()
        for j, other in enumerate(traj.ops):
            if j == i:
                continue
            apply_op(other, memory_minus)
        scores_minus = []
        for q in queries:
            ans = answer_with_memory(q["question"], memory_minus)
            s, _ = judge(q["question"], q["gold"], ans)
            scores_minus.append(s)
        minus_mean = sum(scores_minus) / max(1, len(scores_minus))
        delta = full_mean - minus_mean
        op.counterfactual_credit = delta
        credits[i] = delta
    return credits


# ----------------------------------------------------------------------
# 3. Retrieval-trace credit (medium cost, regulariser)
# ----------------------------------------------------------------------
def credit_retrieval_trace(
    traj: WriteTrajectory,
    retrieved_uids_per_query: list[list[str]],   # one list per query in the QA battery
    answer_correct_per_query: list[float],       # judge score per query
    positive_credit_correct: float = 1.0,
    negative_credit_unused: float = -0.05,
) -> list[float]:
    """Items the read policy retrieved during a query that was answered
    correctly get positive credit. Items stored but never retrieved get a
    small negative credit (storage cost penalty).

    Credit is split across the ops that produced the retrieved items.
    """
    # 1. Build uid → contributing op indices
    uid_to_ops: dict[str, list[int]] = {}
    for i, op in enumerate(traj.ops):
        for uid in op.produced_uids:
            uid_to_ops.setdefault(uid, []).append(i)

    credits: list[float] = [0.0] * len(traj.ops)
    n_retrieved = [0] * len(traj.ops)

    # 2. Reward retrievals during correctly-answered queries
    for retrieved, correct in zip(retrieved_uids_per_query, answer_correct_per_query):
        if correct < 0.5:
            continue
        for uid in retrieved:
            for i in uid_to_ops.get(uid, []):
                credits[i] += positive_credit_correct
                n_retrieved[i] += 1

    # 3. Penalize ops whose units were stored but never retrieved
    all_retrieved_uids = set().union(*retrieved_uids_per_query) if retrieved_uids_per_query else set()
    for i, op in enumerate(traj.ops):
        if not op.produced_uids:
            continue
        unused = [u for u in op.produced_uids if u not in all_retrieved_uids]
        credits[i] += negative_credit_unused * len(unused)

    for i, op in enumerate(traj.ops):
        op.retrieval_trace_credit = credits[i]
    return credits


# ----------------------------------------------------------------------
# Composite — what the trainer actually calls
# ----------------------------------------------------------------------
def assign_all_credits(
    traj: WriteTrajectory,
    queries: list[dict],
    *,
    answer_with_memory: Callable[[str, Backend], str],
    judge: Callable[[str, str, str], tuple[float, str]],
    backend_factory: Callable[[], Backend],
    apply_op: Callable[[WriteOpRecord, Backend], None],
    retrieved_uids_per_query: list[list[str]] | None = None,
    correct_per_query: list[float] | None = None,
    use_episode: bool = True,
    use_counterfactual: bool = True,
    use_trace: bool = True,
    n_ablate: int | None = 16,
) -> list[float]:
    """Run all enabled credit modes and return composed per-op credit."""
    # Episode reward (requires we already know total_reward — derive from full
    # battery score)
    if use_episode:
        # Recompute via the full memory if needed; cheaper to do inside
        # counterfactual since we'll build memory_full there too.
        pass

    if use_counterfactual:
        credit_counterfactual(
            traj, queries,
            answer_with_memory=answer_with_memory,
            judge=judge,
            backend_factory=backend_factory,
            apply_op=apply_op,
            n_ablate=n_ablate,
        )

    if use_trace and retrieved_uids_per_query and correct_per_query:
        credit_retrieval_trace(
            traj, retrieved_uids_per_query, correct_per_query,
        )

    if use_episode:
        # Use the avg of counterfactual where available, else 0; bake in cost.
        valid = [r.counterfactual_credit for r in traj.ops
                 if r.counterfactual_credit is not None and r.counterfactual_credit == r.counterfactual_credit]
        avg = sum(valid) / len(valid) if valid else 0.0
        credit_episode(traj, avg * len(traj.ops))   # per-op = avg

    return [r.total_credit() for r in traj.ops]
