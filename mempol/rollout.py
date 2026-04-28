"""Framework-agnostic rollout collector + group-relative advantages.

The same code Tinker, TRL, or verl will plug into. We provide:

  collect_rollouts(question, gold, backend, policy_factory, G, cost_lambda)
     → list[Trajectory]   each with .reward and .step_records

  compute_advantages(rewards) → list[float]    (group-relative, normalized)

For the LLM-as-policy case (when we have a small SFT'd LM emitting ops), pass
a `policy_factory` that returns G stochastic copies. For the heuristic teacher
case (used to build SFT data), pass a single deterministic + perturbations.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Iterable

from .backends.base import Backend, Hit
from .eval.judge import judge
from .policies.base import ReadPolicy, Step, Trace


@dataclass
class StepRecord:
    """One op decision in a trajectory. Format is generic enough for Tinker
    Datums (sampled_tokens + logprobs) AND for our pre-RL teacher traces
    (where tokens/logprobs may be None)."""
    step_index: int
    state_text: str
    op: str
    args: dict
    obs_summary: str
    sampled_tokens: list[int] | None = None
    logprobs: list[float] | None = None
    prompt_tokens: list[int] | None = None


@dataclass
class Trajectory:
    qid: str
    question: str
    gold: str
    answer: str
    judge_score: float
    cost: float
    reward: float
    step_records: list[StepRecord] = field(default_factory=list)


def cost_of(trace: Trace, lambda_step: float = 0.005, lambda_retrieval: float = 0.01) -> float:
    return lambda_step * len(trace.steps) + lambda_retrieval * trace.n_retrievals


def trace_to_records(trace: Trace, question: str) -> list[StepRecord]:
    """Convert a heuristic Trace (no token-level info) into StepRecords.
    Used during SFT-data collection. RL trainers will replace the empty
    token/logprob fields by sampling through the LM."""
    out = []
    state_so_far = f"<task>read</task>\n<query>{question}</query>\n"
    for i, s in enumerate(trace.steps):
        out.append(
            StepRecord(
                step_index=i,
                state_text=state_so_far,
                op=s.op,
                args=s.args,
                obs_summary=s.obs_summary,
            )
        )
        state_so_far += f"<op>{s.op}</op><obs>{s.obs_summary}</obs>\n"
    return out


def collect_rollouts(
    question: str,
    gold: str,
    backend: Backend,
    policy_sampler: Callable[[], ReadPolicy],
    G: int = 8,
    cost_lambda_step: float = 0.005,
    cost_lambda_retrieve: float = 0.01,
) -> list[Trajectory]:
    """Run G policy rollouts and score each. The policy_sampler is called G
    times so it can return G different stochastic policy instances (e.g.
    different temperatures, dropouts, or Tinker `sample()` branches)."""
    trajs = []
    for _ in range(G):
        policy = policy_sampler()
        trace = policy.run(question, backend)
        s, _ = judge(question, gold, trace.answer)
        c = cost_of(trace, cost_lambda_step, cost_lambda_retrieve)
        traj = Trajectory(
            qid="",
            question=question,
            gold=gold,
            answer=trace.answer,
            judge_score=s,
            cost=c,
            reward=s - c,
            step_records=trace_to_records(trace, question),
        )
        trajs.append(traj)
    return trajs


def compute_advantages(rewards: list[float], normalise: bool = True) -> list[float]:
    """Group-relative advantage. If normalise=True, divide by std (DeepSeek-style)."""
    if not rewards:
        return []
    mean = sum(rewards) / len(rewards)
    advs = [r - mean for r in rewards]
    if normalise and len(advs) > 1:
        # Population std
        var = sum(a * a for a in advs) / len(advs)
        std = var ** 0.5 or 1.0
        advs = [a / std for a in advs]
    return advs
