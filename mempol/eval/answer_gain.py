"""Answer-gain reward — judge margin over a random-K baseline.

Replaces the dia_id-level reader-overlap signal that we shipped first.
That signal was structurally biased low: the reader retrieves chunks
of ~6 dia_ids each from full text, while the post-W KG stores entities
with 1 dia_id each, so even a perfect write policy could not exceed
~0.5 overlap, and most rollouts landed near zero.

Answer-gain measures what the write policy actually buys you:

    gain(q) = judge(R(q, post_W_KG), gold)
            - judge(R(q, random_K_baseline), gold)

with K = K_max (the same retention budget the policy is operating
under). The random baseline is the must-beat content-agnostic frontier
we ship in `random_baseline.py`. A learned policy that ties random
gets reward 0; that beats random gets a positive signal; that loses to
random gets a negative signal.

The baseline scores are cached per (conv_id, question, K) — random is
sampled once per (conv, K) using a deterministic seed. So the runtime
cost amortises to one judge call per question over the full training
run, regardless of how many GRPO rollouts hit that conv.
"""
from __future__ import annotations
import logging
import random
from dataclasses import dataclass, field
from typing import Any

from mempol.backends.base import Backend, Unit
from mempol.backends.flat import FlatBackend
from mempol.backends.pie_kg import PIEBackend
from mempol.eval.judge import judge as _judge

logger = logging.getLogger(__name__)


@dataclass
class GainResult:
    mean_gain: float
    per_question: list[tuple[str, float]] = field(default_factory=list)
    n_random_baseline_calls: int = 0   # incremented on cache misses


def _build_random_K_backend(
    full_text_backend: FlatBackend, K: int, seed: int,
) -> FlatBackend:
    """Take K random units from `full_text_backend` and rebuild a fresh
    FlatBackend with just those. Used as the random-baseline memory."""
    rng = random.Random(seed)
    units = list(full_text_backend.units)
    if len(units) <= K:
        return full_text_backend
    sampled = rng.sample(units, K)
    out = FlatBackend()
    out.ingest([
        Unit(uid=u.uid, text=u.text, metadata=dict(u.metadata or {}))
        for u in sampled
    ])
    return out


def battery_answer_gain(
    backend: PIEBackend,
    battery: list[tuple[str, str, list[str]]],
    full_text_backend: FlatBackend,
    reader: Any,
    K: int,
    conv_id: str,
    baseline_cache: dict[tuple[str, str, int], float] | None = None,
    seed: int = 0,
) -> GainResult:
    """Score a write trajectory by judge-margin over a random-K baseline.

    Args:
        backend: the post-W KG (already budget-enforced before we get here).
        battery: list of (question, gold, evidence_dia_ids).
        full_text_backend: the full-conversation FlatBackend, used to build
            the random-K baseline by sampling K of its units.
        reader: frozen R; must have `.run(q, backend) -> trace` with
            trace.answer.
        K: retention budget (same K as the policy's hard cap).
        conv_id: opaque key for the random-baseline cache.
        baseline_cache: shared dict[(conv_id, question, K) -> baseline_score].
            Caller passes a dict that persists across all rollouts of this
            conv so the random-baseline score is judged at most once per
            (conv, q, K) over the entire training run.
        seed: deterministic seed for the random sample. Should be stable
            across runs so cached baselines stay consistent.

    Returns:
        GainResult with mean_gain in [-1, 1].
    """
    if baseline_cache is None:
        baseline_cache = {}

    # Random-K backend is built once per (conv_id, K) — its identity is
    # fixed by the seed.
    random_backend = _build_random_K_backend(full_text_backend, K, seed=hash(
        (conv_id, K, seed)) & 0xFFFFFFFF)

    per_q: list[tuple[str, float]] = []
    cache_misses = 0
    for q, gold, _ev in battery:
        # Score on the post-W KG (fresh judge call every rollout)
        try:
            w_trace = reader.run(q, backend)
            w_ans = w_trace.answer or "not in context"
            w_score, _ = _judge(q, gold, w_ans)
        except Exception as e:
            logger.warning("answer_gain: post-W judge failed (%s)", e)
            w_score = 0.0

        # Random-K baseline — cached per (conv, q, K)
        cache_key = (conv_id, q, K)
        if cache_key in baseline_cache:
            base_score = baseline_cache[cache_key]
        else:
            try:
                r_trace = reader.run(q, random_backend)
                r_ans = r_trace.answer or "not in context"
                base_score, _ = _judge(q, gold, r_ans)
            except Exception as e:
                logger.warning("answer_gain: random-baseline judge failed (%s)", e)
                base_score = 0.0
            baseline_cache[cache_key] = base_score
            cache_misses += 1

        gain = float(w_score) - float(base_score)
        per_q.append((q, gain))

    mean = sum(g for _, g in per_q) / max(len(per_q), 1)
    return GainResult(
        mean_gain=mean,
        per_question=per_q,
        n_random_baseline_calls=cache_misses,
    )
