"""WriteReward — deferred + dense reward for the write policy.

Side note: if the env var `MEMPOL_TRAJECTORY_DUMP_DIR` is set, every call
to `WriteReward.__call__` writes a JSON snapshot of the trajectory
(messages, metrics, KG snapshot, per-question coverage) to that directory.
Used by the Streamlit dashboard. We do this here rather than via Tinker's
own logging because the cookbook's `rollout_json_export` writes to an
internal store we can't tail; doing it ourselves gives a flat directory
of one-JSON-per-rollout that any tool can read.


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
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from mempol.backends.base import Backend
from mempol.backends.pie_kg import PIEBackend
from mempol.eval.evidence_coverage import battery_coverage
from mempol.eval.judge import judge as _judge_sync
from mempol.eval.reader_overlap import (
    battery_reader_overlap, enforce_budget, OverlapResult,
)
from mempol.policies.v1_heuristic import HeuristicPolicy

logger = logging.getLogger(__name__)


def _maybe_dump_trajectory(history: list, reward: float,
                            metrics_full: dict) -> None:
    """If MEMPOL_TRAJECTORY_DUMP_DIR is set, write this rollout to disk
    for the dashboard. Failures are swallowed so logging never crashes the
    training loop."""
    out_dir = os.environ.get("MEMPOL_TRAJECTORY_DUMP_DIR", "").strip()
    if not out_dir:
        return
    try:
        d = Path(out_dir)
        d.mkdir(parents=True, exist_ok=True)
        # Normalize messages to plain dicts so json.dumps works regardless
        # of whether tinker passed in dataclasses or message objects.
        msgs_out = []
        for m in history:
            if isinstance(m, dict):
                msgs_out.append({k: v for k, v in m.items() if v is not None})
                continue
            md: dict = {}
            for fld in ("role", "content", "tool_calls"):
                v = getattr(m, fld, None)
                if v is not None:
                    md[fld] = v
            msgs_out.append(md)
        payload = {
            "ts":       time.time(),
            "reward":   float(reward),
            "metrics":  {k: float(v) if isinstance(v, (int, float)) else v
                         for k, v in metrics_full.items()
                         if not isinstance(v, list) and not isinstance(v, dict)},
            "per_question_coverage":
                metrics_full.get("per_question_coverage", []),
            "kg_snapshot": metrics_full.get("kg_snapshot", {}),
            "messages":   msgs_out,
        }
        # Filename: <step-time>_<short-uuid>.json so the dashboard can sort
        # by mtime and pick the most recent batch.
        name = f"{int(payload['ts']*1000)}_{uuid.uuid4().hex[:6]}.json"
        (d / name).write_text(json.dumps(payload, ensure_ascii=False, default=str))
    except Exception as e:
        logger.warning("trajectory dump failed: %s", e)


# Cost coefficients. Mostly redundant now that we enforce a hard retention
# budget — kept for ablation purposes (cost-only training as a comparison).
DEFAULT_COST_PER_OP = 0.001
DEFAULT_COST_PER_LOOKUP = 0.0
DEFAULT_COST_PER_ENTITY = 0.0

# Reward mix. The dense signal is reader-overlap (label-free, validated to
# correlate with gold coverage at within-turn ρ ≈ 0.61 on LoCoMo). We blend
# with the LLM-judge QA term so the policy is anchored to actual downstream
# answer quality, not just retrieval shape.
DEFAULT_W_OVERLAP = 0.7
DEFAULT_W_QA = 0.3

# Hard retention budget: the post-W KG is pruned to at most this many
# entities before scoring. The number 12 is calibrated to LoCoMo turn-level
# episodes (typical conversation has ~50 turns × ~0.25 entities/turn = ~12)
# but should be reported as an ablation. The budget makes "store everything"
# structurally impossible.
DEFAULT_K_MAX = 12


@dataclass
class WriteReward:
    """Reader-overlap + judge reward for one write trajectory, scored
    under a hard retention budget.

    Lifetime: one instance per env (per group member). Bound to a specific
    PIEBackend (the one being mutated by the W tools in that env), a
    pre-computed QA battery, a frozen reader, and a full-text reference
    backend the reader queries to compute the overlap target.

    Attributes:
        backend: the post-W KG. Pruned to ≤ k_max entities before scoring.
        query_battery: list of (question, gold, evidence_dia_ids) tuples.
            evidence_dia_ids is unused by the overlap reward — kept for
            backward compat with optional logging of legacy coverage.
        full_text_backend: a Backend (typically FlatBackend) holding the
            FULL conversation chunks. The reader queries this to define
            the overlap target.
        reader: the frozen read policy with `.run(question, backend)`.
        r_runner: callable(question, backend) -> answer_str for the judge
            term. Defaults to the heuristic R if not provided AND w_qa > 0.
        write_tool: WriteTool instance — source of truth for op counts.
        w_overlap, w_qa: blend weights for the two reward terms.
        k_max: hard retention budget; KG is pruned to this many entities
            before scoring.
        cost_*: residual op costs (kept for ablations).
        full_text_cache: shared dict[question -> set[dia_id]] across
            rollouts of the same conv to amortise the full-text retrieval
            cost. Caller passes one cache per (conv, episode-batch).
    """
    backend: PIEBackend
    query_battery: list[tuple[str, str, list[str]]]
    full_text_backend: Backend | None = None
    reader: Any = None
    r_runner: Callable[[str, PIEBackend], str] | None = None
    write_tool: Any = None
    w_overlap: float = DEFAULT_W_OVERLAP
    w_qa: float = DEFAULT_W_QA
    k_max: int = DEFAULT_K_MAX
    cost_per_op: float = DEFAULT_COST_PER_OP
    cost_per_lookup: float = DEFAULT_COST_PER_LOOKUP
    cost_per_entity: float = DEFAULT_COST_PER_ENTITY
    full_text_cache: dict[str, set[str]] | None = None
    # Internal: per-evaluation metrics for logging
    _last_metrics: dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        # Default the judge runner to the heuristic R only if QA is in the mix.
        if self.r_runner is None and self.w_qa > 0:
            self.r_runner = _default_r_runner
        # Default reader for overlap to the heuristic if not provided.
        if self.reader is None:
            self.reader = _HEURISTIC_R

    async def __call__(self, history: list[dict]) -> tuple[float, dict[str, float]]:
        """Tinker-compatible reward signature.

        Returns (reward, metrics) where reward is a scalar in roughly [-0.01, 1.0]
        and metrics is logged per-step by the trainer. The reward decomposes:

            reward = w_overlap * reader_overlap(R, full_text, post_W_KG, Q)
                   + w_qa      * mean_q judge(R(q, post_W_KG), gold_q)
                   - cost(ops)

        Budget enforcement happens BEFORE scoring: the post-W KG is pruned
        to at most k_max entities (lowest-importance first). This makes
        "store everything" structurally impossible.
        """
        # 1. Empty battery → no signal.
        if not self.query_battery:
            self._last_metrics = {"battery_size": 0.0}
            return -0.01, dict(self._last_metrics)

        # 2. Enforce hard retention budget on the post-W KG.
        n_pruned = enforce_budget(self.backend, k_max=self.k_max)

        # 3. Reader-overlap — label-free dense signal. If full_text_backend
        #    isn't supplied (e.g. unit tests, smoke), fall back to legacy
        #    coverage so the reward still trains.
        overlap_result: OverlapResult | None = None
        cov_result = battery_coverage(self.backend, self.query_battery)
        if self.full_text_backend is not None and self.reader is not None:
            try:
                overlap_result = battery_reader_overlap(
                    backend=self.backend,
                    battery=self.query_battery,
                    full_text_backend=self.full_text_backend,
                    reader=self.reader,
                    full_text_cache=self.full_text_cache,
                )
                mean_overlap = overlap_result.mean_overlap
            except Exception as e:
                logger.warning("WriteReward: reader-overlap failed (%s); "
                                "falling back to coverage signal", e)
                mean_overlap = cov_result.mean_coverage
        else:
            # No full-text backend: fall back to coverage. Used during the
            # transition period and for ablations.
            mean_overlap = cov_result.mean_coverage

        # 4. QA judge — optional anchoring term.
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

        # 4. Cost from the trajectory. Op counts come from the WriteTool's
        # in-method counters (ground truth) when available; fall back to
        # history scraping (fragile across renderer versions) otherwise.
        if self.write_tool is not None:
            wt = self.write_tool
            n_lookups   = int(getattr(wt, "n_lookups", 0))
            n_mutations = int(
                getattr(wt, "n_creates", 0) + getattr(wt, "n_updates", 0)
                + getattr(wt, "n_merges", 0)  + getattr(wt, "n_relations", 0)
                + getattr(wt, "n_contradictions", 0) + getattr(wt, "n_forgets", 0)
            )
            n_noops    = int(getattr(wt, "n_noops", 0))
            n_ops_total = n_lookups + n_mutations + n_noops
        else:
            n_ops_total = self._count_ops(history)
            n_lookups = self._count_lookups(history)
            n_mutations = max(0, n_ops_total - n_lookups)
            n_noops = 0
        n_entities = len(self.backend.wm.entities)
        cost = (
            self.cost_per_op * n_mutations
            + self.cost_per_lookup * n_lookups
            + self.cost_per_entity * n_entities
        )

        reward = self.w_overlap * mean_overlap + self.w_qa * mean_qa - cost
        self._last_metrics = {
            "reader_overlap_mean": mean_overlap,
            "coverage_mean": cov_result.mean_coverage,    # legacy, kept for log
            "qa_mean": mean_qa,
            "cost_total": cost,
            "n_ops": float(n_ops_total),
            "n_lookups": float(n_lookups),
            "n_mutations": float(n_mutations),
            "n_noops": float(n_noops),
            "n_entities": float(n_entities),
            "n_pruned": float(n_pruned),
            "k_max": float(self.k_max),
            "battery_size": float(len(self.query_battery)),
            "stored_dia_ids": float(cov_result.n_stored_dia_ids),
            "evidence_hit_frac": (
                cov_result.n_evidence_dia_ids_hit
                / max(cov_result.n_evidence_dia_ids_total, 1)
            ),
        }
        if overlap_result is not None:
            self._last_metrics["full_text_dia_ids_recovered_frac"] = (
                overlap_result.n_full_text_dia_ids_recovered
                / max(overlap_result.n_full_text_dia_ids_total, 1)
            )
        # Side-channel state for the dashboard: serialize the per-question
        # coverage list and a compact KG snapshot. Tinker's rollout JSON
        # exporter reads `metrics` as a flat dict, so we attach these as
        # extra members on `_last_metrics` only — string lists are dropped
        # by the metric uploader but a wrapper script can read them off
        # the rollout dump.
        try:
            self._last_metrics_full = dict(self._last_metrics)
            self._last_metrics_full["per_question_coverage"] = [
                (q[:120], float(s)) for q, s in cov_result.per_question
            ]
            self._last_metrics_full["kg_snapshot"] = self._kg_snapshot()
            _maybe_dump_trajectory(history, reward, self._last_metrics_full)
        except Exception as e:
            logger.warning("metrics-side-channel build failed: %s", e)
        return reward, dict(self._last_metrics)

    def _kg_snapshot(self, max_entities: int = 30) -> dict:
        """Serialise the post-W KG for dashboard display. Compact — capped
        to `max_entities` rows. Reads the same backend that just got scored."""
        wm = self.backend.wm
        entities = []
        for uid, e in list(wm.entities.items())[:max_entities]:
            tids = wm._entity_transitions.get(uid, [])
            entities.append({
                "uid": uid,
                "name": e.name,
                "type": e.type.value if hasattr(e.type, "value") else str(e.type),
                "current_state": dict(e.current_state or {}),
                "n_transitions": len(tids),
                "source_dia_id": e.created_from or "",
            })
        return {
            "n_entities": len(wm.entities),
            "stored_dia_ids": sorted(self._collect_dia_ids()),
            "entities": entities,
        }

    def _collect_dia_ids(self) -> set[str]:
        """Re-collect stored dia_ids (mirrors evidence_coverage.stored_dia_ids
        but kept inline so this method has no extra import dependency)."""
        out: set[str] = set()
        for e in self.backend.wm.entities.values():
            if e.created_from:
                out.add(e.created_from)
        for trans_list in self.backend.wm._entity_transitions.values():
            for tid in trans_list:
                tr = self.backend.wm.transitions.get(tid)
                if tr and tr.trigger_conversation_id:
                    out.add(tr.trigger_conversation_id)
        return out

    @staticmethod
    def _assistant_tool_calls(msg: Any) -> list[dict]:
        """Return the structured tool_calls list from an assistant message.

        Tinker's renderers parse `<tool_call>` tags out of the raw content
        into a structured field. The substring fallback catches older
        renderers that leave the tag in `content`. Returns one dict per
        tool call with at least a `name` key."""
        if isinstance(msg, dict):
            tcs = msg.get("tool_calls") or []
            content = msg.get("content") or ""
        else:
            tcs = getattr(msg, "tool_calls", None) or []
            content = getattr(msg, "content", "") or ""
        # Normalize: structured form first (tinker-cookbook >=0.4)
        out: list[dict] = []
        for tc in tcs:
            if isinstance(tc, dict):
                name = tc.get("name") or (tc.get("function") or {}).get("name", "")
                out.append({"name": str(name)})
        if out:
            return out
        # Fallback: substring scrape of `<tool_call>{"name": "..."}` blocks.
        if isinstance(content, str) and "<tool_call>" in content:
            import re
            for m in re.finditer(r'"name"\s*:\s*"([^"]+)"', content):
                out.append({"name": m.group(1)})
        return out

    @classmethod
    def _count_ops(cls, history: list[dict]) -> int:
        """Total assistant tool calls in the trajectory (mutations + lookups)."""
        n = 0
        for msg in history:
            role = (msg.get("role") if isinstance(msg, dict)
                    else getattr(msg, "role", None))
            if role != "assistant":
                continue
            n += len(cls._assistant_tool_calls(msg))
        return n

    @classmethod
    def _count_lookups(cls, history: list[dict]) -> int:
        """Subset of assistant tool calls that are lookup ops."""
        n = 0
        for msg in history:
            role = (msg.get("role") if isinstance(msg, dict)
                    else getattr(msg, "role", None))
            if role != "assistant":
                continue
            for tc in cls._assistant_tool_calls(msg):
                if tc.get("name") in ("lookup_entity", "lookup_relation"):
                    n += 1
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
