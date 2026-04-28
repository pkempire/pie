"""Reader-overlap reward and retention-budget enforcement.

Motivation
----------
Phase B's earlier dense reward was *evidence coverage* — the fraction of a
question's gold dia_ids preserved in the post-W KG. Coverage requires
LoCoMo's per-question evidence labels, which research benchmarks ship and
real personal-AI deployments do not. Worse, on its own coverage rewards
"store more dia_ids" without distinguishing useful storage from bloat: a
follow-up audit found `n_kept` correlates with gold coverage at ρ ≈ 0.98.

This module replaces coverage with a *reader-overlap* signal validated in
an independent study (Discovery report, Apr 2026): for each question in a
held-out battery, take what a frozen reader retrieves from the full
conversation text (the gold "useful turns" set as the reader sees it) and
score how much of that the post-W memory actually preserves. No gold
evidence labels are needed; the signal correlates with gold coverage at
within-turn Spearman ρ ≈ 0.61 and is robust to backend swaps.

Crucially, we pair the reward with a *hard retention budget*. The write
policy is allowed at most K_max entities per episode; if it exceeds the
budget, the lowest-importance entities are pruned before scoring. This
makes "store everything" structurally impossible and turns the problem
into a budgeted-OS optimisation rather than a soft cost-vs-gain trade.

Reward formulation
------------------
For one write episode with question battery Q = {q_i}:

    full_text_dia_ids(q)  = {dia_id of each turn R retrieves from the
                              full conversation backend for q}
    post_W_dia_ids        = union of provenance dia_ids in the budgeted
                              post-W KG
    overlap(q)            = |full_text_dia_ids(q) ∩ post_W_dia_ids|
                            / max(|full_text_dia_ids(q)|, 1)
    reward                = mean_q overlap(q)

Notes:
  - We use *recall* (full-text dia_ids that survived) rather than Jaccard
    because the post-W set is intentionally smaller (under budget).
  - The full-text dia_ids are computed once per battery (cached) since
    they don't depend on the W trajectory.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Any, Iterable

from mempol.backends.base import Backend, Hit
from mempol.backends.pie_kg import PIEBackend

logger = logging.getLogger(__name__)


# ─── Hard retention-budget enforcement ───────────────────────────────────────
def enforce_budget(backend: PIEBackend, k_max: int) -> int:
    """Prune the KG so it contains at most k_max entities.

    Pruning rule: lowest-importance entities first; ties broken by
    last_seen (older drops first). This mimics what a real production
    LRU/importance-mix eviction policy would do.

    Returns the number of entities removed. A no-op if already under
    budget; safe to call multiple times.
    """
    wm = backend.wm
    n = len(wm.entities)
    if n <= k_max:
        return 0
    # Rank by (importance asc, last_seen asc) — least-valued first
    ranked = sorted(
        wm.entities.items(),
        key=lambda kv: (kv[1].importance or 0.0, kv[1].last_seen or 0.0),
    )
    to_drop = ranked[: n - k_max]
    for uid, _ in to_drop:
        wm.entities.pop(uid, None)
        # Also drop their transitions and relationships so retrieval is
        # consistent with the reduced KG.
        if hasattr(wm, "_entity_transitions"):
            for tid in wm._entity_transitions.pop(uid, []):
                wm.transitions.pop(tid, None)
    # Rebuild embedding matrix once after the bulk drop
    try:
        wm.rebuild_embedding_matrix()
    except Exception:
        pass
    return len(to_drop)


# ─── Reader-overlap scoring ──────────────────────────────────────────────────
@dataclass
class OverlapResult:
    """Per-trajectory overlap breakdown."""
    mean_overlap: float
    per_question: list[tuple[str, float]] = field(default_factory=list)
    n_full_text_dia_ids_total: int = 0
    n_full_text_dia_ids_recovered: int = 0


def _hits_to_dia_ids(hits: Iterable[Hit]) -> set[str]:
    """Map a list of Hits to the set of dia_ids they touch.

    Works for both flat (Unit.metadata['dia_ids']) and KG (entity provenance
    via metadata['source_dia_id'] when available, else fallback empty).
    """
    out: set[str] = set()
    for h in hits:
        m = h.unit.metadata or {}
        dia_ids = m.get("dia_ids") or m.get("dia_id")
        if isinstance(dia_ids, str):
            out.add(dia_ids)
        elif isinstance(dia_ids, list):
            out.update(d for d in dia_ids if isinstance(d, str))
        # KG-Hit fallback: source_dia_id is set by _entity_to_hit
        sd = m.get("source_dia_id")
        if isinstance(sd, str) and sd:
            out.add(sd)
    return out


def _stored_dia_ids(backend: PIEBackend) -> set[str]:
    """All dia_ids referenced by the post-W KG (entity provenance + transitions)."""
    out: set[str] = set()
    for e in backend.wm.entities.values():
        if e.created_from:
            out.add(e.created_from)
    for trans_list in backend.wm._entity_transitions.values():
        for tid in trans_list:
            tr = backend.wm.transitions.get(tid)
            if tr and tr.trigger_conversation_id:
                out.add(tr.trigger_conversation_id)
    return out


def battery_reader_overlap(
    backend: PIEBackend,
    battery: list[tuple[str, str, list[str]]],
    full_text_backend: Backend,
    reader: Any,
    full_text_cache: dict[str, set[str]] | None = None,
) -> OverlapResult:
    """Score a write trajectory by reader-overlap.

    Args:
        backend: the post-W KG. Already budget-enforced before calling.
        battery: list of (question, gold_answer, evidence_dia_ids). Evidence
            is unused here — kept in the signature so this function is
            drop-in compatible with the coverage signature.
        full_text_backend: a Backend (typically FlatBackend) populated with
            the FULL conversation chunks. Built once per conversation.
        reader: a frozen read policy with `.run(question, backend) -> trace`
            and `trace.final_hits` of type list[Hit].
        full_text_cache: optional dict[question -> set[dia_id]] to skip
            re-running R on full text across rollouts of the same battery.

    Returns:
        OverlapResult.mean_overlap is the scalar reward in [0, 1].
    """
    stored = _stored_dia_ids(backend)
    per_q: list[tuple[str, float]] = []
    total_ref, total_hit = 0, 0

    for q, _gold, _ev in battery:
        if full_text_cache is not None and q in full_text_cache:
            ref = full_text_cache[q]
        else:
            try:
                trace = reader.run(q, full_text_backend)
                ref = _hits_to_dia_ids(trace.final_hits or [])
            except Exception as e:
                logger.warning("reader-overlap: full-text reader failed on q=%r: %s",
                               q[:60], e)
                ref = set()
            if full_text_cache is not None:
                full_text_cache[q] = ref

        if not ref:
            per_q.append((q, 0.0))
            continue
        hit = sum(1 for d in ref if d in stored)
        per_q.append((q, hit / len(ref)))
        total_ref += len(ref)
        total_hit += hit

    mean = sum(s for _, s in per_q) / max(len(per_q), 1)
    return OverlapResult(
        mean_overlap=mean,
        per_question=per_q,
        n_full_text_dia_ids_total=total_ref,
        n_full_text_dia_ids_recovered=total_hit,
    )


# ─── Smoke test ──────────────────────────────────────────────────────────────
def _smoke():
    """End-to-end smoke against a hand-built KG and a small Flat backend."""
    from mempol.backends.flat import FlatBackend
    from mempol.backends.base import Unit
    from mempol.policies.v1_heuristic import HeuristicPolicy

    # Build a flat full-text backend with 3 chunks
    flat = FlatBackend()
    flat.ingest([
        Unit(uid="c1", text="Caroline lives in Boston and works at Whoop.",
             metadata={"dia_ids": ["D1:5"]}),
        Unit(uid="c2", text="On 7 May Caroline went to an LGBTQ support group.",
             metadata={"dia_ids": ["D1:7"]}),
        Unit(uid="c3", text="Caroline is excited to start her new role.",
             metadata={"dia_ids": ["D2:3"]}),
    ])

    # Build a KG that preserves only D1:5 and D2:3 (missed D1:7)
    kg = PIEBackend()
    kg.create_entity(name="Caroline", type="person",
                      state={"city": "Boston", "job": "Whoop"},
                      source="D1:5", timestamp=1.0)
    kg.create_entity(name="excitement", type="belief",
                      state={"target": "new role"},
                      source="D2:3", timestamp=3.0)

    enforce_budget(kg, k_max=10)         # under budget already
    reader = HeuristicPolicy(first_k=3, final_k=2,
                              do_reformulate=False, do_expand=False)

    battery = [
        ("Where does Caroline live?", "Boston", []),
        ("What did Caroline do on 7 May?", "support group", []),
    ]
    r = battery_reader_overlap(kg, battery, flat, reader)
    print(f"mean_overlap = {r.mean_overlap:.3f}")
    for q, s in r.per_question:
        print(f"  {s:.2f}  {q}")
    print(f"recovered {r.n_full_text_dia_ids_recovered}"
          f" / {r.n_full_text_dia_ids_total} full-text dia_ids")


if __name__ == "__main__":
    _smoke()
