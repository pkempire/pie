"""Evidence-coverage scorer — deterministic, dense, judge-free reward signal
for the write policy.

Motivation
----------
Phase B's deferred reward currently runs a frozen R policy + LLM judge against
a held-out QA battery for every write trajectory. That signal is noisy and
expensive (~3 s and ~$0.001 per question per rollout). Worse, the QA reward
collapses to 0 whenever R fails to retrieve a chunk, even if W stored the
right thing — the noise comes from R, not from the W action sequence we're
trying to credit.

LoCoMo annotates every QA with an `evidence` field: the list of dia_ids whose
content the question depends on. Every PIE entity records the dia_id it was
created from (`Entity.created_from`), and every state transition records the
dia_id that triggered it (`StateTransition.trigger_conversation_id`). So we
can compute a *coverage* score per question:

    cov(q) = |evidence(q) ∩ stored_dia_ids(M_τ)| / |evidence(q)|

That is: of the dia_ids the question needs, how many did the W policy
actually preserve in the memory state?

Why this is a good reward
-------------------------
- **Free**: pure dict lookups, no LLM call.
- **Dense**: per-question signal between 0 and 1, not bucketed to {0, 0.5, 1}.
- **Deterministic**: removes the R + judge variance entirely from training.
- **Per-op credit**: if op_i was the create that tagged dia_id d into the KG
  and Q needs d, then op_i provably contributed (used by counterfactual
  ablation in `mempol.rewards.credit`).
- **Decoupled from R**: training W against coverage doesn't entangle W with
  R's bugs.

Why we still keep judge as eval-only
------------------------------------
Coverage scores "did you store the source", not "can the answer be
reconstructed". A pathological W could store a verbatim chunk of every dia_id
and get coverage=1.0 while the resulting KG is unreadable. We watch the judge
score on a holdout to make sure coverage gains transfer to actual answer
quality. If they diverge we change the reward mix or add a structure penalty.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterable

from mempol.backends.pie_kg import PIEBackend


def stored_dia_ids(backend: PIEBackend) -> set[str]:
    """Collect every dia_id referenced by an entity or transition currently in
    the PIE world model.

    An entity contributes its `created_from` (set when the W policy emitted
    `create_entity` with `source=current_dia_id`). Each state transition
    contributes its `trigger_conversation_id` (set on every
    `update_state` / `mark_contradiction` / `forget` op). Both are
    individual dia_ids — we union them.
    """
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


def coverage(evidence: Iterable[str], stored: set[str]) -> float:
    """Fraction of an evidence list that appears in `stored`.

    Returns 0.0 when evidence is empty (no signal — caller handles).
    """
    ev = [e for e in evidence if e]
    if not ev:
        return 0.0
    hits = sum(1 for e in ev if e in stored)
    return hits / len(ev)


@dataclass
class CoverageResult:
    """Per-trajectory coverage breakdown."""
    mean_coverage: float                            # mean over the battery
    per_question: list[tuple[str, float]] = field(default_factory=list)
    n_stored_dia_ids: int = 0
    n_evidence_dia_ids_total: int = 0
    n_evidence_dia_ids_hit: int = 0


def battery_coverage(
    backend: PIEBackend,
    battery: list[tuple[str, str, list[str]]],     # (question, gold, evidence)
) -> CoverageResult:
    """Score a write trajectory against a QA battery.

    Args:
        backend: the PIEBackend mutated by the W trajectory.
        battery: list of (question, gold_answer, evidence_dia_ids).

    Returns:
        CoverageResult with mean_coverage and per-question scores.
    """
    stored = stored_dia_ids(backend)
    per_q: list[tuple[str, float]] = []
    total_ev, total_hit = 0, 0
    for q, _gold, ev in battery:
        ev_clean = [e for e in (ev or []) if e]
        if not ev_clean:
            per_q.append((q, 0.0))
            continue
        hits = sum(1 for e in ev_clean if e in stored)
        per_q.append((q, hits / len(ev_clean)))
        total_ev += len(ev_clean)
        total_hit += hits
    mean = sum(s for _, s in per_q) / max(len(per_q), 1)
    return CoverageResult(
        mean_coverage=mean,
        per_question=per_q,
        n_stored_dia_ids=len(stored),
        n_evidence_dia_ids_total=total_ev,
        n_evidence_dia_ids_hit=total_hit,
    )


# ── Smoke ────────────────────────────────────────────────────────────────────
def _smoke():
    """End-to-end check on a hand-built PIE backend."""
    b = PIEBackend()
    e1 = b.create_entity(name="LGBTQ support group", type="event",
                         state={"date": "7 May 2023"},
                         source="D1:5", timestamp=1.0)
    e2 = b.create_entity(name="Caroline", type="person",
                         state={"city": "Boston"},
                         source="D1:7", timestamp=2.0)
    b.update_state(uid=e2, new_state={"mood": "excited"},
                   transition_type="update", source="D2:3", timestamp=3.0)
    battery = [
        ("Why did Caroline visit the support group?", "to find community",
         ["D1:5", "D1:7"]),                # both stored → cov=1.0
        ("When did Caroline move to Boston?", "May 2023", ["D9:1"]),
        # not stored → cov=0.0
        ("How did Caroline feel after the meeting?", "excited",
         ["D2:3", "D2:9"]),                # 1 of 2 → cov=0.5
    ]
    r = battery_coverage(b, battery)
    print(f"stored dia_ids: {sorted(stored_dia_ids(b))}")
    print(f"mean_coverage: {r.mean_coverage:.3f}")
    for q, s in r.per_question:
        print(f"  {s:.2f}  {q}")


if __name__ == "__main__":
    _smoke()
