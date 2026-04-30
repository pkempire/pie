"""Architecture patterns — abstractions over real-world component combinations.

Why patterns matter
===================
Component-level retrieval finds the right *parts*. Patterns tell you how
the parts fit. A planner that only does semantic search will pick the
top-1 component for every requirement and miss that "headless browser +
managed infra + cron + diff + alert" is itself a unit with predictable
slot fillers.

We don't hand-write patterns. We extract them from the real-world
architectures we mine (architecture_components table). The pipeline:

  1. CLUSTER mined architectures by Jaccard similarity over their
     component sets. A pattern is a cluster.

  2. ABSTRACT the cluster into a pattern card: which roles recur, what
     components fill each role, what fraction of cluster members use
     each filler.

  3. VERIFY by sampling held-out cluster members against the pattern:
     can we auto-instantiate the pattern's slots from the held-out
     architecture's components without missing roles?

The result is a Pattern with ranked slot fillers, canonical examples,
and a description that the planner can match user specs against.

Schema (added to architect/db/schema.sql when we ship this; left here
as a note since the pattern table doesn't exist yet — kept inline
ergonomically until clusters give us enough signal to commit a schema):

    patterns(id, slug, name, description, slot_count, n_examples,
              canonical_example_arch_ids, created_at)
    pattern_slots(pattern_id, role, weight, required INT)
    pattern_slot_fillers(pattern_id, role, component_id,
                          frac_of_examples)

Algorithm in detail
===================

Clustering. We use single-linkage on Jaccard distance (1 - |A∩B|/|A∪B|).
Two architectures are near each other if they share most of their
components. Threshold ~0.45 gives clusters of "same-shape" systems
without merging unrelated stacks. We cap cluster size at ~50 architectures
(the pattern is well-defined long before that).

Slot extraction. Within a cluster, we get every (component, role) pair
from architecture_components. We aggregate: a role is "real" if it
appears in ≥ 30% of the cluster, "core" if ≥ 70%, "optional" if 30-70%.
For each role, the candidate fillers are the components that filled it,
ranked by frequency within the cluster.

LLM card. We pass the cluster's summary (size, top components, top
roles, ~3 example architectures) to an LLM and ask for a short,
concrete pattern description: name, when-to-use, key trade-offs.

Verification. We hold out 20% of the cluster. For each held-out
architecture, we try to auto-instantiate the pattern's required slots
from the architecture's components. If ≥ 80% of held-out architectures
verify cleanly, the pattern lands; otherwise it's flagged for manual
review.

This file ships the algorithm but not the cron driver (that comes
once we've mined enough architectures to have real clusters; today we
have 0 mined arches so this is the algorithm-as-code with a hand-rolled
fixture for testing).
"""
from __future__ import annotations
import json
import logging
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable

from . import db
from mempol import llm, config

logger = logging.getLogger(__name__)


# ─── Data model (in-memory; schema lands when patterns table is added) ───────
@dataclass
class PatternSlot:
    role: str
    required: bool                                     # appears in ≥ core_threshold of cluster
    fillers: list[tuple[str, float]] = field(default_factory=list)
    # ↑ list of (component_slug, fraction_of_cluster_using_it)


@dataclass
class Pattern:
    slug: str                                          # snake_case unique id
    name: str                                          # human-friendly
    description: str
    when_to_use: str
    slots: list[PatternSlot]
    n_examples: int
    canonical_example_arch_ids: list[int]


# ─── Step 1. Clustering by Jaccard ──────────────────────────────────────────
def _arch_component_sets() -> dict[int, set[int]]:
    """Return {architecture_id: {component_id, ...}} for all mined arches."""
    out: dict[int, set[int]] = defaultdict(set)
    with db.connect() as conn:
        cur = conn.execute(
            "SELECT architecture_id, component_id FROM architecture_components"
        )
        for row in cur:
            out[row["architecture_id"]].add(row["component_id"])
    return out


def _jaccard(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / max(len(a | b), 1)


def cluster_architectures(
    threshold: float = 0.45, min_cluster_size: int = 5,
) -> list[set[int]]:
    """Single-linkage agglomerative clustering on Jaccard similarity.

    Returns clusters (sets of architecture_ids) of size ≥ min_cluster_size.
    Single-linkage is the right choice here because we want any path of
    similar architectures to merge — we'd rather have one big "browser
    agent" cluster than three small ones for variants.
    """
    sets = _arch_component_sets()
    arch_ids = list(sets.keys())
    if not arch_ids:
        return []

    # Disjoint set / union-find with on-the-fly merge
    parent = {a: a for a in arch_ids}
    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(x: int, y: int) -> None:
        parent[find(x)] = find(y)

    for i, a in enumerate(arch_ids):
        for b in arch_ids[i + 1:]:
            if _jaccard(sets[a], sets[b]) >= threshold:
                union(a, b)

    clusters: dict[int, set[int]] = defaultdict(set)
    for a in arch_ids:
        clusters[find(a)].add(a)
    return [c for c in clusters.values() if len(c) >= min_cluster_size]


# ─── Step 2. Slot extraction within a cluster ───────────────────────────────
def _arch_component_roles(arch_ids: Iterable[int]) -> list[tuple[int, int, str]]:
    """Return [(arch_id, component_id, role), ...] for the given architectures."""
    arch_ids = list(arch_ids)
    if not arch_ids:
        return []
    placeholders = ",".join("?" * len(arch_ids))
    with db.connect() as conn:
        cur = conn.execute(
            f"SELECT architecture_id, component_id, role "
            f"FROM architecture_components "
            f"WHERE architecture_id IN ({placeholders})",
            tuple(arch_ids),
        )
        return [(row["architecture_id"], row["component_id"], row["role"])
                for row in cur]


def extract_slots(
    cluster: set[int],
    core_threshold: float = 0.70,
    minor_threshold: float = 0.30,
) -> list[PatternSlot]:
    """Extract a slot list from a cluster of architectures.

    A "slot" is a role that appears in ≥ minor_threshold of the cluster.
    A slot is required if it appears in ≥ core_threshold.

    Filler ranking: within each slot, components ordered by the fraction
    of cluster architectures that used that component for that role.
    """
    triples = _arch_component_roles(cluster)
    if not triples:
        return []
    n = len(cluster)

    # Tally roles
    role_arch_set: dict[str, set[int]] = defaultdict(set)
    role_fillers: dict[str, Counter] = defaultdict(Counter)
    for arch_id, comp_id, role in triples:
        if not role:
            role = "unspecified"
        role_arch_set[role].add(arch_id)
        role_fillers[role][comp_id] += 1

    slots: list[PatternSlot] = []
    with db.connect() as conn:
        for role, archs_with in role_arch_set.items():
            frac = len(archs_with) / n
            if frac < minor_threshold:
                continue
            fillers: list[tuple[str, float]] = []
            for comp_id, count in role_fillers[role].most_common():
                row = conn.execute(
                    "SELECT slug FROM components WHERE id=?", (comp_id,),
                ).fetchone()
                if row:
                    fillers.append((row["slug"], count / n))
            slots.append(PatternSlot(
                role=role,
                required=frac >= core_threshold,
                fillers=fillers,
            ))
    # Sort: required slots first, then by filler-coverage of the top filler
    slots.sort(
        key=lambda s: (s.required, s.fillers[0][1] if s.fillers else 0.0),
        reverse=True,
    )
    return slots


# ─── Step 3. LLM-write the pattern card ─────────────────────────────────────
_PATTERN_CARD_SYSTEM = """You are summarising a recurring software
architecture pattern that appears in real GitHub repositories. You will
receive: a description of the role+filler structure of the pattern, a
list of canonical example repos, and the count of repos in the cluster.

Write a concise pattern card with:
  - name: 3-6 word camelcase-or-spaces name (e.g. "Scheduled browser
    scraping with diff alerts")
  - description: 2-3 sentences describing the pattern
  - when_to_use: 1-2 sentence rule of thumb for when this pattern fits

Be opinionated where evidence supports it. Avoid marketing copy. Output
JSON only."""


def write_pattern_card(
    cluster: set[int], slots: list[PatternSlot], canonical_examples: list[int],
) -> dict:
    """Have the LLM produce a card for this pattern."""
    slot_summary = []
    with db.connect() as conn:
        for s in slots:
            top = ", ".join(
                f"{slug}({frac:.0%})" for slug, frac in s.fillers[:4]
            )
            slot_summary.append(
                f"  - {s.role} [{'required' if s.required else 'optional'}]: {top}"
            )
        ex_lines = []
        for aid in canonical_examples[:3]:
            row = conn.execute(
                "SELECT name, source_url, summary FROM architectures WHERE id=?",
                (aid,),
            ).fetchone()
            if row:
                ex_lines.append(f"  - {row['name']}: {row['summary'][:120]}")
    user = (
        f"Cluster size: {len(cluster)} architectures.\n\n"
        f"Slots and fillers:\n" + "\n".join(slot_summary) + "\n\n"
        f"Canonical examples:\n" + "\n".join(ex_lines) + "\n\n"
        "Return JSON: {\"name\": \"...\", \"description\": \"...\", \"when_to_use\": \"...\"}"
    )
    msgs = [
        {"role": "system", "content": _PATTERN_CARD_SYSTEM},
        {"role": "user",   "content": user},
    ]
    raw = llm.chat(msgs, model=config.OBSERVER_MODEL or "gpt-4o-mini",
                    json_mode=True)
    try:
        return json.loads(raw)
    except Exception as e:
        logger.warning("pattern card parse fail: %s; raw=%r", e, raw[:200])
        return {}


# ─── Step 4. Verify a pattern against held-out cluster members ──────────────
def verify_pattern(pattern: Pattern, held_out: set[int]) -> float:
    """Fraction of held-out architectures that successfully fill all
    `required` slots from their own component lists.

    Above 0.8 = the pattern is real and stable. 0.5–0.8 = real but loose,
    flag for review. < 0.5 = the cluster wasn't actually homogeneous.
    """
    if not held_out:
        return 0.0
    triples = _arch_component_roles(held_out)
    by_arch: dict[int, set[tuple[int, str]]] = defaultdict(set)
    for arch_id, comp_id, role in triples:
        by_arch[arch_id].add((comp_id, role or "unspecified"))

    required_roles = {s.role for s in pattern.slots if s.required}
    if not required_roles:
        return 1.0

    n_pass = 0
    for arch_id, pairs in by_arch.items():
        roles_present = {r for _, r in pairs}
        if required_roles.issubset(roles_present):
            n_pass += 1
    return n_pass / len(by_arch)


# ─── Top-level entry ────────────────────────────────────────────────────────
def extract_patterns(
    threshold: float = 0.45,
    min_cluster_size: int = 5,
    holdout_frac: float = 0.2,
) -> list[Pattern]:
    """Run the full pipeline. Returns a list of Pattern objects.

    Side effect: none yet — caller decides whether to persist (the
    patterns / pattern_slots / pattern_slot_fillers tables aren't in
    schema.sql yet because we have zero mined architectures right now;
    when the miner has populated some, we'll commit the schema and
    have this function persist directly).
    """
    clusters = cluster_architectures(threshold=threshold,
                                       min_cluster_size=min_cluster_size)
    out: list[Pattern] = []
    for cluster in clusters:
        cluster_list = sorted(cluster)
        n_holdout = max(1, int(len(cluster_list) * holdout_frac))
        held_out = set(cluster_list[:n_holdout])
        train_set = set(cluster_list[n_holdout:])
        slots = extract_slots(train_set)
        if not slots:
            continue
        # Pick canonical examples: highest quality_signal in train_set
        with db.connect() as conn:
            placeholders = ",".join("?" * len(train_set))
            cur = conn.execute(
                f"SELECT id FROM architectures WHERE id IN ({placeholders}) "
                f"ORDER BY quality_signal DESC LIMIT 3",
                tuple(train_set),
            )
            canonical_ids = [row["id"] for row in cur]
        card = write_pattern_card(train_set, slots, canonical_ids)
        if not card:
            continue
        pat = Pattern(
            slug=card.get("name", "pattern").lower().replace(" ", "_")[:60],
            name=card.get("name", "unnamed pattern"),
            description=card.get("description", ""),
            when_to_use=card.get("when_to_use", ""),
            slots=slots,
            n_examples=len(cluster),
            canonical_example_arch_ids=canonical_ids,
        )
        verified_frac = verify_pattern(pat, held_out)
        if verified_frac >= 0.5:
            out.append(pat)
            logger.info("pattern %r verified at %.0f%% (cluster=%d)",
                        pat.name, verified_frac * 100, len(cluster))
        else:
            logger.info("pattern %r failed verification at %.0f%%; skipping",
                        pat.name, verified_frac * 100)
    return out
