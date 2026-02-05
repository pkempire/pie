"""
Resolution Ablation — measure entity resolution quality across tiers.

Tests Hypothesis 2 (rolling context) and directly evaluates the
three-tier resolution system (string → embedding → LLM) from
ARCHITECTURE-FINAL.md.

Metrics:
  - Unique entity count at each resolution tier
  - Estimated duplication rate via string matching
  - Resolution method distribution
  - Resolution method distribution over time (do later batches match more?)

Usage:
    python3 -m pie.eval.resolution_ablation [--world-model output/world_model.json]

The ablation simulates what would happen if only lower tiers were available:
  Tier 0: No resolution (every extracted mention = new entity)
  Tier 1: String matching only
  Tier 2: String + embedding similarity
  Tier 3: String + embedding + LLM verification (full system)
"""

from __future__ import annotations
import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("pie.eval.resolution_ablation")


# ── String matching (mirrors world_model._fuzzy_ratio) ───────────────────────

def _normalize(name: str) -> str:
    return name.lower().strip().replace("-", " ").replace("_", " ")


def _fuzzy_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, _normalize(a), _normalize(b)).ratio()


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class DuplicationCluster:
    """A group of entity names that likely refer to the same real entity."""
    canonical: str
    members: list[str] = field(default_factory=list)
    max_similarity: float = 0.0


@dataclass
class TemporalBucket:
    """Resolution stats for a time period."""
    period_label: str
    total_transitions: int = 0
    unique_entities_touched: int = 0
    # How many of those entities were created (new) vs updated (matched existing)
    new_entities: int = 0
    matched_entities: int = 0

    @property
    def match_rate(self) -> float:
        if self.unique_entities_touched == 0:
            return 0.0
        return self.matched_entities / self.unique_entities_touched


@dataclass
class ResolutionReport:
    """Full resolution ablation report."""
    # Entity counts per tier
    tier_0_entities: int = 0  # no resolution: raw extracted count
    tier_1_entities: int = 0  # after string dedup
    tier_2_entities: int = 0  # after string + embedding (approximated)
    tier_3_entities: int = 0  # actual count (full resolution)

    # Duplication clusters found by string matching
    duplication_clusters: list[DuplicationCluster] = field(default_factory=list)
    estimated_duplication_rate: float = 0.0

    # Alias coverage
    entities_with_aliases: int = 0
    total_aliases: int = 0
    avg_aliases_per_aliased_entity: float = 0.0

    # Web grounding
    web_verified_entities: int = 0
    web_verification_rate: float = 0.0

    # Temporal buckets
    temporal_buckets: list[TemporalBucket] = field(default_factory=list)

    # Method distribution (if resolution logs exist — estimated from data)
    name_match_count: int = 0
    embedding_match_estimate: int = 0
    new_entity_count: int = 0

    def to_dict(self) -> dict:
        return {
            "tier_entity_counts": {
                "tier_0_no_resolution": self.tier_0_entities,
                "tier_1_string_only": self.tier_1_entities,
                "tier_2_string_plus_embedding": self.tier_2_entities,
                "tier_3_full_resolution": self.tier_3_entities,
            },
            "duplication_analysis": {
                "clusters_found": len(self.duplication_clusters),
                "estimated_duplication_rate": round(self.estimated_duplication_rate, 4),
                "top_clusters": [
                    {
                        "canonical": c.canonical,
                        "members": c.members,
                        "max_similarity": round(c.max_similarity, 3),
                    }
                    for c in sorted(
                        self.duplication_clusters,
                        key=lambda c: len(c.members),
                        reverse=True,
                    )[:20]
                ],
            },
            "alias_coverage": {
                "entities_with_aliases": self.entities_with_aliases,
                "total_aliases": self.total_aliases,
                "avg_aliases_per_aliased_entity": round(self.avg_aliases_per_aliased_entity, 2),
            },
            "web_grounding": {
                "verified_entities": self.web_verified_entities,
                "verification_rate": round(self.web_verification_rate, 4),
            },
            "temporal_match_rate": [
                {
                    "period": b.period_label,
                    "total_transitions": b.total_transitions,
                    "entities_touched": b.unique_entities_touched,
                    "new_entities": b.new_entities,
                    "matched_entities": b.matched_entities,
                    "match_rate": round(b.match_rate, 3),
                }
                for b in self.temporal_buckets
            ],
        }


# ── Tier 0: Estimate raw extraction count ────────────────────────────────────

def estimate_tier_0_count(data: dict) -> int:
    """
    Estimate how many entities would exist with zero resolution.

    Without resolution, every extraction mention is a separate entity.
    We approximate this as: unique entities + all their aliases
    (since aliases represent mentions that WERE resolved).
    Plus transition count as a proxy for re-mentions.
    """
    entities = data.get("entities", {})
    total_aliases = sum(len(e.get("aliases", [])) for e in entities.values())

    # Each alias represents a mention that was resolved instead of becoming
    # a separate entity. Without resolution, each would be its own entity.
    # Also count transitions as evidence of re-mentions.
    transitions = data.get("transitions", {})
    creation_count = sum(
        1 for t in transitions.values()
        if t.get("transition_type") == "creation"
    )
    update_count = sum(
        1 for t in transitions.values()
        if t.get("transition_type") != "creation"
    )

    # Tier 0: every creation + every alias + fraction of updates
    # (updates to existing entities would be new entities without resolution)
    return creation_count + total_aliases


# ── Tier 1: String-only dedup ─────────────────────────────────────────────────

def find_string_duplication_clusters(
    entities: dict[str, dict],
    threshold: float = 0.85,
) -> list[DuplicationCluster]:
    """
    Find groups of entity names that are likely duplicates via string matching.

    This runs ALL entity names against each other to find pairs that the
    string-matching tier would merge. Returns clusters of similar names.
    """
    names = []
    for eid, entity in entities.items():
        names.append((eid, entity.get("name", ""), entity.get("type", "")))

    # Union-find for clustering
    parent = {eid: eid for eid, _, _ in names}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    # Compare all pairs (O(n²) — fine for <10K entities)
    max_sims: dict[tuple[str, str], float] = {}
    for i, (eid_a, name_a, type_a) in enumerate(names):
        for j, (eid_b, name_b, type_b) in enumerate(names):
            if j <= i:
                continue
            # Only compare same type (different types shouldn't merge)
            if type_a != type_b:
                continue

            score = _fuzzy_ratio(name_a, name_b)

            # Also check containment
            norm_a, norm_b = _normalize(name_a), _normalize(name_b)
            if norm_a and norm_b and (norm_a in norm_b or norm_b in norm_a):
                score = max(score, 0.90)

            if score >= threshold:
                union(eid_a, eid_b)
                key = (min(eid_a, eid_b), max(eid_a, eid_b))
                max_sims[key] = max(max_sims.get(key, 0), score)

    # Build clusters
    clusters_map: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for eid, name, etype in names:
        root = find(eid)
        clusters_map[root].append((eid, name))

    clusters = []
    for root, members in clusters_map.items():
        if len(members) <= 1:
            continue
        # Pick the longest name as canonical
        canonical = max(members, key=lambda m: len(m[1]))[1]
        member_names = [m[1] for m in members if m[1] != canonical]

        # Find max similarity in this cluster
        max_sim = 0.0
        for m1 in members:
            for m2 in members:
                if m1[0] >= m2[0]:
                    continue
                key = (min(m1[0], m2[0]), max(m1[0], m2[0]))
                max_sim = max(max_sim, max_sims.get(key, 0))

        clusters.append(DuplicationCluster(
            canonical=canonical,
            members=member_names,
            max_similarity=max_sim,
        ))

    return clusters


def estimate_tier_1_count(current_count: int, clusters: list[DuplicationCluster]) -> int:
    """
    After string-only dedup, how many entities remain?
    Each cluster collapses to 1 entity, so we subtract (cluster_size - 1) per cluster.
    """
    duplicates_removed = sum(len(c.members) for c in clusters)
    return current_count - duplicates_removed


# ── Temporal analysis ─────────────────────────────────────────────────────────

def analyze_temporal_resolution(data: dict, bucket_months: int = 3) -> list[TemporalBucket]:
    """
    Track resolution behavior over time.

    For each time bucket, count:
    - How many transitions occurred
    - How many unique entities were touched
    - How many of those were CREATIONS (new) vs UPDATES (matched existing)

    The hypothesis: later buckets should have higher match rates because
    the world model has more entities to match against.
    """
    import datetime

    transitions = data.get("transitions", {})
    if not transitions:
        return []

    # Group transitions by time bucket
    buckets_raw: dict[str, dict] = defaultdict(lambda: {
        "transitions": 0,
        "entities": set(),
        "creations": set(),
        "updates": set(),
    })

    for tid, t in transitions.items():
        ts = t.get("timestamp", 0)
        if ts <= 0:
            continue
        dt = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc)
        # Bucket by quarter
        quarter = (dt.month - 1) // bucket_months
        bucket_key = f"{dt.year}-Q{quarter + 1}"

        eid = t.get("entity_id", "")
        buckets_raw[bucket_key]["transitions"] += 1
        buckets_raw[bucket_key]["entities"].add(eid)

        if t.get("transition_type") == "creation":
            buckets_raw[bucket_key]["creations"].add(eid)
        else:
            buckets_raw[bucket_key]["updates"].add(eid)

    # Convert to TemporalBucket objects
    result = []
    for key in sorted(buckets_raw.keys()):
        b = buckets_raw[key]
        all_entities = b["entities"]
        new_entities = b["creations"]
        matched = all_entities - new_entities  # entities that existed before this bucket

        result.append(TemporalBucket(
            period_label=key,
            total_transitions=b["transitions"],
            unique_entities_touched=len(all_entities),
            new_entities=len(new_entities),
            matched_entities=len(matched),
        ))

    return result


# ── Main analysis ─────────────────────────────────────────────────────────────

def analyze(data: dict) -> ResolutionReport:
    """Run the full resolution ablation analysis."""
    report = ResolutionReport()
    entities = data.get("entities", {})

    if not entities:
        logger.warning("World model is empty — nothing to analyze.")
        return report

    # Tier 3 (actual) = current entity count
    report.tier_3_entities = len(entities)

    # Tier 0 (no resolution) = estimated raw extraction count
    report.tier_0_entities = estimate_tier_0_count(data)

    # Find remaining duplication clusters (things string matching would catch
    # that are currently separate entities — i.e., missed merges)
    clusters = find_string_duplication_clusters(entities, threshold=0.85)
    report.duplication_clusters = clusters

    # Tier 1 (string only) = current minus clusters that exist
    # But also current count already has some merges done.
    # Estimate: without embedding/LLM tiers, we'd have more duplicates.
    # The clusters we found represent entities that ARE separate currently
    # but string matching thinks they should merge.
    report.tier_1_entities = estimate_tier_1_count(report.tier_3_entities, clusters)

    # Tier 2 is between tier 1 and tier 3 — estimate as midpoint
    # (true measurement requires re-running resolution with embedding only)
    report.tier_2_entities = (report.tier_1_entities + report.tier_3_entities) // 2

    # Duplication rate: fraction of current entities that appear to be duplicates
    total_in_clusters = sum(1 + len(c.members) for c in clusters)
    duplicate_count = sum(len(c.members) for c in clusters)
    report.estimated_duplication_rate = (
        duplicate_count / report.tier_3_entities if report.tier_3_entities else 0
    )

    # Alias coverage
    aliased = [e for e in entities.values() if e.get("aliases")]
    report.entities_with_aliases = len(aliased)
    report.total_aliases = sum(len(e.get("aliases", [])) for e in entities.values())
    report.avg_aliases_per_aliased_entity = (
        report.total_aliases / len(aliased) if aliased else 0
    )

    # Web grounding
    web_verified = [e for e in entities.values() if e.get("web_verified", False)]
    report.web_verified_entities = len(web_verified)
    report.web_verification_rate = (
        len(web_verified) / report.tier_3_entities if report.tier_3_entities else 0
    )

    # Temporal analysis
    report.temporal_buckets = analyze_temporal_resolution(data)

    return report


# ── Pretty printing ───────────────────────────────────────────────────────────

def print_report(report: ResolutionReport):
    """Print the resolution ablation report."""
    print("\n" + "=" * 70)
    print("  PIE ENTITY RESOLUTION ABLATION REPORT")
    print("=" * 70)

    print(f"\n{'─'*50}")
    print("  ENTITY COUNTS BY RESOLUTION TIER")
    print(f"{'─'*50}")
    tiers = [
        ("Tier 0 — No resolution", report.tier_0_entities),
        ("Tier 1 — String match only", report.tier_1_entities),
        ("Tier 2 — String + embedding", report.tier_2_entities),
        ("Tier 3 — Full (actual)", report.tier_3_entities),
    ]
    max_count = max(t[1] for t in tiers) if tiers else 1
    for label, count in tiers:
        bar_len = int(count / max(max_count, 1) * 30)
        bar = "█" * bar_len
        reduction = ""
        if count != tiers[0][1] and tiers[0][1] > 0:
            pct = (1 - count / tiers[0][1]) * 100
            reduction = f"  (−{pct:.0f}%)"
        print(f"  {label:<30} {count:>6}  {bar}{reduction}")

    print(f"\n{'─'*50}")
    print("  DUPLICATION ANALYSIS")
    print(f"{'─'*50}")
    print(f"  Clusters found:        {len(report.duplication_clusters)}")
    print(f"  Estimated dup rate:    {report.estimated_duplication_rate * 100:.1f}%")
    if report.duplication_clusters:
        print(f"\n  Top potential duplicates:")
        for c in sorted(report.duplication_clusters, key=lambda c: len(c.members), reverse=True)[:10]:
            members_str = ", ".join(c.members[:3])
            if len(c.members) > 3:
                members_str += f" (+{len(c.members) - 3} more)"
            print(f"    • {c.canonical} ↔ {members_str}  (sim={c.max_similarity:.2f})")

    print(f"\n{'─'*50}")
    print("  ALIAS COVERAGE")
    print(f"{'─'*50}")
    print(f"  Entities with aliases:   {report.entities_with_aliases}")
    print(f"  Total aliases:           {report.total_aliases}")
    print(f"  Avg aliases/entity:      {report.avg_aliases_per_aliased_entity:.1f}")

    print(f"\n{'─'*50}")
    print("  WEB GROUNDING")
    print(f"{'─'*50}")
    print(f"  Verified entities:   {report.web_verified_entities}")
    print(f"  Verification rate:   {report.web_verification_rate * 100:.1f}%")

    if report.temporal_buckets:
        print(f"\n{'─'*50}")
        print("  RESOLUTION OVER TIME (match rate by quarter)")
        print(f"{'─'*50}")
        print(f"  {'Period':<12} {'Trans':>6} {'Entities':>9} {'New':>5} {'Matched':>8} {'Rate':>7}")
        print(f"  {'─'*12} {'─'*6} {'─'*9} {'─'*5} {'─'*8} {'─'*7}")
        for b in report.temporal_buckets:
            bar = "█" * int(b.match_rate * 20)
            print(
                f"  {b.period_label:<12} {b.total_transitions:>6} "
                f"{b.unique_entities_touched:>9} {b.new_entities:>5} "
                f"{b.matched_entities:>8} {b.match_rate:>6.1%}  {bar}"
            )

    print("\n" + "=" * 70)


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PIE Resolution Ablation — measure entity resolution quality across tiers"
    )
    parser.add_argument(
        "--world-model",
        type=Path,
        default=Path("output/world_model.json"),
        help="Path to world_model.json (default: output/world_model.json)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.85,
        help="String match threshold for duplication detection (default: 0.85)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON instead of human-readable",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write report to file",
    )
    args = parser.parse_args()

    path = args.world_model
    if not path.exists():
        logger.error(f"World model not found at {path}")
        logger.error("Run the ingestion pipeline first: python3 -m pie.ingestion.pipeline")
        sys.exit(1)

    with open(path) as f:
        data = json.load(f)

    logger.info(f"Loaded world model from {path}: {len(data.get('entities', {}))} entities")

    report = analyze(data)

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print_report(report)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        logger.info(f"\nReport saved to {args.output}")


if __name__ == "__main__":
    main()
