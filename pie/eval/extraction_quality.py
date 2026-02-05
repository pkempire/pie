"""
Extraction Quality Report — score the world model's extraction quality.

Loads world_model.json and computes:
  - Entity type distribution
  - Avg transitions per entity
  - Orphan entities (0 relationships)
  - Noise indicators (short names, generic names, single-transition entities)
  - Flagged entities for manual review

Usage:
    python3 -m pie.eval.extraction_quality [--world-model output/world_model.json]

Hypothesis tested: N/A (diagnostic, not hypothesis-driven).
Supports Hypothesis 2 (rolling context) and Hypothesis 8 (daily batch)
by providing quality baselines before/after those changes.
"""

from __future__ import annotations
import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, median, stdev

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("pie.eval.extraction_quality")

# ── Noise detection heuristics ────────────────────────────────────────────────

# Names ≤ this length are suspicious (likely abbreviations extracted wrong)
MIN_NAME_LENGTH = 2

# Names that are almost certainly noise — overly generic terms
GENERIC_NAMES = {
    "user", "assistant", "system", "it", "they", "this", "that",
    "code", "data", "file", "thing", "stuff", "api", "app",
    "function", "class", "variable", "module", "script", "test",
    "project", "task", "issue", "problem", "solution", "idea",
    "the user", "the assistant", "the project", "the system",
    "i", "you", "he", "she", "we", "me", "my",
}

# Entity types where single-word generic names are more acceptable
GENERIC_OK_TYPES = {"concept", "tool"}


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class NoiseFlag:
    """A flagged entity with a reason."""
    entity_id: str
    entity_name: str
    entity_type: str
    reasons: list[str] = field(default_factory=list)
    severity: str = "low"  # low | medium | high


@dataclass
class QualityReport:
    """Full quality report."""
    # Counts
    total_entities: int = 0
    total_transitions: int = 0
    total_relationships: int = 0

    # Type distribution
    type_distribution: dict[str, int] = field(default_factory=dict)

    # Transition stats
    avg_transitions_per_entity: float = 0.0
    median_transitions_per_entity: float = 0.0
    max_transitions_per_entity: int = 0
    max_transitions_entity_name: str = ""
    entities_with_0_transitions_beyond_creation: int = 0
    entities_with_only_creation: int = 0

    # Relationship stats
    orphan_count: int = 0
    orphan_rate: float = 0.0
    avg_relationships_per_entity: float = 0.0

    # Noise
    noise_flags: list[NoiseFlag] = field(default_factory=list)
    noise_rate: float = 0.0

    # Temporal span
    earliest_timestamp: float = 0.0
    latest_timestamp: float = 0.0
    temporal_span_days: float = 0.0

    # Top entities by transitions
    top_entities_by_transitions: list[tuple[str, str, int]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "counts": {
                "entities": self.total_entities,
                "transitions": self.total_transitions,
                "relationships": self.total_relationships,
            },
            "type_distribution": self.type_distribution,
            "transition_stats": {
                "avg_per_entity": round(self.avg_transitions_per_entity, 2),
                "median_per_entity": round(self.median_transitions_per_entity, 2),
                "max_per_entity": self.max_transitions_per_entity,
                "max_entity": self.max_transitions_entity_name,
                "entities_only_creation": self.entities_with_only_creation,
                "entities_no_update": self.entities_with_0_transitions_beyond_creation,
            },
            "relationship_stats": {
                "orphan_count": self.orphan_count,
                "orphan_rate": round(self.orphan_rate, 4),
                "avg_per_entity": round(self.avg_relationships_per_entity, 2),
            },
            "noise": {
                "flagged_count": len(self.noise_flags),
                "noise_rate": round(self.noise_rate, 4),
                "flags": [
                    {
                        "name": f.entity_name,
                        "type": f.entity_type,
                        "reasons": f.reasons,
                        "severity": f.severity,
                    }
                    for f in sorted(self.noise_flags, key=lambda f: f.severity, reverse=True)
                ],
            },
            "temporal": {
                "span_days": round(self.temporal_span_days, 1),
            },
            "top_entities_by_transitions": [
                {"name": name, "type": etype, "transitions": count}
                for name, etype, count in self.top_entities_by_transitions
            ],
        }


# ── Core analysis ─────────────────────────────────────────────────────────────

def load_world_model(path: Path) -> dict:
    """Load the world_model.json file."""
    if not path.exists():
        logger.error(f"World model not found at {path}")
        logger.error("Run the ingestion pipeline first: python3 -m pie.ingestion.pipeline")
        sys.exit(1)

    with open(path) as f:
        data = json.load(f)

    logger.info(f"Loaded world model from {path}")
    return data


def count_transitions_per_entity(data: dict) -> dict[str, list[dict]]:
    """Group transitions by entity_id."""
    by_entity: dict[str, list[dict]] = defaultdict(list)
    for tid, t in data.get("transitions", {}).items():
        by_entity[t["entity_id"]].append(t)
    return by_entity


def count_relationships_per_entity(data: dict) -> dict[str, int]:
    """Count relationships per entity (both directions)."""
    counts: Counter = Counter()
    for rid, r in data.get("relationships", {}).items():
        counts[r["source_id"]] += 1
        counts[r["target_id"]] += 1
    return dict(counts)


def detect_noise(entity_id: str, entity: dict, transitions: list[dict]) -> NoiseFlag | None:
    """Check if an entity is likely noise. Returns a NoiseFlag or None."""
    name = entity.get("name", "")
    etype = entity.get("type", "unknown")
    reasons = []

    # Short name
    if len(name.strip()) <= MIN_NAME_LENGTH:
        reasons.append(f"very_short_name (len={len(name.strip())})")

    # Generic name
    if name.lower().strip() in GENERIC_NAMES:
        if etype not in GENERIC_OK_TYPES:
            reasons.append("generic_name")

    # Single transition (creation only → never referenced again)
    non_creation = [t for t in transitions if t.get("transition_type") != "creation"]
    if len(transitions) <= 1 and len(non_creation) == 0:
        reasons.append("single_transition_only_creation")

    # Empty or trivial state
    state = entity.get("current_state", {})
    if not state or state == {}:
        reasons.append("empty_state")
    elif isinstance(state, dict):
        desc = state.get("description", "")
        if isinstance(desc, str) and len(desc.strip()) < 10:
            reasons.append("trivial_state_description")

    # No aliases, no web grounding, low importance, single occurrence
    # → likely an incidental mention
    if (not entity.get("aliases") and
            not entity.get("web_verified", False) and
            entity.get("importance", 0) < 0.05 and
            len(transitions) <= 1):
        reasons.append("incidental_mention")

    if not reasons:
        return None

    severity = "low"
    if len(reasons) >= 3:
        severity = "high"
    elif len(reasons) >= 2 or "generic_name" in reasons:
        severity = "medium"

    return NoiseFlag(
        entity_id=entity_id,
        entity_name=name,
        entity_type=etype,
        reasons=reasons,
        severity=severity,
    )


def analyze(data: dict) -> QualityReport:
    """Run full quality analysis on the world model."""
    report = QualityReport()

    entities = data.get("entities", {})
    transitions_raw = data.get("transitions", {})
    relationships_raw = data.get("relationships", {})

    report.total_entities = len(entities)
    report.total_transitions = len(transitions_raw)
    report.total_relationships = len(relationships_raw)

    if report.total_entities == 0:
        logger.warning("World model is empty — nothing to analyze.")
        return report

    # ── Type distribution ─────────────────────────────────────────────────
    type_counter = Counter(e.get("type", "unknown") for e in entities.values())
    report.type_distribution = dict(type_counter.most_common())

    # ── Transitions per entity ────────────────────────────────────────────
    trans_by_entity = count_transitions_per_entity(data)
    transition_counts = []
    for eid, entity in entities.items():
        t_list = trans_by_entity.get(eid, [])
        transition_counts.append(len(t_list))

        non_creation = [t for t in t_list if t.get("transition_type") != "creation"]
        if len(non_creation) == 0:
            report.entities_with_0_transitions_beyond_creation += 1
        if len(t_list) <= 1 and all(
            t.get("transition_type") == "creation" for t in t_list
        ):
            report.entities_with_only_creation += 1

    report.avg_transitions_per_entity = mean(transition_counts) if transition_counts else 0
    report.median_transitions_per_entity = median(transition_counts) if transition_counts else 0
    report.max_transitions_per_entity = max(transition_counts) if transition_counts else 0

    # Find entity with max transitions
    if transition_counts:
        max_idx = transition_counts.index(max(transition_counts))
        max_eid = list(entities.keys())[max_idx]
        report.max_transitions_entity_name = entities[max_eid].get("name", max_eid)

    # ── Relationships per entity ──────────────────────────────────────────
    rel_counts = count_relationships_per_entity(data)
    orphans = [eid for eid in entities if rel_counts.get(eid, 0) == 0]
    report.orphan_count = len(orphans)
    report.orphan_rate = len(orphans) / report.total_entities if report.total_entities else 0

    all_rel_counts = [rel_counts.get(eid, 0) for eid in entities]
    report.avg_relationships_per_entity = mean(all_rel_counts) if all_rel_counts else 0

    # ── Noise detection ───────────────────────────────────────────────────
    for eid, entity in entities.items():
        t_list = trans_by_entity.get(eid, [])
        flag = detect_noise(eid, entity, t_list)
        if flag:
            report.noise_flags.append(flag)
    report.noise_rate = len(report.noise_flags) / report.total_entities if report.total_entities else 0

    # ── Temporal span ─────────────────────────────────────────────────────
    all_timestamps = []
    for t in transitions_raw.values():
        ts = t.get("timestamp", 0)
        if ts and ts > 0:
            all_timestamps.append(ts)
    if all_timestamps:
        report.earliest_timestamp = min(all_timestamps)
        report.latest_timestamp = max(all_timestamps)
        report.temporal_span_days = (report.latest_timestamp - report.earliest_timestamp) / 86400

    # ── Top entities by transition count ──────────────────────────────────
    entity_transition_counts = []
    for eid, entity in entities.items():
        t_count = len(trans_by_entity.get(eid, []))
        entity_transition_counts.append((entity.get("name", eid), entity.get("type", "?"), t_count))
    entity_transition_counts.sort(key=lambda x: x[2], reverse=True)
    report.top_entities_by_transitions = entity_transition_counts[:20]

    return report


# ── Pretty printing ───────────────────────────────────────────────────────────

def print_report(report: QualityReport):
    """Print the quality report in a human-readable format."""
    print("\n" + "=" * 70)
    print("  PIE EXTRACTION QUALITY REPORT")
    print("=" * 70)

    print(f"\n{'─'*40}")
    print("  COUNTS")
    print(f"{'─'*40}")
    print(f"  Entities:       {report.total_entities:,}")
    print(f"  Transitions:    {report.total_transitions:,}")
    print(f"  Relationships:  {report.total_relationships:,}")

    print(f"\n{'─'*40}")
    print("  TYPE DISTRIBUTION")
    print(f"{'─'*40}")
    for etype, count in sorted(report.type_distribution.items(), key=lambda x: -x[1]):
        pct = count / report.total_entities * 100 if report.total_entities else 0
        bar = "█" * int(pct / 2)
        print(f"  {etype:<15} {count:>5}  ({pct:5.1f}%)  {bar}")

    print(f"\n{'─'*40}")
    print("  TRANSITION STATS")
    print(f"{'─'*40}")
    print(f"  Avg per entity:      {report.avg_transitions_per_entity:.2f}")
    print(f"  Median per entity:   {report.median_transitions_per_entity:.1f}")
    print(f"  Max per entity:      {report.max_transitions_per_entity}  ({report.max_transitions_entity_name})")
    print(f"  Creation-only:       {report.entities_with_only_creation:,}"
          f"  ({report.entities_with_only_creation / report.total_entities * 100:.1f}% of entities)"
          if report.total_entities else "")
    print(f"  No updates:          {report.entities_with_0_transitions_beyond_creation:,}")

    print(f"\n{'─'*40}")
    print("  RELATIONSHIP STATS")
    print(f"{'─'*40}")
    print(f"  Avg per entity:  {report.avg_relationships_per_entity:.2f}")
    print(f"  Orphans:         {report.orphan_count:,}  ({report.orphan_rate * 100:.1f}% of entities)")

    print(f"\n{'─'*40}")
    print("  NOISE DETECTION")
    print(f"{'─'*40}")
    high = [f for f in report.noise_flags if f.severity == "high"]
    med = [f for f in report.noise_flags if f.severity == "medium"]
    low = [f for f in report.noise_flags if f.severity == "low"]
    print(f"  Flagged entities: {len(report.noise_flags):,}  ({report.noise_rate * 100:.1f}%)")
    print(f"    High severity:    {len(high)}")
    print(f"    Medium severity:  {len(med)}")
    print(f"    Low severity:     {len(low)}")

    if high:
        print(f"\n  {'─'*36}")
        print("  HIGH-SEVERITY FLAGS (likely noise)")
        print(f"  {'─'*36}")
        for f in high[:15]:
            print(f"    • {f.entity_name} ({f.entity_type}): {', '.join(f.reasons)}")
    if med:
        print(f"\n  {'─'*36}")
        print("  MEDIUM-SEVERITY FLAGS (review)")
        print(f"  {'─'*36}")
        for f in med[:15]:
            print(f"    • {f.entity_name} ({f.entity_type}): {', '.join(f.reasons)}")

    if report.top_entities_by_transitions:
        print(f"\n{'─'*40}")
        print("  TOP ENTITIES (by transition count)")
        print(f"{'─'*40}")
        for name, etype, count in report.top_entities_by_transitions[:15]:
            print(f"  {count:>4}  {name} ({etype})")

    if report.temporal_span_days:
        print(f"\n{'─'*40}")
        print("  TEMPORAL SPAN")
        print(f"{'─'*40}")
        print(f"  {report.temporal_span_days:.0f} days of data")

    print("\n" + "=" * 70)


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PIE Extraction Quality Report — analyze world model quality"
    )
    parser.add_argument(
        "--world-model",
        type=Path,
        default=Path("output/world_model.json"),
        help="Path to world_model.json (default: output/world_model.json)",
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
        help="Write report to file (in addition to stdout)",
    )
    args = parser.parse_args()

    data = load_world_model(args.world_model)
    report = analyze(data)

    if args.json:
        output = json.dumps(report.to_dict(), indent=2)
        print(output)
    else:
        print_report(report)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        logger.info(f"\nReport saved to {args.output}")


if __name__ == "__main__":
    main()
