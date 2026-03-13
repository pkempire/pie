"""
Procedural Memory Extraction — detect behavioral patterns from entity lifecycle analysis.

This is the key novel capability: analyzing transition sequences across entities
to extract recurring behavioral patterns the user may not be aware of.
"""

from __future__ import annotations
import json
import logging
from datetime import datetime, timedelta
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("pie.analysis")


@dataclass
class TransitionPattern:
    """A detected behavioral pattern from cross-entity lifecycle analysis."""
    name: str
    description: str
    sequence: list[str]            # e.g., ["creation", "update", "update", "contradiction", "update"]
    entity_ids: list[str]          # entities that match this pattern
    entity_names: list[str]
    occurrences: int
    avg_duration_days: float       # average time from first to last transition
    avg_steps: float               # average number of transitions
    confidence: float              # how consistent the pattern is


@dataclass
class EntityLifecycle:
    """Summary of an entity's lifecycle for pattern analysis."""
    entity_id: str
    entity_name: str
    entity_type: str
    transitions: list[dict]
    duration_days: float
    transition_types: list[str]    # ordered sequence of transition types
    has_contradiction: bool
    has_resolution: bool
    is_active: bool                # had activity in last 30 days of data


def load_world_model(path: str = "output/world_model.json") -> dict:
    """Load the world model from JSON."""
    with open(path) as f:
        return json.load(f)


def extract_lifecycles(wm: dict) -> list[EntityLifecycle]:
    """Extract entity lifecycles from world model data."""
    entities = wm.get("entities", {})
    transitions = wm.get("transitions", {})

    # Group transitions by entity
    entity_transitions = defaultdict(list)
    for t in transitions.values():
        entity_transitions[t["entity_id"]].append(t)

    # Sort each entity's transitions by timestamp
    for eid in entity_transitions:
        entity_transitions[eid].sort(key=lambda t: t.get("timestamp", 0))

    lifecycles = []
    max_timestamp = max((e.get("last_seen", 0) for e in entities.values()), default=0)

    for eid, entity in entities.items():
        trans = entity_transitions.get(eid, [])
        if len(trans) < 2:
            continue  # Need at least 2 transitions for a lifecycle

        first_ts = trans[0].get("timestamp", 0)
        last_ts = trans[-1].get("timestamp", 0)
        duration = (last_ts - first_ts) / 86400 if first_ts and last_ts else 0

        transition_types = [t.get("transition_type", "unknown") for t in trans]

        lifecycle = EntityLifecycle(
            entity_id=eid,
            entity_name=entity.get("name", "unknown"),
            entity_type=entity.get("type", "unknown"),
            transitions=trans,
            duration_days=duration,
            transition_types=transition_types,
            has_contradiction="contradiction" in transition_types,
            has_resolution="resolution" in transition_types,
            is_active=(max_timestamp - last_ts) < 30 * 86400,
        )
        lifecycles.append(lifecycle)

    return lifecycles


def detect_patterns(lifecycles: list[EntityLifecycle], min_occurrences: int = 2) -> list[TransitionPattern]:
    """
    Detect recurring patterns in entity lifecycles.

    Patterns detected:
    1. Rapid exploration → commitment (multiple updates then stable)
    2. Evaluation → contradiction → pivot (tried something, it failed, switched)
    3. Creation-only entities (mentioned once, never updated — noise?)
    4. High-velocity entities (many transitions in short time)
    5. Decision reversal patterns
    """
    patterns = []

    # Pattern 1: Evaluate → Commit — entities with 3+ updates that stabilized
    evaluate_commit = []
    for lc in lifecycles:
        if (lc.entity_type in ("tool", "project", "decision") and
            len(lc.transitions) >= 3 and
            not lc.has_contradiction and
            lc.transition_types.count("update") >= 2):
            evaluate_commit.append(lc)

    if len(evaluate_commit) >= min_occurrences:
        patterns.append(TransitionPattern(
            name="Evaluate → Commit",
            description="User explores a technology/project through multiple updates without contradiction, indicating steady progression",
            sequence=["creation", "update", "update+", "stable"],
            entity_ids=[lc.entity_id for lc in evaluate_commit],
            entity_names=[lc.entity_name for lc in evaluate_commit],
            occurrences=len(evaluate_commit),
            avg_duration_days=sum(lc.duration_days for lc in evaluate_commit) / len(evaluate_commit),
            avg_steps=sum(len(lc.transitions) for lc in evaluate_commit) / len(evaluate_commit),
            confidence=min(1.0, len(evaluate_commit) / 5),
        ))

    # Pattern 2: Try → Fail → Pivot — entities with contradiction
    try_fail_pivot = []
    for lc in lifecycles:
        if lc.has_contradiction:
            try_fail_pivot.append(lc)

    if len(try_fail_pivot) >= min_occurrences:
        patterns.append(TransitionPattern(
            name="Try → Fail → Pivot",
            description="User committed to an approach, hit a contradiction, and changed course",
            sequence=["creation", "update*", "contradiction", "update"],
            entity_ids=[lc.entity_id for lc in try_fail_pivot],
            entity_names=[lc.entity_name for lc in try_fail_pivot],
            occurrences=len(try_fail_pivot),
            avg_duration_days=sum(lc.duration_days for lc in try_fail_pivot) / len(try_fail_pivot),
            avg_steps=sum(len(lc.transitions) for lc in try_fail_pivot) / len(try_fail_pivot),
            confidence=min(1.0, len(try_fail_pivot) / 3),
        ))

    # Pattern 3: High velocity entities — lots of changes in short time
    high_velocity = []
    for lc in lifecycles:
        if lc.duration_days > 0:
            velocity = len(lc.transitions) / max(lc.duration_days, 1)
            if velocity > 0.5 and len(lc.transitions) >= 3:  # More than 1 transition per 2 days
                high_velocity.append((lc, velocity))

    if len(high_velocity) >= min_occurrences:
        hvs = [lc for lc, _ in high_velocity]
        patterns.append(TransitionPattern(
            name="High Velocity Iteration",
            description="Entities that change state rapidly — indicates active experimentation or rapid iteration",
            sequence=["creation", "update", "update", "update..."],
            entity_ids=[lc.entity_id for lc in hvs],
            entity_names=[lc.entity_name for lc in hvs],
            occurrences=len(hvs),
            avg_duration_days=sum(lc.duration_days for lc in hvs) / len(hvs),
            avg_steps=sum(len(lc.transitions) for lc in hvs) / len(hvs),
            confidence=min(1.0, len(hvs) / 5),
        ))

    # Pattern 4: Type-specific patterns — group by entity type
    by_type = defaultdict(list)
    for lc in lifecycles:
        by_type[lc.entity_type].append(lc)

    for etype, type_lcs in by_type.items():
        if len(type_lcs) >= 3:
            avg_transitions = sum(len(lc.transitions) for lc in type_lcs) / len(type_lcs)
            avg_duration = sum(lc.duration_days for lc in type_lcs) / len(type_lcs)
            contradiction_rate = sum(1 for lc in type_lcs if lc.has_contradiction) / len(type_lcs)

            patterns.append(TransitionPattern(
                name=f"{etype.title()} Lifecycle Profile",
                description=f"Typical lifecycle for {etype} entities: avg {avg_transitions:.1f} transitions over {avg_duration:.1f} days, {contradiction_rate:.0%} hit contradictions",
                sequence=[f"avg {avg_transitions:.1f} transitions"],
                entity_ids=[lc.entity_id for lc in type_lcs],
                entity_names=[lc.entity_name for lc in type_lcs],
                occurrences=len(type_lcs),
                avg_duration_days=avg_duration,
                avg_steps=avg_transitions,
                confidence=min(1.0, len(type_lcs) / 5),
            ))

    return patterns


def predict_future_state(
    entity_type: str,
    current_transitions: int,
    patterns: list[TransitionPattern],
) -> dict:
    """
    Given an entity's current state, predict likely future based on patterns.

    This is basic future state prediction from procedural memory.
    """
    relevant_patterns = [p for p in patterns if entity_type.lower() in p.name.lower()]

    if not relevant_patterns:
        return {"prediction": "insufficient data", "confidence": 0.0}

    # Use the most confident relevant pattern
    best = max(relevant_patterns, key=lambda p: p.confidence)

    remaining_transitions = best.avg_steps - current_transitions
    remaining_days = best.avg_duration_days * (remaining_transitions / best.avg_steps) if best.avg_steps > 0 else 0

    return {
        "prediction": f"Based on '{best.name}' pattern ({best.occurrences} observations)",
        "expected_total_transitions": best.avg_steps,
        "remaining_transitions": max(0, remaining_transitions),
        "expected_remaining_days": max(0, remaining_days),
        "contradiction_likely": any("contradiction" in p.sequence for p in relevant_patterns),
        "confidence": best.confidence,
    }


def run_analysis(wm_path: str = "output/world_model.json"):
    """Run full procedural memory analysis and print results."""
    print("\n" + "=" * 70)
    print("PROCEDURAL MEMORY ANALYSIS")
    print("Extracting behavioral patterns from entity lifecycles")
    print("=" * 70)

    wm = load_world_model(wm_path)
    lifecycles = extract_lifecycles(wm)

    print(f"\n  Entities with 2+ transitions: {len(lifecycles)}")
    print(f"  Total entities: {len(wm.get('entities', {}))}")

    # Lifecycle summary
    print("\n--- Entity Lifecycles ---")
    for lc in sorted(lifecycles, key=lambda l: len(l.transitions), reverse=True)[:15]:
        marker = " ⚠️ HAS CONTRADICTION" if lc.has_contradiction else ""
        types_str = " → ".join(lc.transition_types)
        print(f"  {lc.entity_name:40} ({lc.entity_type:10}) {len(lc.transitions)} steps, {lc.duration_days:.0f}d{marker}")
        print(f"    Sequence: {types_str}")

    # Pattern detection
    patterns = detect_patterns(lifecycles)

    print(f"\n--- Detected Patterns ({len(patterns)}) ---")
    for p in patterns:
        print(f"\n  📋 {p.name}")
        print(f"     {p.description}")
        print(f"     Occurrences: {p.occurrences} | Avg duration: {p.avg_duration_days:.1f}d | Avg steps: {p.avg_steps:.1f}")
        print(f"     Confidence: {p.confidence:.0%}")
        print(f"     Entities: {', '.join(p.entity_names[:5])}")
        if len(p.entity_names) > 5:
            print(f"                ... and {len(p.entity_names) - 5} more")

    # Future state prediction demo
    print(f"\n--- Future State Predictions ---")
    test_cases = [
        ("tool", 2, "New tool being evaluated (2 transitions so far)"),
        ("project", 3, "Project in active development (3 transitions)"),
        ("decision", 1, "Recent decision (1 transition)"),
    ]

    for etype, current_trans, desc in test_cases:
        pred = predict_future_state(etype, current_trans, patterns)
        print(f"\n  Scenario: {desc}")
        print(f"  Prediction: {pred['prediction']}")
        if pred['confidence'] > 0:
            print(f"  Expected remaining transitions: {pred['remaining_transitions']:.1f}")
            print(f"  Expected remaining days: {pred['expected_remaining_days']:.0f}")
            print(f"  Contradiction likely: {pred['contradiction_likely']}")
            print(f"  Confidence: {pred['confidence']:.0%}")


if __name__ == "__main__":
    run_analysis()
