#!/usr/bin/env python3
from __future__ import annotations
"""
Consolidation Engine — PIE's "sleep cycle."

Instead of processing observations in real-time, this runs OFFLINE over the
full world model to find patterns that only emerge across many entities and
long time horizons.  Biological analogy: hippocampal replay during sleep.

Three layers:
  Layer 1 — Pattern Extraction:  Find recurring behavioral patterns across
            entity timelines using LLM-based semantic analysis.
  Layer 2 — Concept Compression: Cluster similar entities into archetypes
            that represent "the same kind of thing."
  Layer 3 — Forward Simulation:  Given patterns + archetypes + current state,
            predict what happens next.

Usage:
    python tools/consolidate.py                        # Full analysis (dry run)
    python tools/consolidate.py --apply                # Write patterns back to WM
    python tools/consolidate.py --layer 1              # Only run pattern extraction
    python tools/consolidate.py --layer 2              # Only run compression
    python tools/consolidate.py --layer 3              # Only run prediction
"""

import argparse
import json
import sys
import time
import logging
from pathlib import Path
from collections import defaultdict, Counter
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pie.core.world_model import WorldModel
from pie.core.models import Entity, StateTransition, EntityType
from pie.core.llm import LLMClient

logger = logging.getLogger("pie.consolidation")


# ═══════════════════════════════════════════════════════════════════════════
# Data structures for consolidation output
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class BehavioralPattern:
    """A recurring pattern discovered across entity timelines."""
    id: str
    name: str                          # "Exploration-then-pivot"
    description: str                   # Human-readable description
    evidence_entity_ids: list[str]     # Entities that exhibit this pattern
    pattern_type: str                  # "lifecycle", "cascade", "habit", "rhythm"
    frequency: int = 0                 # How many entities show this
    avg_duration_days: float = 0.0     # How long the pattern typically takes
    triggers: list[str] = field(default_factory=list)  # What starts this pattern
    phases: list[dict] = field(default_factory=list)    # Ordered phase descriptions
    confidence: float = 0.0


@dataclass
class EntityArchetype:
    """A cluster of similar entities compressed into a prototype."""
    id: str
    name: str                          # "Weekend side project"
    description: str
    entity_type: str                   # EntityType value
    member_entity_ids: list[str]       # Which entities belong here
    member_count: int = 0
    avg_transitions: float = 0.0
    avg_lifespan_days: float = 0.0
    typical_lifecycle: str = ""        # Pattern name this archetype follows
    defining_features: list[str] = field(default_factory=list)


@dataclass
class Prediction:
    """A forward prediction about what will happen next."""
    entity_id: str
    entity_name: str
    prediction: str                    # What will happen
    timeframe: str                     # When
    confidence: float = 0.0
    reasoning: str = ""
    based_on_pattern: str = ""         # Which pattern drives this prediction
    based_on_archetype: str = ""       # Which archetype this entity matches


@dataclass
class ConsolidationReport:
    """Full output of a consolidation cycle."""
    timestamp: float = field(default_factory=time.time)
    patterns: list[BehavioralPattern] = field(default_factory=list)
    archetypes: list[EntityArchetype] = field(default_factory=list)
    predictions: list[Prediction] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════
# Layer 1: Pattern Extraction
# ═══════════════════════════════════════════════════════════════════════════

def _build_entity_timeline(wm: WorldModel, entity_id: str) -> dict:
    """Build a rich timeline summary for an entity."""
    entity = wm.entities.get(entity_id)
    if not entity:
        return {}

    transitions = wm.get_transitions(entity_id, ordered=True)
    if not transitions:
        return {}

    first_ts = transitions[0].timestamp
    last_ts = transitions[-1].timestamp
    span_days = (last_ts - first_ts) / 86400

    # Extract meaningful trigger summaries (skip generic "Updated from X batch")
    meaningful_triggers = []
    for t in transitions:
        trigger = t.trigger_summary or ""
        if trigger and not trigger.startswith("Updated from ") and not trigger.startswith("Fields updated"):
            meaningful_triggers.append({
                "date": datetime.fromtimestamp(t.timestamp, tz=timezone.utc).strftime("%Y-%m-%d"),
                "type": t.transition_type.value,
                "summary": trigger[:200],
            })

    # Compute activity density over time
    if span_days > 0 and len(transitions) >= 3:
        third = len(transitions) // 3
        first_third_density = third / max((transitions[third].timestamp - first_ts) / 86400, 0.1)
        last_third_density = third / max((last_ts - transitions[-third].timestamp) / 86400, 0.1)
    else:
        first_third_density = last_third_density = 0

    # Relationships
    rels = []
    for rid in wm._entity_relationships.get(entity_id, []):
        rel = wm.relationships.get(rid)
        if rel:
            other_id = rel.target_id if rel.source_id == entity_id else rel.source_id
            other = wm.entities.get(other_id)
            if other:
                rels.append(f"{rel.type.value}: {other.name}")

    return {
        "name": entity.name,
        "type": entity.type.value,
        "transitions": len(transitions),
        "span_days": round(span_days, 1),
        "first_seen": datetime.fromtimestamp(first_ts, tz=timezone.utc).strftime("%Y-%m-%d"),
        "last_seen": datetime.fromtimestamp(last_ts, tz=timezone.utc).strftime("%Y-%m-%d"),
        "current_state": {k: str(v)[:100] for k, v in entity.current_state.items()},
        "key_events": meaningful_triggers[:10],
        "early_density": round(first_third_density, 2),
        "late_density": round(last_third_density, 2),
        "relationships": rels[:8],
    }


PATTERN_EXTRACTION_PROMPT = """You are analyzing entity timelines from a personal knowledge graph.
Each entity represents something in a person's life (projects, tools, decisions, etc.) with a
recorded history of state changes over time.

Your task: find BEHAVIORAL PATTERNS that recur across multiple entities. Not surface-level
observations ("things get updated") — deep structural patterns about HOW this person engages
with things over time.

Examples of real patterns:
- "Exploration burst → scope narrowing → build phase → ship or abandon"
- "Intense 3-day deep dive, then 2+ weeks of silence, then either return or forget"
- "Starts with tool evaluation (3+ options), picks one, goes all-in, then switches"
- "Cascading project spawning — one project idea triggers 2-3 related ones within a week"

## Entity Timelines:

{timelines}

## Task:
Analyze these timelines and extract 3-8 behavioral patterns. For each pattern:

1. Give it a clear name
2. Describe the phases it goes through
3. List which entities from above demonstrate it
4. Estimate how long it typically takes
5. What triggers it to start
6. How confident are you (0-1)

Respond as JSON:
{{
  "patterns": [
    {{
      "name": "pattern name",
      "description": "what this pattern looks like",
      "type": "lifecycle|cascade|habit|rhythm",
      "phases": ["phase 1 description", "phase 2 description", ...],
      "evidence_entities": ["entity name 1", "entity name 2"],
      "avg_duration_days": 14,
      "triggers": ["what starts this pattern"],
      "confidence": 0.8
    }}
  ]
}}"""


def extract_patterns(wm: WorldModel, llm: LLMClient) -> list[BehavioralPattern]:
    """Layer 1: Extract behavioral patterns from entity timelines."""
    print("\n  Layer 1: Pattern Extraction")
    print("  " + "─" * 50)

    # Select entities with rich enough histories (5+ transitions)
    rich_entities = []
    for eid, entity in wm.entities.items():
        transitions = wm.get_transitions(eid, ordered=True)
        if len(transitions) >= 5:
            rich_entities.append(eid)

    print(f"    Entities with 5+ transitions: {len(rich_entities)}")

    if len(rich_entities) < 3:
        print("    Not enough rich entities for pattern extraction.")
        return []

    # Build timelines
    timelines = []
    for eid in rich_entities:
        tl = _build_entity_timeline(wm, eid)
        if tl:
            timelines.append(tl)

    # Process in batches to fit context window (15 entities per batch)
    all_patterns: list[BehavioralPattern] = []
    BATCH_SIZE = 15

    for i in range(0, len(timelines), BATCH_SIZE):
        batch = timelines[i:i + BATCH_SIZE]
        timelines_text = json.dumps(batch, indent=2)

        prompt = PATTERN_EXTRACTION_PROMPT.format(timelines=timelines_text)

        print(f"    Analyzing batch {i // BATCH_SIZE + 1} ({len(batch)} entities)...")

        try:
            result = llm.chat(
                messages=[{"role": "user", "content": prompt}],
                model="gpt-5-mini",
                json_mode=True,
            )
            raw = result["content"]
            patterns = raw.get("patterns", [])

            for j, p in enumerate(patterns):
                # Map evidence entity names to IDs
                evidence_ids = []
                for name in p.get("evidence_entities", []):
                    found = wm.find_by_name(name)
                    if found:
                        evidence_ids.append(found.id)

                bp = BehavioralPattern(
                    id=f"pattern_{i}_{j}",
                    name=p.get("name", ""),
                    description=p.get("description", ""),
                    evidence_entity_ids=evidence_ids,
                    pattern_type=p.get("type", "lifecycle"),
                    frequency=len(evidence_ids),
                    avg_duration_days=p.get("avg_duration_days", 0),
                    triggers=p.get("triggers", []),
                    phases=[{"description": ph} for ph in p.get("phases", [])],
                    confidence=p.get("confidence", 0.5),
                )
                all_patterns.append(bp)
                print(f"      → {bp.name} ({bp.frequency} entities, conf={bp.confidence})")

        except Exception as e:
            print(f"    ERROR in batch {i // BATCH_SIZE + 1}: {e}")
            continue

    # Merge similar patterns across batches
    # (Two patterns with similar names/descriptions from different batches = same pattern)
    merged = _merge_similar_patterns(all_patterns)
    print(f"    Extracted {len(merged)} unique patterns ({len(all_patterns)} before merge)")
    return merged


def _merge_similar_patterns(patterns: list[BehavioralPattern]) -> list[BehavioralPattern]:
    """Merge patterns with similar names across batches."""
    if len(patterns) <= 1:
        return patterns

    from difflib import SequenceMatcher

    merged = []
    used = set()

    for i, p1 in enumerate(patterns):
        if i in used:
            continue
        # Find similar patterns
        group = [p1]
        for j, p2 in enumerate(patterns):
            if j <= i or j in used:
                continue
            ratio = SequenceMatcher(None, p1.name.lower(), p2.name.lower()).ratio()
            if ratio > 0.6:
                group.append(p2)
                used.add(j)
        used.add(i)

        # Merge group into one pattern
        if len(group) == 1:
            merged.append(p1)
        else:
            # Combine evidence entities
            all_evidence = []
            for p in group:
                all_evidence.extend(p.evidence_entity_ids)
            all_evidence = list(set(all_evidence))

            best = max(group, key=lambda p: p.confidence)
            best.evidence_entity_ids = all_evidence
            best.frequency = len(all_evidence)
            merged.append(best)

    return merged


# ═══════════════════════════════════════════════════════════════════════════
# Layer 2: Concept Compression
# ═══════════════════════════════════════════════════════════════════════════

COMPRESSION_PROMPT = """You are analyzing a set of entities from a personal knowledge graph,
all of the same type: {entity_type}.

Your task: group these into ARCHETYPES — clusters of entities that represent "the same kind of
thing" in this person's life. Not by topic, but by behavioral signature.

For example, among "project" entities you might find:
- "Weekend spike" — intense 2-5 day burst, 5-15 transitions, then silence
- "Long-running workhorse" — steady activity over months, core to the person's work
- "Exploratory pivot chain" — started as one thing, morphed into something else 2-3 times
- "Tool evaluation" — appeared briefly while comparing options, then either adopted or dropped

## Entities ({entity_type}):

{entities_json}

## Discovered behavioral patterns (from Layer 1):

{patterns_json}

## Task:
Group these entities into 2-6 archetypes. For each:

1. Name the archetype
2. Describe what defines it (behavioral signature, not topic)
3. List which entities belong
4. What lifecycle pattern does it follow?
5. Key distinguishing features

Respond as JSON:
{{
  "archetypes": [
    {{
      "name": "archetype name",
      "description": "behavioral definition",
      "member_entities": ["entity name 1", "entity name 2"],
      "typical_lifecycle": "pattern name from above or new",
      "avg_transitions": 8.5,
      "avg_lifespan_days": 30,
      "defining_features": ["feature 1", "feature 2"]
    }}
  ]
}}"""


def compress_entities(
    wm: WorldModel,
    llm: LLMClient,
    patterns: list[BehavioralPattern],
) -> list[EntityArchetype]:
    """Layer 2: Cluster entities into behavioral archetypes."""
    print("\n  Layer 2: Concept Compression")
    print("  " + "─" * 50)

    # Group entities by type, only types with 5+ entities worth clustering
    type_groups: dict[str, list[str]] = defaultdict(list)
    for eid, entity in wm.entities.items():
        type_groups[entity.type.value].append(eid)

    all_archetypes: list[EntityArchetype] = []
    patterns_json = json.dumps([
        {"name": p.name, "description": p.description, "phases": [ph["description"] for ph in p.phases]}
        for p in patterns
    ], indent=2)

    for etype, eids in type_groups.items():
        if len(eids) < 5:
            continue

        print(f"    Clustering {etype} ({len(eids)} entities)...")

        # Build entity summaries for the LLM
        entity_summaries = []
        for eid in eids:
            entity = wm.entities[eid]
            transitions = wm.get_transitions(eid, ordered=True)
            if not transitions:
                continue

            span = (transitions[-1].timestamp - transitions[0].timestamp) / 86400 if len(transitions) > 1 else 0
            triggers = [
                t.trigger_summary[:100]
                for t in transitions
                if t.trigger_summary
                and not t.trigger_summary.startswith("Updated from ")
                and not t.trigger_summary.startswith("Fields updated")
            ]

            entity_summaries.append({
                "name": entity.name,
                "transitions": len(transitions),
                "span_days": round(span, 1),
                "first_seen": datetime.fromtimestamp(
                    transitions[0].timestamp, tz=timezone.utc
                ).strftime("%Y-%m-%d"),
                "key_events": triggers[:5],
                "current_state_summary": str(entity.current_state.get("description", ""))[:150],
            })

        # Only send entities with 2+ transitions for meaningful clustering
        entity_summaries = [e for e in entity_summaries if e["transitions"] >= 2]
        if len(entity_summaries) < 5:
            print(f"      Skipped — too few entities with 2+ transitions")
            continue

        # Batch if needed (30 entities per call)
        BATCH_SIZE = 30
        for i in range(0, len(entity_summaries), BATCH_SIZE):
            batch = entity_summaries[i:i + BATCH_SIZE]

            prompt = COMPRESSION_PROMPT.format(
                entity_type=etype,
                entities_json=json.dumps(batch, indent=2),
                patterns_json=patterns_json,
            )

            try:
                result = llm.chat(
                    messages=[{"role": "user", "content": prompt}],
                    model="gpt-5-mini",
                    json_mode=True,
                )
                raw = result["content"]

                for j, a in enumerate(raw.get("archetypes", [])):
                    member_ids = []
                    for name in a.get("member_entities", []):
                        found = wm.find_by_name(name)
                        if found:
                            member_ids.append(found.id)

                    archetype = EntityArchetype(
                        id=f"archetype_{etype}_{i}_{j}",
                        name=a.get("name", ""),
                        description=a.get("description", ""),
                        entity_type=etype,
                        member_entity_ids=member_ids,
                        member_count=len(member_ids),
                        avg_transitions=a.get("avg_transitions", 0),
                        avg_lifespan_days=a.get("avg_lifespan_days", 0),
                        typical_lifecycle=a.get("typical_lifecycle", ""),
                        defining_features=a.get("defining_features", []),
                    )
                    all_archetypes.append(archetype)
                    print(f"      → {archetype.name}: {archetype.member_count} entities")

            except Exception as e:
                print(f"      ERROR: {e}")
                continue

    print(f"    Total archetypes: {len(all_archetypes)}")
    return all_archetypes


# ═══════════════════════════════════════════════════════════════════════════
# Layer 3: Forward Simulation
# ═══════════════════════════════════════════════════════════════════════════

PREDICTION_PROMPT = """You are a behavioral prediction engine. Given a person's discovered
behavioral patterns, entity archetypes, and the current state of their active entities,
predict what is likely to happen in the next 2-4 weeks.

## Behavioral Patterns Discovered:

{patterns_text}

## Entity Archetypes:

{archetypes_text}

## Currently Active Entities (recent activity in last 30 days):

{active_entities}

## Task:
Generate 5-10 specific, actionable predictions. Each should:

1. Name the specific entity affected
2. What will happen to it
3. When (relative timeframe)
4. Why (which pattern/archetype drives this prediction)
5. Confidence (0-1)

Be specific and non-obvious. "Projects will be updated" is useless.
"Pulse-Fi will hit a compute wall within 2 weeks and pivot to a lighter
sensor approach based on the exploration-then-simplify pattern" is useful.

Respond as JSON:
{{
  "predictions": [
    {{
      "entity_name": "exact entity name",
      "prediction": "what will happen",
      "timeframe": "when",
      "confidence": 0.7,
      "reasoning": "why — which pattern/archetype",
      "based_on_pattern": "pattern name",
      "based_on_archetype": "archetype name or empty"
    }}
  ]
}}"""


def predict_forward(
    wm: WorldModel,
    llm: LLMClient,
    patterns: list[BehavioralPattern],
    archetypes: list[EntityArchetype],
) -> list[Prediction]:
    """Layer 3: Forward simulation — predict what happens next."""
    print("\n  Layer 3: Forward Simulation")
    print("  " + "─" * 50)

    # Find currently active entities (transitions in last 30 days relative to latest data)
    all_timestamps = [t.timestamp for t in wm.transitions.values()]
    if not all_timestamps:
        print("    No transitions found.")
        return []
    latest_ts = max(all_timestamps)
    cutoff = latest_ts - (30 * 86400)  # 30 days before latest

    active_entities = []
    for eid, entity in wm.entities.items():
        transitions = wm.get_transitions(eid, ordered=True)
        if transitions and transitions[-1].timestamp >= cutoff:
            recent_triggers = [
                t.trigger_summary[:100]
                for t in transitions[-5:]
                if t.trigger_summary
                and not t.trigger_summary.startswith("Updated from ")
                and not t.trigger_summary.startswith("Fields updated")
            ]
            active_entities.append({
                "name": entity.name,
                "type": entity.type.value,
                "total_transitions": len(transitions),
                "last_activity": datetime.fromtimestamp(
                    transitions[-1].timestamp, tz=timezone.utc
                ).strftime("%Y-%m-%d"),
                "recent_events": recent_triggers,
                "current_state": str(entity.current_state.get("description", ""))[:200],
            })

    active_entities.sort(key=lambda e: e["total_transitions"], reverse=True)
    print(f"    Active entities (last 30 days): {len(active_entities)}")

    if not active_entities:
        print("    No active entities found.")
        return []

    # Format inputs
    patterns_text = "\n".join(
        f"- **{p.name}** ({p.pattern_type}): {p.description}\n"
        f"  Phases: {' → '.join(ph['description'] for ph in p.phases)}\n"
        f"  Duration: ~{p.avg_duration_days:.0f} days | Frequency: {p.frequency} entities"
        for p in patterns
    )

    archetypes_text = "\n".join(
        f"- **{a.name}** ({a.entity_type}): {a.description}\n"
        f"  Lifecycle: {a.typical_lifecycle} | Avg span: {a.avg_lifespan_days:.0f} days\n"
        f"  Features: {', '.join(a.defining_features[:3])}"
        for a in archetypes
    )

    # Cap at 30 most active for context window
    active_json = json.dumps(active_entities[:30], indent=2)

    prompt = PREDICTION_PROMPT.format(
        patterns_text=patterns_text or "(No patterns extracted yet)",
        archetypes_text=archetypes_text or "(No archetypes extracted yet)",
        active_entities=active_json,
    )

    try:
        result = llm.chat(
            messages=[{"role": "user", "content": prompt}],
            model="gpt-5-mini",
            json_mode=True,
        )
        raw = result["content"]

        predictions = []
        for p in raw.get("predictions", []):
            # Resolve entity
            entity = wm.find_by_name(p.get("entity_name", ""))

            pred = Prediction(
                entity_id=entity.id if entity else "",
                entity_name=p.get("entity_name", ""),
                prediction=p.get("prediction", ""),
                timeframe=p.get("timeframe", ""),
                confidence=p.get("confidence", 0.5),
                reasoning=p.get("reasoning", ""),
                based_on_pattern=p.get("based_on_pattern", ""),
                based_on_archetype=p.get("based_on_archetype", ""),
            )
            predictions.append(pred)
            print(f"    → [{pred.confidence:.0%}] {pred.entity_name}: {pred.prediction[:80]}")

        return predictions

    except Exception as e:
        print(f"    ERROR: {e}")
        return []


# ═══════════════════════════════════════════════════════════════════════════
# Main orchestrator
# ═══════════════════════════════════════════════════════════════════════════

def run_consolidation(
    wm: WorldModel,
    llm: LLMClient,
    layers: list[int] | None = None,
) -> ConsolidationReport:
    """Run the full consolidation cycle (or specific layers)."""
    run_layers = layers or [1, 2, 3]
    report = ConsolidationReport()

    t0 = time.time()

    # Layer 1: Pattern extraction
    patterns = []
    if 1 in run_layers:
        patterns = extract_patterns(wm, llm)
        report.patterns = patterns
    else:
        print("\n  Layer 1: Skipped")

    # Layer 2: Concept compression
    archetypes = []
    if 2 in run_layers:
        archetypes = compress_entities(wm, llm, patterns)
        report.archetypes = archetypes
    else:
        print("\n  Layer 2: Skipped")

    # Layer 3: Forward simulation
    if 3 in run_layers:
        predictions = predict_forward(wm, llm, patterns, archetypes)
        report.predictions = predictions
    else:
        print("\n  Layer 3: Skipped")

    elapsed = time.time() - t0
    report.stats = {
        "layers_run": run_layers,
        "patterns_found": len(report.patterns),
        "archetypes_found": len(report.archetypes),
        "predictions_made": len(report.predictions),
        "elapsed_seconds": round(elapsed, 1),
        "llm_calls": llm.stats["total_calls"],
        "tokens_used": llm.stats["total_tokens"],
    }

    return report


def main():
    parser = argparse.ArgumentParser(description="PIE Consolidation — offline sleep cycle")
    parser.add_argument("--output", type=str, default="./output", help="World model directory")
    parser.add_argument("--layer", type=int, nargs="*", help="Run specific layers (1, 2, 3)")
    parser.add_argument("--apply", action="store_true", help="Write patterns back to world model")
    parser.add_argument("--report", type=str, default=None, help="Save report to JSON file")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s", datefmt="%H:%M:%S")

    wm_path = Path(args.output) / "world_model.json"
    if not wm_path.exists():
        print(f"No world model at {wm_path}")
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print(f"  PIE CONSOLIDATION — Sleep Cycle")
    print(f"{'=' * 60}")

    wm = WorldModel(persist_path=wm_path)
    llm = LLMClient()
    print(f"  World model: {len(wm.entities)} entities, {len(wm.transitions)} transitions")

    layers = args.layer if args.layer else None
    report = run_consolidation(wm, llm, layers)

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"  CONSOLIDATION COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Patterns found:    {report.stats['patterns_found']}")
    print(f"  Archetypes found:  {report.stats['archetypes_found']}")
    print(f"  Predictions made:  {report.stats['predictions_made']}")
    print(f"  Time:              {report.stats['elapsed_seconds']:.0f}s")
    print(f"  Tokens used:       {report.stats['tokens_used']:,}")

    if report.patterns:
        print(f"\n  Behavioral Patterns:")
        for p in report.patterns:
            print(f"    • {p.name} ({p.pattern_type}) — {p.frequency} entities, conf={p.confidence:.0%}")
            if p.phases:
                phases_str = " → ".join(ph["description"][:40] for ph in p.phases[:4])
                print(f"      {phases_str}")

    if report.archetypes:
        print(f"\n  Entity Archetypes:")
        for a in report.archetypes:
            print(f"    • {a.name} ({a.entity_type}): {a.member_count} entities, ~{a.avg_lifespan_days:.0f}d lifespan")

    if report.predictions:
        print(f"\n  Forward Predictions:")
        for p in report.predictions:
            print(f"    • [{p.confidence:.0%}] {p.entity_name}: {p.prediction[:80]}")

    # Save report
    report_path = args.report or str(Path(args.output) / "consolidation_report.json")
    report_dict = {
        "timestamp": report.timestamp,
        "stats": report.stats,
        "patterns": [asdict(p) for p in report.patterns],
        "archetypes": [asdict(a) for a in report.archetypes],
        "predictions": [asdict(p) for p in report.predictions],
    }
    Path(report_path).write_text(json.dumps(report_dict, indent=2))
    print(f"\n  Report saved to {report_path}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
