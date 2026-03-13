#!/usr/bin/env python3
"""
PIE Daily Briefing — executive assistant mode.

Loads the world model, runs dynamics analysis, and produces a prioritized
daily briefing: what to focus on, what's stale, what's coming up.

Usage:
    python briefing.py                    # Full briefing
    python briefing.py --focus projects   # Focus on projects only
    python briefing.py --json             # Machine-readable output
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent))

from pie.core.world_model import WorldModel
from pie.core.dynamics import TransitionDynamics
from pie.core.models import EntityType


def load_world_model(path: str = "./output/world_model.json") -> WorldModel:
    wm = WorldModel(persist_path=path)
    if not wm.entities:
        print(f"No world model found at {path}")
        sys.exit(1)
    return wm


def compute_importance(wm: WorldModel, report) -> dict[str, float]:
    """Compute importance scores for all entities."""
    scores = {}
    for eid, profile in report.entity_profiles.items():
        entity = wm.entities.get(eid)
        if not entity:
            continue
        base = math.log2(1 + profile.total_transitions) / 8.0
        base = min(base, 1.0)
        recency = 1.0 - 0.8 * profile.staleness_score
        scores[eid] = round(base * recency, 4)
        entity.importance = scores[eid]
    return scores


def briefing_active_projects(wm, report, top_n=10):
    """What projects need attention?"""
    projects = []
    for eid, entity in wm.entities.items():
        if entity.type != EntityType.PROJECT:
            continue
        profile = report.entity_profiles.get(eid)
        if not profile:
            continue
        projects.append({
            "name": entity.name,
            "importance": entity.importance,
            "staleness": profile.staleness_score,
            "transitions": profile.total_transitions,
            "last_active": profile.last_transition_ts,
            "state": entity.current_state.get("description", str(entity.current_state))[:200],
            "status": entity.current_state.get("status", "unknown"),
        })

    # Sort by importance (high first), then by staleness (stale = needs attention)
    projects.sort(key=lambda p: (-p["importance"], -p["staleness"]))
    return projects[:top_n]


def briefing_goals(wm, report, top_n=15):
    """What goals/commitments exist?"""
    goals = []
    for eid, entity in wm.entities.items():
        if entity.type == EntityType.GOAL:
            goals.append({
                "name": entity.name,
                "deadline": entity.current_state.get("deadline"),
                "status": entity.current_state.get("status", "unknown"),
                "priority": entity.current_state.get("priority", "unknown"),
                "description": entity.current_state.get("description", "")[:200],
                "importance": entity.importance,
            })

    # Also extract goal-like decisions
    for eid, entity in wm.entities.items():
        if entity.type == EntityType.DECISION:
            state_desc = entity.current_state.get("description", "").lower()
            # Heuristic: decisions with forward-looking language are goals
            if any(w in state_desc for w in ["plan to", "want to", "need to", "going to", "will ", "target", "aiming"]):
                goals.append({
                    "name": entity.name,
                    "deadline": entity.current_state.get("deadline"),
                    "status": "inferred",
                    "priority": "medium",
                    "description": entity.current_state.get("description", "")[:200],
                    "importance": entity.importance,
                })

    goals.sort(key=lambda g: (
        {"high": 0, "medium": 1, "low": 2}.get(g.get("priority", ""), 3),
        -g.get("importance", 0),
    ))
    return goals[:top_n]


def briefing_stale_entities(wm, report, top_n=15):
    """What's been neglected?"""
    stale = []
    for eid in report.stale_entities:
        entity = wm.entities.get(eid)
        profile = report.entity_profiles.get(eid)
        if not entity or not profile:
            continue
        if entity.type in (EntityType.CONCEPT, EntityType.PERIOD):
            continue  # Skip concepts/periods — staleness isn't actionable

        days_since = (time.time() - profile.last_transition_ts) / 86400
        stale.append({
            "name": entity.name,
            "type": entity.type.value,
            "staleness": profile.staleness_score,
            "days_since_update": round(days_since, 1),
            "transitions": profile.total_transitions,
            "importance": entity.importance,
        })

    stale.sort(key=lambda s: (-s["importance"], -s["staleness"]))
    return stale[:top_n]


def briefing_recent_activity(wm, days=7):
    """What changed recently?"""
    cutoff = time.time() - days * 86400
    recent = []
    for tid, t in wm.transitions.items():
        if t.timestamp >= cutoff:
            entity = wm.entities.get(t.entity_id)
            if entity:
                recent.append({
                    "entity": entity.name,
                    "type": entity.type.value,
                    "trigger": t.trigger_summary[:100],
                    "transition_type": t.transition_type.value,
                    "timestamp": t.timestamp,
                })

    recent.sort(key=lambda r: -r["timestamp"])
    return recent[:20]


def briefing_cooccurrences(report, top_n=10):
    """What entities move together? (causal signals)"""
    results = []
    for co in report.cooccurrences[:top_n]:
        results.append({
            "entity_a": co.entity_a_name,
            "entity_b": co.entity_b_name,
            "count": co.cooccurrence_count,
            "confidence": co.confidence,
            "typical_lag_hours": round(co.typical_lag_s / 3600, 1),
        })
    return results


def briefing_people(wm):
    """Who's in the network?"""
    people = []
    for eid, entity in wm.entities.items():
        if entity.type != EntityType.PERSON:
            continue
        rels = wm.get_relationships(eid)
        people.append({
            "name": entity.name,
            "relationships": len(rels),
            "state": entity.current_state.get("description", "")[:150],
            "importance": entity.importance,
        })
    people.sort(key=lambda p: -p["importance"])
    return people


def print_briefing(wm, report):
    """Print a human-readable daily briefing."""
    now = datetime.now()
    print(f"\n{'='*60}")
    print(f"  PIE DAILY BRIEFING — {now.strftime('%A, %B %d, %Y')}")
    print(f"{'='*60}")
    print(f"  World model: {len(wm.entities)} entities, {len(wm.transitions)} transitions")
    print(f"  Stale: {len(report.stale_entities)} | Volatile: {len(report.volatile_entities)}")

    # Active Projects
    projects = briefing_active_projects(wm, report)
    if projects:
        print(f"\n{'─'*60}")
        print("  ACTIVE PROJECTS (by importance)")
        print(f"{'─'*60}")
        for i, p in enumerate(projects, 1):
            days = (time.time() - p["last_active"]) / 86400
            stale_tag = " ⚠ STALE" if p["staleness"] > 0.7 else ""
            print(f"  {i}. {p['name']} (importance: {p['importance']:.3f}{stale_tag})")
            print(f"     Status: {p['status']} | Last active: {days:.0f} days ago | {p['transitions']} transitions")
            if p["state"]:
                print(f"     State: {p['state'][:120]}")

    # Goals
    goals = briefing_goals(wm, report)
    if goals:
        print(f"\n{'─'*60}")
        print("  GOALS & COMMITMENTS")
        print(f"{'─'*60}")
        for g in goals:
            deadline = f" (deadline: {g['deadline']})" if g.get("deadline") else ""
            print(f"  [{g['priority'].upper():6s}] {g['name']}{deadline}")
            if g["description"]:
                print(f"           {g['description'][:120]}")

    # Stale / Needs Attention
    stale = briefing_stale_entities(wm, report)
    if stale:
        print(f"\n{'─'*60}")
        print("  NEEDS ATTENTION (stale but important)")
        print(f"{'─'*60}")
        for s in stale[:10]:
            print(f"  ⚠ {s['name']} ({s['type']}) — {s['days_since_update']:.0f} days since update, importance={s['importance']:.3f}")

    # Co-occurrences
    coocs = briefing_cooccurrences(report)
    if coocs:
        print(f"\n{'─'*60}")
        print("  CAUSAL PATTERNS (entities that change together)")
        print(f"{'─'*60}")
        for co in coocs[:7]:
            print(f"  {co['entity_a']} ↔ {co['entity_b']} ({co['count']}x, lag: {co['typical_lag_hours']:.1f}h)")

    # Predictions
    if report.predicted_next_transitions:
        print(f"\n{'─'*60}")
        print("  PREDICTED NEXT CHANGES")
        print(f"{'─'*60}")
        for pred in report.predicted_next_transitions[:7]:
            print(f"  → {pred['entity_name']}: likely {pred['predicted_type']} (conf: {pred['confidence']:.2f})")
            print(f"    {pred['reason']}")

    # People
    people = briefing_people(wm)
    if people:
        print(f"\n{'─'*60}")
        print("  NETWORK ({} people)".format(len(people)))
        print(f"{'─'*60}")
        for p in people[:8]:
            print(f"  {p['name']} — {p['relationships']} connections, importance={p['importance']:.3f}")

    print(f"\n{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="PIE Daily Briefing")
    parser.add_argument("--output", type=str, default="./output", help="World model directory")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--focus", type=str, default=None, help="Focus: projects, goals, stale, people")
    args = parser.parse_args()

    wm_path = Path(args.output) / "world_model.json"
    wm = load_world_model(str(wm_path))

    print("Analyzing dynamics...", end=" ", flush=True)
    dynamics = TransitionDynamics(wm)
    report = dynamics.analyze()
    compute_importance(wm, report)
    print(f"done. ({len(report.stale_entities)} stale, {len(report.volatile_entities)} volatile)")

    if args.json:
        output = {
            "projects": briefing_active_projects(wm, report),
            "goals": briefing_goals(wm, report),
            "stale": briefing_stale_entities(wm, report),
            "cooccurrences": briefing_cooccurrences(report),
            "predictions": report.predicted_next_transitions[:10],
            "people": briefing_people(wm),
        }
        print(json.dumps(output, indent=2, default=str))
    else:
        print_briefing(wm, report)


if __name__ == "__main__":
    main()
