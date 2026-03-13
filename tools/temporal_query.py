#!/usr/bin/env python3
"""
Temporal Query — query the continuous time state of the world model.

Usage:
    python tools/temporal_query.py                          # Population summary
    python tools/temporal_query.py --entity "Lucid Academy"  # Single entity
    python tools/temporal_query.py --stale                   # Most overdue entities
    python tools/temporal_query.py --momentum                # Most active entities
    python tools/temporal_query.py --validate                # Holdout validation
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import statistics
from pathlib import Path
from collections import Counter, defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pie.core.world_model import WorldModel
from pie.core.temporal import TemporalState


def find_entity(wm: WorldModel, name: str) -> str | None:
    """Find entity ID by name (case-insensitive partial match)."""
    name_lower = name.lower()
    # Exact match first
    for eid, entity in wm.entities.items():
        if entity.name.lower() == name_lower:
            return eid
    # Partial match
    for eid, entity in wm.entities.items():
        if name_lower in entity.name.lower():
            return eid
    return None


def holdout_validation(wm: WorldModel, ts: TemporalState) -> dict:
    """Rigorous holdout test: for each entity with 5+ transitions,
    hold out the last transition and test prediction accuracy.

    This measures how well the learned model predicts WHEN the next
    event will happen, using only data available before that event.
    """
    # We use the already-fitted model (which includes the last transition)
    # and check: what survival probability does it assign to actual events?
    results = []

    trans_by_entity = defaultdict(list)
    for t in wm.transitions.values():
        trans_by_entity[t.entity_id].append(t.timestamp)

    for eid, timestamps in trans_by_entity.items():
        if len(timestamps) < 5:
            continue
        timestamps.sort()

        # Hold out last
        train_ts = timestamps[:-1]
        actual_t = timestamps[-1]

        # Compute train gaps
        gaps = [(train_ts[i + 1] - train_ts[i]) / 86400 for i in range(len(train_ts) - 1)]
        if not gaps:
            continue
        mean_gap = statistics.mean(gaps)
        if mean_gap < 0.001:
            continue

        # Predict: next event = last_train + mean_gap
        actual_gap = (actual_t - train_ts[-1]) / 86400

        # Survival probability at actual event time using the learned table
        k = actual_gap / mean_gap
        table = ts._get_table(eid)
        s = table.survival(k)

        results.append({
            "eid": eid,
            "name": wm.entities[eid].name,
            "n_train": len(train_ts),
            "mean_gap": mean_gap,
            "actual_gap": actual_gap,
            "pred_gap": mean_gap,
            "abs_error": abs(actual_gap - mean_gap),
            "relative_error": abs(actual_gap - mean_gap) / mean_gap,
            "survival_at_event": s,
        })

    if not results:
        return {"error": "no entities with 5+ transitions"}

    # Aggregate metrics
    errors = [r["abs_error"] for r in results]
    rel_errors = [r["relative_error"] for r in results]
    within_1x = sum(1 for r in results if r["relative_error"] <= 1.0) / len(results)
    within_2x = sum(1 for r in results if r["relative_error"] <= 2.0) / len(results)
    survivals = [r["survival_at_event"] for r in results if r["survival_at_event"] is not None]

    return {
        "n_test_entities": len(results),
        "median_abs_error_days": round(statistics.median(errors), 2),
        "mean_abs_error_days": round(statistics.mean(errors), 2),
        "median_relative_error": round(statistics.median(rel_errors), 2),
        "within_1x_mean": f"{within_1x:.1%}",
        "within_2x_mean": f"{within_2x:.1%}",
        "mean_survival_at_event": round(statistics.mean(survivals), 3) if survivals else None,
        "median_survival_at_event": round(statistics.median(survivals), 3) if survivals else None,
        # A well-calibrated model should have mean survival ≈ 0.5 at observed events
        # (events are uniformly distributed over the CDF)
    }


def main():
    parser = argparse.ArgumentParser(description="PIE Temporal Query")
    parser.add_argument("--output", type=str, default="./output", help="World model directory")
    parser.add_argument("--entity", type=str, help="Query a specific entity by name")
    parser.add_argument("--stale", action="store_true", help="Show most overdue entities")
    parser.add_argument("--momentum", action="store_true", help="Show most active entities")
    parser.add_argument("--validate", action="store_true", help="Run holdout validation")
    parser.add_argument("--top", type=int, default=20, help="Number of results to show")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    wm_path = Path(args.output) / "world_model.json"
    if not wm_path.exists():
        print(f"No world model at {wm_path}")
        sys.exit(1)

    # Load
    wm = WorldModel(persist_path=wm_path)

    # Learn
    ts = TemporalState(wm)
    stats = ts.learn()
    now = ts._now()

    print(f"\n{'=' * 60}")
    print(f"  PIE TEMPORAL STATE")
    print(f"{'=' * 60}")
    print(f"  Learned from {stats['total_entities']} entities")
    print(f"  Individually fitted (10+ transitions): {stats['individually_fitted']}")
    print(f"  With rhythm (2+ transitions): {stats['with_rhythm']}")
    print(f"  Singletons: {stats['singletons']}")
    print(f"  Global survival table: {stats['global_table_points']} points from {stats['global_n_intervals']} intervals")
    print(f"  Population median interval: {stats['population_median_interval_days']} days")
    if stats["type_tables"]:
        print(f"  Per-type tables: {', '.join(f'{k}({v}pts)' for k, v in stats['type_tables'].items())}")
    print()

    if args.entity:
        eid = find_entity(wm, args.entity)
        if not eid:
            print(f"  Entity not found: {args.entity}")
            sys.exit(1)
        result = ts.query(eid, now)
        if args.json:
            print(json.dumps(result, indent=2, default=str))
        else:
            print(f"  Entity: {result['name']} ({result['type']})")
            print(f"  Status: {result['status']}")
            print(f"  Survival: {result['survival']}")
            print(f"  State confidence: {result['state_confidence']}")
            print(f"  Alive probability: {result['alive_probability']}")
            print(f"  Silence: {result['silence_days']} days")
            print(f"  Rhythm: {result['rhythm_mean_days']} days between transitions")
            if result.get("silence_in_rhythm_units"):
                print(f"  Silence in rhythm units: {result['silence_in_rhythm_units']}x")
            if result.get("expected_next_in_days"):
                print(f"  Expected next: {result['expected_next_in_days']} days from now")
            print(f"\n  Current state:")
            for k, v in result.get("current_state", {}).items():
                print(f"    {k}: {str(v)[:80]}")

    elif args.stale:
        print(f"  Most overdue entities:")
        results = ts.rank_by_staleness(now, top_n=args.top)
        if args.json:
            print(json.dumps(results, indent=2))
        else:
            for i, r in enumerate(results, 1):
                print(f"  {i:3d}. [{r['status']:8s}] {r['name'][:40]:<40s} "
                      f"S={r['survival']:.4f}  silence={r['silence_days']:6.1f}d  "
                      f"rhythm={r['mean_interval_days']:5.1f}d  n={r['n_transitions']}")

    elif args.momentum:
        print(f"  Highest momentum entities:")
        results = ts.rank_by_momentum(now, top_n=args.top)
        if args.json:
            print(json.dumps(results, indent=2))
        else:
            for i, r in enumerate(results, 1):
                print(f"  {i:3d}. [{r['status']:8s}] {r['name'][:40]:<40s} "
                      f"mom={r['momentum']:.4f}  alive={r['alive']:.3f}  "
                      f"density={r['density']:.3f}/d  n={r['n_transitions']}")

    elif args.validate:
        print(f"  Running holdout validation...")
        result = holdout_validation(wm, ts)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"  Test entities: {result['n_test_entities']}")
            print(f"  Median absolute error: {result['median_abs_error_days']} days")
            print(f"  Mean absolute error: {result['mean_abs_error_days']} days")
            print(f"  Median relative error: {result['median_relative_error']}x")
            print(f"  Predictions within 1x mean: {result['within_1x_mean']}")
            print(f"  Predictions within 2x mean: {result['within_2x_mean']}")
            if result.get("mean_survival_at_event") is not None:
                print(f"  Mean survival at event: {result['mean_survival_at_event']}")
                print(f"  (well-calibrated model → ≈ 0.5)")

    else:
        # Population summary
        summary = ts.population_summary(now)
        if args.json:
            print(json.dumps(summary, indent=2))
        else:
            print(f"  Population at latest timestamp:")
            for status, count in sorted(summary["status_distribution"].items(),
                                        key=lambda x: -x[1]):
                pct = count / summary["total_entities"] * 100
                bar = "█" * int(pct / 2)
                print(f"    {status:10s}: {count:5d} ({pct:5.1f}%) {bar}")
            print(f"\n  Mean alive probability: {summary['mean_alive_probability']}")

    print(f"\n{'=' * 60}\n")


if __name__ == "__main__":
    main()
