#!/usr/bin/env python3
"""
PIE Agent — the single entry point for using PIE.

Usage:
    # Generate your system prompt (copy-paste into any LLM)
    python tools/pie_agent.py prompt

    # Focus on a specific project
    python tools/pie_agent.py prompt --focus sponsorFind

    # See what needs attention right now
    python tools/pie_agent.py brief

    # Schedule a wake-up for yourself
    python tools/pie_agent.py wake "Check if agency emails got replies" --in 3d
    python tools/pie_agent.py wake "Review Lucid Labs landing page" --in 6h

    # Record a prediction
    python tools/pie_agent.py predict "sponsorFind" "Will close first paid pilot" --in 14d

    # See all pending wake-ups and predictions
    python tools/pie_agent.py schedule

    # Predict next state changes from the world model
    python tools/pie_agent.py predict-next

    # Full status: prompt + brief + predictions
    python tools/pie_agent.py status
"""

import sys
import os
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pie.core.world_model import WorldModel
from pie.core.temporal import TemporalState
from pie.core.prompt_engine import PromptEngine, PromptConfig, generate_prompt
from pie.core.scheduler import Scheduler


def parse_duration(s: str) -> float:
    """Parse '3d', '6h', '30m', '2w' into hours."""
    s = s.strip().lower()
    if s.endswith('d'):
        return float(s[:-1]) * 24
    if s.endswith('h'):
        return float(s[:-1])
    if s.endswith('m'):
        return float(s[:-1]) / 60
    if s.endswith('w'):
        return float(s[:-1]) * 24 * 7
    return float(s)  # assume hours


def load_world_model(path: str) -> WorldModel:
    wm = WorldModel(persist_path=path)
    return wm


def cmd_prompt(args):
    """Generate the system prompt."""
    prompt = generate_prompt(args.world_model, focus=args.focus)
    if args.output:
        Path(args.output).write_text(prompt)
        print(f"Wrote {len(prompt)} chars (~{len(prompt)//4} tokens) to {args.output}")
    else:
        print(prompt)
        print(f"\n--- {len(prompt)} chars, ~{len(prompt)//4} tokens ---")


def cmd_brief(args):
    """Show what needs attention right now."""
    wm = load_world_model(args.world_model)
    ts = TemporalState(wm)
    stats = ts.learn()
    ref_t = time.time()

    config = PromptConfig()

    print("=" * 60)
    print(f"  PIE BRIEFING — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    # Scheduler briefing
    sched = Scheduler(args.schedule_file)
    sched_brief = sched.generate_briefing()
    if "No wake-ups" not in sched_brief:
        print(f"\n{sched_brief}")

    # Overdue entities
    stale = ts.rank_by_staleness(ref_t, top_n=15)
    overdue = []
    for item in stale:
        eid = item.get('entity_id', '')
        entity = wm.entities.get(eid)
        if not entity:
            continue
        name = entity.name

        # Skip Pranay's (brother's) entities
        name_lower = name.lower()
        if any(p.lower() in name_lower for p in ['pulse-fi', 'pulsefi', 'pulse\u2011fi', 'whisper', 'wifi csi']):
            continue

        silence = item.get('silence_days', 0)
        mean_interval = item.get('mean_interval_days', 0)
        survival = item.get('survival', 0)
        classification = item.get('status', item.get('classification', ''))

        if mean_interval > 0 and silence > mean_interval * 1.5 and classification not in ('dead',):
            overdue.append({
                'name': name,
                'type': entity.type.value,
                'silence': silence,
                'expected': mean_interval,
                'ratio': silence / mean_interval,
                'survival': survival,
                'status': classification,
            })

    if overdue:
        print("\n## Overdue (needs attention)\n")
        for o in overdue[:10]:
            print(f"  {o['status']:>8}  {o['name'][:40]:<40}  {o['silence']:.0f}d silent (expected {o['expected']:.0f}d, {o['ratio']:.1f}x overdue)")

    # High momentum
    momentum = ts.rank_by_momentum(ref_t, top_n=10)
    hot = []
    for item in momentum:
        eid = item.get('entity_id', '')
        entity = wm.entities.get(eid)
        if not entity:
            continue
        name = entity.name
        name_lower = name.lower()
        if any(p.lower() in name_lower for p in ['pulse-fi', 'pulsefi', 'pulse\u2011fi']):
            continue

        alive = item.get('alive', item.get('alive_probability', 0))
        if alive > 0.6:
            hot.append({
                'name': name,
                'type': entity.type.value,
                'alive': alive,
            })

    if hot:
        print("\n## High Momentum (actively evolving)\n")
        for h in hot[:8]:
            print(f"  {h['alive']:.0%} alive  {h['name'][:50]}")

    # Population summary
    pop = ts.population_summary(ref_t)
    statuses = pop.get('status_counts', {})
    print(f"\n## Population: {pop.get('total_entities', '?')} entities")
    for status, count in sorted(statuses.items(), key=lambda x: -x[1]):
        pct = count / pop.get('total_entities', 1) * 100
        bar = '█' * int(pct / 3)
        print(f"  {status:>10}: {count:>4} ({pct:.0f}%) {bar}")


def cmd_wake(args):
    """Schedule a wake-up."""
    sched = Scheduler(args.schedule_file)
    hours = parse_duration(args.delay)
    w = sched.schedule(
        message=args.message,
        delay_hours=hours,
        entity_name=args.entity,
        priority=args.priority,
        context=args.context or "",
    )
    print(f"Scheduled: \"{w.message}\"")
    print(f"  Fires: {w.trigger_dt.strftime('%Y-%m-%d %H:%M')} ({w.time_until()} from now)")
    if w.entity_name:
        print(f"  Entity: {w.entity_name}")
    print(f"  ID: {w.id}")


def cmd_predict(args):
    """Record a prediction."""
    sched = Scheduler(args.schedule_file)
    days = parse_duration(args.timeframe) / 24  # convert hours to days
    p = sched.predict(
        entity_id="",
        entity_name=args.entity,
        prediction=args.prediction,
        expected_in_days=days,
        confidence=args.confidence,
        reasoning=args.reasoning or "",
        based_on="manual",
    )
    expected_dt = datetime.fromtimestamp(p.expected_by)
    print(f"Predicted: \"{p.entity_name}\" → {p.prediction}")
    print(f"  Expected by: {expected_dt.strftime('%Y-%m-%d')} ({days:.0f}d)")
    print(f"  Confidence: {p.confidence:.0%}")


def cmd_predict_next(args):
    """Predict next state changes from temporal patterns."""
    wm = load_world_model(args.world_model)
    ts = TemporalState(wm)
    ts.learn()
    ref_t = time.time()

    print("=" * 60)
    print("  PREDICTED NEXT STATE CHANGES")
    print("=" * 60)

    # For each entity with rhythm data, predict when next transition happens
    predictions = []
    for eid, rhythm in ts.rhythms.items():
        if rhythm.n_transitions < 3:
            continue

        entity = wm.entities.get(eid)
        if not entity:
            continue

        name = entity.name
        name_lower = name.lower()
        if any(p.lower() in name_lower for p in ['pulse-fi', 'pulsefi', 'pulse\u2011fi']):
            continue

        query = ts.query(eid, ref_t)
        expected_next_days = query.get('expected_next_in_days')
        survival = query.get('survival', 0)
        classification = query.get('status', query.get('classification', ''))

        if expected_next_days is not None and classification not in ('dead',):
            days_until = expected_next_days

            # Get last state for context
            trans_ids = wm._entity_transitions.get(eid, [])
            last_summary = ""
            if trans_ids:
                last_trans = max(
                    (wm.transitions[tid] for tid in trans_ids if tid in wm.transitions),
                    key=lambda t: t.timestamp,
                    default=None,
                )
                if last_trans:
                    last_summary = last_trans.trigger_summary or ""

            predictions.append({
                'name': name,
                'type': entity.type.value,
                'days_until': days_until,
                'survival': survival,
                'status': classification,
                'transitions': rhythm.n_transitions,
                'rhythm': query.get('rhythm_mean_days', rhythm.mean_interval / 86400),
                'last_summary': last_summary[:80],
            })

    # Sort by soonest expected
    predictions.sort(key=lambda p: p['days_until'])

    # Show overdue first
    overdue = [p for p in predictions if p['days_until'] < 0]
    if overdue:
        print("\n## OVERDUE (predicted transition already passed)\n")
        for p in overdue[:15]:
            print(f"  {p['status']:>8}  {p['name'][:40]:<40}  {abs(p['days_until']):.1f}d overdue  (rhythm: {p['rhythm']:.1f}d)")
            if p['last_summary']:
                print(f"           Last: {p['last_summary']}")

    # Then upcoming
    upcoming = [p for p in predictions if 0 <= p['days_until'] <= 30]
    if upcoming:
        print("\n## EXPECTED NEXT (within 30 days)\n")
        for p in upcoming[:15]:
            print(f"  {p['status']:>8}  {p['name'][:40]:<40}  in {p['days_until']:.1f}d  (S={p['survival']:.2f})")

    # Far out
    far = [p for p in predictions if p['days_until'] > 30]
    if far:
        print(f"\n  + {len(far)} entities predicted beyond 30 days")


def cmd_schedule(args):
    """Show all pending wake-ups and predictions."""
    sched = Scheduler(args.schedule_file)
    print(sched.generate_briefing())
    s = sched.summary()
    print(f"\nStats: {s['pending_wakeups']} pending, {s['due_now']} due now, "
          f"{s['active_predictions']} predictions, {s['expired_predictions']} expired")


def cmd_status(args):
    """Full status: everything at a glance."""
    cmd_brief(args)
    print("\n")
    cmd_predict_next(args)


def main():
    parser = argparse.ArgumentParser(description="PIE Agent — your AI thinking partner")
    parser.add_argument("--world-model", default="output/world_model.json")
    parser.add_argument("--schedule-file", default="output/schedule.json")

    sub = parser.add_subparsers(dest="command")

    # prompt
    p_prompt = sub.add_parser("prompt", help="Generate system prompt")
    p_prompt.add_argument("--focus", default=None, help="Project to expand")
    p_prompt.add_argument("--output", "-o", default=None, help="Write to file")

    # brief
    sub.add_parser("brief", help="What needs attention now")

    # wake
    p_wake = sub.add_parser("wake", help="Schedule a wake-up")
    p_wake.add_argument("message", help="What to do when it fires")
    p_wake.add_argument("--in", dest="delay", default="1d", help="When to fire (e.g., 3d, 6h, 30m)")
    p_wake.add_argument("--entity", default=None, help="Related entity name")
    p_wake.add_argument("--priority", default="normal", choices=["low", "normal", "high", "urgent"])
    p_wake.add_argument("--context", default=None, help="Why this matters")

    # predict
    p_pred = sub.add_parser("predict", help="Record a prediction")
    p_pred.add_argument("entity", help="Entity name")
    p_pred.add_argument("prediction", help="What you predict will happen")
    p_pred.add_argument("--in", dest="timeframe", default="7d", help="Expected by (e.g., 14d, 2w)")
    p_pred.add_argument("--confidence", type=float, default=0.5)
    p_pred.add_argument("--reasoning", default=None)

    # predict-next
    sub.add_parser("predict-next", help="Predict next state changes from patterns")

    # schedule
    sub.add_parser("schedule", help="Show pending wake-ups & predictions")

    # status
    sub.add_parser("status", help="Full status overview")

    args = parser.parse_args()

    if args.command == "prompt":
        cmd_prompt(args)
    elif args.command == "brief":
        cmd_brief(args)
    elif args.command == "wake":
        cmd_wake(args)
    elif args.command == "predict":
        cmd_predict(args)
    elif args.command == "predict-next":
        cmd_predict_next(args)
    elif args.command == "schedule":
        cmd_schedule(args)
    elif args.command == "status":
        cmd_status(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
