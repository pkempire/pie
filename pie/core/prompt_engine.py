"""
PIE Prompt Engine — generates a living system prompt from the world model.

This is the core of PIE's usefulness: every conversation gets injected with
a compressed, temporally-aware briefing that makes the LLM actually know
your full context.

The prompt is NOT a data dump. It's a structured briefing:
1. Who you are + current priorities
2. Per-project state snapshots (compressed)
3. Predictions: what SHOULD change next
4. Attention flags: what's overdue, what's surprising
5. Temporal awareness: "it's been X days since you touched Y"

Target: ~3-4K tokens. Rich enough to be useful, small enough for every conversation.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from pie.core.world_model import WorldModel
from pie.core.temporal import TemporalState
from pie.core.dynamics import TransitionDynamics


@dataclass
class PromptConfig:
    """Controls what goes into the system prompt."""
    owner: str = "Parth Kocheta"
    # Entities owned by someone else (brother Pranay's projects etc.)
    # Pranay Kocheta = younger brother, built PulseFi at UCSC with Nayan Bhatia (PhD) & Prof. Obraczka
    exclude_owners: dict[str, list[str]] = field(default_factory=lambda: {
        "Pranay (brother)": [
            "Pulse-Fi",
            "PulseFi",
            "Pulse\u2011Fi",
            "Whisper",
            "WiFi CSI",
        ]
    })
    max_projects: int = 8
    max_goals: int = 10
    max_predictions: int = 6
    max_attention_flags: int = 5
    include_temporal: bool = True
    include_predictions: bool = True
    include_goals: bool = True
    include_relationships: bool = True
    # Time context
    reference_time: float | None = None  # defaults to now


def _days_ago(ts: float, ref: float) -> str:
    """Human-readable time delta."""
    days = (ref - ts) / 86400
    if days < 0.04:  # < 1 hour
        return "just now"
    if days < 1:
        hours = int(days * 24)
        return f"{hours}h ago"
    if days < 7:
        return f"{days:.1f}d ago"
    if days < 30:
        weeks = int(days / 7)
        return f"{weeks}w ago"
    if days < 365:
        months = int(days / 30)
        return f"{months}mo ago"
    years = days / 365
    return f"{years:.1f}y ago"


def _compress_state(state: dict, max_keys: int = 8) -> dict:
    """Pick the most informative keys from an entity's state dict."""
    priority_keys = [
        'status', 'phase', 'stage', 'current_focus', 'next_steps',
        'short_term_target', 'one_year_target', 'revenue', 'traction',
        'business_model', 'pricing_targets', 'current_goals', 'monetization_paths',
        'ICP_target', 'positioning', 'description', 'last_activity',
        'subscribers', 'students', 'brands', 'creators', 'scale',
        'pipeline', 'offerings', 'next_action', 'priority',
    ]
    result = {}
    for k in priority_keys:
        if k in state and state[k] is not None:
            v = state[k]
            if isinstance(v, list):
                v = ", ".join(str(x) for x in v[:5])
            elif isinstance(v, dict):
                v = json.dumps(v)
            v = str(v)
            if len(v) > 150:
                v = v[:147] + "..."
            result[k] = v
            if len(result) >= max_keys:
                break
    return result


def _is_nayan_entity(entity_name: str, config: PromptConfig) -> bool:
    """Check if entity belongs to Nayan (brother) based on name matching."""
    name_lower = entity_name.lower()
    for owner, patterns in config.exclude_owners.items():
        for pattern in patterns:
            if pattern.lower() in name_lower:
                return True
    return False


class PromptEngine:
    """Generates a living system prompt from the world model."""

    def __init__(self, wm: WorldModel, config: PromptConfig | None = None):
        self.wm = wm
        self.config = config or PromptConfig()
        self.temporal: TemporalState | None = None
        self.dynamics: TransitionDynamics | None = None

    def initialize(self):
        """Learn temporal patterns and dynamics from the world model."""
        self.temporal = TemporalState(self.wm)
        self.temporal.learn()
        self.dynamics = TransitionDynamics(self.wm)

    def generate(self, focus_project: str | None = None) -> str:
        """
        Generate the full system prompt.

        Args:
            focus_project: If set, expand context for this specific project.

        Returns:
            A system prompt string ready to inject.
        """
        if self.temporal is None:
            self.initialize()

        ref_t = self.config.reference_time or time.time()
        now_str = datetime.fromtimestamp(ref_t).strftime("%Y-%m-%d %H:%M")

        sections = []

        # ── Header ──
        sections.append(self._header(now_str, ref_t))

        # ── Project snapshots ──
        sections.append(self._project_snapshots(ref_t, focus_project))

        # ── Goals ──
        if self.config.include_goals:
            goals_section = self._goals_section(ref_t)
            if goals_section:
                sections.append(goals_section)

        # ── Predictions ──
        if self.config.include_predictions:
            predictions = self._predictions_section(ref_t)
            if predictions:
                sections.append(predictions)

        # ── Attention flags ──
        attention = self._attention_flags(ref_t)
        if attention:
            sections.append(attention)

        # ── Focus project deep dive ──
        if focus_project:
            deep = self._focus_deep_dive(focus_project, ref_t)
            if deep:
                sections.append(deep)

        # ── Behavioral note ──
        sections.append(self._behavioral_note())

        return "\n\n".join(s for s in sections if s)

    def _header(self, now_str: str, ref_t: float) -> str:
        """Identity + temporal context."""
        # Count Pranay's entities vs total
        total_entities = len(self.wm.entities)
        total_transitions = len(self.wm.transitions)

        # Population summary
        pop = self.temporal.population_summary(ref_t) if self.temporal else {}

        lines = [
            f"You are the AI thinking partner for {self.config.owner}.",
            f"Current time: {now_str}.",
            f"You have access to a world model with {total_entities} entities and {total_transitions} state transitions spanning {pop.get('observation_window_days', '?')} days.",
        ]

        # Add population breakdown
        if pop:
            statuses = pop.get('status_counts', {})
            status_str = ", ".join(f"{k}: {v}" for k, v in sorted(statuses.items(), key=lambda x: -x[1]))
            lines.append(f"Entity health: {status_str}.")

        return "\n".join(lines)

    def _project_snapshots(self, ref_t: float, focus_project: str | None) -> str:
        """Compressed state for each active project."""
        projects = []

        for eid, entity in self.wm.entities.items():
            if entity.type.value.lower() != 'project':
                continue
            if _is_nayan_entity(entity.name, self.config):
                continue

            # Get temporal state
            tq = self.temporal.query(eid, ref_t) if self.temporal else {}
            status = tq.get('status', tq.get('classification', 'unknown'))
            silence = tq.get('silence_days', 0) or 0
            rhythm = tq.get('rhythm_mean_days', tq.get('mean_interval_days', 0)) or 0
            survival = tq.get('survival', 0) or 0
            n_transitions = len(self.wm._entity_transitions.get(eid, []))

            # Compress state
            compressed = _compress_state(entity.current_state)

            projects.append({
                'name': entity.name,
                'id': eid,
                'status': status,
                'silence_days': round(silence, 1),
                'rhythm_days': round(rhythm, 1),
                'survival': round(survival, 2),
                'transitions': n_transitions,
                'state': compressed,
                'is_focus': focus_project and focus_project.lower() in entity.name.lower(),
            })

        # Sort: focus first, then by transition count (most active)
        projects.sort(key=lambda p: (
            not p['is_focus'],
            -p['transitions']
        ))
        projects = projects[:self.config.max_projects]

        if not projects:
            return ""

        lines = ["## Your Projects\n"]
        for p in projects:
            marker = " ← FOCUS" if p['is_focus'] else ""
            silence_str = _days_ago(ref_t - p['silence_days'] * 86400, ref_t)
            lines.append(f"**{p['name']}** [{p['status']}] — {p['transitions']} updates, last {silence_str}, rhythm ~{p['rhythm_days']}d{marker}")

            for k, v in p['state'].items():
                lines.append(f"  {k}: {v}")
            lines.append("")

        return "\n".join(lines)

    def _goals_section(self, ref_t: float) -> str:
        """Active goals from the world model."""
        goals = []
        for eid, entity in self.wm.entities.items():
            if entity.type.value.lower() != 'goal':
                continue
            if _is_nayan_entity(entity.name, self.config):
                continue

            tq = self.temporal.query(eid, ref_t) if self.temporal else {}
            status = tq.get('status', tq.get('classification', 'unknown'))
            silence = tq.get('silence_days', 0) or 0

            # Skip dead goals
            if status == 'dead':
                continue

            goals.append({
                'name': entity.name,
                'status': status,
                'silence_days': round(silence, 1),
                'state': entity.current_state.get('description', ''),
            })

        goals.sort(key=lambda g: g['silence_days'])
        goals = goals[:self.config.max_goals]

        if not goals:
            return ""

        lines = ["## Active Goals\n"]
        for g in goals:
            desc = g['state'][:100] if g['state'] else ''
            lines.append(f"- **{g['name']}** [{g['status']}] {desc}")

        return "\n".join(lines)

    def _predictions_section(self, ref_t: float) -> str:
        """What SHOULD happen next, based on temporal patterns."""
        if not self.temporal:
            return ""

        # Get entities ranked by "most overdue"
        stale = self.temporal.rank_by_staleness(ref_t, top_n=20)

        # Get entities ranked by momentum
        momentum = self.temporal.rank_by_momentum(ref_t, top_n=10)

        predictions = []

        # Overdue entities → predicted to need attention
        for item in stale[:8]:
            eid = item.get('entity_id', '')
            entity = self.wm.entities.get(eid)
            if not entity:
                continue
            if _is_nayan_entity(entity.name, self.config):
                continue

            silence = item.get('silence_days', 0)
            mean_interval = item.get('mean_interval_days', 0)
            survival = item.get('survival', 0)
            status = item.get('status', item.get('classification', 'unknown'))

            if status in ('dead',):
                continue

            # How overdue?
            if mean_interval > 0 and silence > mean_interval * 1.5:
                overdue_factor = silence / mean_interval
                predictions.append({
                    'entity': entity.name,
                    'type': 'overdue',
                    'message': f"Overdue by {overdue_factor:.1f}x its rhythm ({silence:.0f}d silent vs {mean_interval:.0f}d expected). Survival: {survival:.0%}.",
                    'urgency': 1.0 - survival,
                })

        # High momentum → predicted to keep evolving
        for item in momentum[:4]:
            eid = item.get('entity_id', '')
            entity = self.wm.entities.get(eid)
            if not entity:
                continue
            if _is_nayan_entity(entity.name, self.config):
                continue

            alive_p = item.get('alive', item.get('alive_probability', 0))
            if alive_p > 0.7:
                predictions.append({
                    'entity': entity.name,
                    'type': 'momentum',
                    'message': f"High momentum ({alive_p:.0%} alive). Likely to evolve soon.",
                    'urgency': 0.3,
                })

        predictions.sort(key=lambda p: -p['urgency'])
        predictions = predictions[:self.config.max_predictions]

        if not predictions:
            return ""

        lines = ["## Predictions & Attention\n"]
        for p in predictions:
            icon = "⚠️" if p['type'] == 'overdue' else "🔥"
            lines.append(f"{icon} **{p['entity']}**: {p['message']}")

        return "\n".join(lines)

    def _attention_flags(self, ref_t: float) -> str:
        """Things that need immediate attention."""
        flags = []

        # Check for entities with next_steps that are overdue
        for eid, entity in self.wm.entities.items():
            if _is_nayan_entity(entity.name, self.config):
                continue

            state = entity.current_state
            next_steps = state.get('next_steps', [])
            if not next_steps:
                continue

            tq = self.temporal.query(eid, ref_t) if self.temporal else {}
            status = tq.get('status', tq.get('classification', 'unknown'))
            silence = tq.get('silence_days', 0)

            if status in ('dormant', 'fading') and silence > 14:
                flags.append({
                    'entity': entity.name,
                    'message': f"Has defined next_steps but {silence:.0f}d of silence: {next_steps[0] if next_steps else ''}",
                    'urgency': silence / 30,
                })

        flags.sort(key=lambda f: -f['urgency'])
        flags = flags[:self.config.max_attention_flags]

        if not flags:
            return ""

        lines = ["## Needs Attention\n"]
        for f in flags:
            lines.append(f"- **{f['entity']}**: {f['message']}")

        return "\n".join(lines)

    def _focus_deep_dive(self, focus_project: str, ref_t: float) -> str:
        """Expanded context for the focused project."""
        # Find the entity
        target = None
        for eid, entity in self.wm.entities.items():
            if focus_project.lower() in entity.name.lower():
                target = entity
                break

        if not target:
            return ""

        lines = [f"## Deep Dive: {target.name}\n"]

        # Full state (not compressed)
        state = target.current_state
        for k, v in state.items():
            if v is None:
                continue
            v_str = str(v)
            if len(v_str) > 300:
                v_str = v_str[:297] + "..."
            lines.append(f"  {k}: {v_str}")

        # Recent transitions
        trans_ids = self.wm._entity_transitions.get(target.id, [])
        recent = sorted(
            [self.wm.transitions[tid] for tid in trans_ids if tid in self.wm.transitions],
            key=lambda t: t.timestamp,
            reverse=True
        )[:8]

        if recent:
            lines.append("\n### Recent Timeline")
            for t in recent:
                ago = _days_ago(t.timestamp, ref_t)
                summary = t.trigger_summary or t.transition_type.value
                lines.append(f"  [{ago}] {summary}")

        # Related entities
        rel_ids = self.wm._entity_relationships.get(target.id, [])
        if rel_ids:
            lines.append("\n### Connected Entities")
            for rid in rel_ids[:12]:
                rel = self.wm.relationships.get(rid)
                if not rel:
                    continue
                other_id = rel.target_id if rel.source_id == target.id else rel.source_id
                other = self.wm.entities.get(other_id)
                if other:
                    lines.append(f"  - {other.name} ({other.type.value})")

        return "\n".join(lines)

    def _behavioral_note(self) -> str:
        """Instructions for how the LLM should use this context."""
        return """## How to Use This Context

You are not just answering questions — you are a thinking partner with memory.
- Reference specific entities and their states when relevant.
- When the user mentions a project, you have its full history. Use it.
- Flag when something is overdue or predicted to change.
- If the user is working on something, check if related entities need attention.
- Predict what should happen next based on patterns, don't just describe what happened.
- Be specific: use numbers, dates, entity names. Not vague advice.
- Note: Pulse-Fi / PulseFi / WiFi CSI projects belong to Pranay (younger brother), not Parth. Nayan Bhatia is the PhD student collaborator on PulseFi.
- Parth = the user. UMD CS. Built Lucid Academy, sponsorFind, Lucid Labs, PIE, Hermes. Won MA State Science Fair. Research at CMU AirLab. Interned at Sanofi.
- The user wants to systematically track progress and make forward motion. Help with that."""


def generate_prompt(
    world_model_path: str = "output/world_model.json",
    focus: str | None = None,
    config: PromptConfig | None = None,
) -> str:
    """
    One-shot: load world model, generate prompt.

    Usage:
        prompt = generate_prompt("output/world_model.json", focus="sponsorFind")
    """
    wm = WorldModel(persist_path=world_model_path)

    engine = PromptEngine(wm, config)
    engine.initialize()
    return engine.generate(focus_project=focus)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PIE Prompt Engine")
    parser.add_argument("--world-model", default="output/world_model.json")
    parser.add_argument("--focus", default=None, help="Project to expand context for")
    parser.add_argument("--output", default=None, help="Write prompt to file")
    args = parser.parse_args()

    prompt = generate_prompt(args.world_model, focus=args.focus)

    if args.output:
        Path(args.output).write_text(prompt)
        print(f"Wrote {len(prompt)} chars to {args.output}")
    else:
        print(prompt)
        print(f"\n--- {len(prompt)} chars, ~{len(prompt)//4} tokens ---")
