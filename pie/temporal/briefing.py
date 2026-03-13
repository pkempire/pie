"""
Temporal Briefing Generator — the core of the MCP product.

Replaces temporal.py's survival function math with raw temporal metadata
that the LLM reasons over directly. Philosophy: compute simple arithmetic
(gaps, ratios, counts), let the LLM do the interpretation.

Two outputs:
1. Full briefing (~3-4K tokens) — injected at conversation start
2. Per-entity temporal metadata — available for individual queries
"""

from __future__ import annotations

import time
import json
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Any

from pie.core.world_model import WorldModel
from pie.core.models import EntityType


# ── Helpers ──────────────────────────────────────────────────────────────────

def _humanize_delta(seconds: float) -> str:
    """Convert seconds to human-readable relative time."""
    days = seconds / 86400
    if days < 0.04:  # < 1 hour
        minutes = int(seconds / 60)
        return f"{minutes}m ago" if minutes > 0 else "just now"
    if days < 1:
        hours = int(days * 24)
        return f"{hours}h ago"
    if days < 2:
        return "yesterday"
    if days < 7:
        return f"{days:.0f} days ago"
    if days < 14:
        return "last week"
    if days < 30:
        weeks = int(days / 7)
        return f"{weeks} weeks ago"
    if days < 60:
        return "last month"
    if days < 365:
        months = int(days / 30)
        return f"{months} months ago"
    years = days / 365
    return f"{years:.1f} years ago"


def _day_of_week(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%A")


def _date_str(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d")


def _datetime_str(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")


# ── Entity Temporal Metadata ─────────────────────────────────────────────────

@dataclass
class EntityTemporalMeta:
    """Raw temporal stats for one entity. Pure arithmetic, no interpretation."""
    entity_id: str
    name: str
    entity_type: str
    first_seen: float
    last_seen: float
    days_silent: float          # days since last update
    total_transitions: int
    avg_gap_days: float | None  # mean gap between transitions
    last_3_gaps_days: list[float]
    gap_ratio: float | None     # days_silent / avg_gap — >1 means overdue
    has_next_steps: bool
    next_steps: list[str]
    status_from_state: str      # whatever "status" field says in current_state
    importance: float
    current_state_summary: str  # first 200 chars of description


class TemporalBriefing:
    """Generate temporal briefings from world model metadata."""

    def __init__(self, world_model: WorldModel):
        self.wm = world_model

    def compute_entity_metadata(self, entity_id: str, ref_time: float) -> EntityTemporalMeta:
        """Compute raw temporal stats for a single entity."""
        entity = self.wm.entities[entity_id]
        transitions = self.wm.get_transitions(entity_id, ordered=True)

        # Compute gaps between consecutive transitions
        timestamps = sorted(t.timestamp for t in transitions)
        gaps = []
        for i in range(1, len(timestamps)):
            gap_days = (timestamps[i] - timestamps[i - 1]) / 86400
            gaps.append(gap_days)

        days_silent = (ref_time - entity.last_seen) / 86400

        avg_gap = sum(gaps) / len(gaps) if gaps else None
        gap_ratio = days_silent / avg_gap if avg_gap and avg_gap > 0 else None

        # Extract useful state fields
        cs = entity.current_state
        status = cs.get("status", cs.get("phase", cs.get("stage", "")))
        description = cs.get("description", "")
        if len(description) > 200:
            description = description[:197] + "..."
        next_steps = cs.get("next_steps", [])
        if isinstance(next_steps, str):
            next_steps = [next_steps]

        return EntityTemporalMeta(
            entity_id=entity_id,
            name=entity.name,
            entity_type=entity.type.value,
            first_seen=entity.first_seen,
            last_seen=entity.last_seen,
            days_silent=round(days_silent, 1),
            total_transitions=len(transitions),
            avg_gap_days=round(avg_gap, 1) if avg_gap else None,
            last_3_gaps_days=[round(g, 1) for g in gaps[-3:]] if gaps else [],
            gap_ratio=round(gap_ratio, 1) if gap_ratio else None,
            has_next_steps=bool(next_steps),
            next_steps=next_steps[:3],
            status_from_state=str(status)[:100] if status else "",
            importance=entity.importance or 0,
            current_state_summary=description,
        )

    def get_all_metadata(self, ref_time: float, min_transitions: int = 2) -> list[EntityTemporalMeta]:
        """Get temporal metadata for all entities with enough history."""
        results = []
        for eid in self.wm.entities:
            transitions = self.wm.get_transitions(eid, ordered=True)
            if len(transitions) < min_transitions:
                continue
            try:
                meta = self.compute_entity_metadata(eid, ref_time)
                results.append(meta)
            except Exception:
                continue
        return results

    def generate_briefing(
        self,
        ref_time: float | None = None,
        focus_project: str | None = None,
        last_interaction_time: float | None = None,
        approaching_deadlines: list[dict] | None = None,
        overdue_commitments: list[dict] | None = None,
    ) -> str:
        """
        Generate the full temporal briefing.

        This is THE core output — 3-4K tokens of structured temporal context
        that gets injected into every conversation.
        """
        now = ref_time or time.time()
        now_str = _datetime_str(now)
        day = _day_of_week(now)

        all_meta = self.get_all_metadata(now)

        sections = []

        # ── Header ──
        sections.append(self._header(now_str, day, last_interaction_time, now))

        # ── Active Projects ──
        sections.append(self._projects_section(all_meta, now, focus_project))

        # ── Goals ──
        goals_section = self._goals_section(all_meta, now)
        if goals_section:
            sections.append(goals_section)

        # ── Deadlines & Commitments ──
        if approaching_deadlines or overdue_commitments:
            sections.append(self._deadlines_section(approaching_deadlines, overdue_commitments))

        # ── Attention Flags ──
        attention = self._attention_section(all_meta, now)
        if attention:
            sections.append(attention)

        # ── Focus Deep Dive ──
        if focus_project:
            deep = self._focus_section(focus_project, now)
            if deep:
                sections.append(deep)

        # ── Behavioral Instructions ──
        sections.append(self._behavioral_note())

        return "\n\n".join(s for s in sections if s)

    def _header(self, now_str: str, day: str, last_interaction: float | None, now: float) -> str:
        """Identity + temporal context + gap analysis."""
        total_entities = len(self.wm.entities)
        total_transitions = len(self.wm.transitions)

        lines = [
            f"## Temporal Briefing — {now_str} ({day})",
            "",
            f"You are the AI thinking partner for Pranay.",
            f"World model: {total_entities} entities, {total_transitions} state transitions.",
        ]

        # Gap analysis
        if last_interaction:
            gap_hours = (now - last_interaction) / 3600
            gap_days = gap_hours / 24

            if gap_hours < 1:
                lines.append(f"Last interaction: {int(gap_hours * 60)}m ago. Continuing current session.")
            elif gap_hours < 24:
                lines.append(f"Last interaction: {gap_hours:.0f}h ago ({_day_of_week(last_interaction)}).")
            elif gap_days < 7:
                lines.append(f"Last interaction: {gap_days:.1f} days ago ({_day_of_week(last_interaction)} {_date_str(last_interaction)}). Short absence.")
            elif gap_days < 30:
                lines.append(f"Last interaction: {gap_days:.0f} days ago ({_date_str(last_interaction)}). Extended absence — ask what happened.")
            else:
                lines.append(f"Last interaction: {gap_days:.0f} days ago ({_date_str(last_interaction)}). Long absence — re-establish context.")
        else:
            lines.append("First interaction recorded.")

        return "\n".join(lines)

    def _projects_section(self, all_meta: list[EntityTemporalMeta], now: float, focus: str | None) -> str:
        """Active projects ranked by temporal urgency."""
        projects = [m for m in all_meta if m.entity_type == "project"]

        # Filter out projects with nonsensical gap ratios (avg_gap < 1 day means data is too noisy)
        projects = [p for p in projects if p.avg_gap_days is None or p.avg_gap_days >= 1.0]

        # Sort: focus first, then by total transitions (most active first), with importance as tiebreak
        def sort_key(m: EntityTemporalMeta):
            is_focus = focus and focus.lower() in m.name.lower()
            return (not is_focus, -m.total_transitions, -m.importance)

        projects.sort(key=sort_key)
        projects = projects[:10]

        if not projects:
            return ""

        lines = ["### Active Projects\n"]
        for p in projects:
            # Temporal status label
            if p.gap_ratio is not None:
                if p.gap_ratio < 0.5:
                    tempo = "on rhythm"
                elif p.gap_ratio < 1.5:
                    tempo = "normal"
                elif p.gap_ratio < 3:
                    tempo = "quiet"
                elif p.gap_ratio < 6:
                    tempo = "dormant"
                else:
                    tempo = "inactive"
            else:
                tempo = "new"

            focus_marker = " ← FOCUS" if focus and focus.lower() in p.name.lower() else ""

            lines.append(f"**{p.name}** [{tempo}]{focus_marker}")
            lines.append(f"  Last update: {_humanize_delta(p.days_silent * 86400)} | "
                         f"Rhythm: ~{p.avg_gap_days}d | "
                         f"Updates: {p.total_transitions} | "
                         f"Gap ratio: {p.gap_ratio}x")

            if p.status_from_state:
                lines.append(f"  Status: {p.status_from_state}")

            if p.has_next_steps:
                steps_str = "; ".join(str(s)[:80] for s in p.next_steps)
                lines.append(f"  Next steps: {steps_str}")

            if p.gap_ratio and p.gap_ratio > 3:
                lines.append(f"  ⚠️ Silent {p.gap_ratio}x its normal rhythm — may need attention or shelving")

            lines.append("")

        return "\n".join(lines)

    def _goals_section(self, all_meta: list[EntityTemporalMeta], now: float) -> str:
        """Active goals from the world model."""
        goals = [m for m in all_meta if m.entity_type == "goal"]
        # Filter out likely dead goals
        goals = [g for g in goals if g.days_silent < 180]
        goals.sort(key=lambda g: g.days_silent)
        goals = goals[:8]

        if not goals:
            return ""

        lines = ["### Goals\n"]
        for g in goals:
            lines.append(f"- **{g.name}** (last touched {_humanize_delta(g.days_silent * 86400)})")
            if g.current_state_summary:
                lines.append(f"  {g.current_state_summary[:120]}")

        return "\n".join(lines)

    def _deadlines_section(self, deadlines: list[dict] | None, commitments: list[dict] | None) -> str:
        """Approaching deadlines and overdue commitments."""
        lines = ["### Deadlines & Commitments\n"]

        if deadlines:
            for d in deadlines:
                lines.append(f"⏰ **{d.get('topic', 'Unknown')}** — due {d.get('due_date', '?')}")
                if d.get('description'):
                    lines.append(f"   {d['description'][:100]}")

        if commitments:
            for c in commitments:
                lines.append(f"⚠️ **OVERDUE**: {c.get('what', '?')} (was due {c.get('due_date', '?')})")

        if not deadlines and not commitments:
            lines.append("No tracked deadlines or commitments.")

        return "\n".join(lines)

    def _attention_section(self, all_meta: list[EntityTemporalMeta], now: float) -> str:
        """Things that deserve proactive mention."""
        flags = []

        for m in all_meta:
            # Skip low-importance stuff
            if m.importance < 0.1 and m.total_transitions < 5:
                continue

            # Flag: has next_steps but gone silent
            if m.has_next_steps and m.days_silent > 14 and m.gap_ratio and m.gap_ratio > 2:
                flags.append({
                    "entity": m.name,
                    "reason": f"Has defined next steps but {m.days_silent:.0f}d of silence ({m.gap_ratio}x rhythm)",
                    "urgency": m.gap_ratio * m.importance,
                    "next_steps": m.next_steps,
                })

            # Flag: high-importance entity gone dormant
            if m.importance > 0.3 and m.gap_ratio and m.gap_ratio > 5:
                flags.append({
                    "entity": m.name,
                    "reason": f"Important entity ({m.importance:.2f}) gone dormant — {m.days_silent:.0f}d silent",
                    "urgency": m.gap_ratio * m.importance,
                })

        flags.sort(key=lambda f: -f["urgency"])
        flags = flags[:6]

        if not flags:
            return ""

        lines = ["### Attention Flags\n"]
        for f in flags:
            lines.append(f"- **{f['entity']}**: {f['reason']}")
            if f.get('next_steps'):
                lines.append(f"  Pending: {'; '.join(str(s)[:60] for s in f['next_steps'])}")

        return "\n".join(lines)

    def _focus_section(self, focus_name: str, now: float) -> str:
        """Deep dive for a specific project."""
        # Find entity — prefer exact match, then substring
        entity = self.wm.find_by_name(focus_name)
        if not entity:
            # Try substring match, preferring shorter names (more specific)
            candidates = []
            for eid, e in self.wm.entities.items():
                if focus_name.lower() in e.name.lower():
                    candidates.append(e)
            if candidates:
                candidates.sort(key=lambda e: len(e.name))
                entity = candidates[0]
        if not entity:
            return ""

        lines = [f"### Deep Dive: {entity.name}\n"]

        # Full state
        for k, v in entity.current_state.items():
            if v is None:
                continue
            v_str = str(v)
            if len(v_str) > 250:
                v_str = v_str[:247] + "..."
            lines.append(f"  {k}: {v_str}")

        # Recent transitions
        transitions = self.wm.get_transitions(entity.id, ordered=True)
        recent = transitions[-8:] if transitions else []
        recent.reverse()

        if recent:
            lines.append("\n**Recent timeline:**")
            for t in recent:
                ago = _humanize_delta(now - t.timestamp)
                summary = t.trigger_summary or t.transition_type.value
                lines.append(f"  [{ago}] {summary}")

        # Relationships
        rels = self.wm.get_relationships(entity.id)
        if rels:
            lines.append("\n**Connected to:**")
            for r in rels[:10]:
                other_id = r.target_id if r.source_id == entity.id else r.source_id
                other = self.wm.entities.get(other_id)
                if other:
                    lines.append(f"  - {other.name} ({r.type.value}: {r.description[:80] if r.description else ''})")

        return "\n".join(lines)

    def _behavioral_note(self) -> str:
        return """### How to Use This Briefing

You have temporal context about Pranay's world. Use it:
- Reference specific projects and their current state when relevant
- If something is overdue or dormant, mention it naturally (don't lecture)
- Track commitments: if he said he'd do X by Y, follow up
- Notice gaps: if he's been away, acknowledge it and ask what changed
- Be specific: use names, timelines, gap ratios — not vague advice
- If a project has next_steps defined, those are actionable items to reference"""
