"""
Context Compiler — Turn retrieved subgraph into LLM-ready markdown.

This is the interface between the graph and the LLM.
Both raw dates AND semantic context are included.
"""

from __future__ import annotations

from datetime import datetime
from dataclasses import dataclass
from typing import Optional

from pie.core.models import Entity, StateTransition, Relationship


@dataclass
class CompiledContext:
    """Compiled context ready for LLM consumption."""
    markdown: str
    entity_count: int
    transition_count: int
    token_estimate: int  # ~4 chars per token


def compile_entity_context(
    entity: Entity,
    transitions: list[StateTransition],
    relationships: list[tuple[Relationship, Entity]],  # (rel, other_entity)
    now: datetime,
) -> str:
    """
    Compile a single entity with its history into markdown.
    
    Includes BOTH raw dates and semantic context.
    """
    lines = []
    
    # Header with dual temporal representation
    first_date = datetime.fromtimestamp(entity.first_seen)
    last_date = datetime.fromtimestamp(entity.last_seen)
    
    lines.append(f"## {entity.name}")
    lines.append(f"**Type:** {entity.type.value}")
    lines.append(f"**First seen:** {first_date.strftime('%Y-%m-%d')} ({_humanize(now, first_date)})")
    lines.append(f"**Last seen:** {last_date.strftime('%Y-%m-%d')} ({_humanize(now, last_date)})")
    
    # Current state
    if entity.current_state:
        lines.append("")
        lines.append("**Current state:**")
        for k, v in entity.current_state.items():
            if k not in ['embedding', 'id']:
                lines.append(f"- {k}: {v}")
    
    # State transitions (the full history)
    if transitions:
        lines.append("")
        lines.append(f"**History ({len(transitions)} changes):**")
        
        # Compute velocity
        if len(transitions) > 1:
            span_days = (entity.last_seen - entity.first_seen) / 86400
            if span_days > 0:
                velocity = len(transitions) / (span_days / 30)  # changes per month
                lines.append(f"_Change velocity: {velocity:.1f}/month_")
        
        # List transitions (most recent first, limit to 10)
        for t in sorted(transitions, key=lambda x: x.timestamp, reverse=True)[:10]:
            t_date = datetime.fromtimestamp(t.timestamp)
            t_relative = _humanize(now, t_date)
            
            icon = "⚠️" if t.transition_type.value == "contradiction" else "•"
            lines.append(f"{icon} **{t_date.strftime('%Y-%m-%d')}** ({t_relative}): {t.trigger_summary}")
            
            if t.transition_type.value == "contradiction":
                lines.append(f"  _This contradicted prior state_")
        
        if len(transitions) > 10:
            lines.append(f"  _... and {len(transitions) - 10} earlier changes_")
    
    # Relationships
    if relationships:
        lines.append("")
        lines.append("**Relationships:**")
        for rel, other in relationships[:10]:
            lines.append(f"- {rel.type.value} → {other.name} ({other.type.value})")
    
    return "\n".join(lines)


def compile_subgraph(
    entities: list[Entity],
    transitions_map: dict[str, list[StateTransition]],  # entity_id -> transitions
    relationships_map: dict[str, list[tuple[Relationship, Entity]]],
    now: datetime,
    query: Optional[str] = None,
) -> CompiledContext:
    """
    Compile a full subgraph into markdown.
    
    This is what the LLM sees when answering a query.
    """
    sections = []
    
    # Header
    if query:
        sections.append(f"# Retrieved Context for: \"{query}\"")
    else:
        sections.append("# Retrieved Context")
    
    sections.append(f"_Retrieved {len(entities)} entities at {now.strftime('%Y-%m-%d %H:%M')}_")
    sections.append("")
    
    # Compile each entity
    total_transitions = 0
    for entity in entities:
        transitions = transitions_map.get(entity.id, [])
        relationships = relationships_map.get(entity.id, [])
        total_transitions += len(transitions)
        
        section = compile_entity_context(entity, transitions, relationships, now)
        sections.append(section)
        sections.append("")  # blank line between entities
    
    markdown = "\n".join(sections)
    
    return CompiledContext(
        markdown=markdown,
        entity_count=len(entities),
        transition_count=total_transitions,
        token_estimate=len(markdown) // 4,
    )


def compile_timeline(
    transitions: list[StateTransition],
    entities_map: dict[str, Entity],  # entity_id -> Entity
    now: datetime,
) -> str:
    """
    Compile a chronological timeline view across all entities.
    
    Useful for "what happened during period X?" queries.
    """
    lines = ["# Timeline", ""]
    
    # Sort by time
    sorted_transitions = sorted(transitions, key=lambda t: t.timestamp)
    
    current_month = None
    for t in sorted_transitions:
        t_date = datetime.fromtimestamp(t.timestamp)
        t_month = t_date.strftime("%B %Y")
        
        # Month header
        if t_month != current_month:
            lines.append(f"## {t_month}")
            current_month = t_month
        
        entity = entities_map.get(t.entity_id)
        entity_name = entity.name if entity else "Unknown"
        
        icon = "⚠️" if t.transition_type.value == "contradiction" else "•"
        lines.append(f"{icon} **{t_date.strftime('%Y-%m-%d')}** [{entity_name}]: {t.trigger_summary}")
    
    return "\n".join(lines)


def _humanize(now: datetime, then: datetime) -> str:
    """Humanize a datetime relative to now."""
    delta = now - then
    days = abs(delta.days)
    
    if days == 0:
        return "today"
    elif days == 1:
        return "yesterday" if delta.days > 0 else "tomorrow"
    elif days < 7:
        return f"{days} days ago"
    elif days < 30:
        weeks = days // 7
        return f"{weeks} week{'s' if weeks > 1 else ''} ago"
    elif days < 365:
        months = days // 30
        return f"{months} month{'s' if months > 1 else ''} ago"
    else:
        years = days // 365
        return f"~{years} year{'s' if years > 1 else ''} ago"


# =============================================================================
# Example output
# =============================================================================

EXAMPLE_OUTPUT = """
# Retrieved Context for: "What's the current state of Science Research Academy?"

_Retrieved 3 entities at 2026-02-07 01:30_

## Science Research Academy
**Type:** project
**First seen:** 2024-03-15 (~22 months ago)
**Last seen:** 2026-01-20 (3 weeks ago)

**Current state:**
- status: active
- focus: research curriculum
- students: 100+
- lead: Pranay

**History (7 changes):**
_Change velocity: 0.3/month_
• **2026-01-20** (3 weeks ago): Pranay now running day-to-day, Parth advisory
• **2025-09-01** (5 months ago): Expanded to 100+ students
⚠️ **2025-05-15** (9 months ago): Pivoted from mentoring to research curriculum
  _This contradicted prior state_
• **2025-02-01** (12 months ago): Launched at scifair.tech with 30 students
• **2024-03-15** (22 months ago): Founded with Pranay as educational platform

**Relationships:**
- works_on → Pranay (person)
- uses → scifair.tech (tool)
- part_of → Edison Scientific (organization)

## Pranay
**Type:** person
**First seen:** 2024-03-15 (~22 months ago)
**Last seen:** 2026-01-20 (3 weeks ago)

**Current state:**
- role: co-founder, running SRA day-to-day
- relationship: brother

**Relationships:**
- works_on → Science Research Academy (project)
- collaborates_with → Parth (person)
"""
