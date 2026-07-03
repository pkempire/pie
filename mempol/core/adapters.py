"""Adapters that map existing repo data into universal memory primitives."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from mempol.data.locomo import load as load_locomo

from .schema import Artifact, MemoryState, Span
from .store import now_iso, stable_id


def _compact_json(obj: object, max_chars: int = 4000) -> str:
    text = json.dumps(obj, ensure_ascii=False, sort_keys=True)
    return text[:max_chars]


def _state_lines(state: dict) -> str:
    lines = []
    for k, v in state.items():
        if isinstance(v, (list, tuple)):
            vv = ", ".join(str(x) for x in v[:8])
        elif isinstance(v, dict):
            vv = _compact_json(v, 800)
        else:
            vv = str(v)
        lines.append(f"{k}: {vv}")
    return "\n".join(lines)


def pie_world_model_items(path: Path, limit: int = 0) -> Iterable[tuple[Artifact, list[Span], MemoryState]]:
    """Convert PIE `output/world_model.json` entities into universal memory.

    PIE remains a legacy domain view. Here each entity becomes one raw artifact,
    one evidence span, and one freeform memory state with provenance.
    """
    obj = json.loads(Path(path).read_text())
    entities = list((obj.get("entities") or {}).values())
    transitions = obj.get("transitions") or {}
    rels = obj.get("relationships") or {}
    if limit > 0:
        entities = entities[:limit]

    transitions_by_entity: dict[str, list[dict]] = {}
    for t in transitions.values():
        transitions_by_entity.setdefault(t.get("entity_id", ""), []).append(t)

    rels_by_entity: dict[str, list[dict]] = {}
    for r in rels.values():
        rels_by_entity.setdefault(r.get("source_id", ""), []).append(r)
        rels_by_entity.setdefault(r.get("target_id", ""), []).append(r)

    for e in entities:
        eid = str(e.get("id") or stable_id("pie_entity", e.get("name", "")))
        title = str(e.get("name") or eid)
        etype = str(e.get("type") or "unknown")
        current_state = e.get("current_state") or {}
        entity_transitions = sorted(
            transitions_by_entity.get(eid, []),
            key=lambda t: float(t.get("timestamp") or 0),
        )
        transition_summaries = [
            f"{t.get('transition_type', 'update')}: {t.get('trigger_summary', '')}"
            for t in entity_transitions[-8:]
        ]
        relationship_summaries = [
            f"{r.get('type', 'related_to')}: {r.get('description', '')}"
            for r in rels_by_entity.get(eid, [])[:8]
        ]
        content = "\n".join(
            part
            for part in [
                f"PIE entity: {title}",
                f"Legacy type/view: {etype}",
                f"Aliases: {', '.join(e.get('aliases') or [])}",
                "Current state:",
                _state_lines(current_state),
                "Recent transitions:",
                "\n".join(transition_summaries),
                "Relationships:",
                "\n".join(relationship_summaries),
            ]
            if part is not None
        )
        artifact = Artifact(
            id=f"pie_artifact_{eid}",
            source="pie_output",
            kind="world_model_entity",
            title=title,
            content=content,
            created_at=now_iso(),
            metadata={
                "legacy_entity_id": eid,
                "legacy_type": etype,
                "first_seen": e.get("first_seen"),
                "last_seen": e.get("last_seen"),
                "importance": e.get("importance"),
            },
        )
        span = Span(
            id=f"pie_span_{eid}",
            artifact_id=artifact.id,
            text=content,
            locator=title,
            metadata={"legacy_entity_id": eid, "legacy_type": etype},
        )
        state = MemoryState(
            id=f"pie_state_{eid}",
            content=content,
            source_span_ids=[span.id],
            created_at=now_iso(),
            updated_at=now_iso(),
            metadata={
                "adapter": "pie_output",
                "view_tags": [etype],
                "legacy_entity_id": eid,
                "name": title,
            },
        )
        yield artifact, [span], state


def architect_seed_items(path: Path, limit: int = 0) -> Iterable[tuple[Artifact, list[Span], MemoryState]]:
    """Convert architect seed components into universal memory."""
    components = json.loads(Path(path).read_text())
    if limit > 0:
        components = components[:limit]
    for c in components:
        slug = str(c.get("slug") or stable_id("component", c.get("name", "")))
        title = str(c.get("name") or slug)
        text = "\n".join(
            part
            for part in [
                f"AI/software component: {title}",
                f"One-liner: {c.get('one_liner', '')}",
                f"Summary: {c.get('summary', '')}",
                f"Kind: {c.get('kind') or c.get('type', '')}",
                f"Runtime: {c.get('runtime', '')}",
                f"Deployment: {c.get('deployment', '')}",
                f"Stack layer: {c.get('stack_layer', '')}",
                f"Tags: {', '.join(c.get('tags') or [])}",
                f"Homepage: {c.get('homepage_url', '')}",
                f"GitHub: {c.get('github_url', '')}",
                f"Integrates with: {', '.join(c.get('integrates_with') or [])}",
                f"Alternatives: {', '.join(c.get('alternative_to') or [])}",
            ]
            if part
        )
        artifact = Artifact(
            id=f"architect_artifact_{slug}",
            source="architect",
            kind="component",
            title=title,
            content=text,
            uri=str(c.get("homepage_url") or c.get("github_url") or ""),
            created_at=now_iso(),
            metadata={k: v for k, v in c.items() if k not in {"summary"}},
        )
        span = Span(
            id=f"architect_span_{slug}",
            artifact_id=artifact.id,
            text=text,
            locator=slug,
            metadata={"slug": slug, "tags": c.get("tags") or []},
        )
        state = MemoryState(
            id=f"architect_state_{slug}",
            content=text,
            source_span_ids=[span.id],
            created_at=now_iso(),
            updated_at=now_iso(),
            metadata={
                "adapter": "architect",
                "view_tags": ["software_component"] + list(c.get("tags") or []),
                "slug": slug,
                "name": title,
            },
        )
        yield artifact, [span], state


def locomo_items(n_convs: int = 1, limit: int = 0) -> Iterable[tuple[Artifact, list[Span], MemoryState]]:
    """Convert LoCoMo turns into universal memory for benchmark smoke tests."""
    count = 0
    for conv, _qas in load_locomo(n_convs=n_convs):
        for turn in conv.turns:
            if limit > 0 and count >= limit:
                return
            uid = f"{conv.sample_id}_{turn.dia_id.replace(':', '_')}"
            text = f"{turn.speaker}: {turn.text}"
            artifact = Artifact(
                id=f"locomo_artifact_{uid}",
                source="locomo",
                kind="conversation_turn",
                title=f"{conv.sample_id} {turn.dia_id}",
                content=text,
                created_at=turn.session_date,
                metadata={
                    "sample_id": conv.sample_id,
                    "dia_id": turn.dia_id,
                    "session": turn.session,
                    "speaker": turn.speaker,
                    "session_date": turn.session_date,
                },
            )
            span = Span(
                id=f"locomo_span_{uid}",
                artifact_id=artifact.id,
                text=text,
                locator=turn.dia_id,
                metadata=artifact.metadata,
            )
            state = MemoryState(
                id=f"locomo_state_{uid}",
                content=text,
                source_span_ids=[span.id],
                created_at=turn.session_date,
                updated_at=turn.session_date,
                metadata={"adapter": "locomo", "view_tags": ["conversation_turn"], **artifact.metadata},
            )
            count += 1
            yield artifact, [span], state
