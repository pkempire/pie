"""PIE knowledge-graph backend.

Wraps `pie.core.world_model.WorldModel` so the same world_model.json that PIE
produces today is the artefact a trained write-policy produces from RL rollouts.

Exposes:
  - The standard read interface (`Backend.retrieve` / `expand` / `filter_by_time`)
    so read-policy training works against this backend.
  - PIE-shaped write operations (`lookup_entity`, `create_entity`, `update_state`,
    `add_relation`, `mark_contradiction`, `forget`, `merge_entities`).

The write ops give us byte-for-byte the same entities/transitions/relationships
schema PIE has been writing all along — no parallel KB, no schema drift.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any

import numpy as np

from pie.core.models import (
    Entity, EntityType, RelationshipType, StateTransition, TransitionType,
)
from pie.core.world_model import WorldModel

from .. import llm
from .base import Backend, Hit, Unit


# ── helpers ──
def _as_entity_type(value: str) -> EntityType:
    try:
        return EntityType(value.lower())
    except ValueError:
        return EntityType.CONCEPT


def _as_transition_type(value: str) -> TransitionType:
    try:
        return TransitionType(value.lower())
    except ValueError:
        return TransitionType.UPDATE


def _as_relationship_type(value: str) -> RelationshipType:
    try:
        return RelationshipType(value.lower())
    except ValueError:
        return RelationshipType.RELATED_TO


def _entity_to_text(e: Entity) -> str:
    state_str = ", ".join(f"{k}: {v}" for k, v in (e.current_state or {}).items())[:300]
    return f"{e.name} ({e.type.value}) — {state_str}"


def _entity_to_hit(e: Entity, score: float, source: str, n_transitions: int = 0) -> Hit:
    """Module-level helper. n_transitions is passed by callers that hold
    the WorldModel reference; default 0 for callers that don't need it."""
    return Hit(
        unit=Unit(
            uid=e.id,
            text=_entity_to_text(e),
            metadata={
                "name": e.name,
                "type": e.type.value,
                "current_state": e.current_state,
                "first_seen": e.first_seen,
                "last_seen": e.last_seen,
                "n_transitions": n_transitions,
                "importance": e.importance,
                "aliases": list(e.aliases or []),
            },
        ),
        score=score,
        source=source,
    )


# ── Backend implementation ──
class PIEBackend(Backend):
    """Read+write backend backed by PIE's WorldModel.

    Each write op returns a JSON-able status string suitable for use as a tool
    observation in the Tinker env.
    """
    name = "pie_kg"

    def __init__(self, world_model: WorldModel | None = None):
        self.wm = world_model or WorldModel()

    # ── Backend.read interface ──
    def ingest(self, units: list[Unit]) -> None:
        """No-op for the typical RL use-case: a write-policy populates the
        world model via tool calls, not via batch ingestion. We keep this
        method for interface compatibility — call it only if you want to
        seed the KB from raw turns via PIE's own pipeline (heavy)."""
        for u in units:
            # Cheap fallback: each turn becomes a low-importance EVENT entity.
            # Real ingestion goes through the write_tools / RL policy.
            self.wm.create_entity(
                name=u.uid,
                type=EntityType.EVENT,
                state={"text": u.text[:500]},
                source_conversation_id=u.metadata.get("conv_id", ""),
                timestamp=float(u.metadata.get("timestamp") or 0.0),
            )

    def retrieve(self, query: str, k: int = 10, source: str = "hybrid") -> list[Hit]:
        # Dense retrieval over entities. WorldModel maintains an embedding
        # matrix internally; we use its `find_by_embedding` API.
        if not self.wm.entities:
            return []
        # Ensure missing embeddings are filled — the policy may create entities
        # at runtime that lack vectors.
        missing = self.wm.get_entities_without_embeddings()
        if missing:
            texts = [_entity_to_text(e) for e in missing]
            embs = llm.embed(texts)
            for e, v in zip(missing, embs):
                self.wm.set_entity_embedding(e.id, v.tolist())
            self.wm.rebuild_embedding_matrix()

        if source == "bm25":
            q = (query or "").lower()
            scored: list[tuple[Entity, float]] = []
            for e in self.wm.entities.values():
                hay = (e.name + " " + " ".join(str(v) for v in (e.current_state or {}).values())).lower()
                if q and q in hay:
                    scored.append((e, 1.0))
            scored.sort(key=lambda x: x[1], reverse=True)
            return [_entity_to_hit(e, s, "bm25") for e, s in scored[:k]]
        # dense / hybrid both go through the embedding API.
        q_emb = llm.embed([query])[0].tolist()
        matches = self.wm.find_by_embedding(q_emb, top_k=k)
        hits = [_entity_to_hit(e, score, "dense") for e, score in matches]
        if source == "hybrid":
            string_matches = self.wm.find_by_string_match(query, threshold=0.7)
            seen = {h.unit.uid for h in hits}
            for e, score in string_matches:
                if e.id not in seen:
                    hits.append(_entity_to_hit(e, score, "string"))
                    seen.add(e.id)
        return hits[:k]

    def expand(self, seed_uids: list[str], k_per: int = 2) -> list[Hit]:
        out, seen = [], set(seed_uids)
        for uid in seed_uids:
            for nb in self.wm.get_neighbors(uid)[:k_per]:
                if nb in seen or nb not in self.wm.entities:
                    continue
                out.append(_entity_to_hit(self.wm.entities[nb], 0.5, "expand_kg"))
                seen.add(nb)
        return out

    # ── PIE-shaped write API (used by write_tools.py) ──
    def _ensure_embeddings(self) -> None:
        """Backfill embeddings for any entities created mid-rollout."""
        missing = self.wm.get_entities_without_embeddings()
        if not missing:
            return
        texts = [_entity_to_text(e) for e in missing]
        embs = llm.embed(texts)
        for e, v in zip(missing, embs):
            self.wm.set_entity_embedding(e.id, v.tolist())
        self.wm.rebuild_embedding_matrix()

    def lookup_entity(
        self, query: str, type: str | None = None, top_k: int = 5
    ) -> list[dict]:
        """Find existing entities matching `query`. Returns a JSONable list.

        Uses BOTH string match and embedding similarity (PIE's tier-1 + tier-2).
        Returns top-k candidates with their key state — designed so the policy
        can decide whether to create a new entity or merge with an existing one.
        """
        candidates: dict[str, float] = {}
        for entity, score in self.wm.find_by_string_match(query, threshold=0.6, entity_type=type):
            candidates[entity.id] = max(candidates.get(entity.id, 0.0), score)
        if self.wm.entities:
            self._ensure_embeddings()
            try:
                q_emb = llm.embed([query])[0].tolist()
                for entity, score in self.wm.find_by_embedding(q_emb, top_k=top_k * 2, entity_type=type):
                    candidates[entity.id] = max(candidates.get(entity.id, 0.0), score)
            except Exception:
                pass
        ranked = sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:top_k]
        out = []
        for uid, score in ranked:
            e = self.wm.entities.get(uid)
            if not e:
                continue
            out.append({
                "uid": uid,
                "name": e.name,
                "type": e.type.value,
                "current_state": e.current_state,
                "n_transitions": len(self.wm.get_transitions(uid)),
                "importance": round(e.importance, 4),
                "aliases": list(e.aliases or []),
                "match_score": round(score, 4),
                "first_seen": e.first_seen,
                "last_seen": e.last_seen,
            })
        return out

    def lookup_relation(self, uid_a: str, uid_b: str | None = None) -> list[dict]:
        out = []
        rels = self.wm.get_relationships(uid_a)
        for r in rels:
            if uid_b and (r.target_id != uid_b and r.source_id != uid_b):
                continue
            out.append({
                "source_id": r.source_id,
                "target_id": r.target_id,
                "type": r.type.value if hasattr(r.type, "value") else str(r.type),
                "description": r.description,
                "timestamp": r.timestamp,
            })
        return out

    def create_entity(
        self, name: str, type: str, state: dict, source: str = "", timestamp: float = 0.0,
    ) -> str:
        e = self.wm.create_entity(
            name=name,
            type=_as_entity_type(type),
            state=state or {},
            source_conversation_id=source,
            timestamp=timestamp,
        )
        return e.id

    def update_state(
        self,
        uid: str,
        new_state: dict,
        transition_type: str = "update",
        source: str = "",
        timestamp: float = 0.0,
        trigger_summary: str = "",
    ) -> bool:
        is_contradiction = transition_type.lower() == "contradiction"
        tr = self.wm.update_entity_state(
            entity_id=uid,
            new_state=new_state,
            source_conversation_id=source,
            timestamp=timestamp,
            trigger_summary=trigger_summary,
            is_contradiction=is_contradiction,
        )
        return tr is not None

    def add_alias(self, uid: str, alias: str) -> bool:
        return bool(self.wm.add_alias(uid, alias))

    def merge_entities(self, canonical_uid: str, alias_uid: str) -> bool:
        """Merge `alias_uid` into `canonical_uid`. Moves transitions and
        relationships only. Does NOT auto-add the alias's name as an alias
        on the canonical — that path leaks garbage through PIE's loose
        word-overlap filter. If the policy wants the alias name preserved,
        it must emit an explicit `add_alias(canonical_uid, alias_text)` op.
        """
        canonical = self.wm.entities.get(canonical_uid)
        alias = self.wm.entities.get(alias_uid)
        if not canonical or not alias or canonical_uid == alias_uid:
            return False
        # Reassign transitions (transitions live in wm.transitions[entity_id])
        for tr in self.wm.get_transitions(alias_uid):
            tr.entity_id = canonical_uid
        if hasattr(self.wm, "transitions") and alias_uid in self.wm.transitions:
            for tr in self.wm.transitions.get(alias_uid, []):
                self.wm.transitions.setdefault(canonical_uid, []).append(tr)
            self.wm.transitions[alias_uid] = []
        # Reassign relationships
        for r in self.wm.get_relationships(alias_uid):
            if r.source_id == alias_uid:
                r.source_id = canonical_uid
            if r.target_id == alias_uid:
                r.target_id = canonical_uid
        # Mark alias archived (don't delete — preserve audit trail)
        self.wm.update_entity_state(
            entity_id=alias_uid,
            new_state={"merged_into": canonical_uid},
            source_conversation_id="merge",
            timestamp=alias.last_seen,
            trigger_summary=f"merged into {canonical.name}",
            is_contradiction=False,
        )
        return True

    def add_relation(
        self, source_uid: str, target_uid: str, rel_type: str, description: str = "",
        timestamp: float = 0.0,
    ) -> bool:
        if source_uid not in self.wm.entities or target_uid not in self.wm.entities:
            return False
        # PIE's WorldModel.add_relationship uses kwarg `rel_type`, not `type`.
        self.wm.add_relationship(
            source_id=source_uid,
            target_id=target_uid,
            rel_type=_as_relationship_type(rel_type),
            description=description,
            timestamp=timestamp,
        )
        return True

    def mark_contradiction(
        self, uid: str, contradicting_state: dict, source: str = "", timestamp: float = 0.0,
    ) -> bool:
        return self.update_state(
            uid=uid,
            new_state=contradicting_state,
            transition_type="contradiction",
            source=source,
            timestamp=timestamp,
            trigger_summary="contradiction logged by policy",
        )

    def forget(self, uid: str, reason: str = "") -> bool:
        e = self.wm.entities.get(uid)
        if not e:
            return False
        return self.update_state(
            uid=uid,
            new_state={"archived": True, "reason": reason},
            transition_type="archival",
            source="",
            timestamp=e.last_seen,
            trigger_summary=f"archived: {reason}",
        )

    # ── Convenience ──
    def stats(self) -> dict:
        return self.wm.stats()

    def save(self, path: str | Path):
        from pathlib import Path as _P
        self.wm.persist_path = _P(path)
        self.wm.save()
