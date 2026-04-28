"""KGmem (PIE) backend with hybrid retrieval.

Wraps `pie.core.world_model.WorldModel` so the same world_model.json that the
prior KGmem system produces is the artefact a trained write-policy produces
from RL rollouts.

Retrieval design (post-Discovery audit, Apr 2026)
=================================================
An earlier version of this file used named-entity-only retrieval as the
default. A controlled study on a 50-question LoCoMo paired sample showed
this single design choice collapses end-to-end QA from 0.50 (FlatBackend
hybrid retrieval) to 0.10 (NER-only KG retrieval), McNemar p = 1.9e-6, and
the gap persists on multi-hop questions. The fix was to fuse NER lookup,
BM25 over entity textual representations, and dense embedding similarity
via reciprocal-rank fusion. That recovered 75 percent of the lost
accuracy. We default to the fused mode here and treat
backend-retriever alignment as a first-class methodological concern: the
read and write sides train and evaluate against the same retriever.

Read interface
  retrieve(query, k, source)   source ∈ {bm25, dense, ner, hybrid}
                                hybrid is the default and fuses all three
  expand, filter_by_time       inherited from Backend

Write interface (driven by the policy's tool calls)
  lookup_entity, lookup_relation
  create_entity, update_state, merge_entities, add_relation,
  mark_contradiction, forget
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
from .flat import BM25Index, _tokens, _rrf


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

    def _build_bm25_index(self) -> tuple[BM25Index, list[str]]:
        """Compact BM25 over entity textual representations.

        Each entity is one document; the document text is name + aliases +
        flattened current_state (skipping container brackets). Tokens are
        the Flat backend's regex tokenizer for consistency. Cached
        per-call (cheap, KGs are small) — for very large KGs we'd build
        and persist incrementally; not worth it at LoCoMo scale.
        """
        index = BM25Index()
        uids: list[str] = []
        for uid, e in self.wm.entities.items():
            parts = [e.name or ""]
            parts.extend(e.aliases or [])
            for v in (e.current_state or {}).values():
                if isinstance(v, (str, int, float, bool)):
                    parts.append(str(v))
            text = " ".join(parts).lower()
            index.add(_tokens(text))
            uids.append(uid)
        return index, uids

    def retrieve(self, query: str, k: int = 10, source: str = "hybrid") -> list[Hit]:
        """Hybrid retrieval over the KG.

        source:
          ner     — name/alias substring match only (the original collapse-
                    inducing default; preserved for ablations)
          bm25    — proper BM25 over entity name + aliases + state values
          dense   — embedding similarity over entity textual rep
          hybrid  — RRF fusion of NER + BM25 + dense (default)

        Empty KG returns empty list. Missing embeddings are backfilled lazily
        because the write policy may create entities at runtime that have no
        vector.
        """
        if not self.wm.entities:
            return []

        # Backfill embeddings created mid-rollout
        missing = self.wm.get_entities_without_embeddings()
        if missing:
            texts = [_entity_to_text(e) for e in missing]
            embs = llm.embed(texts)
            for e, v in zip(missing, embs):
                self.wm.set_entity_embedding(e.id, v.tolist())
            self.wm.rebuild_embedding_matrix()

        # ── Primitive rankers (each returns ordered list of (uid, score)) ──
        def _ner_rank() -> list[tuple[str, float]]:
            out: list[tuple[str, float]] = []
            for e, score in self.wm.find_by_string_match(query, threshold=0.6):
                out.append((e.id, float(score)))
            return out

        def _bm25_rank() -> list[tuple[str, float]]:
            index, uids = self._build_bm25_index()
            if not uids:
                return []
            q_tokens = _tokens(query or "")
            scored = [(uids[i], index.score(q_tokens, i)) for i in range(len(uids))]
            scored = [(u, s) for u, s in scored if s > 0]
            scored.sort(key=lambda x: x[1], reverse=True)
            return scored

        def _dense_rank() -> list[tuple[str, float]]:
            try:
                q_emb = llm.embed([query])[0].tolist()
            except Exception:
                return []
            return [(e.id, float(score))
                    for e, score in self.wm.find_by_embedding(q_emb, top_k=k * 4)]

        if source == "ner":
            ranking = _ner_rank()[:k]
            return [_entity_to_hit(self.wm.entities[uid],
                                    s, "ner",
                                    n_transitions=len(self.wm.get_transitions(uid)))
                    for uid, s in ranking if uid in self.wm.entities]

        if source == "bm25":
            ranking = _bm25_rank()[:k]
            return [_entity_to_hit(self.wm.entities[uid],
                                    s, "bm25",
                                    n_transitions=len(self.wm.get_transitions(uid)))
                    for uid, s in ranking if uid in self.wm.entities]

        if source == "dense":
            ranking = _dense_rank()[:k]
            return [_entity_to_hit(self.wm.entities[uid],
                                    s, "dense",
                                    n_transitions=len(self.wm.get_transitions(uid)))
                    for uid, s in ranking if uid in self.wm.entities]

        # ── Hybrid: RRF over the three rankers ──
        ner_uids   = [u for u, _ in _ner_rank()]
        bm25_uids  = [u for u, _ in _bm25_rank()]
        dense_uids = [u for u, _ in _dense_rank()]
        # _rrf takes integer doc-id ranks; we map uids -> dense ints, then back
        all_uids = list({*ner_uids, *bm25_uids, *dense_uids})
        if not all_uids:
            return []
        u2i = {u: i for i, u in enumerate(all_uids)}
        rank_lists = [
            [u2i[u] for u in ner_uids],
            [u2i[u] for u in bm25_uids],
            [u2i[u] for u in dense_uids],
        ]
        fused = _rrf(rank_lists, k=60)[:k]
        out: list[Hit] = []
        for di, score in fused:
            uid = all_uids[di]
            e = self.wm.entities.get(uid)
            if not e:
                continue
            out.append(_entity_to_hit(
                e, score, "hybrid",
                n_transitions=len(self.wm.get_transitions(uid)),
            ))
        return out

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
