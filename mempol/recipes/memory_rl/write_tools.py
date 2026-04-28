"""Write-side memory ops as Tinker tools.

The WRITE policy emits these tool calls during ingestion of a single conversation
turn. Crucially, the policy is REQUIRED to call `lookup_entity` before creating
new entities — entity deduplication is part of the action sequence, not a
post-hoc resolver step.

Why this matters for the paper:
  • PIE has a 3-tier resolver (string → embedding → LLM verify) that's
    HARDCODED with thresholds.
  • Mem0 uses an LLM-as-judge ADD/UPDATE/DELETE/NONE prompt — also hardcoded.
  • Here, the policy LEARNS the resolution strategy. When to look up, what
    similarity threshold to trust, how aggressively to merge — these become
    learnable behaviours, supervised by downstream QA accuracy.

Drop the @tool decorator from tinker_cookbook.tool_use onto each method when
this file lives in a tinker-cookbook clone.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field

from mempol.backends.pie_kg import PIEBackend
from mempol.recipes.memory_rl.tinker_compat import tool, simple_tool_result, ToolResult


@dataclass
class WriteTool:
    """Per-env write tools wrapping a PIEBackend."""
    backend: PIEBackend
    current_turn_text: str = ""
    current_dia_id: str = ""
    current_timestamp: float = 0.0
    n_lookups: int = 0
    n_creates: int = 0
    n_updates: int = 0
    n_merges: int = 0
    n_relations: int = 0
    n_contradictions: int = 0
    n_forgets: int = 0
    n_noops: int = 0

    # ── 1. Lookup (the policy MUST use these before writing) ──
    @tool
    def lookup_entity(self, query: str, type: str | None = None, top_k: int = 5) -> ToolResult:
        """Find existing entities matching the query. Use BEFORE create_entity.

        Args:
            query: name or short description
            type: optional entity type filter ('person' | 'project' | 'tool' | 'organization' | 'belief' | 'decision' | 'concept' | 'period' | 'event' | 'goal')
            top_k: number of candidates to return (max 10)
        Returns:
            JSON list of {uid, name, type, current_state, match_score, n_transitions, last_seen}
            empty list if no match — safe to create new
        """
        self.n_lookups += 1
        results = self.backend.lookup_entity(query=query, type=type, top_k=min(int(top_k), 10))
        return simple_tool_result(json.dumps({"matches": results}, ensure_ascii=False))

    @tool
    def lookup_relation(self, source_uid: str, target_uid: str | None = None) -> ToolResult:
        """Find existing relationships involving an entity (and optionally a target)."""
        results = self.backend.lookup_relation(source_uid, target_uid)
        return simple_tool_result(json.dumps({"relations": results}, ensure_ascii=False))

    # ── 2. Create / Update ──
    # NOTE: there is no dedup guard. The policy can `lookup_entity` first, or
    # not. Duplicates show up as: (a) more entities → higher storage cost →
    # cost penalty in reward; (b) noisier retrieval at read time → lower QA
    # accuracy → reward penalty. RL learns whether/when to look up.
    @tool
    def create_entity(self, name: str, type: str, state: dict | None = None) -> ToolResult:
        """Create a new entity.

        type ∈ {person, project, tool, organization, belief, decision, concept,
                period, event, goal} (PIE's standard taxonomy — but the policy
                is free to abuse types if a different mapping wins reward;
                that's a learnable choice).
        state is a JSON dict of attributes (e.g. {"status": "active", "description": "..."}).
        """
        uid = self.backend.create_entity(
            name=name,
            type=type,
            state=state or {},
            source=self.current_dia_id,
            timestamp=self.current_timestamp,
        )
        self.n_creates += 1
        return simple_tool_result(json.dumps({"uid": uid, "name": name, "type": type}))

    @tool
    def update_state(
        self,
        uid: str,
        new_state: dict,
        transition_type: str = "update",
        trigger_summary: str = "",
    ) -> ToolResult:
        """Update an existing entity's state.

        transition_type ∈ {update, contradiction, resolution, archival}
        - update: normal state evolution
        - contradiction: new state conflicts with prior; both retained
        - resolution: a prior contradiction is resolved
        - archival: entity is no longer active
        """
        ok = self.backend.update_state(
            uid=uid,
            new_state=new_state,
            transition_type=transition_type,
            source=self.current_dia_id,
            timestamp=self.current_timestamp,
            trigger_summary=trigger_summary,
        )
        if ok:
            self.n_updates += 1
            if transition_type == "contradiction":
                self.n_contradictions += 1
        return simple_tool_result(json.dumps({"ok": ok, "uid": uid, "transition_type": transition_type}))

    # ── 3. Structural ops ──
    @tool
    def merge_entities(self, canonical_uid: str, alias_uid: str) -> ToolResult:
        """Collapse alias_uid into canonical_uid. Moves transitions and
        relationships. Use when lookup returns a high-similarity duplicate."""
        ok = self.backend.merge_entities(canonical_uid, alias_uid)
        if ok:
            self.n_merges += 1
        return simple_tool_result(json.dumps({"ok": ok, "canonical_uid": canonical_uid, "alias_uid": alias_uid}))

    @tool
    def add_relation(
        self, source_uid: str, target_uid: str, rel_type: str, description: str = "",
    ) -> ToolResult:
        """Add a relationship edge between two existing entities.

        rel_type ∈ {uses, works_on, collaborates_with, related_to, part_of,
                    caused_by, during, replaces, integrates_with}
        """
        ok = self.backend.add_relation(
            source_uid=source_uid, target_uid=target_uid, rel_type=rel_type,
            description=description, timestamp=self.current_timestamp,
        )
        if ok:
            self.n_relations += 1
        return simple_tool_result(json.dumps({"ok": ok, "source_uid": source_uid, "target_uid": target_uid, "type": rel_type}))

    @tool
    def mark_contradiction(self, uid: str, contradicting_state: dict) -> ToolResult:
        """Flag that the current turn contradicts the entity's prior state.
        Both states retained — useful when the truth is unclear."""
        ok = self.backend.mark_contradiction(
            uid=uid, contradicting_state=contradicting_state,
            source=self.current_dia_id, timestamp=self.current_timestamp,
        )
        if ok:
            self.n_contradictions += 1
        return simple_tool_result(json.dumps({"ok": ok, "uid": uid}))

    @tool
    def forget(self, uid: str, reason: str = "") -> ToolResult:
        """Archive (soft-delete) an entity. Preserves the transition history
        but marks the entity as no longer active."""
        ok = self.backend.forget(uid, reason)
        if ok:
            self.n_forgets += 1
        return simple_tool_result(json.dumps({"ok": ok, "uid": uid, "reason": reason}))

    @tool
    def noop(self, reason: str = "") -> ToolResult:
        """Mark this turn as not memory-worthy. Use for chitchat, fillers,
        and turns whose content is already represented."""
        self.n_noops += 1
        return simple_tool_result(json.dumps({"ok": True, "reason": reason}))

    # ── stats for reward shaping ──
    def write_stats(self) -> dict:
        return {
            "n_lookups": self.n_lookups,
            "n_creates": self.n_creates,
            "n_updates": self.n_updates,
            "n_merges": self.n_merges,
            "n_relations": self.n_relations,
            "n_contradictions": self.n_contradictions,
            "n_forgets": self.n_forgets,
            "n_noops": self.n_noops,
        }


def smoke():
    """Verify write tools end-to-end against PIEBackend."""
    backend = PIEBackend()
    wt = WriteTool(backend=backend)

    # Turn 1: introduce a project
    wt.current_dia_id = "D1:1"; wt.current_timestamp = 1000.0
    wt.current_turn_text = "I'm building PIE — a memory system for LLM agents."

    print("--- lookup before create (should be empty) ---")
    print(wt.lookup_entity("PIE memory system"))

    print("\n--- create_entity ---")
    pie_uid = json.loads(wt.create_entity(
        name="PIE", type="project",
        state={"description": "memory system for LLM agents", "status": "in progress"},
    ))["uid"]
    print(f"  uid: {pie_uid}")

    # Turn 2: same project again with new info — policy should look up + update
    wt.current_dia_id = "D1:5"; wt.current_timestamp = 1100.0
    print("\n--- lookup again (should find PIE) ---")
    matches = json.loads(wt.lookup_entity("PIE memory system"))
    print(f"  found {len(matches['matches'])} match(es), top score={matches['matches'][0]['match_score']:.2f}")

    print("\n--- create_entity AGAIN — no dedup guard, policy must use lookup ---")
    again = json.loads(wt.create_entity(name="PIE", type="project", state={}))
    print(f"  result: {again}  (we now have 2 entities; policy should have called update_state instead)")

    print("\n--- update_state on existing entity ---")
    upd = json.loads(wt.update_state(
        uid=pie_uid,
        new_state={"status": "v0 shipped", "next_step": "GRPO training"},
        transition_type="update",
    ))
    print(f"  {upd}")

    # Turn 3: introduce a contradiction
    wt.current_dia_id = "D1:9"; wt.current_timestamp = 1200.0
    print("\n--- mark_contradiction ---")
    con = json.loads(wt.mark_contradiction(uid=pie_uid, contradicting_state={"status": "deprecated, switching to mempol"}))
    print(f"  {con}")

    # Turn 4: noop on chitchat
    print("\n--- noop on chitchat ---")
    print(wt.noop(reason="weather small talk"))

    print("\n--- final stats ---")
    print(wt.write_stats())
    n_trans = sum(len(backend.wm.get_transitions(uid)) for uid in backend.wm.entities)
    print(f"backend stats: entities={len(backend.wm.entities)} transitions={n_trans}")


if __name__ == "__main__":
    smoke()
