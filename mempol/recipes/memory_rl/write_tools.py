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
    current_observation_time_text: str = ""
    n_lookups: int = 0
    n_creates: int = 0
    n_updates: int = 0
    n_merges: int = 0
    n_relations: int = 0
    n_contradictions: int = 0
    n_forgets: int = 0
    n_noops: int = 0
    # Append-only log of (op_name, args_dict) per applied tool call.
    # Read by counterfactual.per_op_counterfactual to replay the trajectory
    # minus a chosen op for leave-one-out reward attribution. Args are
    # captured pre-execution so the replay sees the same intent the policy
    # had, regardless of whether the original execution succeeded.
    ops_log: list = field(default_factory=list)

    def _metadata(self) -> dict:
        """Provenance that every write should carry in standalone evals."""
        md = {
            "source_dia_id": self.current_dia_id,
            "observed_at_timestamp": self.current_timestamp,
        }
        if self.current_observation_time_text:
            md["observed_at"] = self.current_observation_time_text
        return {k: v for k, v in md.items() if v not in ("", None)}

    def _with_write_metadata(self, state: dict | None) -> dict:
        out = dict(state or {})
        for k, v in self._metadata().items():
            out.setdefault(k, v)
        return out

    # These plain Python implementations are intentionally separate from the
    # @tool wrappers. When tinker_cookbook is installed, @tool may replace a
    # method with a FunctionTool object, which is correct for RL env specs but
    # not directly callable by local scripts and dashboards.

    def _lookup_entity_impl(self, query: str, type: str | None = None, top_k: int = 5) -> ToolResult:
        self.n_lookups += 1
        results = self.backend.lookup_entity(query=query, type=type, top_k=min(int(top_k), 10))
        return simple_tool_result(json.dumps({"matches": results}, ensure_ascii=False))

    def _lookup_relation_impl(self, source_uid: str, target_uid: str | None = None) -> ToolResult:
        results = self.backend.lookup_relation(source_uid, target_uid)
        return simple_tool_result(json.dumps({"relations": results}, ensure_ascii=False))

    def _create_entity_impl(self, name: str, type: str, state: dict | None = None) -> ToolResult:
        uid = self.backend.create_entity(
            name=name,
            type=type,
            state=self._with_write_metadata(state),
            source=self.current_dia_id,
            timestamp=self.current_timestamp,
        )
        self.n_creates += 1
        return simple_tool_result(json.dumps({"uid": uid, "name": name, "type": type}))

    def _update_state_impl(
        self,
        uid: str,
        new_state: dict,
        transition_type: str = "update",
        trigger_summary: str = "",
    ) -> ToolResult:
        if uid not in self.backend.wm.entities:
            return simple_tool_result(json.dumps({
                "ok": False, "uid": uid, "error": "entity not found",
                "hint": "call lookup_entity first; copy the uid field exactly",
            }))
        ok = self.backend.update_state(
            uid=uid,
            new_state=self._with_write_metadata(new_state),
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

    def _merge_entities_impl(self, canonical_uid: str, alias_uid: str) -> ToolResult:
        missing = [u for u in (canonical_uid, alias_uid)
                   if u not in self.backend.wm.entities]
        if missing:
            return simple_tool_result(json.dumps({
                "ok": False, "error": "entity not found", "missing": missing,
            }))
        ok = self.backend.merge_entities(canonical_uid, alias_uid)
        if ok:
            self.n_merges += 1
        return simple_tool_result(json.dumps({"ok": ok, "canonical_uid": canonical_uid, "alias_uid": alias_uid}))

    def _add_relation_impl(
        self, source_uid: str, target_uid: str, rel_type: str, description: str = "",
    ) -> ToolResult:
        missing = [u for u in (source_uid, target_uid)
                   if u not in self.backend.wm.entities]
        if missing:
            return simple_tool_result(json.dumps({
                "ok": False, "error": "entity not found", "missing": missing,
            }))
        ok = self.backend.add_relation(
            source_uid=source_uid, target_uid=target_uid, rel_type=rel_type,
            description=description, timestamp=self.current_timestamp,
        )
        if ok:
            self.n_relations += 1
        return simple_tool_result(json.dumps({"ok": ok, "source_uid": source_uid, "target_uid": target_uid, "type": rel_type}))

    def _mark_contradiction_impl(self, uid: str, contradicting_state: dict) -> ToolResult:
        if uid not in self.backend.wm.entities:
            return simple_tool_result(json.dumps({
                "ok": False, "uid": uid, "error": "entity not found",
            }))
        ok = self.backend.mark_contradiction(
            uid=uid, contradicting_state=self._with_write_metadata(contradicting_state),
            source=self.current_dia_id, timestamp=self.current_timestamp,
        )
        if ok:
            self.n_contradictions += 1
        return simple_tool_result(json.dumps({"ok": ok, "uid": uid}))

    def _forget_impl(self, uid: str, reason: str = "") -> ToolResult:
        if uid not in self.backend.wm.entities:
            return simple_tool_result(json.dumps({
                "ok": False, "uid": uid, "error": "entity not found",
            }))
        ok = self.backend.forget(uid, reason)
        if ok:
            self.n_forgets += 1
        return simple_tool_result(json.dumps({"ok": ok, "uid": uid, "reason": reason}))

    def _noop_impl(self, reason: str = "") -> ToolResult:
        self.n_noops += 1
        return simple_tool_result(json.dumps({"ok": True, "reason": reason}))

    # ── 1. Lookup (the policy MUST use these before writing) ──
    # Each @tool wrapper appends (op_name, args) to self.ops_log BEFORE
    # delegating to the impl, so per-op counterfactual replay can
    # reconstruct the trajectory minus a chosen op without re-parsing the
    # rendered tool calls. We capture the *intent* (the args the policy
    # emitted), not the post-execution state — replays may produce
    # different `ok` values when prior context has changed.
    @tool
    def lookup_entity(self, query: str, type: str | None = None, top_k: int = 5) -> ToolResult:
        """Find existing entities matching the query. Use BEFORE create_entity."""
        self.ops_log.append(("lookup_entity",
                              {"query": query, "type": type, "top_k": top_k}))
        return self._lookup_entity_impl(query=query, type=type, top_k=top_k)

    @tool
    def lookup_relation(self, source_uid: str, target_uid: str | None = None) -> ToolResult:
        """Find existing relationships involving an entity (and optionally a target)."""
        self.ops_log.append(("lookup_relation",
                              {"source_uid": source_uid, "target_uid": target_uid}))
        return self._lookup_relation_impl(source_uid=source_uid, target_uid=target_uid)

    # ── 2. Create / Update ──
    @tool
    def create_entity(self, name: str, type: str, state: dict | None = None) -> ToolResult:
        """Create a new entity.

        type ∈ {person, project, tool, organization, belief, decision, concept,
                period, event, goal}.
        state is a JSON dict of attributes.
        """
        self.ops_log.append(("create_entity",
                              {"name": name, "type": type, "state": state or {}}))
        return self._create_entity_impl(name=name, type=type, state=state)

    @tool
    def update_state(
        self, uid: str, new_state: dict,
        transition_type: str = "update", trigger_summary: str = "",
    ) -> ToolResult:
        """Update an existing entity's state.

        transition_type ∈ {update, contradiction, resolution, archival}.
        """
        self.ops_log.append(("update_state",
                              {"uid": uid, "new_state": new_state,
                               "transition_type": transition_type,
                               "trigger_summary": trigger_summary}))
        return self._update_state_impl(
            uid=uid, new_state=new_state,
            transition_type=transition_type, trigger_summary=trigger_summary,
        )

    # ── 3. Structural ops ──
    @tool
    def merge_entities(self, canonical_uid: str, alias_uid: str) -> ToolResult:
        """Collapse alias_uid into canonical_uid."""
        self.ops_log.append(("merge_entities",
                              {"canonical_uid": canonical_uid, "alias_uid": alias_uid}))
        return self._merge_entities_impl(canonical_uid=canonical_uid, alias_uid=alias_uid)

    @tool
    def add_relation(
        self, source_uid: str, target_uid: str, rel_type: str, description: str = "",
    ) -> ToolResult:
        """Add a relationship edge between two existing entities."""
        self.ops_log.append(("add_relation",
                              {"source_uid": source_uid, "target_uid": target_uid,
                               "rel_type": rel_type, "description": description}))
        return self._add_relation_impl(
            source_uid=source_uid, target_uid=target_uid,
            rel_type=rel_type, description=description,
        )

    @tool
    def mark_contradiction(self, uid: str, contradicting_state: dict) -> ToolResult:
        """Flag that the current turn contradicts the entity's prior state."""
        self.ops_log.append(("mark_contradiction",
                              {"uid": uid, "contradicting_state": contradicting_state}))
        return self._mark_contradiction_impl(uid=uid, contradicting_state=contradicting_state)

    @tool
    def forget(self, uid: str, reason: str = "") -> ToolResult:
        """Archive (soft-delete) an entity."""
        self.ops_log.append(("forget", {"uid": uid, "reason": reason}))
        return self._forget_impl(uid=uid, reason=reason)

    @tool
    def noop(self, reason: str = "") -> ToolResult:
        """Mark this turn as not memory-worthy."""
        self.ops_log.append(("noop", {"reason": reason}))
        return self._noop_impl(reason=reason)

    # ── Op classification (used by counterfactual to skip non-mutating ops) ──
    NON_MUTATING_OPS: tuple[str, ...] = ("lookup_entity", "lookup_relation", "noop")
    MUTATING_OPS: tuple[str, ...] = (
        "create_entity", "update_state", "merge_entities",
        "add_relation", "mark_contradiction", "forget",
    )

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
