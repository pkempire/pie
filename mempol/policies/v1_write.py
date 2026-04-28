"""v1 heuristic write policy — the teacher we'll imitate at SFT.

Per turn, runs ONE LLM call that:
  1. Decides whether the turn is worth remembering (gate).
  2. If yes, extracts candidate entities / state changes / relations.
  3. Returns a list of ops referencing existing entities by uid where possible
     (the prompt is given the top-K lookup matches as context).

We resolve / dedup against the current memory by passing lookup results into
the prompt — same as how the learned write policy will work at training time.
"""
from __future__ import annotations
import json
import time
from dataclasses import dataclass, field
from typing import Any

from .. import llm, config
from ..backends.pie_kg import PIEBackend
from ..recipes.memory_rl.write_tools import WriteTool


_WRITE_SYS = """You are deciding what to store in a long-term memory of a user's chat history. You see one turn from a long conversation, plus a digest of entities already in memory that might be relevant.

Choose ops to emit. Be DEFAULT noop — most turns are chitchat or transient and should not be stored. Store only durable, specific information that would be useful to recall later.

Op vocabulary (emit a list of ops, in order):

  noop                    — turn isn't memory-worthy
  create_entity           — introduce a new entity
                            args: name, type, state (dict of attrs)
                            type ∈ {person, project, tool, organization, belief, decision, concept, period, event, goal}
  update_state            — update an existing entity (use `existing_uid` from context)
                            args: uid, new_state (dict, will merge), transition_type, trigger_summary
                            transition_type ∈ {update, contradiction, archival, resolution}
  add_relation            — link two existing entities
                            args: source_uid, target_uid, rel_type, description
                            rel_type ∈ {uses, works_on, collaborates_with, related_to, part_of, caused_by, during, replaces, integrates_with}
  mark_contradiction      — flag that this turn conflicts with the entity's prior state
                            args: uid, contradicting_state (dict)
  forget                  — archive an entity
                            args: uid, reason

Rules:
- DO NOT create_entity if a near-duplicate already exists in context — use update_state with that uid.
- For type, prefer the strict taxonomy. If unsure, use `concept`.
- name should be specific (e.g. "PIE memory project", not "the project").
- state is a flat JSON dict of attributes (e.g. {"status": "shipped", "next_step": "GRPO"}).
- When in doubt about whether something is durable, emit `noop` and rely on later turns.

Return JSON only, in this shape:
{"ops": [{"op": "noop", "args": {}}]}
or
{"ops": [
  {"op": "create_entity", "args": {"name": "...", "type": "...", "state": {...}}},
  {"op": "add_relation",  "args": {"source_uid": "...", "target_uid": "...", "rel_type": "...", "description": "..."}}
]}
"""


def _format_lookup_context(matches: list[dict]) -> str:
    if not matches:
        return "(no nearby entities in memory yet)"
    lines = []
    for m in matches:
        st = json.dumps(m.get("current_state") or {}, ensure_ascii=False)[:200]
        aliases = ", ".join((m.get("aliases") or [])[:3])
        lines.append(
            f"  - uid={m['uid'][:8]} name={m['name']!r} type={m['type']} "
            f"state={st} match_score={m['match_score']:.2f} "
            f"n_transitions={m['n_transitions']} aliases=[{aliases}]"
        )
    return "\n".join(lines)


@dataclass
class WriteDecision:
    turn_text: str
    dia_id: str
    timestamp: float
    lookup_query: str
    lookup_matches: list[dict] = field(default_factory=list)
    raw_ops: list[dict] = field(default_factory=list)
    applied_ops: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class HeuristicWritePolicy:
    """Per-turn heuristic write policy. The teacher we'll later SFT against."""
    name = "v1_write"

    def __init__(
        self,
        lookup_top_k: int = 5,
        model: str | None = None,
        gate_with_llm: bool = True,
    ):
        self.lookup_top_k = lookup_top_k
        self.model = model or config.REFORMULATE_MODEL
        self.gate_with_llm = gate_with_llm

    def _lookup_query_from_turn(self, turn_text: str) -> str:
        """Cheap heuristic: take noun-ish content of the turn as the lookup
        query. Long turns are truncated. The policy gets multiple ops anyway,
        so this only seeds the FIRST lookup."""
        return (turn_text or "").strip().replace("\n", " ")[:300]

    def step(
        self,
        turn_text: str,
        dia_id: str,
        timestamp: float,
        backend: PIEBackend,
        write_tool: WriteTool,
    ) -> WriteDecision:
        decision = WriteDecision(
            turn_text=turn_text, dia_id=dia_id, timestamp=timestamp,
            lookup_query=self._lookup_query_from_turn(turn_text),
        )

        # 1. Lookup nearest entities to give the LLM dedup context.
        matches = backend.lookup_entity(decision.lookup_query, top_k=self.lookup_top_k)
        decision.lookup_matches = matches

        # 2. LLM decides ops.
        prompt = (
            f"Turn ({dia_id}, t={timestamp}):\n{turn_text}\n\n"
            f"Existing nearby entities (use uid in update/relation ops):\n"
            f"{_format_lookup_context(matches)}\n\n"
            "Return JSON only."
        )
        raw = llm.chat(
            [
                {"role": "system", "content": _WRITE_SYS},
                {"role": "user", "content": prompt},
            ],
            model=self.model,
            json_mode=True,
        )
        try:
            obj = json.loads(raw)
            decision.raw_ops = list(obj.get("ops", []))
        except Exception as e:
            decision.errors.append(f"parse_error: {e}")
            decision.raw_ops = [{"op": "noop", "args": {"reason": "parse error"}}]

        # 3. Apply ops to the backend via write_tool. Resolve uid prefixes.
        write_tool.current_dia_id = dia_id
        write_tool.current_timestamp = timestamp
        write_tool.current_turn_text = turn_text

        uid_map = {m["uid"][:8]: m["uid"] for m in matches}
        for op_spec in decision.raw_ops:
            op = op_spec.get("op", "noop")
            args = op_spec.get("args") or {}
            # Expand short uid prefixes the LLM might produce
            for k in ("uid", "source_uid", "target_uid", "canonical_uid", "alias_uid"):
                if k in args and isinstance(args[k], str) and len(args[k]) <= 12:
                    args[k] = uid_map.get(args[k][:8], args[k])
            try:
                if op == "noop":
                    write_tool.noop(reason=args.get("reason", ""))
                elif op == "create_entity":
                    write_tool.create_entity(
                        name=args.get("name", ""),
                        type=args.get("type", "concept"),
                        state=args.get("state") or {},
                    )
                elif op == "update_state":
                    write_tool.update_state(
                        uid=args.get("uid", ""),
                        new_state=args.get("new_state") or {},
                        transition_type=args.get("transition_type", "update"),
                        trigger_summary=args.get("trigger_summary", ""),
                    )
                elif op == "add_relation":
                    write_tool.add_relation(
                        source_uid=args.get("source_uid", ""),
                        target_uid=args.get("target_uid", ""),
                        rel_type=args.get("rel_type", "related_to"),
                        description=args.get("description", ""),
                    )
                elif op == "mark_contradiction":
                    write_tool.mark_contradiction(
                        uid=args.get("uid", ""),
                        contradicting_state=args.get("contradicting_state") or {},
                    )
                elif op == "forget":
                    write_tool.forget(uid=args.get("uid", ""), reason=args.get("reason", ""))
                else:
                    decision.errors.append(f"unknown_op: {op}")
                    continue
                decision.applied_ops.append({"op": op, "args": args})
            except Exception as e:
                decision.errors.append(f"apply_error[{op}]: {e}")
        return decision
