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
- If the input contains multiple dia_ids, include source_dia_ids in state for
  every factual write. If the input contains one dia_id, source_dia_id is OK.
- Store concrete dated events, identity facts, plans, preferences, project
  state, decisions, relationships, constraints, and notable emotional
  reactions. A single sentence can be worth storing if it answers a plausible
  future question.
- Time is part of the memory. Resolve relative dates like "yesterday",
  "tomorrow", "last week", and "next month" against the observation time when
  it is provided. Store explicit temporal fields when known:
  observed_at, event_time, valid_from, valid_until, status.
- Prefer soft temporal status over deletion. If something expires, update it
  with valid_until/status instead of forgetting the historical fact.
- In peer conversations, attribute facts to the speaker in the turn text.
  Example: "Caroline: I went to a LGBTQ support group yesterday" should create
  an event for Caroline's support-group attendance, not noop.
- Do NOT noop specific events merely because they are short. Noop only for
  pleasantries, acknowledgments, filler, or content already represented.

Examples:
Turn: "Caroline: I went to a LGBTQ support group yesterday and it was powerful."
Output: {"ops":[{"op":"create_entity","args":{"name":"Caroline's LGBTQ support group visit","type":"event","state":{"description":"Caroline went to an LGBTQ support group yesterday and found it powerful.","speaker":"Caroline"}}}]}

Turn: "user: Let's use Postgres instead of Mongo for the memory store."
Output: {"ops":[{"op":"create_entity","args":{"name":"Memory store database decision","type":"decision","state":{"description":"Use Postgres instead of Mongo for the memory store.","status":"active"}}}]}

Turn: "assistant: Sounds good!"
Output: {"ops":[{"op":"noop","args":{"reason":"acknowledgment only"}}]}

Return JSON only, in this shape:
{"ops": [{"op": "noop", "args": {}}]}
or
{"ops": [
  {"op": "create_entity", "args": {"name": "...", "type": "...", "state": {...}}},
  {"op": "add_relation",  "args": {"source_uid": "...", "target_uid": "...", "rel_type": "...", "description": "..."}}
]}
"""

_RESOLVE_SYS = """You are resolving a proposed memory write against an existing long-term memory.

Decide whether a proposed create_entity operation should instead update one existing entity.

Use semantic identity, not string similarity:
- Same person/project/tool/organization: update the existing entity.
- Same ongoing goal/plan/decision: update the existing entity.
- Same event being elaborated by later turns: update the existing event.
- A genuinely new event, goal, object, or relationship: create_new.
- If unsure, create_new. Never merge merely because names share words.

Return strict JSON:
{"decision":"update_existing","uid":"...","reason":"..."}
or
{"decision":"create_new","reason":"..."}
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


def _flat_state_text(state: dict[str, Any]) -> str:
    parts: list[str] = []
    for v in (state or {}).values():
        if isinstance(v, (str, int, float, bool)):
            parts.append(str(v))
        elif isinstance(v, list):
            parts.extend(str(x) for x in v[:8] if isinstance(x, (str, int, float, bool)))
    return " ".join(parts)


@dataclass
class WriteDecision:
    turn_text: str
    dia_id: str
    timestamp: float
    observation_time_text: str = ""
    lookup_query: str = ""
    recent_context_text: str = ""
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

    def _resolve_create_target(
        self,
        backend: PIEBackend,
        name: str,
        type_: str,
        state: dict[str, Any],
        lookup_matches: list[dict],
    ) -> dict | None:
        """Resolve a proposed create into an existing entity with an LLM judge."""
        query = f"{name} {type_} {_flat_state_text(state)}"[:700]
        candidates: list[dict] = []
        seen: set[str] = set()
        for m in lookup_matches + backend.lookup_entity(query, top_k=max(8, self.lookup_top_k)):
            uid = m.get("uid")
            if not uid or uid in seen:
                continue
            seen.add(uid)
            candidates.append(m)

        if not candidates:
            return None

        candidate_view = [
            {
                "uid": m.get("uid"),
                "name": m.get("name"),
                "type": m.get("type"),
                "current_state": m.get("current_state") or {},
                "n_transitions": m.get("n_transitions"),
            }
            for m in candidates[:8]
        ]
        proposed = {
            "op": "create_entity",
            "args": {"name": name, "type": type_, "state": state},
        }
        raw = llm.chat(
            [
                {"role": "system", "content": _RESOLVE_SYS},
                {"role": "user", "content": json.dumps({
                    "proposed_write": proposed,
                    "candidate_existing_entities": candidate_view,
                }, ensure_ascii=False)},
            ],
            model=self.model,
            json_mode=True,
        )
        try:
            obj = json.loads(raw)
        except Exception:
            return None
        if obj.get("decision") != "update_existing":
            return None
        uid = obj.get("uid")
        for m in candidates:
            if m.get("uid") == uid or str(m.get("uid", "")).startswith(str(uid)):
                m = dict(m)
                m["resolver_reason"] = obj.get("reason", "")
                return m
        return None

    def step(
        self,
        turn_text: str,
        dia_id: str,
        timestamp: float,
        backend: PIEBackend,
        write_tool: WriteTool,
        observation_time_text: str = "",
        recent_context_text: str = "",
    ) -> WriteDecision:
        decision = WriteDecision(
            turn_text=turn_text, dia_id=dia_id, timestamp=timestamp,
            observation_time_text=observation_time_text,
            lookup_query=self._lookup_query_from_turn(turn_text),
            recent_context_text=recent_context_text,
        )

        # 1. Lookup nearest entities to give the LLM dedup context. Go through
        # WriteTool's plain implementation so dashboard stats match behavior.
        try:
            lookup_result = write_tool._lookup_entity_impl(
                query=decision.lookup_query,
                top_k=self.lookup_top_k,
            )
            content = lookup_result.get("content") if hasattr(lookup_result, "get") else getattr(lookup_result, "content", "{}")
            lookup_obj = json.loads(content or "{}")
            matches = list(lookup_obj.get("matches", []))
        except Exception:
            matches = backend.lookup_entity(decision.lookup_query, top_k=self.lookup_top_k)
        decision.lookup_matches = matches

        # 2. LLM decides ops.
        prompt = (
            f"Turn ({dia_id}, t={timestamp}):\n{turn_text}\n\n"
            f"Observation time: {observation_time_text or '(unknown)'}\n"
            f"Recent context before this turn:\n{recent_context_text or '(none)'}\n\n"
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
        write_tool.current_observation_time_text = observation_time_text

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
                    write_tool._noop_impl(reason=args.get("reason", ""))
                elif op == "create_entity":
                    name = args.get("name", "")
                    type_ = args.get("type", "concept")
                    state = args.get("state") or {}
                    resolved = self._resolve_create_target(
                        backend=backend,
                        name=name,
                        type_=type_,
                        state=state,
                        lookup_matches=matches,
                    )
                    if resolved:
                        uid = resolved["uid"]
                        update_state = dict(state)
                        update_state.setdefault("name_variant", name)
                        write_tool._update_state_impl(
                            uid=uid,
                            new_state=update_state,
                            transition_type="update",
                            trigger_summary=f"deduped create_entity into existing {resolved.get('name', uid)}",
                        )
                        applied_args = {
                            "uid": uid,
                            "new_state": update_state,
                            "transition_type": "update",
                            "trigger_summary": f"deduped create_entity from proposed {name}",
                            "resolved_from_create": True,
                            "proposed_name": name,
                            "matched_name": resolved.get("name"),
                            "match_score": resolved.get("match_score"),
                        }
                        decision.applied_ops.append({"op": "update_state", "args": applied_args})
                        continue
                    write_tool._create_entity_impl(name=name, type=type_, state=state)
                elif op == "update_state":
                    write_tool._update_state_impl(
                        uid=args.get("uid", ""),
                        new_state=args.get("new_state") or {},
                        transition_type=args.get("transition_type", "update"),
                        trigger_summary=args.get("trigger_summary", ""),
                    )
                elif op == "add_relation":
                    write_tool._add_relation_impl(
                        source_uid=args.get("source_uid", ""),
                        target_uid=args.get("target_uid", ""),
                        rel_type=args.get("rel_type", "related_to"),
                        description=args.get("description", ""),
                    )
                elif op == "mark_contradiction":
                    write_tool._mark_contradiction_impl(
                        uid=args.get("uid", ""),
                        contradicting_state=args.get("contradicting_state") or {},
                    )
                elif op == "forget":
                    write_tool._forget_impl(uid=args.get("uid", ""), reason=args.get("reason", ""))
                else:
                    decision.errors.append(f"unknown_op: {op}")
                    continue
                decision.applied_ops.append({"op": op, "args": args})
            except Exception as e:
                decision.errors.append(f"apply_error[{op}]: {e}")
        return decision
