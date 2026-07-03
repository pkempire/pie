"""PIE baseline on a single LoCoMo conversation.

Extracts entities via a Reflector-style prompt (single LLM call per session),
wraps in PIEBackend for hybrid retrieval, and evaluates with HeuristicPolicy.

Usage:
    python -m mempol.scripts.run_pie_baseline --conv-idx 1 --run-name pie_conv1
"""
from __future__ import annotations
import argparse, json, time, sys
from pathlib import Path

from mempol import config, llm
from mempol.backends.pie_kg import PIEBackend
from mempol.data.locomo import load as load_locomo
from mempol.eval.judge import judge
from mempol.eval.runner import conv_to_units
from mempol.policies.v1_heuristic import HeuristicPolicy

# ── Reflector extraction prompt ──────────────────────────────────────────
_EXTRACT_SYS = """You extract structured entities from a conversation session.
For each person, project, decision, event, belief, tool, or organization mentioned,
produce a JSON entity with a unique id, type, name, and current state.

Entity types: person, project, tool, organization, belief, decision, concept, event, goal

For each entity:
- "id": a unique short slug (e.g. "caroline_person", "reunion_decision")
- "type": one of the types above
- "name": human-readable name
- "state": dict of key attributes (e.g. {"location": "Boston", "status": "planning"})
- "relations": list of {"target_id": "...", "type": "related_to|uses|works_on|part_of|motivated_by|contradicts", "description": "..."}

Also detect STATE CHANGES: if an entity's state differs from a prior session, include:
- "transitions": [{"from_state": {...}, "to_state": {...}, "type": "update|contradiction|resolution", "summary": "..."}]

Return valid JSON: {"entities": [...], "session_summary": "1-2 sentences about this session"}"""


def _extract_entities_from_session(session_text: str, existing_summary: str = "") -> dict:
    """One LLM call per session to extract entities."""
    user_msg = f"Conversation session:\n\n{session_text[:8000]}"
    if existing_summary:
        user_msg = (
            f"EXISTING ENTITIES (use these ids for updates, create new ids for new entities):\n"
            f"{existing_summary[:2000]}\n\n---\n\n{user_msg}"
        )
    raw = llm.chat(
        [
            {"role": "system", "content": _EXTRACT_SYS},
            {"role": "user", "content": user_msg},
        ],
        model="gpt-4o-mini",
        json_mode=True,
    )
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        print(f"  [warn] JSON parse failed, raw len={len(raw)}")
        return {"entities": [], "session_summary": ""}


def _entity_summary(wm) -> str:
    """Compact summary of existing entities for context injection."""
    lines = []
    for eid, e in sorted(wm.entities.items(), key=lambda x: x[1].last_seen or 0, reverse=True):
        state_str = ", ".join(f"{k}={v}" for k, v in (e.current_state or {}).items())[:120]
        lines.append(f"  [{eid}] {e.type.value}: {e.name} — {state_str}")
        if len(lines) >= 30:
            break
    return "\n".join(lines)


def build_world_model_for_conversation(conv, max_sessions: int = 0) -> PIEBackend:
    """Extract entities from LoCoMo conversation sessions, build PIEBackend."""
    from pie.core.world_model import WorldModel
    from pie.core.models import EntityType, RelationshipType, TransitionType

    wm = WorldModel()
    # Group turns by session
    sessions: dict[int, list] = {}
    for t in conv.turns:
        sessions.setdefault(t.session, []).append(t)

    session_keys = sorted(sessions.keys())
    if max_sessions > 0:
        session_keys = session_keys[:max_sessions]

    print(f"  Extracting from {len(session_keys)} sessions...")
    for si, sk in enumerate(session_keys):
        turns = sessions[sk]
        session_text = "\n".join(
            f"[{t.dia_id} | {t.speaker}] {t.text}" for t in turns
        )
        existing = _entity_summary(wm) if wm.entities else ""
        result = _extract_entities_from_session(session_text, existing)

        # Phase 1: Create/ensure all entities first
        created_this_session = set()
        for ent in result.get("entities", []):
            eid = ent.get("id", "")
            if not eid:
                continue
            etype_str = ent.get("type", "concept")
            name = ent.get("name", eid)
            state = ent.get("state", {})
            try:
                etype = EntityType(etype_str.lower())
            except ValueError:
                etype = EntityType.CONCEPT

            if eid in wm.entities:
                # Detect state change on existing entity
                existing_ent = wm.entities[eid]
                if state and state != existing_ent.current_state:
                    wm.update_entity_state(
                        entity_id=eid, new_state=state,
                        source_conversation_id=f"session_{sk}",
                        timestamp=float(si),
                        trigger_summary=f"updated in session {sk}",
                    )
                existing_ent.last_seen = float(si)
            else:
                wm.create_entity(
                    name=name, type=etype, state=state,
                    source_conversation_id=f"session_{sk}",
                    timestamp=float(si),
                )
                created_this_session.add(eid)

        # Phase 2: Apply transitions (only for entities that exist)
        for ent in result.get("entities", []):
            eid = ent.get("id", "")
            if not eid or eid not in wm.entities:
                continue
            for tr in ent.get("transitions", []):
                try:
                    tr_type = TransitionType(tr.get("type", "update").lower())
                except ValueError:
                    tr_type = TransitionType.UPDATE
                wm.update_entity_state(
                    entity_id=eid, new_state=tr.get("to_state", {}),
                    source_conversation_id=f"session_{sk}",
                    timestamp=float(si),
                    trigger_summary=tr.get("summary", ""),
                    is_contradiction=(tr_type == TransitionType.CONTRADICTION),
                )

        # Phase 3: Apply relations (only when both entities exist)
        for ent in result.get("entities", []):
            eid = ent.get("id", "")
            if not eid or eid not in wm.entities:
                continue
            for rel in ent.get("relations", []):
                target_id = rel.get("target_id", "")
                if not target_id or target_id not in wm.entities:
                    continue
                try:
                    rel_type = RelationshipType(rel.get("type", "related_to").lower())
                except ValueError:
                    rel_type = RelationshipType.RELATED_TO
                wm.add_relationship(
                    source_id=eid, target_id=target_id,
                    rel_type=rel_type,
                    description=rel.get("description", ""),
                    timestamp=float(si),
                )

        print(f"    session {sk}: {len(result.get('entities',[]))} entities, "
              f"total={len(wm.entities)}")

    print(f"  Built world model: {len(wm.entities)} entities, "
          f"{len(wm.transitions)} transitions, {len(wm.relationships)} relationships")
    return PIEBackend(world_model=wm)


def run(conv_idx: int = 1, max_qs: int = 0, max_sessions: int = 0, run_name: str = "pie_baseline"):
    convs = load_locomo(n_convs=conv_idx + 1)
    if conv_idx >= len(convs):
        raise SystemExit(f"Only {len(convs)} convs, need idx {conv_idx}")
    conv, qas = convs[conv_idx]

    print(f"[{conv.sample_id}] {len(conv.turns)} turns, {len(qas)} questions")

    # 1. Build world model + backend
    t0 = time.time()
    backend = build_world_model_for_conversation(conv, max_sessions=max_sessions)
    ingest_time = time.time() - t0
    print(f"  Ingestion: {ingest_time:.1f}s")

    # 2. Eval
    qas_to_run = qas if max_qs == 0 else qas[:max_qs]
    policy = HeuristicPolicy(do_reformulate=False, do_route=False, do_expand=True)
    print(f"\n  Evaluating {policy.name} on {len(qas_to_run)} questions...")

    out_dir = config.RESULTS_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    traces_path = out_dir / "traces.jsonl"

    results = []
    t_eval = time.time()
    with traces_path.open("w", buffering=1) as f:
        for i, qa in enumerate(qas_to_run):
            trace = policy.run(qa.question, backend)
            score, reason = judge(qa.question, qa.answer, trace.answer)
            r = {
                "qid": qa.qid, "category_name": qa.category_name,
                "question": qa.question, "gold": qa.answer,
                "answer": trace.answer, "score": score,
                "judge_reason": reason, "n_retrievals": trace.n_retrievals,
                "n_steps": len(trace.steps),
            }
            results.append(r)
            f.write(json.dumps(r) + "\n")
            f.flush()
            if (i + 1) % 10 == 0:
                acc = sum(x["score"] for x in results) / len(results)
                print(f"    q {i+1}/{len(qas_to_run)}  running_acc={acc:.3f}")

    eval_time = time.time() - t_eval

    # 3. Summarise
    from collections import defaultdict
    by_cat = defaultdict(list)
    for r in results:
        by_cat[r["category_name"]].append(r["score"])

    summary = {
        "conv_id": conv.sample_id,
        "n_entities": len(backend.wm.entities),
        "n_transitions": len(backend.wm.transitions),
        "n_relationships": len(backend.wm.relationships),
        "ingest_time_s": round(ingest_time, 1),
        "eval_time_s": round(eval_time, 1),
        "n_qs": len(results),
        "overall_acc": round(sum(r["score"] for r in results) / max(1, len(results)), 4),
        "avg_retrievals": round(sum(r["n_retrievals"] for r in results) / max(1, len(results)), 2),
        "by_category": {
            k: {"n": len(v), "acc": round(sum(v) / len(v), 4)}
            for k, v in sorted(by_cat.items())
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n=== PIE BASELINE SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nFiles: {out_dir}/")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--conv-idx", type=int, default=1)
    ap.add_argument("--max-qs", type=int, default=0, help="0 = all")
    ap.add_argument("--max-sessions", type=int, default=0, help="0 = all sessions")
    ap.add_argument("--run-name", default="pie_baseline")
    args = ap.parse_args()
    run(
        conv_idx=args.conv_idx,
        max_qs=None if args.max_qs == 0 else args.max_qs,
        max_sessions=args.max_sessions if args.max_sessions > 0 else 0,
        run_name=args.run_name,
    )
