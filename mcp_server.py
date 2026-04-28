#!/usr/bin/env python3
"""
PIE MCP Server — gives any MCP-compatible AI (Claude, Cursor, etc.)
direct access to your personal knowledge graph.

Tools available:
  search                — hybrid BM25+dense search, returns top-k entities
  broad_scan            — LLM decomposes topic into 12 sub-queries, returns 60 entities
  get_entity            — full detail: state + complete transition history + relationships
  get_entities_by_type  — all entities of a given type (tool/project/person/org/goal/…)
  get_recent_entities   — most recently updated entities (optional type filter)
  answer                — full RAG pipeline: retrieve → compile temporal context → gpt-5.4
  get_architecture      — generates a full architecture doc for a project from the KG
  enrich_entity         — enriches a tool/org entity with current web-grounded knowledge
  get_briefing          — active projects, goals, stale threads, people network
  get_beliefs           — all extracted beliefs and preferences
  get_decisions         — all significant decisions
  get_stats             — entity counts, type breakdown

Usage (Claude Desktop or Cursor):
    command: python3.11
    args: ["/path/to/personal-intelligence-system/mcp_server.py"]
"""

import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from mcp.server.fastmcp import FastMCP

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("pie.mcp")

_ANSWER_MODEL = "gpt-5.4"
_WM_PATH = PROJECT_ROOT / "output" / "world_model.json"
_SESSIONS_DIR = PROJECT_ROOT / "output" / "sessions"

# ── Lazy singletons ────────────────────────────────────────────────────────────

_wm = None
_llm = None
_retriever = None


def _get_wm():
    global _wm
    if _wm is None:
        from pie.core.world_model import WorldModel
        _wm = WorldModel(persist_path=_WM_PATH)
        if not _wm.entities:
            raise RuntimeError(f"World model empty or not found at {_WM_PATH}")
    return _wm


def _get_llm():
    global _llm
    if _llm is None:
        from pie.core.llm import LLMClient
        _llm = LLMClient()
    return _llm


def _get_retriever():
    global _retriever
    if _retriever is None:
        from pie.retrieval.hybrid_retriever import HybridRetriever
        from pie.config import PIEConfig
        _retriever = HybridRetriever(_get_wm(), _get_llm(), PIEConfig())
    return _retriever


def _fmt_ts(ts: float) -> str:
    if not ts:
        return "unknown"
    from datetime import datetime
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d")


def _humanize(ts: float) -> str:
    if not ts:
        return ""
    days = (time.time() - ts) / 86400
    if days < 1:   return "today"
    if days < 7:   return f"{int(days)}d ago"
    if days < 30:  return f"{int(days/7)}w ago"
    if days < 365: return f"{int(days/30)}mo ago"
    return f"{days/365:.1f}y ago"


def _entity_to_dict(entity, wm, include_history: bool = False) -> dict:
    out = {
        "id": entity.id,
        "name": entity.name,
        "type": entity.type.value,
        "aliases": entity.aliases[:5],
        "current_state": entity.current_state,
        "first_seen": _fmt_ts(entity.first_seen),
        "last_seen": _fmt_ts(entity.last_seen),
        "last_seen_relative": _humanize(entity.last_seen),
        "importance": entity.importance or 0,
    }
    if include_history:
        transitions = wm.get_transitions(entity.id)
        out["transition_count"] = len(transitions)
        out["history"] = [
            {
                "date": _fmt_ts(t.timestamp),
                "type": t.transition_type.value,
                "summary": t.trigger_summary,
            }
            for t in transitions
        ]
        rels = wm.get_relationships(entity.id)
        out["relationships"] = [
            {
                "type": r.type.value,
                "entity": (wm.entities.get(
                    r.target_id if r.source_id == entity.id else r.source_id
                ) or type("_", (), {"name": "unknown"})()).name,
                "direction": "out" if r.source_id == entity.id else "in",
                "description": r.description[:100],
            }
            for r in rels[:15]
        ]
    return out


# ── MCP Server ─────────────────────────────────────────────────────────────────

mcp = FastMCP(
    "PIE — Personal Intelligence Engine",
    instructions=(
        "You have direct access to a personal temporal knowledge graph with ~4000 entities "
        "and ~6700 state transitions spanning over a year of the user's life.\n\n"
        "STRATEGY FOR BEST RESULTS:\n"
        "1. For specific questions: use `search` or `get_entity` to find precise facts.\n"
        "2. For broad questions ('tell me about all my projects / health / etc'): use `broad_scan` "
        "   which decomposes your query into 12 specific sub-queries and retrieves 60 entities.\n"
        "3. For a full LLM-synthesized answer: use `answer` — it runs retrieval + temporal context "
        "   compilation + gpt-5.4 synthesis in one call.\n"
        "4. For session context: call `get_briefing` at conversation start.\n\n"
        "The knowledge graph has typed entities: project, person, organization, tool, belief, "
        "goal, event, decision, concept, period. Use entity_type filter in `search` to narrow down.\n\n"
        "State transitions track HOW things changed over time — contradictions, updates, resolutions. "
        "This is the core differentiator vs. flat memory systems."
    ),
)


@mcp.tool()
def search(query: str, entity_type: str = "", top_k: int = 15) -> str:
    """Hybrid BM25+dense search over the knowledge graph.

    Returns ranked entities with current state and metadata.
    Fast (< 1s) — use for specific entity lookups.

    Args:
        query: What to search for. E.g. "Lucid Academy revenue", "sleep supplements", "BJJ training"
        entity_type: Optional filter. One of: project, person, organization, tool, belief,
                     goal, event, decision, concept, period
        top_k: Number of results (default 15, max 30)
    """
    wm = _get_wm()
    r = _get_retriever()
    entity_ids = r.retrieve(query, top_k=min(top_k, 30))

    results = []
    for eid in entity_ids:
        entity = wm.entities.get(eid)
        if not entity:
            continue
        if entity_type and entity.type.value != entity_type.lower():
            continue
        results.append(_entity_to_dict(entity, wm))

    return json.dumps({"query": query, "count": len(results), "entities": results},
                      indent=2, default=str)


@mcp.tool()
def broad_scan(topic: str, top_k: int = 60) -> str:
    """Exhaustive memory scan — decomposes topic into 12 sub-queries, returns up to 60 entities.

    USE THIS for: "tell me everything about my health", "list all my projects",
    "what do you know about my diet", "scan all health memories", etc.

    Internally: LLM generates 12 specific search queries from your topic →
    each runs BM25+dense retrieval → all results merged via multi-source RRF.
    One LLM call + fast local retrieval. Takes ~5-10s.

    Args:
        topic: Broad topic or question. Can be a full paragraph for maximum recall.
        top_k: Max entities to return (default 60)
    """
    wm = _get_wm()
    r = _get_retriever()
    entity_ids = r.broad_retrieve(topic, top_k=min(top_k, 80))

    results = []
    for eid in entity_ids:
        entity = wm.entities.get(eid)
        if entity:
            results.append(_entity_to_dict(entity, wm))

    return json.dumps({
        "topic": topic,
        "entities_found": len(results),
        "note": "Use get_entity(name) for full history of any specific entity.",
        "entities": results,
    }, indent=2, default=str)


@mcp.tool()
def get_entity(name: str) -> str:
    """Get complete details for an entity: full state + every transition + all relationships.

    Use this after search/broad_scan when you need the complete history of a specific entity.
    Fuzzy-matches the name if exact match not found.

    Args:
        name: Entity name (e.g. "Lucid Academy", "ADHD", "carnivore diet")
    """
    wm = _get_wm()
    entity = wm.find_by_name(name)

    if not entity:
        # Try fuzzy fallback
        candidates = []
        name_lower = name.lower()
        for e in wm.entities.values():
            if name_lower in e.name.lower() or any(name_lower in a.lower() for a in e.aliases):
                candidates.append(e)
        if candidates:
            entity = sorted(candidates, key=lambda e: -(e.importance or 0))[0]

    if not entity:
        return json.dumps({"error": f"No entity found matching '{name}'. Try search() first."})

    return json.dumps(_entity_to_dict(entity, wm, include_history=True),
                      indent=2, default=str)


@mcp.tool()
def answer(question: str, mode: str = "auto") -> str:
    """Full RAG pipeline: retrieve → compile temporal context → gpt-5.4 answer.

    This is the highest-quality tool: it retrieves relevant entities, compiles
    rich temporal context (state history, contradictions, dates), and synthesizes
    a full answer using gpt-5.4.

    Args:
        question: Any natural language question about the user's world.
        mode: "auto" (default) | "broad" (force 60-entity scan) | "focused" (top 10 only).
              "auto" picks broad for long/scan queries, focused for specific questions.
    """
    from pie.eval.query_interface import answer_query, _is_scan_query
    from datetime import datetime

    llm = _get_llm()
    r = _get_retriever()

    force_broad = (mode == "broad") or (mode == "auto" and _is_scan_query(question))

    result = answer_query(
        question, r, llm,
        model=_ANSWER_MODEL,
        top_k=60 if force_broad else 15,
        force_broad=force_broad,
    )

    return json.dumps({
        "question": question,
        "answer": result.answer,
        "entities_used": result.entities_used,
        "retrieval_mode": result.retrieval_method,
        "latency_ms": round(result.latency_ms),
    }, indent=2, default=str)


@mcp.tool()
def get_briefing() -> str:
    """Get an executive briefing: active projects, goals, stale entities, people network.

    Call this at the start of a session to load temporal context.
    Returns: top projects by importance, all goals, stale entities, people.
    """
    wm = _get_wm()
    from pie.core.models import EntityType

    now = time.time()
    type_counts: dict[str, int] = defaultdict(int)
    for e in wm.entities.values():
        type_counts[e.type.value] += 1

    def _top(etype, limit=20):
        entities = [e for e in wm.entities.values() if e.type.value == etype]
        entities.sort(key=lambda e: -(e.importance or 0))
        return [
            {
                "name": e.name,
                "last_seen": _humanize(e.last_seen),
                "status": e.current_state.get("status", "") if isinstance(e.current_state, dict) else "",
                "description": (e.current_state.get("description", "") if isinstance(e.current_state, dict) else str(e.current_state))[:200],
            }
            for e in entities[:limit]
        ]

    stale = [
        e for e in wm.entities.values()
        if e.last_seen and (now - e.last_seen) / 86400 > 30
        and e.type.value in ("project", "goal", "decision")
    ]
    stale.sort(key=lambda e: -(now - (e.last_seen or 0)))

    return json.dumps({
        "entity_counts": dict(type_counts),
        "active_projects": _top("project", 20),
        "goals": _top("goal", 15),
        "people": _top("person", 20),
        "stale_projects": [
            {"name": e.name, "last_seen": _humanize(e.last_seen)}
            for e in stale[:10]
        ],
    }, indent=2, default=str)


@mcp.tool()
def get_beliefs() -> str:
    """Get all extracted beliefs, preferences, and stated opinions.

    These are positions the user has stated — values, worldview, preferences —
    that can change over time and have state transition history.
    """
    wm = _get_wm()
    beliefs = [e for e in wm.entities.values() if e.type.value == "belief"]
    beliefs.sort(key=lambda e: -(e.importance or 0))
    return json.dumps([
        {
            "name": e.name,
            "description": (e.current_state.get("description", "") if isinstance(e.current_state, dict) else str(e.current_state))[:300],
            "last_seen": _humanize(e.last_seen),
        }
        for e in beliefs
    ], indent=2, default=str)


@mcp.tool()
def get_decisions() -> str:
    """Get all significant decisions with their reasoning and context."""
    wm = _get_wm()
    decisions = [e for e in wm.entities.values() if e.type.value == "decision"]
    decisions.sort(key=lambda e: -(e.importance or 0))
    return json.dumps([
        {
            "name": e.name,
            "description": (e.current_state.get("description", "") if isinstance(e.current_state, dict) else str(e.current_state))[:300],
            "last_seen": _humanize(e.last_seen),
        }
        for e in decisions
    ], indent=2, default=str)


@mcp.tool()
def chat_session(
    message: str,
    session_id: str = "default",
    top_k: int = 20,
) -> str:
    """Conversational interface to PIE with persistent memory across calls.

    Unlike `answer()` (single-shot RAG), this maintains a full conversation
    history in a session file. Each call adds to the history, so you can ask
    follow-up questions and build on previous answers — same as talking to a
    person who remembers everything you said in the conversation.

    The session persists between MCP calls and even between Cursor restarts.
    Use session_id to maintain separate conversations (default: "default").

    Args:
        message: Your message or question.
        session_id: Conversation thread name (default: "default").
        top_k: Number of entities to retrieve for context (default: 20).
    """
    _SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    session_file = _SESSIONS_DIR / f"{session_id}.json"

    # Load existing history
    history: list[dict] = []
    if session_file.exists():
        try:
            history = json.loads(session_file.read_text())
        except Exception:
            history = []

    # Retrieve relevant PIE context for this message
    retriever = _get_retriever()
    try:
        entity_ids = retriever.broad_retrieve(message, top_k=top_k, n_subqueries=6)
        context_md = retriever.compile_context(entity_ids, query=message, max_transitions=10)
        if len(context_md) > 40_000:
            context_md = context_md[:40_000] + "\n\n[...truncated...]"
    except Exception:
        context_md = ""

    # Build messages: system + history + new context + user message
    system = (
        "You are a personal knowledge assistant with access to the user's temporal knowledge graph "
        "(PIE — Personal Intelligence Engine). You have full conversation memory within this session.\n\n"
        "Use the CONTEXT BLOCK below (retrieved from PIE for this specific message) to ground your answer. "
        "Also use anything discussed earlier in the conversation. "
        "Be direct, specific, and make use of temporal detail (dates, transitions, contradictions).\n\n"
        f"PIE Context for this message:\n\n{context_md}"
    )

    messages = [{"role": "system", "content": system}]
    # Include last 20 turns of history to stay within context limits
    for turn in history[-20:]:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": message})

    llm = _get_llm()
    result = llm.chat(messages=messages, model=_ANSWER_MODEL, max_tokens=2000)
    answer = (result.get("content") or "").strip()

    # Save updated history
    history.append({"role": "user", "content": message, "ts": time.time()})
    history.append({"role": "assistant", "content": answer, "ts": time.time()})
    session_file.write_text(json.dumps(history, ensure_ascii=False, indent=2))

    return json.dumps({
        "answer": answer,
        "session_id": session_id,
        "turn": len(history) // 2,
        "entities_used": len(entity_ids) if 'entity_ids' in dir() else 0,
    }, ensure_ascii=False)


@mcp.tool()
def list_sessions() -> str:
    """List all active conversation sessions with their message counts and last activity."""
    _SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    sessions = []
    for f in sorted(_SESSIONS_DIR.glob("*.json")):
        try:
            history = json.loads(f.read_text())
            if not isinstance(history, list):
                continue
            last_ts = max((m.get("ts", 0) for m in history), default=0)
            sessions.append({
                "session_id": f.stem,
                "turns": len(history) // 2,
                "last_active": _humanize(last_ts),
                "last_message": next(
                    (m["content"][:80] for m in reversed(history) if m.get("role") == "user"), ""
                ),
            })
        except Exception:
            pass
    return json.dumps(sessions, indent=2)


@mcp.tool()
def ingest_conversation(
    text: str,
    title: str = "Untitled conversation",
    source: str = "mcp_ingest",
) -> str:
    """Ingest a new conversation or notes into the knowledge graph.

    Use this to keep PIE up to date with recent conversations, decisions,
    projects, or events. Provide the raw conversation text (or a structured
    summary) and it will be extracted, resolved against existing entities,
    and saved permanently.

    Args:
        text: Raw conversation text, meeting notes, or structured summary.
              Supports any format — ChatGPT-style, plain prose, bullet points.
        title: Human-readable title for this conversation/note.
        source: Source identifier (default: 'mcp_ingest').

    Returns:
        Summary of what was extracted and added.
    """
    import uuid as _uuid
    from pie.core.models import Conversation, Turn, DailyBatch
    from pie.core.llm import LLMClient, parse_extraction_result
    from pie.ingestion.prompts import (
        EXTRACTION_SYSTEM_PROMPT,
        build_extraction_user_message,
        format_conversations_for_extraction,
    )
    from pie.resolution.resolver import EntityResolver
    from pie.config import PIEConfig

    wm = _get_wm()
    llm = _get_llm()
    config = PIEConfig()

    now = time.time()
    conv_id = str(_uuid.uuid4())

    # Wrap raw text into a minimal Conversation object
    conv = Conversation(
        id=conv_id,
        title=title,
        created_at=now,
        updated_at=now,
        model=source,
        turns=[Turn(role="user", text=text, timestamp=now)],
    )

    batch = DailyBatch(
        date=__import__("datetime").datetime.now().strftime("%Y-%m-%d"),
        conversations=[conv],
    )

    # Build context preamble from current world model
    context_preamble = ""
    if len(wm.entities) > 0:
        context_preamble = wm.build_context_preamble(now)

    from pie.ingestion.prompts import format_conversations_for_extraction
    conversations_text = format_conversations_for_extraction(
        batch.conversations, max_chars_per_turn=8000, max_turns_per_conversation=50
    )
    user_message = build_extraction_user_message(
        batch_date=batch.date,
        conversations_text=conversations_text,
        context_preamble=context_preamble,
        num_conversations=1,
    )

    result = llm.chat(
        messages=[
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        model="gpt-5.4",
        json_mode=True,
        max_tokens=4000,
    )

    extraction = parse_extraction_result(
        raw=result["content"],
        conversation_ids=[conv_id],
        tokens=result["tokens"],
    )

    # Resolve entities against existing world model
    resolver = EntityResolver(world_model=wm, llm=llm, config=config.resolution)
    resolved = resolver.resolve(extraction.entities)

    # Apply to world model
    creates, updates = 0, 0
    for r in resolved:
        if r.action == "create":
            from pie.core.models import Entity, EntityType
            entity = Entity(
                id=str(_uuid.uuid4()),
                type=EntityType(r.extracted.type),
                name=r.extracted.name,
                aliases=[],
                current_state=r.extracted.state or {},
                first_seen=now,
                last_seen=now,
                importance=r.extracted.confidence or 0.5,
            )
            wm.entities[entity.id] = entity
            creates += 1
        elif r.action == "update" and r.matched_id:
            wm.update_entity_state(
                entity_id=r.matched_id,
                new_state=r.extracted.state or {},
                source_conversation_id=conv_id,
                timestamp=now,
                trigger_summary=f"[{source}] {title}",
            )
            updates += 1

    # Apply state changes
    for sc in extraction.state_changes:
        # Find entity by name
        match = wm.find_by_name(sc.entity_name)
        if match:
            wm.update_entity_state(
                entity_id=match.id,
                new_state={"description": sc.new_state} if isinstance(sc.new_state, str) else (sc.new_state or {}),
                source_conversation_id=conv_id,
                timestamp=now,
                trigger_summary=sc.what_changed,
                is_contradiction=sc.is_contradiction,
            )

    wm.rebuild_embedding_matrix()
    wm.save()

    global _retriever
    _retriever = None  # Reset so it rebuilds with new data on next query

    return json.dumps({
        "status": "ok",
        "title": title,
        "entities_created": creates,
        "entities_updated": updates,
        "state_changes_applied": len(extraction.state_changes),
        "relationships_found": len(extraction.relationships),
        "tokens_used": result["tokens"]["total"],
        "summary": extraction.summary or f"Ingested '{title}': {creates} new entities, {updates} updated.",
    }, indent=2)


@mcp.tool()
def get_entities_by_type(entity_type: str, limit: int = 50) -> str:
    """Retrieve all entities of a specific type from the knowledge graph.

    Useful for browsing the full inventory of tools, projects, people, goals, etc.
    without needing a search query.

    Args:
        entity_type: One of: tool, project, person, organization, goal, decision,
                     belief, concept, event, period
        limit: Max entities to return (default 50, max 200)
    """
    from pie.core.models import EntityType
    wm = _get_wm()

    try:
        target_type = EntityType(entity_type.lower())
    except ValueError:
        valid = [e.value for e in EntityType]
        return json.dumps({"error": f"Unknown type '{entity_type}'. Valid: {valid}"})

    matches = [
        e for e in wm.entities.values()
        if e.type == target_type
    ]
    # Sort by last_seen descending
    matches.sort(key=lambda e: e.last_seen or 0, reverse=True)
    matches = matches[:min(limit, 200)]

    results = [_entity_to_dict(e, wm) for e in matches]
    return json.dumps({
        "entity_type": entity_type,
        "total_found": len(results),
        "entities": results,
    }, indent=2, default=str)


@mcp.tool()
def get_recent_entities(limit: int = 30, entity_type: str | None = None) -> str:
    """Get the most recently updated entities — useful for a 'what's new' pulse check.

    Args:
        limit: Number of entities to return (default 30)
        entity_type: Optional — filter by type (tool, project, person, etc.)
    """
    from pie.core.models import EntityType
    wm = _get_wm()

    entities = list(wm.entities.values())

    if entity_type:
        try:
            target_type = EntityType(entity_type.lower())
            entities = [e for e in entities if e.type == target_type]
        except ValueError:
            pass

    entities.sort(key=lambda e: e.last_seen or 0, reverse=True)
    entities = entities[:limit]

    results = [_entity_to_dict(e, wm) for e in entities]
    return json.dumps({
        "limit": limit,
        "entity_type_filter": entity_type,
        "entities": results,
    }, indent=2, default=str)


@mcp.tool()
def get_architecture(project_name: str) -> str:
    """
    Generate a full architecture document for a project from the knowledge graph.

    Retrieves the project entity, all related tool/technology entities, architectural
    decisions, and key relationships — then synthesizes a structured architecture doc
    showing what tools were chosen for each layer and why.

    Args:
        project_name: Name of the project (e.g. "Lucid Academy", "Hermes", "PIE")
    """
    wm = _get_wm()
    llm = _get_llm()
    retriever = _get_retriever()

    # Broad retrieve: project + all tech choices, decisions, tools in its orbit
    entity_ids = retriever.broad_retrieve(
        f"{project_name} tech stack architecture tools decisions database framework API",
        top_k=40,
        n_subqueries=8,
    )

    if not entity_ids:
        return json.dumps({"error": f"No entities found for project: {project_name}"})

    context = retriever.compile_context(entity_ids, max_transitions=8)

    # Also pull direct relationships for the project entity
    project_entity = wm.find_by_name(project_name)
    rel_context = ""
    if project_entity:
        rels = [r for r in wm.relationships.values()
                if r.from_entity_id == project_entity.id or r.to_entity_id == project_entity.id]
        if rels:
            lines = []
            for r in rels[:20]:
                other_id = r.to_entity_id if r.from_entity_id == project_entity.id else r.from_entity_id
                other = wm.entities.get(other_id)
                other_name = other.name if other else other_id
                direction = "→" if r.from_entity_id == project_entity.id else "←"
                lines.append(f"  {project_name} {direction} [{r.relationship_type}] {other_name}")
            rel_context = "\nDirect relationships:\n" + "\n".join(lines)

    system = """\
You are an expert software architect with deep knowledge of AI and full-stack systems.
Given context from a personal knowledge graph about a project, produce a structured
architecture document. Focus on:
1. What the project does (1-2 sentences)
2. Tech stack by layer (frontend, backend, AI/ML, data, infra, integrations)
3. Key architectural decisions made and the reasoning
4. Tool choices per component and why they were selected
5. Current status and open architectural questions

Format as clean markdown. Be specific — cite exact tools and decisions from the context.
Where the knowledge graph has gaps, note them explicitly rather than guessing."""

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Project: {project_name}\n\n{context}{rel_context}\n\nGenerate the architecture document."},
    ]

    result = llm.chat(messages=messages, model=_ANSWER_MODEL, max_tokens=2000)
    content = result["content"]
    if isinstance(content, dict):
        content = json.dumps(content)

    return content


@mcp.tool()
def enrich_entity(entity_name: str, force: bool = False) -> str:
    """
    Enrich a tool/organization entity with current web-grounded knowledge.

    Looks up the entity in PIE, then uses LLM knowledge + any available web search
    to produce an up-to-date description covering: what it does, current version/status,
    how it fits into the broader ecosystem, and notes for your specific use cases.
    Saves the enrichment as a new state transition so it persists.

    Args:
        entity_name: Name of the tool or org to enrich (e.g. "LangGraph", "Supabase")
        force: If True, re-enriches even if recently updated (default: False)
    """
    import os
    from datetime import datetime, timezone

    wm = _get_wm()
    llm = _get_llm()

    entity = wm.find_by_name(entity_name)
    if not entity:
        # Try fuzzy search
        results = wm.find_by_embedding(
            llm.embed_single(entity_name), top_k=1
        )
        if results:
            entity = wm.entities.get(results[0])
        if not entity:
            return json.dumps({"error": f"Entity not found: {entity_name}"})

    current_state = entity.current_state or {}
    current_desc = current_state.get("description", "")
    entity_type = entity.type.value

    # Check if recently enriched (within 7 days) unless force
    enriched_at = current_state.get("web_enriched_at")
    if enriched_at and not force:
        from datetime import datetime
        try:
            days_ago = (datetime.now() - datetime.fromisoformat(enriched_at)).days
            if days_ago < 7:
                return json.dumps({
                    "status": "skipped",
                    "reason": f"Already enriched {days_ago}d ago. Use force=True to re-enrich.",
                    "entity": entity_name,
                })
        except Exception:
            pass

    # Gather context: all transitions + relationships
    retriever = _get_retriever()
    entity_ids = retriever.broad_retrieve(
        f"{entity_name} tool capabilities use cases ecosystem", top_k=10, n_subqueries=3
    )
    context = retriever.compile_context(entity_ids, max_transitions=5)

    # PIE usage context: how this entity appears in the user's history
    transitions = [t for t in wm.transitions.values() if t.entity_id == entity.id]
    usage_notes = []
    for t in sorted(transitions, key=lambda x: x.timestamp, reverse=True)[:5]:
        s = t.new_state if isinstance(t.new_state, dict) else {"description": str(t.new_state)}
        if s.get("description"):
            usage_notes.append(f"- {s['description'][:200]}")
    usage_context = "\n".join(usage_notes) if usage_notes else "No prior usage notes."

    system = """\
You are an expert technical researcher. Given information about a tool/technology from
a personal knowledge graph, produce an enriched, structured description that covers:
1. What it is and what it does (current as of 2026)
2. Key capabilities and differentiators vs alternatives
3. How it fits the user's specific use cases based on their history
4. Current maturity, ecosystem health, and any important recent changes
5. Best practices for integration

Be specific and factual. Today's date is 2026-04-13. Where you're uncertain about
very recent changes (post-2025), say so explicitly."""

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": (
            f"Tool/Entity: {entity_name} (type: {entity_type})\n\n"
            f"Current PIE description:\n{current_desc}\n\n"
            f"User's usage history:\n{usage_context}\n\n"
            f"Related context from knowledge graph:\n{context[:3000]}\n\n"
            "Produce an enriched description."
        )},
    ]

    result = llm.chat(messages=messages, model=_ANSWER_MODEL, max_tokens=1500)
    enriched = result["content"]
    if isinstance(enriched, dict):
        enriched = json.dumps(enriched)

    # Save as state transition
    now = datetime.now(timezone.utc).timestamp()
    new_state = dict(current_state)
    new_state["description"] = enriched
    new_state["web_enriched_at"] = datetime.now(timezone.utc).date().isoformat()

    wm.update_entity_state(
        entity_id=entity.id,
        new_state=new_state,
        source_conversation_id="mcp_enrich",
        timestamp=now,
        trigger_summary=f"Web-enriched description for {entity_name}",
    )
    wm.save()

    return json.dumps({
        "status": "enriched",
        "entity": entity_name,
        "entity_id": entity.id,
        "enriched_description": enriched,
    }, indent=2)


@mcp.tool()
def get_stats() -> str:
    """Get high-level statistics about the knowledge graph."""
    wm = _get_wm()
    type_counts: dict[str, int] = defaultdict(int)
    for e in wm.entities.values():
        type_counts[e.type.value] += 1

    return json.dumps({
        "total_entities": len(wm.entities),
        "total_transitions": len(wm.transitions),
        "total_relationships": len(wm.relationships),
        "entities_by_type": dict(sorted(type_counts.items(), key=lambda x: -x[1])),
        "world_model_path": str(_WM_PATH),
    }, indent=2)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run(transport="stdio")
