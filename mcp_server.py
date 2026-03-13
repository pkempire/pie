#!/usr/bin/env python3
"""
PIE MCP Server — temporal awareness engine for Claude.

Gives any LLM client (Claude Desktop, Cursor, etc.) awareness of your world:
what projects exist, what's stale, what deadlines are approaching, how long
since you last talked, and what deserves proactive mention.

Two key operations:
1. BEFORE conversation: get_temporal_briefing() — structured temporal context
2. AFTER conversation: update_world() — extract entities/transitions, track threads

Plus: search, entity lookup, timeline, deadlines, commitments, world stats.

Usage:
    python mcp_server.py                  # stdio mode (for Claude Desktop)

Configure in Claude Desktop settings.json:
    "mcpServers": {
        "pie": {
            "command": "python",
            "args": ["/path/to/personal-intelligence-system/mcp_server.py"],
            "env": {}
        }
    }
"""

import sys
import math
import time
import json
import logging
import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from mcp.server.fastmcp import FastMCP

from pie.core.world_model import WorldModel
from pie.core.dynamics import TransitionDynamics
from pie.core.models import EntityType
from pie.temporal.briefing import TemporalBriefing
from pie.temporal.gaps import GapAnalyzer
from pie.temporal.threads import ThreadTracker

logger = logging.getLogger("pie.mcp")

# ── Initialize ────────────────────────────────────────────────────────────────

WM_PATH = PROJECT_ROOT / "output" / "world_model.json"
THREADS_PATH = PROJECT_ROOT / "output" / "threads.json"
INTERACTIONS_PATH = PROJECT_ROOT / "output" / "interactions.json"

mcp = FastMCP(
    "PIE — Personal Intelligence Engine",
    instructions=(
        "You have access to Pranay's personal world model — a temporal knowledge graph "
        "with 4000 entities and 6700 state transitions spanning a year of his life. "
        "IMPORTANT: Call get_temporal_briefing() at the START of every conversation to "
        "load temporal context. Call update_world() at the END to keep the model current. "
        "Use the temporal briefing to proactively mention relevant deadlines, stale threads, "
        "and context — don't wait to be asked."
    ),
)

_wm: WorldModel | None = None
_dynamics: TransitionDynamics | None = None
_report = None
_briefing_engine: TemporalBriefing | None = None
_gap_analyzer: GapAnalyzer | None = None
_thread_tracker: ThreadTracker | None = None

# Semantic search index
_tfidf: TfidfVectorizer | None = None
_svd: TruncatedSVD | None = None
_embeddings: np.ndarray | None = None
_entity_id_order: list | None = None


def _get_wm() -> WorldModel:
    global _wm
    if _wm is None:
        _wm = WorldModel(persist_path=str(WM_PATH))
        if not _wm.entities:
            raise RuntimeError(f"No world model at {WM_PATH}")
    return _wm


def _get_briefing_engine() -> TemporalBriefing:
    global _briefing_engine
    if _briefing_engine is None:
        _briefing_engine = TemporalBriefing(_get_wm())
    return _briefing_engine


def _get_gap_analyzer() -> GapAnalyzer:
    global _gap_analyzer
    if _gap_analyzer is None:
        _gap_analyzer = GapAnalyzer(INTERACTIONS_PATH)
    return _gap_analyzer


def _get_thread_tracker() -> ThreadTracker:
    global _thread_tracker
    if _thread_tracker is None:
        _thread_tracker = ThreadTracker(THREADS_PATH)
    return _thread_tracker


def _get_search_index():
    """Build TF-IDF + SVD semantic search index (lazy, ~1s)."""
    global _tfidf, _svd, _embeddings, _entity_id_order
    if _tfidf is not None:
        return _tfidf, _svd, _embeddings, _entity_id_order

    wm = _get_wm()
    _entity_id_order = []
    texts = []

    for eid, entity in wm.entities.items():
        _entity_id_order.append(eid)
        cs = entity.current_state
        parts = [
            entity.name,
            entity.type.value,
            ' '.join(entity.aliases),
        ]
        for k, v in cs.items():
            if isinstance(v, str):
                parts.append(f"{k}: {v}")
            elif isinstance(v, list):
                parts.append(f"{k}: {' '.join(str(x) for x in v)}")
            elif isinstance(v, dict):
                parts.append(f"{k}: {json.dumps(v)}")
        texts.append(' '.join(parts))

    _tfidf = TfidfVectorizer(
        max_features=5000,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.8,
        sublinear_tf=True,
    )
    tfidf_matrix = _tfidf.fit_transform(texts)

    _svd = TruncatedSVD(n_components=128, random_state=42)
    _embeddings = _svd.fit_transform(tfidf_matrix)
    # Normalize
    norms = np.linalg.norm(_embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    _embeddings = _embeddings / norms

    logger.info(f"Search index built: {len(_entity_id_order)} entities, 128d embeddings")
    return _tfidf, _svd, _embeddings, _entity_id_order


def _semantic_search(query: str, top_k: int = 10, entity_type: str = "") -> list:
    """Semantic search using TF-IDF + SVD cosine similarity."""
    wm = _get_wm()
    tfidf, svd, embeddings, eid_order = _get_search_index()

    # Transform query
    q_tfidf = tfidf.transform([query])
    q_emb = svd.transform(q_tfidf)
    q_norm = np.linalg.norm(q_emb)
    if q_norm > 0:
        q_emb = q_emb / q_norm

    # Cosine similarity
    scores = embeddings @ q_emb.T
    scores = scores.flatten()

    # Filter by type if needed
    if entity_type:
        for i, eid in enumerate(eid_order):
            e = wm.entities.get(eid)
            if e and e.type.value != entity_type.lower():
                scores[i] = -1

    # Get top-k
    top_indices = np.argsort(scores)[::-1][:top_k]
    results = []
    for idx in top_indices:
        if scores[idx] <= 0:
            break
        eid = eid_order[idx]
        entity = wm.entities.get(eid)
        if entity:
            results.append((entity, float(scores[idx])))

    return results


def _get_dynamics():
    global _dynamics, _report
    wm = _get_wm()
    if _dynamics is None:
        _dynamics = TransitionDynamics(wm)
        _report = _dynamics.analyze()
    return _dynamics, _report


def _importance(eid: str) -> float:
    """Get entity importance (computes if needed)."""
    _, report = _get_dynamics()
    profile = report.entity_profiles.get(eid)
    if not profile:
        return 0.0
    base = math.log2(1 + profile.total_transitions) / 8.0
    base = min(base, 1.0)
    recency = 1.0 - 0.8 * profile.staleness_score
    return round(base * recency, 4)


def _entity_summary(entity, include_state=True, include_transitions=False) -> dict:
    """Serialize an entity for tool output."""
    wm = _get_wm()
    result = {
        "name": entity.name,
        "type": entity.type.value,
        "importance": entity.importance or _importance(entity.id),
        "first_seen": entity.first_seen,
        "last_seen": entity.last_seen,
        "aliases": entity.aliases,
        "web_verified": entity.web_verified,
    }
    if include_state:
        result["current_state"] = entity.current_state
    if include_transitions:
        transitions = wm.get_transitions(entity.id, ordered=True)
        result["transitions"] = [
            {
                "type": t.transition_type.value,
                "trigger": t.trigger_summary,
                "timestamp": t.timestamp,
                "from_state_keys": list((t.from_state or {}).keys()),
                "to_state_keys": list((t.to_state or {}).keys()),
            }
            for t in transitions[-10:]  # last 10
        ]
        result["total_transitions"] = len(transitions)
    return result


# ── Tools ─────────────────────────────────────────────────────────────────────


@mcp.tool()
def search_entities(query: str, entity_type: str = "", top_k: int = 10) -> str:
    """Semantic search the world model for entities matching a query.

    Uses TF-IDF + SVD embeddings for semantic matching — understands meaning,
    not just keywords. Use this to find projects, tools, people, beliefs, etc.

    Args:
        query: Search terms (e.g. "Lucid Academy", "browser automation", "BJJ", "revenue strategy")
        entity_type: Optional filter: person, project, tool, organization, belief, decision, concept, period, event, goal
        top_k: Max results to return
    """
    results_raw = _semantic_search(query, top_k=top_k, entity_type=entity_type)
    results = []
    for entity, score in results_raw:
        summary = _entity_summary(entity)
        summary["relevance_score"] = round(score, 3)
        results.append(summary)
    return json.dumps(results, indent=2, default=str)


@mcp.tool()
def get_entity(name: str) -> str:
    """Get full details for a specific entity by name.

    Returns the complete current state, transition history, and relationships.

    Args:
        name: Entity name (fuzzy matched)
    """
    wm = _get_wm()
    entity = wm.find_by_name(name)
    if not entity:
        # Try fuzzy
        matches = wm.find_by_fuzzy_name(name, threshold=0.6)
        if matches:
            entity = matches[0][0]
        else:
            return json.dumps({"error": f"No entity found matching '{name}'"})

    result = _entity_summary(entity, include_state=True, include_transitions=True)

    # Add relationships
    rels = wm.get_relationships(entity.id)
    result["relationships"] = []
    for r in rels:
        if r:
            other_id = r.target_id if r.source_id == entity.id else r.source_id
            other = wm.entities.get(other_id)
            other_name = other.name if other else "unknown"
            direction = "outgoing" if r.source_id == entity.id else "incoming"
            result["relationships"].append({
                "type": r.type.value,
                "entity": other_name,
                "direction": direction,
                "description": r.description,
            })

    return json.dumps(result, indent=2, default=str)


@mcp.tool()
def get_briefing() -> str:
    """Get today's executive briefing.

    Returns: active projects ranked by importance, goals & commitments,
    stale entities needing attention, causal co-occurrence patterns,
    predicted next transitions, and the people network.
    """
    wm = _get_wm()
    _, report = _get_dynamics()

    briefing = {}

    # Active projects
    projects = []
    for eid, entity in wm.entities.items():
        if entity.type not in (EntityType.PROJECT, EntityType.ORGANIZATION):
            continue
        profile = report.entity_profiles.get(eid)
        if not profile or profile.total_transitions < 2:
            continue
        days_since = (time.time() - profile.last_transition_ts) / 86400
        projects.append({
            "name": entity.name,
            "type": entity.type.value,
            "importance": entity.importance or _importance(eid),
            "staleness": round(profile.staleness_score, 3),
            "days_since_update": round(days_since, 1),
            "transitions": profile.total_transitions,
            "status": entity.current_state.get("status", "unknown"),
            "description": entity.current_state.get("description", "")[:200],
        })
    projects.sort(key=lambda p: -p["importance"])
    briefing["active_projects"] = projects[:15]

    # Goals
    goals = []
    for eid, entity in wm.entities.items():
        if entity.type == EntityType.GOAL:
            goals.append({
                "name": entity.name,
                "deadline": entity.current_state.get("deadline"),
                "status": entity.current_state.get("status", "unknown"),
                "priority": entity.current_state.get("priority", "unknown"),
                "description": entity.current_state.get("description", "")[:200],
            })
    # Also infer goals from decisions
    for eid, entity in wm.entities.items():
        if entity.type == EntityType.DECISION:
            desc = entity.current_state.get("description", "").lower()
            if any(w in desc for w in ["plan to", "want to", "need to", "target", "aiming", "will "]):
                goals.append({
                    "name": entity.name,
                    "deadline": entity.current_state.get("deadline"),
                    "status": "inferred_from_decision",
                    "priority": "medium",
                    "description": entity.current_state.get("description", "")[:200],
                })
    briefing["goals"] = goals[:20]

    # Stale
    stale = []
    for eid in report.stale_entities:
        entity = wm.entities.get(eid)
        profile = report.entity_profiles.get(eid)
        if not entity or not profile:
            continue
        if entity.type in (EntityType.CONCEPT, EntityType.PERIOD):
            continue
        days_since = (time.time() - profile.last_transition_ts) / 86400
        stale.append({
            "name": entity.name,
            "type": entity.type.value,
            "staleness": round(profile.staleness_score, 3),
            "days_since_update": round(days_since, 1),
            "importance": entity.importance or _importance(eid),
        })
    stale.sort(key=lambda s: (-s.get("importance", 0), -s["staleness"]))
    briefing["stale_entities"] = stale[:15]

    # Predictions
    briefing["predicted_next_transitions"] = report.predicted_next_transitions[:10]

    # People
    people = []
    for eid, entity in wm.entities.items():
        if entity.type == EntityType.PERSON:
            people.append({
                "name": entity.name,
                "description": entity.current_state.get("description", "")[:150],
                "importance": entity.importance or _importance(eid),
            })
    people.sort(key=lambda p: -p["importance"])
    briefing["people"] = people

    return json.dumps(briefing, indent=2, default=str)


@mcp.tool()
def get_beliefs() -> str:
    """Get all of Parth's extracted beliefs, preferences, and opinions.

    These are things he has stated positions on — values, preferences,
    personality traits, and worldview that could change over time.
    """
    wm = _get_wm()
    beliefs = []
    for eid, entity in wm.entities.items():
        if entity.type == EntityType.BELIEF:
            beliefs.append({
                "name": entity.name,
                "description": entity.current_state.get("description", "")[:300],
                "importance": entity.importance or _importance(eid),
            })
    beliefs.sort(key=lambda b: -b["importance"])
    return json.dumps(beliefs, indent=2, default=str)


@mcp.tool()
def get_decisions() -> str:
    """Get all significant decisions Parth has made.

    Includes technical, business, product, and personal decisions with reasoning.
    """
    wm = _get_wm()
    decisions = []
    for eid, entity in wm.entities.items():
        if entity.type == EntityType.DECISION:
            decisions.append({
                "name": entity.name,
                "description": entity.current_state.get("description", "")[:300],
                "importance": entity.importance or _importance(eid),
            })
    decisions.sort(key=lambda d: -d["importance"])
    return json.dumps(decisions, indent=2, default=str)


@mcp.tool()
def get_entity_history(name: str) -> str:
    """Get the full state transition history for an entity.

    Shows how an entity evolved over time — every state change with timestamps,
    triggers, and before/after states.

    Args:
        name: Entity name (fuzzy matched)
    """
    wm = _get_wm()
    entity = wm.find_by_name(name)
    if not entity:
        matches = wm.find_by_fuzzy_name(name, threshold=0.6)
        if matches:
            entity = matches[0][0]
        else:
            return json.dumps({"error": f"No entity found matching '{name}'"})

    transitions = wm.get_transitions(entity.id, ordered=True)
    history = {
        "entity": entity.name,
        "type": entity.type.value,
        "total_transitions": len(transitions),
        "timeline": [],
    }

    for t in transitions:
        entry = {
            "type": t.transition_type.value,
            "trigger": t.trigger_summary,
            "timestamp": t.timestamp,
        }
        # Show what changed (new keys or changed values)
        if t.from_state and t.to_state:
            changes = {}
            for k in set(list(t.to_state.keys()) + list(t.from_state.keys())):
                old_v = t.from_state.get(k)
                new_v = t.to_state.get(k)
                if str(old_v) != str(new_v):
                    changes[k] = {"from": str(old_v)[:100], "to": str(new_v)[:100]}
            if changes:
                entry["changes"] = changes
        elif t.to_state:
            entry["initial_state"] = {k: str(v)[:100] for k, v in t.to_state.items()}

        history["timeline"].append(entry)

    return json.dumps(history, indent=2, default=str)


@mcp.tool()
def get_related_entities(name: str) -> str:
    """Find all entities connected to a given entity via relationships.

    Shows the network around an entity — what it uses, is part of, integrates with, etc.

    Args:
        name: Entity name (fuzzy matched)
    """
    wm = _get_wm()
    entity = wm.find_by_name(name)
    if not entity:
        matches = wm.find_by_fuzzy_name(name, threshold=0.6)
        if matches:
            entity = matches[0][0]
        else:
            return json.dumps({"error": f"No entity found matching '{name}'"})

    rels = wm.get_relationships(entity.id)
    connections = []
    for r in rels:
        other_id = r.target_id if r.source_id == entity.id else r.source_id
        other = wm.entities.get(other_id)
        if not other:
            continue
        connections.append({
            "entity": other.name,
            "type": other.type.value,
            "relationship": r.type.value,
            "direction": "outgoing" if r.source_id == entity.id else "incoming",
            "description": r.description,
            "importance": other.importance or 0,
        })

    connections.sort(key=lambda c: -c["importance"])
    return json.dumps({
        "entity": entity.name,
        "total_connections": len(connections),
        "connections": connections,
    }, indent=2, default=str)


@mcp.tool()
def get_world_model_stats() -> str:
    """Get high-level statistics about the world model.

    Returns entity counts by type, transition counts, date range, etc.
    """
    wm = _get_wm()
    _, report = _get_dynamics()

    type_counts = defaultdict(int)
    for entity in wm.entities.values():
        type_counts[entity.type.value] += 1

    return json.dumps({
        "total_entities": len(wm.entities),
        "total_transitions": len(wm.transitions),
        "total_relationships": len(wm.relationships),
        "entities_by_type": dict(sorted(type_counts.items(), key=lambda x: -x[1])),
        "stale_count": len(report.stale_entities),
        "volatile_count": len(report.volatile_entities),
        "cooccurrence_patterns": len(report.cooccurrences),
    }, indent=2)


@mcp.tool()
def query_world_model(question: str) -> str:
    """Ask a natural-language question about Parth's world model.

    Uses semantic search (TF-IDF + SVD embeddings) to find the most relevant
    entities, then returns full context including state and relationships.

    Examples:
    - "What tools is Parth using for browser automation?"
    - "What's the status of Lucid Academy?"
    - "Who does Parth work with?"
    - "What decisions has he made about monetization?"
    - "What content ideas has he discussed?"

    Args:
        question: Natural language question
    """
    wm = _get_wm()
    results_raw = _semantic_search(question, top_k=10)

    results = []
    for entity, score in results_raw:
        entry = {
            "name": entity.name,
            "type": entity.type.value,
            "relevance_score": round(score, 3),
            "current_state": entity.current_state,
            "importance": entity.importance or _importance(entity.id),
        }
        # Add relationships for highly relevant results
        if score > 0.3:
            rels = wm.get_relationships(entity.id)
            entry["relationships"] = [
                {
                    "type": r.type.value,
                    "entity": getattr(wm.entities.get(
                        r.target_id if r.source_id == entity.id else r.source_id
                    ), 'name', 'unknown'),
                    "description": r.description,
                }
                for r in rels[:5]
            ]
        results.append(entry)

    return json.dumps({
        "question": question,
        "results_count": len(results),
        "results": results,
    }, indent=2, default=str)


# ── Temporal Tools (THE NEW STUFF) ────────────────────────────────────────────


@mcp.tool()
def get_temporal_briefing(focus_project: str = "", current_context: str = "") -> str:
    """Get a temporal briefing — CALL THIS AT THE START OF EVERY CONVERSATION.

    Returns structured temporal context: what's active, what's stale, approaching
    deadlines, how long since last interaction, and what deserves proactive mention.
    This is the core tool that gives you temporal awareness.

    Args:
        focus_project: Optional project name to expand context for (e.g. "PIE", "sponsorFind")
        current_context: Optional brief description of what the user is working on right now
    """
    now = time.time()
    gap = _get_gap_analyzer()
    tracker = _get_thread_tracker()
    engine = _get_briefing_engine()

    # Record this interaction
    gap.record_interaction(now)

    # Get temporal data
    deadlines = tracker.get_approaching_deadlines(window_days=14)
    commitments = tracker.get_overdue_commitments()

    briefing = engine.generate_briefing(
        ref_time=now,
        focus_project=focus_project if focus_project else None,
        last_interaction_time=gap.last_interaction_time,
        approaching_deadlines=deadlines,
        overdue_commitments=commitments,
    )

    return briefing


@mcp.tool()
def update_world(
    conversation_summary: str,
    entities_mentioned: str = "",
    deadlines_mentioned: str = "",
    commitments_made: str = "",
) -> str:
    """Update the world model after a conversation — CALL THIS AT THE END.

    Records what was discussed so future briefings stay current. Also updates
    thread tracking for deadlines and commitments.

    Args:
        conversation_summary: Brief summary of what was discussed in this conversation.
        entities_mentioned: Comma-separated names of projects/people/tools discussed.
        deadlines_mentioned: JSON array of deadlines: [{"topic": "...", "due_date": "...", "entity": "..."}]
        commitments_made: JSON array of commitments: [{"what": "...", "who": "user", "due_date": "..."}]
    """
    tracker = _get_thread_tracker()
    now = time.time()

    # Parse entities
    entity_names = [e.strip() for e in entities_mentioned.split(",") if e.strip()] if entities_mentioned else []

    # Parse deadlines
    parsed_deadlines = None
    if deadlines_mentioned:
        try:
            parsed_deadlines = json.loads(deadlines_mentioned)
        except json.JSONDecodeError:
            pass

    # Parse commitments
    parsed_commitments = None
    if commitments_made:
        try:
            parsed_commitments = json.loads(commitments_made)
        except json.JSONDecodeError:
            pass

    # Update thread tracker
    tracker.update_from_conversation(
        mentioned_entities=entity_names,
        new_deadlines=parsed_deadlines,
        new_commitments=parsed_commitments,
    )

    # Touch any matching entities in the world model to update last_seen
    wm = _get_wm()
    touched = []
    for name in entity_names:
        entity = wm.find_by_name(name)
        if entity:
            entity.last_seen = now
            touched.append(entity.name)

    # Save world model if we touched anything
    if touched:
        wm.save()

    result = {
        "status": "updated",
        "entities_touched": touched,
        "deadlines_added": len(parsed_deadlines) if parsed_deadlines else 0,
        "commitments_added": len(parsed_commitments) if parsed_commitments else 0,
        "summary_recorded": conversation_summary[:200],
    }

    return json.dumps(result, indent=2)


@mcp.tool()
def get_approaching_deadlines(window_days: int = 14) -> str:
    """Get deadlines approaching in the next N days.

    Returns tracked deadlines and commitments, sorted by urgency.

    Args:
        window_days: How many days ahead to look (default 14)
    """
    tracker = _get_thread_tracker()
    deadlines = tracker.get_approaching_deadlines(window_days)
    overdue = tracker.get_overdue_commitments()

    return json.dumps({
        "approaching_deadlines": deadlines,
        "overdue_commitments": overdue,
    }, indent=2, default=str)


@mcp.tool()
def get_stale_threads(threshold_days: int = 14) -> str:
    """Get active threads that haven't been mentioned recently.

    These are topics that were opened but have gone quiet — potential
    forgotten commitments or abandoned threads.

    Args:
        threshold_days: How many days of silence before a thread is "stale" (default 14)
    """
    tracker = _get_thread_tracker()
    stale = tracker.get_stale_threads(threshold_days)

    return json.dumps({
        "stale_threads": stale,
        "total_active_threads": len(tracker.get_active_threads()),
    }, indent=2, default=str)


@mcp.tool()
def track_deadline(
    topic: str,
    due_date: str,
    entity_name: str = "",
    notes: str = "",
) -> str:
    """Explicitly track a deadline for future proactive mention.

    Use this when the user mentions a deadline during conversation.
    It will appear in future temporal briefings as it approaches.

    Args:
        topic: What the deadline is for (e.g. "API demo", "paper submission")
        due_date: When it's due (e.g. "2026-03-20", "next Friday")
        entity_name: Optional project/entity name this relates to
        notes: Optional additional context
    """
    tracker = _get_thread_tracker()

    # Try to parse the date
    due_timestamp = None
    try:
        from datetime import datetime
        # Try common formats
        for fmt in ["%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%m/%d/%Y"]:
            try:
                dt = datetime.strptime(due_date, fmt)
                due_timestamp = dt.timestamp()
                break
            except ValueError:
                continue
    except Exception:
        pass

    thread = tracker.open_thread(
        topic=topic,
        entity_name=entity_name if entity_name else None,
        deadline=due_date,
        deadline_timestamp=due_timestamp,
    )
    if notes:
        thread.notes = notes
        tracker._save()

    return json.dumps({
        "status": "deadline_tracked",
        "thread_id": thread.id,
        "topic": topic,
        "due_date": due_date,
        "parsed_timestamp": due_timestamp,
    }, indent=2)


@mcp.tool()
def track_commitment(
    what: str,
    due_date: str = "",
    who: str = "user",
) -> str:
    """Track a commitment (something the user said they'd do).

    Will appear in future briefings when the due date passes.

    Args:
        what: What was committed to (e.g. "finish integration tests")
        due_date: When it's due (optional, e.g. "2026-03-20")
        who: Who committed — usually "user" (default)
    """
    tracker = _get_thread_tracker()

    due_timestamp = None
    if due_date:
        try:
            from datetime import datetime
            for fmt in ["%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%m/%d/%Y"]:
                try:
                    dt = datetime.strptime(due_date, fmt)
                    due_timestamp = dt.timestamp()
                    break
                except ValueError:
                    continue
        except Exception:
            pass

    tracker.add_commitment(
        thread_id=None,
        what=what,
        who=who,
        due_date=due_date,
        due_timestamp=due_timestamp,
    )

    return json.dumps({
        "status": "commitment_tracked",
        "what": what,
        "who": who,
        "due_date": due_date,
    }, indent=2)


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run(transport="stdio")
