"""
PIE Wikipedia UI Server

Run:
    python3 -m pie.ui.server --world-model output/world_model.json

Then open http://localhost:7331 in your browser.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger("pie.ui")

app = Flask(__name__, static_folder=None)
CORS(app)

# Module-level state loaded once at startup
_world_model = None
_retriever = None
_llm = None
_ui_dir = Path(__file__).parent


# ── Startup loader ─────────────────────────────────────────────────────────

def load(world_model_path: Path):
    global _world_model, _retriever, _llm

    from pie.core.world_model import WorldModel
    from pie.core.llm import LLMClient
    from pie.retrieval.hybrid_retriever import HybridRetriever
    from pie.config import PIEConfig

    logger.warning(f"Loading world model from {world_model_path}...")
    _world_model = WorldModel(persist_path=world_model_path)
    logger.warning(f"  {len(_world_model.entities)} entities, {len(_world_model.transitions)} transitions")

    _llm = LLMClient()
    logger.warning("Building retrieval index...")
    _retriever = HybridRetriever(_world_model, _llm, PIEConfig())
    logger.warning("Ready.")


# ── Static: serve wiki.html ────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(str(Path(__file__).parent.parent.parent), "wiki.html")


# ── API: entity tree ────────────────────────────────────────────────────────

@app.route("/api/entities")
def get_entities():
    """Return all entities grouped by type, sorted by importance/recency."""
    if _world_model is None:
        return jsonify({"error": "World model not loaded"}), 503

    now = datetime.now().timestamp()
    grouped: dict[str, list] = {}

    for eid, entity in _world_model.entities.items():
        etype = entity.type.value
        if etype not in grouped:
            grouped[etype] = []

        desc = ""
        if isinstance(entity.current_state, dict):
            desc = entity.current_state.get("description", "")
        if not desc:
            desc = str(entity.current_state)[:120]

        n_transitions = len(_world_model._entity_transitions.get(eid, []))
        recency_days = (now - entity.last_seen) / 86400 if entity.last_seen else 9999

        grouped[etype].append({
            "id": eid,
            "name": entity.name,
            "desc": desc[:120],
            "aliases": entity.aliases[:3],
            "n_transitions": n_transitions,
            "last_seen": entity.last_seen,
            "recency_days": round(recency_days, 1),
            "importance": entity.importance or 0,
        })

    # Sort each group: most recently seen first
    for etype in grouped:
        grouped[etype].sort(key=lambda e: e["last_seen"], reverse=True)

    return jsonify(grouped)


# ── API: entity raw data ────────────────────────────────────────────────────

@app.route("/api/entity/<entity_id>")
def get_entity(entity_id: str):
    """Return full entity data + transitions + relationships."""
    if _world_model is None:
        return jsonify({"error": "World model not loaded"}), 503

    entity = _world_model.entities.get(entity_id)
    if entity is None:
        return jsonify({"error": "Entity not found"}), 404

    now = datetime.now()
    transitions = _world_model.get_transitions(entity_id)
    relationships = _world_model.get_relationships(entity_id)

    def fmt_ts(ts: float) -> str:
        if not ts:
            return ""
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d")

    return jsonify({
        "id": entity.id,
        "name": entity.name,
        "type": entity.type.value,
        "aliases": entity.aliases,
        "current_state": entity.current_state,
        "first_seen": fmt_ts(entity.first_seen),
        "last_seen": fmt_ts(entity.last_seen),
        "importance": entity.importance,
        "transitions": [
            {
                "id": t.id,
                "type": t.transition_type.value,
                "summary": t.trigger_summary,
                "date": fmt_ts(t.timestamp),
                "to_state": t.to_state,
                "confidence": t.confidence,
            }
            for t in transitions
        ],
        "relationships": [
            {
                "id": r.id,
                "type": r.type.value,
                "description": r.description,
                "other_id": r.target_id if r.source_id == entity_id else r.source_id,
                "other_name": (
                    _world_model.entities[
                        r.target_id if r.source_id == entity_id else r.source_id
                    ].name
                    if (r.target_id if r.source_id == entity_id else r.source_id)
                    in _world_model.entities
                    else "unknown"
                ),
                "direction": "out" if r.source_id == entity_id else "in",
            }
            for r in relationships
        ],
    })


# ── API: LLM-generated wiki page ───────────────────────────────────────────

@app.route("/api/entity/<entity_id>/page", methods=["POST"])
def generate_page(entity_id: str):
    """Generate an LLM wiki page for an entity."""
    if _world_model is None or _llm is None:
        return jsonify({"error": "World model not loaded"}), 503

    entity = _world_model.entities.get(entity_id)
    if entity is None:
        return jsonify({"error": "Entity not found"}), 404

    model = (request.json or {}).get("model", "gpt-4o-mini")
    now = datetime.now()

    transitions = _world_model.get_transitions(entity_id)
    relationships = _world_model.get_relationships(entity_id)

    # Build rich context for the LLM
    context_lines = [
        f"Entity: {entity.name}",
        f"Type: {entity.type.value}",
        f"Aliases: {', '.join(entity.aliases) if entity.aliases else 'none'}",
        f"First seen: {datetime.fromtimestamp(entity.first_seen).strftime('%B %Y') if entity.first_seen else 'unknown'}",
        f"Last seen: {datetime.fromtimestamp(entity.last_seen).strftime('%B %d, %Y') if entity.last_seen else 'unknown'}",
        f"\nCurrent state:\n{json.dumps(entity.current_state, indent=2, default=str)}",
    ]

    if transitions:
        context_lines.append(f"\nState history ({len(transitions)} transitions):")
        for t in transitions[-20:]:
            dt = datetime.fromtimestamp(t.timestamp).strftime("%Y-%m-%d") if t.timestamp else "unknown"
            flag = " ⚠ CONTRADICTION" if t.transition_type.value == "contradiction" else ""
            context_lines.append(f"  [{dt}] {t.transition_type.value}{flag}: {t.trigger_summary}")

    if relationships:
        context_lines.append(f"\nRelationships:")
        for r in relationships[:15]:
            other_id = r.target_id if r.source_id == entity_id else r.source_id
            other = _world_model.entities.get(other_id)
            other_name = other.name if other else "unknown"
            direction = "→" if r.source_id == entity_id else "←"
            context_lines.append(f"  {direction} {r.type.value}: {other_name} — {r.description}")

    context = "\n".join(context_lines)

    system_prompt = """You are writing a personal Wikipedia page for one entry in someone's life knowledge graph.

Your output must be a JSON object with these fields:
{
  "intro": "2-3 sentence paragraph — current situation, most important thing to know right now",
  "summary": "markdown — what this entity is, its full history, key milestones (use ## subheadings)",
  "recent": "markdown — what changed in the last 90 days (bullet points with dates)",
  "todos": ["list", "of", "action items or open questions extracted from the history"],
  "status": "one word: active | paused | completed | abandoned | unclear",
  "tags": ["3-5 thematic tags"]
}

Write for the owner of this knowledge — first person perspective. Be concrete and direct.
If there are contradictions in the history, highlight them."""

    try:
        result = _llm.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Write a wiki page for:\n\n{context}"},
            ],
            model=model,
            json_mode=True,
        )
        page = json.loads(result["content"])
        page["entity_id"] = entity_id
        page["entity_name"] = entity.name
        page["generated_at"] = now.isoformat()
        return jsonify(page)
    except Exception as e:
        logger.exception(f"Page generation failed for {entity_id}")
        return jsonify({"error": str(e)}), 500


# ── API: hybrid search ──────────────────────────────────────────────────────

@app.route("/api/search", methods=["POST"])
def search():
    """Hybrid retrieval: BM25 + dense + RRF."""
    if _retriever is None:
        return jsonify({"error": "Retriever not loaded"}), 503

    body = request.json or {}
    query = body.get("query", "").strip()
    top_k = int(body.get("top_k", 15))
    if not query:
        return jsonify({"entities": []}), 200

    entity_ids = _retriever.retrieve(query, top_k=top_k)
    results = []
    for eid in entity_ids:
        entity = _world_model.entities.get(eid)
        if entity:
            desc = ""
            if isinstance(entity.current_state, dict):
                desc = entity.current_state.get("description", "")
            results.append({
                "id": eid,
                "name": entity.name,
                "type": entity.type.value,
                "desc": desc[:200],
                "last_seen": entity.last_seen,
            })

    return jsonify({"entities": results, "query": query})


# ── API: smart query (LLM answer + suggested follow-ups) ───────────────────

@app.route("/api/query", methods=["POST"])
def smart_query():
    """Full query pipeline: hybrid retrieval → temporal context → LLM answer."""
    if _retriever is None or _llm is None:
        return jsonify({"error": "Retriever not loaded"}), 503

    body = request.json or {}
    query = body.get("query", "").strip()
    model = body.get("model", "gpt-4o-mini")
    top_k = int(body.get("top_k", 10))
    if not query:
        return jsonify({"error": "Empty query"}), 400

    from pie.eval.query_interface import answer_query
    result = answer_query(query, _retriever, _llm, model=model, top_k=top_k)

    # Generate follow-up suggestions
    followups = []
    try:
        fu_result = _llm.chat(
            messages=[
                {
                    "role": "system",
                    "content": "Given a question and answer from a personal knowledge graph, suggest 3 short follow-up questions. Return JSON: {\"questions\": [\"...\", \"...\", \"...\"]}"
                },
                {
                    "role": "user",
                    "content": f"Q: {query}\n\nA: {result.answer[:500]}"
                }
            ],
            model=model,
            json_mode=True,
        )
        followups = json.loads(fu_result["content"]).get("questions", [])
    except Exception:
        pass

    return jsonify({
        "answer": result.answer,
        "entities_used": result.entities_used,
        "latency_ms": result.latency_ms,
        "followups": followups,
    })


# ── API: LLM-generate smart search terms ───────────────────────────────────

@app.route("/api/smart-search", methods=["POST"])
def smart_search():
    """LLM generates optimal search queries, then runs hybrid retrieval for each."""
    if _retriever is None or _llm is None:
        return jsonify({"error": "Retriever not loaded"}), 503

    body = request.json or {}
    topic = body.get("topic", "").strip()
    if not topic:
        return jsonify({"error": "Empty topic"}), 400

    model = body.get("model", "gpt-4o-mini")

    # Get a sample of entity names to ground the LLM
    sample_names = [e.name for e in list(_world_model.entities.values())[:80]]

    try:
        gen_result = _llm.chat(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are helping search a personal knowledge graph. "
                        "Given a topic, generate 4-6 specific search queries that will find relevant entities. "
                        "Queries should be short keyword phrases, not full sentences. "
                        "Return JSON: {\"queries\": [\"...\", \"...\"]}"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Topic: {topic}\n\n"
                        f"Some entities in this knowledge graph: {', '.join(sample_names[:50])}\n\n"
                        "Generate the best search queries to find everything relevant to this topic."
                    ),
                },
            ],
            model=model,
            json_mode=True,
        )
        queries = json.loads(gen_result["content"]).get("queries", [topic])
    except Exception:
        queries = [topic]

    # Run hybrid retrieval for each generated query, merge by entity_id
    seen: dict[str, dict] = {}
    for q in queries[:6]:
        for eid in _retriever.retrieve(q, top_k=8):
            if eid not in seen:
                entity = _world_model.entities.get(eid)
                if entity:
                    desc = ""
                    if isinstance(entity.current_state, dict):
                        desc = entity.current_state.get("description", "")
                    seen[eid] = {
                        "id": eid,
                        "name": entity.name,
                        "type": entity.type.value,
                        "desc": desc[:200],
                        "last_seen": entity.last_seen,
                    }

    return jsonify({
        "topic": topic,
        "generated_queries": queries,
        "entities": list(seen.values()),
    })


# ── API: stats ──────────────────────────────────────────────────────────────

@app.route("/api/stats")
def stats():
    if _world_model is None:
        return jsonify({"error": "Not loaded"}), 503
    return jsonify(_world_model.stats)


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PIE Wikipedia UI server")
    parser.add_argument(
        "--world-model", type=Path, default=Path("output/world_model.json"),
    )
    parser.add_argument("--port", type=int, default=7331)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    args = parser.parse_args()

    if not args.world_model.exists():
        print(f"World model not found at {args.world_model}. Run ingestion first.")
        sys.exit(1)

    load(args.world_model)
    print(f"\n  Open http://{args.host}:{args.port} in your browser\n")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
