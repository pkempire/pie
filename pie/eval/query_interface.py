"""
Query Interface — end-to-end test of PIE's temporal query answering.

The basic "can PIE answer temporal queries?" test:
  1. Take a natural language question
  2. Find relevant entities via embedding search
  3. Compile semantic temporal context for those entities
  4. Ask an LLM to answer using the compiled context
  5. Return the answer with sources

Tests Hypothesis 5 (world model retrieval > standard RAG for temporal queries)
by providing a functional query path that can be compared against RAG baselines.

Usage:
    # Interactive mode
    python3 -m pie.eval.query_interface --world-model output/world_model.json

    # Single query
    python3 -m pie.eval.query_interface --query "How has the SRA project evolved?"

    # Batch evaluation from file
    python3 -m pie.eval.query_interface --batch queries.jsonl --output results.jsonl
"""

from __future__ import annotations
import argparse
import datetime
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("pie.eval.query_interface")


# ── Config ────────────────────────────────────────────────────────────────────

TOP_K_ENTITIES = 10         # how many entities to retrieve per query
MAX_CONTEXT_CHARS = 8000    # cap on compiled context length


# ── Temporal context compiler (mirrors ARCHITECTURE-FINAL.md) ─────────────────

def _humanize_delta(seconds: float) -> str:
    """Convert seconds to human-readable duration."""
    days = seconds / 86400
    if days < 1:
        return "today"
    elif days < 2:
        return "yesterday"
    elif days < 7:
        return f"{int(days)} days ago"
    elif days < 30:
        weeks = int(days / 7)
        return f"{weeks} week{'s' if weeks != 1 else ''} ago"
    elif days < 365:
        months = int(days / 30)
        return f"about {months} month{'s' if months != 1 else ''} ago"
    else:
        years = days / 365
        if years < 2:
            months = int(days / 30)
            return f"about {months} months ago"
        return f"about {years:.1f} years ago"


def _guess_period(timestamp: float) -> str:
    """
    Guess a life period from timestamp.
    In production, Period entities in the graph replace this.
    """
    dt = datetime.datetime.fromtimestamp(timestamp, tz=datetime.timezone.utc)
    year, month = dt.year, dt.month
    if month < 4:
        season = "early"
    elif month < 7:
        season = "mid"
    elif month < 10:
        season = "late summer/fall"
    else:
        season = "late"
    return f"{season} {year}"


def compile_entity_context(
    entity: dict,
    transitions: list[dict],
    relationships: list[dict],
    all_entities: dict[str, dict],
    now: float,
) -> str:
    """
    Compile semantic temporal context for one entity.
    This is the core of PIE's context compilation — the LLM never sees raw timestamps.
    """
    name = entity.get("name", "unknown")
    etype = entity.get("type", "unknown")
    first_seen = entity.get("first_seen", now)
    last_seen = entity.get("last_seen", now)

    first_ago = _humanize_delta(now - first_seen)
    last_ago = _humanize_delta(now - last_seen)
    first_period = _guess_period(first_seen)

    lines = []

    # Header with temporal metadata
    change_count = len(transitions)
    months_span = max((last_seen - first_seen) / (30 * 86400), 1)
    velocity = change_count / months_span

    lines.append(f"## {name} ({etype})")
    lines.append(f"First appeared {first_ago} ({first_period}), last referenced {last_ago}.")
    lines.append(f"Changed state {change_count} times (~{velocity:.1f}x/month).")

    # Aliases
    aliases = entity.get("aliases", [])
    if aliases:
        lines.append(f"Also known as: {', '.join(aliases)}")

    # Timeline of changes
    if transitions:
        lines.append("")
        lines.append("Timeline:")
        for t in transitions:
            ts = t.get("timestamp", now)
            t_ago = _humanize_delta(now - ts)
            t_period = _guess_period(ts)
            ttype = t.get("transition_type", "update")
            summary = t.get("trigger_summary", "")

            prefix = "  •"
            if ttype == "contradiction":
                prefix = "  ⚠"
            elif ttype == "creation":
                prefix = "  ★"

            lines.append(f"{prefix} {t_ago} ({t_period}): {summary}")

    # Current state
    state = entity.get("current_state", {})
    if state:
        desc = state.get("description", "")
        if not desc and isinstance(state, dict):
            desc = "; ".join(f"{k}: {v}" for k, v in state.items() if k != "description")
        if desc:
            lines.append(f"\nCurrent state: {desc}")

    # Relationships
    if relationships:
        rel_lines = []
        for r in relationships:
            source_id = r.get("source_id", "")
            target_id = r.get("target_id", "")
            rtype = r.get("type", "related_to")
            desc = r.get("description", "")

            # Resolve the other entity's name
            other_id = target_id if source_id == entity.get("id", "") else source_id
            other = all_entities.get(other_id, {})
            other_name = other.get("name", "unknown")

            direction = "→" if source_id == entity.get("id", "") else "←"
            rel_str = f"  {direction} {rtype}: {other_name}"
            if desc:
                rel_str += f" ({desc})"
            rel_lines.append(rel_str)

        if rel_lines:
            lines.append(f"\nRelationships:")
            lines.extend(rel_lines[:8])  # cap at 8 relationships per entity

    return "\n".join(lines)


# ── Entity retrieval ──────────────────────────────────────────────────────────

def retrieve_entities_by_embedding(
    query: str,
    data: dict,
    llm_client,
    top_k: int = TOP_K_ENTITIES,
) -> list[tuple[str, dict, float]]:
    """
    Find relevant entities for a query via embedding similarity.
    Returns list of (entity_id, entity_dict, similarity_score).
    """
    from pie.core.world_model import cosine_similarity

    # Embed the query
    query_embedding = llm_client.embed_single(query)

    # Score all entities that have embeddings
    # For entities without embeddings, compute on the fly from name + state
    entities = data.get("entities", {})
    scores = []

    # Batch compute missing embeddings
    needs_embedding = []
    needs_embedding_texts = []
    for eid, entity in entities.items():
        if not entity.get("embedding"):
            state = entity.get("current_state", {})
            desc = state.get("description", str(state)[:200]) if isinstance(state, dict) else str(state)[:200]
            text = f"{entity.get('name', '')} ({entity.get('type', '')}): {desc}"
            aliases = entity.get("aliases", [])
            if aliases:
                text += f" (aka: {', '.join(aliases)})"
            needs_embedding.append(eid)
            needs_embedding_texts.append(text)

    if needs_embedding_texts:
        try:
            embeddings = llm_client.embed(needs_embedding_texts)
            for eid, emb in zip(needs_embedding, embeddings):
                entities[eid]["_computed_embedding"] = emb
        except Exception as e:
            logger.warning(f"Batch embedding failed: {e}")

    # Score all entities
    for eid, entity in entities.items():
        emb = entity.get("embedding") or entity.get("_computed_embedding")
        if emb:
            sim = cosine_similarity(query_embedding, emb)
            scores.append((eid, entity, sim))

    scores.sort(key=lambda x: x[2], reverse=True)
    return scores[:top_k]


def retrieve_entities_by_name(
    query: str,
    data: dict,
    top_k: int = TOP_K_ENTITIES,
) -> list[tuple[str, dict, float]]:
    """
    Fallback retrieval: find entities whose names appear in the query.
    Used when embeddings aren't available.
    """
    from pie.core.world_model import _normalize, _fuzzy_ratio

    entities = data.get("entities", {})
    query_lower = query.lower()
    matches = []

    for eid, entity in entities.items():
        name = entity.get("name", "")
        score = 0.0

        # Check if entity name appears in query
        if _normalize(name) in query_lower:
            score = max(score, 0.9)

        # Check aliases
        for alias in entity.get("aliases", []):
            if _normalize(alias) in query_lower:
                score = max(score, 0.85)

        # Fuzzy match against query words
        query_words = query_lower.split()
        for word in query_words:
            if len(word) > 3:
                ratio = _fuzzy_ratio(word, name)
                score = max(score, ratio)

        if score > 0.3:
            matches.append((eid, entity, score))

    matches.sort(key=lambda x: x[2], reverse=True)
    return matches[:top_k]


# ── Full query pipeline ──────────────────────────────────────────────────────

@dataclass
class QueryResult:
    """Result of a single query through PIE."""
    query: str
    answer: str
    entities_used: list[str]       # entity names included in context
    context_compiled: str           # the actual context sent to the LLM
    retrieval_method: str           # "embedding" or "name_match"
    model: str
    latency_ms: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "answer": self.answer,
            "entities_used": self.entities_used,
            "retrieval_method": self.retrieval_method,
            "model": self.model,
            "latency_ms": round(self.latency_ms, 1),
            "error": self.error,
        }


def answer_query(
    query: str,
    data: dict,
    model: str = "gpt-5-mini",
    use_embeddings: bool = True,
    top_k: int = TOP_K_ENTITIES,
) -> QueryResult:
    """
    Full query pipeline:
      1. Retrieve relevant entities
      2. Compile semantic temporal context
      3. Ask LLM to answer
    """
    from pie.core.llm import LLMClient

    t0 = time.time()
    llm = LLMClient()
    now = time.time()

    entities_data = data.get("entities", {})
    transitions_data = data.get("transitions", {})
    relationships_data = data.get("relationships", {})

    # Step 1: Retrieve relevant entities
    retrieval_method = "embedding"
    try:
        if use_embeddings:
            retrieved = retrieve_entities_by_embedding(query, data, llm, top_k=top_k)
        else:
            raise RuntimeError("Embeddings disabled")
    except Exception as e:
        logger.warning(f"Embedding retrieval failed ({e}), falling back to name match")
        retrieved = retrieve_entities_by_name(query, data, top_k=top_k)
        retrieval_method = "name_match"

    if not retrieved:
        return QueryResult(
            query=query,
            answer="I don't have enough information in the world model to answer this question.",
            entities_used=[],
            context_compiled="",
            retrieval_method=retrieval_method,
            model=model,
            latency_ms=(time.time() - t0) * 1000,
        )

    # Step 2: Compile context for retrieved entities
    # Group transitions and relationships by entity
    trans_by_entity = {}
    for tid, t in transitions_data.items():
        eid = t.get("entity_id", "")
        if eid not in trans_by_entity:
            trans_by_entity[eid] = []
        trans_by_entity[eid].append(t)
    # Sort each entity's transitions
    for eid in trans_by_entity:
        trans_by_entity[eid].sort(key=lambda t: t.get("timestamp", 0))

    rels_by_entity = {}
    for rid, r in relationships_data.items():
        for eid in (r.get("source_id", ""), r.get("target_id", "")):
            if eid not in rels_by_entity:
                rels_by_entity[eid] = []
            rels_by_entity[eid].append(r)

    context_parts = []
    entity_names = []
    total_chars = 0

    for eid, entity, score in retrieved:
        transitions = trans_by_entity.get(eid, [])
        relationships = rels_by_entity.get(eid, [])

        part = compile_entity_context(
            entity=entity,
            transitions=transitions,
            relationships=relationships,
            all_entities=entities_data,
            now=now,
        )

        # Respect context cap
        if total_chars + len(part) > MAX_CONTEXT_CHARS:
            break

        context_parts.append(part)
        entity_names.append(entity.get("name", eid))
        total_chars += len(part)

    compiled_context = "\n\n".join(context_parts)

    # Step 3: Ask LLM
    messages = [
        {
            "role": "system",
            "content": (
                "You are a personal knowledge assistant. You answer questions about "
                "the user's world — their projects, people, decisions, beliefs, and "
                "how things have changed over time.\n\n"
                "Use ONLY the provided context to answer. The context includes "
                "temporal information (when things happened, how they evolved, change "
                "velocity). Use this temporal information in your answer.\n\n"
                "If the context doesn't contain enough information to answer, say so. "
                "Don't make up information."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Context from world model:\n\n{compiled_context}\n\n"
                f"---\n\nQuestion: {query}"
            ),
        },
    ]

    try:
        result = llm.chat(messages=messages, model=model, max_tokens=500)
        answer = result["content"]
    except Exception as e:
        answer = f"Error generating answer: {e}"

    latency = (time.time() - t0) * 1000

    return QueryResult(
        query=query,
        answer=answer,
        entities_used=entity_names,
        context_compiled=compiled_context,
        retrieval_method=retrieval_method,
        model=model,
        latency_ms=latency,
    )


# ── Batch evaluation ──────────────────────────────────────────────────────────

def run_batch(
    queries_path: Path,
    data: dict,
    model: str = "gpt-5-mini",
    use_embeddings: bool = True,
) -> list[QueryResult]:
    """
    Run a batch of queries from a JSONL file.
    Each line: {"query": "...", "expected_answer": "..."} (expected_answer optional)
    """
    results = []

    with open(queries_path) as f:
        lines = [json.loads(line) for line in f if line.strip()]

    logger.info(f"Running {len(lines)} queries from {queries_path}")

    for i, item in enumerate(lines):
        query = item.get("query", "")
        if not query:
            continue

        logger.info(f"  [{i + 1}/{len(lines)}] {query[:60]}...")
        result = answer_query(query, data, model=model, use_embeddings=use_embeddings)
        results.append(result)

        # If expected answer provided, note it
        if "expected_answer" in item:
            result_dict = result.to_dict()
            result_dict["expected_answer"] = item["expected_answer"]

    return results


# ── Interactive mode ──────────────────────────────────────────────────────────

def interactive(data: dict, model: str = "gpt-5-mini", use_embeddings: bool = True):
    """Interactive query loop."""
    print("\n" + "=" * 60)
    print("  PIE Query Interface — Interactive Mode")
    print("=" * 60)
    print(f"  Model: {model}")
    print(f"  Entities: {len(data.get('entities', {}))}")
    print(f"  Transitions: {len(data.get('transitions', {}))}")
    print(f"  Retrieval: {'embedding' if use_embeddings else 'name_match'}")
    print(f"\n  Type a question, or 'quit' to exit.")
    print(f"  Type 'debug' before a query to see full context.")
    print("=" * 60)

    debug = False

    while True:
        try:
            query = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if query.lower() == "debug":
            debug = not debug
            print(f"  Debug mode: {'ON' if debug else 'OFF'}")
            continue

        result = answer_query(query, data, model=model, use_embeddings=use_embeddings)

        if debug:
            print(f"\n{'─'*40}")
            print(f"  Retrieval: {result.retrieval_method}")
            print(f"  Entities: {', '.join(result.entities_used)}")
            print(f"  Latency: {result.latency_ms:.0f}ms")
            print(f"{'─'*40}")
            print(f"  Context ({len(result.context_compiled)} chars):")
            print(result.context_compiled[:2000])
            if len(result.context_compiled) > 2000:
                print(f"  ... ({len(result.context_compiled) - 2000} more chars)")
            print(f"{'─'*40}")

        print(f"\n{result.answer}")
        print(f"\n  [{result.latency_ms:.0f}ms | {len(result.entities_used)} entities | {result.retrieval_method}]")


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PIE Query Interface — temporal query answering over the world model"
    )
    parser.add_argument(
        "--world-model",
        type=Path,
        default=Path("output/world_model.json"),
        help="Path to world_model.json",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5-mini",
        help="LLM model for answering (default: gpt-5-mini)",
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        default=None,
        help="Single query to answer (non-interactive)",
    )
    parser.add_argument(
        "--batch",
        type=Path,
        default=None,
        help="Batch queries from JSONL file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write results to file (JSONL for batch, JSON for single)",
    )
    parser.add_argument(
        "--no-embeddings",
        action="store_true",
        help="Disable embedding retrieval, use name matching only",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K_ENTITIES,
        help=f"Number of entities to retrieve (default: {TOP_K_ENTITIES})",
    )
    args = parser.parse_args()

    # Load world model
    if not args.world_model.exists():
        logger.error(f"World model not found at {args.world_model}")
        logger.error("Run the ingestion pipeline first.")
        sys.exit(1)

    with open(args.world_model) as f:
        data = json.load(f)

    entity_count = len(data.get("entities", {}))
    logger.info(f"Loaded world model: {entity_count} entities")

    if entity_count == 0:
        logger.error("World model is empty. Run ingestion first.")
        sys.exit(1)

    use_embeddings = not args.no_embeddings

    if args.batch:
        # Batch mode
        results = run_batch(args.batch, data, model=args.model, use_embeddings=use_embeddings)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, "w") as f:
                for r in results:
                    f.write(json.dumps(r.to_dict()) + "\n")
            logger.info(f"Results written to {args.output}")
        else:
            for r in results:
                print(f"\nQ: {r.query}")
                print(f"A: {r.answer}")
                print(f"  [{r.latency_ms:.0f}ms | {len(r.entities_used)} entities]")

    elif args.query:
        # Single query mode
        result = answer_query(
            args.query, data, model=args.model, use_embeddings=use_embeddings, top_k=args.top_k
        )
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, "w") as f:
                json.dump(result.to_dict(), f, indent=2)
            logger.info(f"Result written to {args.output}")

        print(f"\nQ: {result.query}")
        print(f"A: {result.answer}")
        print(f"\nEntities: {', '.join(result.entities_used)}")
        print(f"Retrieval: {result.retrieval_method}")
        print(f"Latency: {result.latency_ms:.0f}ms")

    else:
        # Interactive mode
        interactive(data, model=args.model, use_embeddings=use_embeddings)


if __name__ == "__main__":
    main()
