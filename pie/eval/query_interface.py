"""
Query Interface — end-to-end temporal query answering over the PIE world model.

Pipeline per query:
  1. Hybrid retrieval: BM25 (sparse) + dense (embedding matrix matmul) → RRF merge
  2. Temporal boost: date hints parsed from query text (no extra API call)
  3. One-hop graph expansion for multi-hop queries
  4. Context compilation via context_compiler (rich temporal markdown)
  5. LLM answer generation

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
import json
import logging
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger("pie.eval.query_interface")

TOP_K_ENTITIES = 30        # always retrieve this many via smart LLM decomposition
MAX_CONTEXT_CHARS = 60_000  # generous; LLM sub-queries return rich context
DEFAULT_MODEL = "gpt-5.4"


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class QueryResult:
    query: str
    answer: str
    entities_used: list[str]
    context_compiled: str
    retrieval_method: str
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


# ── Core query function ───────────────────────────────────────────────────────

def answer_query(
    query: str,
    retriever,          # HybridRetriever
    llm,                # LLMClient
    model: str = DEFAULT_MODEL,
    top_k: int = TOP_K_ENTITIES,
    now: datetime | None = None,
    force_broad: bool = False,  # kept for back-compat, ignored
) -> QueryResult:
    """Run the full hybrid retrieval → context compile → LLM answer pipeline.

    Always uses broad_retrieve() which:
      - Decomposes the query into sub-queries via LLM
      - Retrieves across BM25 + dense, merged via RRF
      - Includes up to 25 transitions per entity (full history)
    """
    t0 = time.time()
    now = now or datetime.now()

    entity_ids = retriever.broad_retrieve(query, top_k=top_k, now=now)

    if not entity_ids:
        return QueryResult(
            query=query,
            answer="I don't have enough information in the world model to answer this.",
            entities_used=[],
            context_compiled="",
            retrieval_method="hybrid",
            model=model,
            latency_ms=(time.time() - t0) * 1000,
        )

    context_md = retriever.compile_context(
        entity_ids, query=query, now=now, max_transitions=25
    )
    if len(context_md) > MAX_CONTEXT_CHARS:
        context_md = context_md[:MAX_CONTEXT_CHARS] + "\n\n[...context truncated...]"

    entity_names = [
        retriever.world_model.entities[eid].name
        for eid in entity_ids
        if eid in retriever.world_model.entities
    ]

    messages = [
        {
            "role": "system",
            "content": (
                "You are a personal knowledge assistant with access to the user's "
                "structured world model — their projects, people, decisions, beliefs, "
                "and how those things have changed over time.\n\n"
                "Answer using ONLY the provided context. Make full use of the "
                "temporal information (dates, change history, contradictions). "
                "If the context is insufficient, say so clearly."
            ),
        },
        {
            "role": "user",
            "content": f"Context:\n\n{context_md}\n\n---\n\nQuestion: {query}",
        },
    ]

    try:
        result = llm.chat(messages=messages, model=model, max_tokens=2000)
        answer = (result.get("content") or "").strip()
        if not answer:
            answer = f"[Model returned empty response]\n\n{context_md}"
    except Exception as e:
        answer = f"Error generating answer: {e}"

    return QueryResult(
        query=query,
        answer=answer,
        entities_used=entity_names,
        context_compiled=context_md,
        retrieval_method="hybrid",
        model=model,
        latency_ms=(time.time() - t0) * 1000,
    )


# ── Batch mode ────────────────────────────────────────────────────────────────

def run_batch(
    queries_path: Path,
    retriever,
    llm,
    model: str = DEFAULT_MODEL,
    top_k: int = TOP_K_ENTITIES,
) -> list[QueryResult]:
    with open(queries_path) as f:
        items = [json.loads(line) for line in f if line.strip()]

    logger.info(f"Running {len(items)} queries from {queries_path}")
    results = []
    for i, item in enumerate(items):
        query = item.get("query", "")
        if not query:
            continue
        logger.info(f"  [{i+1}/{len(items)}] {query[:60]}...")
        results.append(answer_query(query, retriever, llm, model=model, top_k=top_k))
    return results


# ── Interactive mode ──────────────────────────────────────────────────────────

def interactive(retriever, llm, model: str = DEFAULT_MODEL, top_k: int = TOP_K_ENTITIES):
    wm = retriever.world_model
    print("\n" + "=" * 60)
    print("  PIE Query Interface — Interactive Mode")
    print("=" * 60)
    print(f"  Model:       {model}")
    print(f"  Entities:    {len(wm.entities)}")
    print(f"  Transitions: {len(wm.transitions)}")
    print(f"  Retrieval:   smart (LLM sub-queries + BM25 + dense + RRF)")
    print(f"\n  Type a question, or 'quit' to exit.")
    print(f"  Type 'debug' to toggle retrieved context preview.")
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

        result = answer_query(query, retriever, llm, model=model, top_k=top_k)

        if debug:
            print(f"\n{'─'*40}")
            print(f"  Entities retrieved: {len(result.entities_used)} — {', '.join(result.entities_used[:10])}")
            print(f"  Latency: {result.latency_ms:.0f}ms")
            print(f"{'─'*40}")
            preview = result.context_compiled[:3000]
            print(preview)
            if len(result.context_compiled) > 3000:
                print(f"  ... ({len(result.context_compiled) - 3000} more chars)")
            print(f"{'─'*40}")

        print(f"\n{result.answer}")
        print(f"\n  [{result.latency_ms:.0f}ms | {len(result.entities_used)} entities]")


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PIE Query Interface — hybrid temporal query answering"
    )
    parser.add_argument(
        "--world-model", type=Path, default=Path("output/world_model.json"),
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help=f"LLM model for answering (default: {DEFAULT_MODEL})",
    )
    parser.add_argument("--query", "-q", type=str, default=None)
    parser.add_argument("--batch", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=TOP_K_ENTITIES)
    args = parser.parse_args()

    if not args.world_model.exists():
        print(f"World model not found at {args.world_model}. Run ingestion first.")
        sys.exit(1)

    # --- Load world model (auto-loads embeddings from .npy if present) ---
    from pie.core.world_model import WorldModel
    from pie.core.llm import LLMClient
    from pie.retrieval.hybrid_retriever import HybridRetriever

    wm = WorldModel(persist_path=args.world_model)
    print(f"Loaded world model: {len(wm.entities)} entities")

    if len(wm.entities) == 0:
        print("World model is empty. Run ingestion first.")
        sys.exit(1)

    llm = LLMClient()

    # --- Build hybrid retriever (BM25 index + matrix already loaded from .npy) ---
    print("Building retrieval index...", end=" ", flush=True)
    retriever = HybridRetriever(wm, llm)
    print("done.")

    if args.batch:
        results = run_batch(args.batch, retriever, llm, model=args.model, top_k=args.top_k)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, "w") as f:
                for r in results:
                    f.write(json.dumps(r.to_dict()) + "\n")
            print(f"Results written to {args.output}")
        else:
            for r in results:
                print(f"\nQ: {r.query}\nA: {r.answer}")
                print(f"  [{r.latency_ms:.0f}ms | {len(r.entities_used)} entities]")

    elif args.query:
        result = answer_query(args.query, retriever, llm, model=args.model, top_k=args.top_k)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, "w") as f:
                json.dump(result.to_dict(), f, indent=2)
        print(f"\nQ: {result.query}")
        print(f"A: {result.answer}")
        print(f"\nEntities: {', '.join(result.entities_used)}")
        print(f"Latency: {result.latency_ms:.0f}ms")

    else:
        interactive(retriever, llm, model=args.model, top_k=args.top_k)


if __name__ == "__main__":
    main()
