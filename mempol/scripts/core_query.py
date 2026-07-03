"""Query the universal memory core with provenance and budget metrics."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from mempol import config, llm
from mempol.core.retrieval import format_context, retrieve_budgeted
from mempol.core.schema import TraceEvent
from mempol.core.store import estimate_tokens, now_iso, store_for_run, trace_id


def _fallback_answer(query: str, context: str) -> str:
    lines = [ln.strip() for ln in context.splitlines() if ln.strip()]
    evidence = []
    seen = set()
    for ln in lines:
        if ln.startswith("[") or ln.startswith("source:") or ln.startswith("evidence:"):
            continue
        if len(ln) > 40:
            key = ln[:180]
            if key in seen:
                continue
            seen.add(key)
            evidence.append(ln)
        if len(evidence) >= 8:
            break
    if not evidence:
        return f"I found no strong matches for: {query}"
    return (
        "No OPENAI_API_KEY is available, so this is an extractive answer from retrieved memory.\n\n"
        + "\n".join(f"- {e[:500]}" for e in evidence)
    )


def _llm_answer(query: str, context: str) -> str:
    if not os.getenv("OPENAI_API_KEY"):
        return _fallback_answer(query, context)
    prompt = f"""You answer using only the retrieved universal-memory context.

Be concrete. If the question asks for a plan, recommend the next actions and cite which retrieved memories/evidence drove the answer.

Question:
{query}

Retrieved context:
{context}
"""
    return llm.chat(
        [
            {"role": "system", "content": "You are a precise memory-backed planning assistant. Ground every important claim in retrieved context."},
            {"role": "user", "content": prompt},
        ],
        model=config.ANSWER_MODEL,
    )


def query(run_name: str, query_text: str, k: int = 8, token_budget: int = 3000, no_llm: bool = False) -> dict:
    store = store_for_run(run_name)
    hits, metrics = retrieve_budgeted(store, query_text, k=k, token_budget=token_budget)
    context = format_context(store, hits)
    answer = _fallback_answer(query_text, context) if no_llm else _llm_answer(query_text, context)
    result = {
        "run_name": run_name,
        "query": query_text,
        "answer": answer,
        "metrics": {
            **metrics,
            "answer_tokens_est": estimate_tokens(answer),
            "context_tokens_est": estimate_tokens(context),
        },
        "retrieved": hits,
        "context": context,
    }
    store.log_trace(
        TraceEvent(
            id=trace_id("query", run_name),
            run_name=run_name,
            op="query",
            input={"query": query_text, "k": k, "token_budget": token_budget, "no_llm": no_llm},
            output={"answer": answer, "retrieved_ids": [h["id"] for h in hits]},
            metrics=result["metrics"],
            created_at=now_iso(),
        )
    )
    store.commit()
    out_dir = config.RESULTS_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "latest_core_query.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    with (out_dir / "core_queries.jsonl").open("a") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")
    store.close()
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="universal_smoke")
    ap.add_argument("--query", required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--token-budget", type=int, default=3000)
    ap.add_argument("--no-llm", action="store_true", help="Force extractive local answer")
    args = ap.parse_args()
    result = query(args.run_name, args.query, args.k, args.token_budget, args.no_llm)

    print("\nANSWER\n======")
    print(result["answer"])
    print("\nMETRICS\n=======")
    print(json.dumps(result["metrics"], indent=2))
    print("\nRETRIEVED\n=========")
    for i, hit in enumerate(result["retrieved"], 1):
        print(f"{i}. {hit['kind']} {hit['id']} score={hit['score']:.3f} tokens~{hit['token_estimate']}")
        print(f"   source={hit.get('source', '')}")
        print(hit["text"][:500].replace("\n", " "))
        print()
    print(f"Wrote {config.RESULTS_DIR / args.run_name / 'latest_core_query.json'}")


if __name__ == "__main__":
    main()
