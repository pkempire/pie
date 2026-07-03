"""Generic retrieval helpers for universal memory."""
from __future__ import annotations

from .store import SQLiteMemoryStore, estimate_tokens


def retrieve_budgeted(
    store: SQLiteMemoryStore,
    query: str,
    k: int = 8,
    token_budget: int = 3000,
    include_spans: bool = True,
    diversify_sources: bool = True,
) -> tuple[list[dict], dict]:
    """Retrieve top results while respecting an approximate context budget."""
    raw = store.retrieve(query, k=max(k * 8, k), include_spans=include_spans)
    ranked = _diversify(raw, k=max(k * 3, k)) if diversify_sources else raw
    kept: list[dict] = []
    used = 0
    for hit in ranked:
        n = int(hit.get("token_estimate") or estimate_tokens(hit.get("text", "")))
        if kept and used + n > token_budget:
            continue
        kept.append(hit)
        used += n
        if len(kept) >= k:
            break
    return kept, {
        "retrieved": len(kept),
        "candidate_count": len(raw),
        "retrieval_tokens_est": used,
        "token_budget": token_budget,
        "diversify_sources": diversify_sources,
        "sources": sorted({str(h.get("source", "")) for h in kept if h.get("source")}),
    }


def format_context(store: SQLiteMemoryStore, hits: list[dict], provenance_limit: int = 3) -> str:
    """Format retrieved memory and source evidence for an LLM or terminal."""
    chunks = []
    for i, hit in enumerate(hits, 1):
        title = f"[{i}] {hit['kind']} {hit['id']} score={hit['score']:.3f}"
        chunks.append(title)
        chunks.append(hit.get("text", "")[:2200])
        if hit["kind"] == "memory_state":
            state = store.get_memory_state(hit["id"])
            if state:
                prov = store.provenance_for_state(state, limit=provenance_limit)
                for p in prov:
                    chunks.append(
                        f"  source: {p['artifact_source']} / {p['artifact_title']} / {p['locator']}\n"
                        f"  evidence: {p['text'][:600]}"
                    )
        elif hit["kind"] == "span":
            artifact = store.get_artifact(hit.get("artifact_id", ""))
            if artifact:
                chunks.append(f"  source: {artifact.source} / {artifact.title} / {hit.get('locator', '')}")
    return "\n\n".join(chunks)


def _diversify(hits: list[dict], k: int) -> list[dict]:
    """Interleave high-scoring results by source so one corpus does not drown out
    every other corpus in universal-memory queries.

    This is a retrieval policy, not a schema. It is intentionally simple and
    logged in metrics so a learned retriever can replace it later.
    """
    buckets: dict[str, list[dict]] = {}
    for h in hits:
        buckets.setdefault(str(h.get("source") or h.get("kind") or "unknown"), []).append(h)
    for rows in buckets.values():
        rows.sort(key=lambda x: x["score"], reverse=True)
    ordered_sources = sorted(
        buckets,
        key=lambda src: buckets[src][0]["score"] if buckets[src] else 0.0,
        reverse=True,
    )
    out: list[dict] = []
    seen = set()
    while len(out) < min(k, len(hits)):
        progressed = False
        for src in ordered_sources:
            while buckets[src]:
                h = buckets[src].pop(0)
                if h["id"] in seen:
                    continue
                out.append(h)
                seen.add(h["id"])
                progressed = True
                break
            if len(out) >= k:
                break
        if not progressed:
            break
    return out
