"""
Side-by-side comparison of embedding-only vs graph-aware retrieval.

Outputs detailed logs showing exactly what each approach retrieves
and how the contexts differ.
"""

from __future__ import annotations
import json
import time
import logging
from pathlib import Path
from dataclasses import dataclass

from pie.core.llm import LLMClient
from pie.core.world_model import WorldModel, cosine_similarity
from pie.retrieval.graph_retriever import (
    retrieve_subgraph, 
    parse_query_intent,
    QueryIntent,
)
from pie.eval.query_interface import (
    retrieve_entities_by_embedding,
    compile_entity_context,
    answer_query,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("retrieval_compare")


@dataclass
class RetrievalResult:
    method: str
    entities: list[dict]  # [{name, type, score, source}]
    context: str
    answer: str
    latency_ms: float
    intent: QueryIntent | None = None


def load_world_model_data(path: Path) -> tuple[dict, WorldModel]:
    """Load both raw data dict and WorldModel object."""
    with open(path) as f:
        data = json.load(f)
    wm = WorldModel(path)
    return data, wm


def embedding_only_retrieve(
    query: str,
    data: dict,
    llm: LLMClient,
    top_k: int = 10,
) -> RetrievalResult:
    """The old approach: pure embedding similarity."""
    t0 = time.time()
    
    retrieved = retrieve_entities_by_embedding(query, data, llm, top_k=top_k)
    
    entities = []
    for eid, entity, score in retrieved:
        entities.append({
            "id": eid,
            "name": entity.get("name", "?"),
            "type": entity.get("type", "?"),
            "score": round(score, 3),
            "source": "embedding",
        })
    
    # Compile context (reusing existing function structure)
    now = time.time()
    context_parts = []
    
    transitions_data = data.get("transitions", {})
    relationships_data = data.get("relationships", {})
    entities_data = data.get("entities", {})
    
    trans_by_entity = {}
    for tid, t in transitions_data.items():
        eid = t.get("entity_id", "")
        if eid not in trans_by_entity:
            trans_by_entity[eid] = []
        trans_by_entity[eid].append(t)
    for eid in trans_by_entity:
        trans_by_entity[eid].sort(key=lambda t: t.get("timestamp", 0))
    
    rels_by_entity = {}
    for rid, r in relationships_data.items():
        for eid in (r.get("source_id", ""), r.get("target_id", "")):
            if eid not in rels_by_entity:
                rels_by_entity[eid] = []
            rels_by_entity[eid].append(r)
    
    for eid, entity, score in retrieved:
        part = compile_entity_context(
            entity=entity,
            transitions=trans_by_entity.get(eid, []),
            relationships=rels_by_entity.get(eid, []),
            all_entities=entities_data,
            now=now,
        )
        context_parts.append(part)
    
    context = "\n\n".join(context_parts)
    latency = (time.time() - t0) * 1000
    
    return RetrievalResult(
        method="embedding_only",
        entities=entities,
        context=context,
        answer="",  # filled in later
        latency_ms=latency,
    )


def graph_aware_retrieve(
    query: str,
    data: dict,
    wm: WorldModel,
    llm: LLMClient,
    max_entities: int = 10,
    max_hops: int = 2,
) -> RetrievalResult:
    """The new approach: LLM-guided graph traversal."""
    t0 = time.time()
    
    result = retrieve_subgraph(query, wm, llm, max_entities=max_entities, max_hops=max_hops)
    
    entities = []
    for eid in result.entity_ids:
        entity = wm.get_entity(eid)
        if entity:
            # Find the path that brought this entity
            source = "seed"
            for path in result.paths:
                if eid in path.entity_ids:
                    idx = path.entity_ids.index(eid)
                    if idx > 0:
                        source = f"hop_{idx}"
                    break
            
            entities.append({
                "id": eid,
                "name": entity.name,
                "type": entity.type.value,
                "score": 0,  # graph retriever doesn't expose per-entity scores cleanly
                "source": source,
            })
    
    # Compile context using same function
    now = time.time()
    context_parts = []
    
    transitions_data = data.get("transitions", {})
    relationships_data = data.get("relationships", {})
    entities_data = data.get("entities", {})
    
    trans_by_entity = {}
    for tid, t in transitions_data.items():
        eid = t.get("entity_id", "")
        if eid not in trans_by_entity:
            trans_by_entity[eid] = []
        trans_by_entity[eid].append(t)
    for eid in trans_by_entity:
        trans_by_entity[eid].sort(key=lambda t: t.get("timestamp", 0))
    
    rels_by_entity = {}
    for rid, r in relationships_data.items():
        for eid in (r.get("source_id", ""), r.get("target_id", "")):
            if eid not in rels_by_entity:
                rels_by_entity[eid] = []
            rels_by_entity[eid].append(r)
    
    for eid in result.entity_ids:
        entity = entities_data.get(eid)
        if entity:
            part = compile_entity_context(
                entity=entity,
                transitions=trans_by_entity.get(eid, []),
                relationships=rels_by_entity.get(eid, []),
                all_entities=entities_data,
                now=now,
            )
            context_parts.append(part)
    
    context = "\n\n".join(context_parts)
    latency = (time.time() - t0) * 1000
    
    return RetrievalResult(
        method="graph_aware",
        entities=entities,
        context=context,
        answer="",
        latency_ms=latency,
        intent=result.intent,
    )


def compare_single_query(
    query: str,
    data: dict,
    wm: WorldModel,
    llm: LLMClient,
    top_k: int = 10,
    verbose: bool = True,
) -> dict:
    """Run both retrievers on a single query and compare."""
    
    if verbose:
        print("\n" + "=" * 80)
        print(f"QUERY: {query}")
        print("=" * 80)
    
    # Run both
    emb_result = embedding_only_retrieve(query, data, llm, top_k=top_k)
    graph_result = graph_aware_retrieve(query, data, wm, llm, max_entities=top_k)
    
    if verbose:
        # Log intent parsing
        if graph_result.intent:
            intent = graph_result.intent
            print(f"\n📋 PARSED INTENT:")
            print(f"   entity_types: {intent.entity_types}")
            print(f"   named_entities: {intent.named_entities}")
            print(f"   temporal_pattern: {intent.temporal_pattern}")
            print(f"   relationship_types: {intent.relationship_types}")
            print(f"   hop_pattern: {intent.hop_pattern}")
            print(f"   query_type: {intent.query_type}")
        
        # Compare retrieved entities
        print(f"\n{'─' * 40}")
        print("EMBEDDING-ONLY RETRIEVAL:")
        print(f"{'─' * 40}")
        print(f"   Latency: {emb_result.latency_ms:.0f}ms")
        print(f"   Entities ({len(emb_result.entities)}):")
        for e in emb_result.entities:
            print(f"      • {e['name']} ({e['type']}) — score: {e['score']}")
        
        print(f"\n{'─' * 40}")
        print("GRAPH-AWARE RETRIEVAL:")
        print(f"{'─' * 40}")
        print(f"   Latency: {graph_result.latency_ms:.0f}ms")
        print(f"   Entities ({len(graph_result.entities)}):")
        for e in graph_result.entities:
            print(f"      • {e['name']} ({e['type']}) — via: {e['source']}")
        
        # Show overlap
        emb_names = {e['name'] for e in emb_result.entities}
        graph_names = {e['name'] for e in graph_result.entities}
        overlap = emb_names & graph_names
        only_emb = emb_names - graph_names
        only_graph = graph_names - emb_names
        
        print(f"\n{'─' * 40}")
        print("COMPARISON:")
        print(f"{'─' * 40}")
        print(f"   Overlap: {len(overlap)} entities")
        if only_emb:
            print(f"   Only in embedding: {only_emb}")
        if only_graph:
            print(f"   Only in graph: {only_graph}")
        
        # Context lengths
        print(f"\n   Context lengths:")
        print(f"      Embedding: {len(emb_result.context):,} chars")
        print(f"      Graph: {len(graph_result.context):,} chars")
    
    # Get answers from both
    # (We'd need to call the LLM with each context)
    
    return {
        "query": query,
        "embedding": {
            "entities": [e['name'] for e in emb_result.entities],
            "latency_ms": emb_result.latency_ms,
            "context_chars": len(emb_result.context),
        },
        "graph": {
            "entities": [e['name'] for e in graph_result.entities],
            "latency_ms": graph_result.latency_ms,
            "context_chars": len(graph_result.context),
            "intent": {
                "entity_types": graph_result.intent.entity_types if graph_result.intent else [],
                "temporal_pattern": graph_result.intent.temporal_pattern if graph_result.intent else None,
            },
        },
        "overlap_count": len(emb_names & graph_names) if verbose else 0,
    }


TEST_QUERIES = [
    # Temporal evolution
    "How has the PIE project evolved over time?",
    "What was the first project I worked on?",
    "What changed about my beliefs on AI safety?",
    
    # Relationship-based
    "What tools did I use for the SRA project?",
    "Who did I collaborate with on research?",
    "What concepts are related to temporal reasoning?",
    
    # Factual
    "What is the current status of the health protocol?",
    "What decision did I make about TRT dosing?",
    
    # Aggregation
    "What were my main projects in 2025?",
    "What patterns do you see in my tool choices?",
]


def run_comparison(
    wm_path: Path = Path("output/world_model.json"),
    queries: list[str] | None = None,
):
    """Run full comparison on test queries."""
    
    data, wm = load_world_model_data(wm_path)
    llm = LLMClient()
    
    print(f"\nWorld Model: {len(data.get('entities', {}))} entities, "
          f"{len(data.get('transitions', {}))} transitions, "
          f"{len(data.get('relationships', {}))} relationships")
    
    queries = queries or TEST_QUERIES
    results = []
    
    for query in queries:
        result = compare_single_query(query, data, wm, llm)
        results.append(result)
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    total_overlap = sum(r['overlap_count'] for r in results)
    total_entities = sum(len(r['embedding']['entities']) for r in results)
    
    print(f"Queries: {len(results)}")
    print(f"Average overlap: {total_overlap / len(results):.1f} / 10 entities")
    print(f"Average graph latency overhead: "
          f"{sum(r['graph']['latency_ms'] - r['embedding']['latency_ms'] for r in results) / len(results):.0f}ms")
    
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Single query mode
        query = " ".join(sys.argv[1:])
        data, wm = load_world_model_data(Path("output/world_model.json"))
        llm = LLMClient()
        compare_single_query(query, data, wm, llm)
    else:
        run_comparison()
