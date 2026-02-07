"""
Graph-Aware Retrieval — LLM-guided subgraph selection.

The key insight: we have a graph, so use it as a graph.
Embeddings find seeds. LLM parses query intent. Graph traversal expands.

Pipeline:
  1. Query Understanding — LLM extracts structured intent
  2. Seed Selection — embeddings + constraints find starting nodes
  3. Graph Traversal — expand along query-relevant edges
  4. Subgraph Scoring — rank paths by relevance
  5. Context Assembly — compile from scored subgraph
"""

from __future__ import annotations
import json
import logging
from dataclasses import dataclass, field
from typing import Literal

from pie.core.llm import LLMClient
from pie.core.world_model import WorldModel, cosine_similarity

logger = logging.getLogger("pie.retrieval")


# ── Query Understanding ───────────────────────────────────────────────────────

@dataclass
class QueryIntent:
    """Structured representation of what the query is asking for."""
    
    # Entity constraints
    entity_types: list[str] = field(default_factory=list)  # ["project", "person"]
    named_entities: list[str] = field(default_factory=list)  # explicit names in query
    
    # Temporal constraints
    time_anchor: str | None = None  # "early 2025", "last month", etc.
    time_range: tuple[float, float] | None = None  # (start_ts, end_ts) if parseable
    temporal_pattern: str | None = None  # "evolution", "first", "last", "during", "before"
    
    # Relationship patterns
    relationship_types: list[str] = field(default_factory=list)  # edge types to follow
    hop_pattern: str | None = None  # "direct", "transitive", "all_connected"
    
    # Query type (affects retrieval strategy)
    query_type: str = "factual"  # factual, temporal, comparative, aggregation
    
    # Raw for fallback
    raw_query: str = ""


QUERY_UNDERSTANDING_PROMPT = """Analyze this query about a personal knowledge graph and extract structured intent.

The graph contains entities of types: project, person, tool, concept, decision, belief, organization, life_period, event

Query: {query}

Return JSON with these fields:
- entity_types: list of entity types likely relevant (empty = any)
- named_entities: explicit names/titles mentioned in query
- time_anchor: temporal reference if any ("early 2025", "last month", "before X happened")
- temporal_pattern: null, "evolution", "first", "last", "during", "before", "after", "sequence"
- relationship_types: edge types to traverse ("uses", "part_of", "related_to", "influences", "collaborates_with")
- hop_pattern: "direct" (1-hop), "transitive" (follow chains), "neighborhood" (all connected)
- query_type: "factual", "temporal", "comparative", "aggregation"

JSON only, no explanation:"""


def parse_query_intent(query: str, llm: LLMClient) -> QueryIntent:
    """Use LLM to extract structured intent from natural language query."""
    
    try:
        result = llm.chat(
            messages=[{"role": "user", "content": QUERY_UNDERSTANDING_PROMPT.format(query=query)}],
            model="gpt-4o-mini",
            max_tokens=300,
            temperature=0,
        )
        
        content = result["content"].strip()
        # Handle markdown code blocks
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        
        data = json.loads(content)
        
        return QueryIntent(
            entity_types=data.get("entity_types", []),
            named_entities=data.get("named_entities", []),
            time_anchor=data.get("time_anchor"),
            temporal_pattern=data.get("temporal_pattern"),
            relationship_types=data.get("relationship_types", []),
            hop_pattern=data.get("hop_pattern", "direct"),
            query_type=data.get("query_type", "factual"),
            raw_query=query,
        )
    except Exception as e:
        logger.warning(f"Query parsing failed: {e}, using defaults")
        return QueryIntent(raw_query=query)


# ── Seed Selection ────────────────────────────────────────────────────────────

@dataclass
class ScoredEntity:
    """Entity with retrieval score and source."""
    entity_id: str
    score: float
    source: str  # "named", "embedding", "type_match"


def select_seeds(
    intent: QueryIntent,
    world_model: WorldModel,
    llm: LLMClient,
    max_seeds: int = 10,
    embedding_threshold: float = 0.7,
) -> list[ScoredEntity]:
    """
    Find seed entities using intent constraints.
    Priority: named entities > embedding match > type-constrained search
    """
    seeds: dict[str, ScoredEntity] = {}
    
    # 1. Named entity lookup (highest confidence)
    for name in intent.named_entities:
        # Exact match
        entity = world_model.find_by_name(name)
        if entity:
            seeds[entity.id] = ScoredEntity(entity.id, 1.0, "named")
            continue
        
        # Fuzzy match
        matches = world_model.find_by_string_match(name, threshold=0.8)
        for entity, score in matches[:2]:
            if entity.id not in seeds:
                seeds[entity.id] = ScoredEntity(entity.id, score * 0.95, "named_fuzzy")
    
    # 2. Embedding search (constrained by intent)
    # NOTE: WorldModel.find_by_embedding only works if entities have embeddings.
    # Since we don't persist embeddings, compute them on the fly.
    query_embedding = llm.embed_single(intent.raw_query)
    
    # Filter by entity type if specified
    type_filter = set(intent.entity_types) if intent.entity_types else None
    candidate_ids = set(world_model.entities.keys()) - set(seeds.keys())
    
    if type_filter:
        candidate_ids = {eid for eid in candidate_ids 
                        if world_model.entities[eid].type.value in type_filter}
    
    # Batch compute embeddings for candidates
    if candidate_ids:
        candidates = [(eid, world_model.entities[eid]) for eid in list(candidate_ids)[:200]]
        texts = []
        for eid, entity in candidates:
            state = entity.current_state
            desc = state.get("description", str(state)[:200]) if isinstance(state, dict) else str(state)[:200]
            text = f"{entity.name} ({entity.type.value}): {desc}"
            texts.append(text)
        
        try:
            embeddings = llm.embed(texts)
            for (eid, entity), emb in zip(candidates, embeddings):
                sim = cosine_similarity(query_embedding, emb)
                if sim >= embedding_threshold and eid not in seeds:
                    seeds[eid] = ScoredEntity(eid, sim, "embedding")
        except Exception as e:
            logger.warning(f"Batch embedding failed: {e}")
    
    # 3. Type-based retrieval if we have type constraints but few seeds
    if intent.entity_types and len(seeds) < max_seeds // 2:
        for etype in intent.entity_types:
            type_entities = world_model._type_index.get(etype, set())
            for eid in list(type_entities)[:max_seeds]:
                if eid not in seeds:
                    entity = world_model.get_entity(eid)
                    if entity:
                        # Score by recency/importance
                        seeds[eid] = ScoredEntity(eid, entity.importance * 0.5, "type_match")
    
    # Sort by score, take top seeds
    sorted_seeds = sorted(seeds.values(), key=lambda s: s.score, reverse=True)
    return sorted_seeds[:max_seeds]


# ── Graph Traversal ───────────────────────────────────────────────────────────

@dataclass
class TraversalPath:
    """A path through the graph from a seed."""
    entity_ids: list[str]
    edge_types: list[str]  # relationship types traversed
    score: float


def traverse_from_seeds(
    seeds: list[ScoredEntity],
    intent: QueryIntent,
    world_model: WorldModel,
    max_hops: int = 2,
    max_paths: int = 50,
) -> list[TraversalPath]:
    """
    Expand from seeds following query-relevant edges.
    
    Traversal strategy depends on intent:
    - "direct": 1-hop only
    - "transitive": follow chains of same relationship type
    - "neighborhood": BFS up to max_hops
    """
    paths: list[TraversalPath] = []
    
    # Start with seed entities as 0-hop paths
    for seed in seeds:
        paths.append(TraversalPath(
            entity_ids=[seed.entity_id],
            edge_types=[],
            score=seed.score,
        ))
    
    hop_pattern = intent.hop_pattern or "direct"
    relevant_rel_types = set(intent.relationship_types) if intent.relationship_types else None
    
    if hop_pattern == "direct" or max_hops == 0:
        return paths
    
    # BFS expansion
    frontier = list(paths)
    visited_edges: set[tuple[str, str]] = set()
    
    for hop in range(max_hops):
        next_frontier = []
        
        for path in frontier:
            current_id = path.entity_ids[-1]
            relationships = world_model.get_relationships(current_id)
            
            for rel in relationships:
                # Get the other end of the relationship
                other_id = rel.target_id if rel.source_id == current_id else rel.source_id
                
                # Skip if we've traversed this edge
                edge_key = (min(current_id, other_id), max(current_id, other_id))
                if edge_key in visited_edges:
                    continue
                visited_edges.add(edge_key)
                
                # Skip if relationship type doesn't match intent
                rel_type = rel.type.value
                if relevant_rel_types and rel_type not in relevant_rel_types:
                    # Penalize but don't skip entirely
                    rel_score_mult = 0.3
                else:
                    rel_score_mult = 1.0
                
                # For transitive pattern, only follow same edge type
                if hop_pattern == "transitive" and path.edge_types:
                    if rel_type != path.edge_types[-1]:
                        continue
                
                # Create new path
                new_path = TraversalPath(
                    entity_ids=path.entity_ids + [other_id],
                    edge_types=path.edge_types + [rel_type],
                    score=path.score * 0.8 * rel_score_mult,  # decay per hop
                )
                next_frontier.append(new_path)
                
                if len(paths) + len(next_frontier) >= max_paths:
                    break
            
            if len(paths) + len(next_frontier) >= max_paths:
                break
        
        paths.extend(next_frontier)
        frontier = next_frontier
        
        if not frontier or len(paths) >= max_paths:
            break
    
    return paths


# ── Temporal Filtering ────────────────────────────────────────────────────────

def filter_by_temporal_intent(
    paths: list[TraversalPath],
    intent: QueryIntent,
    world_model: WorldModel,
) -> list[TraversalPath]:
    """
    Re-score paths based on temporal constraints.
    
    Temporal patterns:
    - "evolution": prioritize entities with many transitions
    - "first"/"last": prioritize by first_seen/last_seen
    - "during" + time_anchor: filter to time range
    """
    if not intent.temporal_pattern and not intent.time_anchor:
        return paths
    
    rescored = []
    
    for path in paths:
        temporal_score = 1.0
        
        for eid in path.entity_ids:
            entity = world_model.get_entity(eid)
            if not entity:
                continue
            
            transitions = world_model.get_transitions(eid)
            
            if intent.temporal_pattern == "evolution":
                # Boost entities with rich history
                temporal_score *= min(1.0 + len(transitions) * 0.1, 2.0)
            
            elif intent.temporal_pattern in ("first", "earliest"):
                # Score by how early this entity appeared
                # Lower first_seen = higher score
                if entity.first_seen > 0:
                    # Normalize: entities from 2+ years ago get max boost
                    import time
                    age_days = (time.time() - entity.first_seen) / 86400
                    temporal_score *= min(1.0 + age_days / 365, 2.0)
            
            elif intent.temporal_pattern in ("last", "latest", "recent"):
                # Score by recency
                if entity.last_seen > 0:
                    import time
                    days_ago = (time.time() - entity.last_seen) / 86400
                    # Recent = higher score
                    temporal_score *= max(0.3, 1.0 - days_ago / 180)
            
            # TODO: parse time_anchor into time_range and filter
        
        rescored.append(TraversalPath(
            entity_ids=path.entity_ids,
            edge_types=path.edge_types,
            score=path.score * temporal_score,
        ))
    
    rescored.sort(key=lambda p: p.score, reverse=True)
    return rescored


# ── Subgraph Assembly ─────────────────────────────────────────────────────────

@dataclass 
class RetrievedSubgraph:
    """The result of graph-aware retrieval."""
    entity_ids: list[str]  # unique entities, ordered by relevance
    paths: list[TraversalPath]  # how we got there
    intent: QueryIntent
    
    def get_entities(self, world_model: WorldModel) -> list:
        """Get actual entity objects."""
        return [world_model.get_entity(eid) for eid in self.entity_ids if world_model.get_entity(eid)]


def retrieve_subgraph(
    query: str,
    world_model: WorldModel,
    llm: LLMClient | None = None,
    max_entities: int = 15,
    max_hops: int = 2,
) -> RetrievedSubgraph:
    """
    Full graph-aware retrieval pipeline.
    
    1. Parse query intent
    2. Select seeds
    3. Traverse graph
    4. Filter by temporal constraints
    5. Dedupe and rank entities
    """
    if llm is None:
        llm = LLMClient()
    
    # 1. Query understanding
    intent = parse_query_intent(query, llm)
    logger.info(f"Query intent: types={intent.entity_types}, temporal={intent.temporal_pattern}, hops={intent.hop_pattern}")
    
    # 2. Seed selection
    seeds = select_seeds(intent, world_model, llm, max_seeds=8)
    logger.info(f"Seeds: {[s.entity_id[:8] for s in seeds]}")
    
    if not seeds:
        return RetrievedSubgraph(entity_ids=[], paths=[], intent=intent)
    
    # 3. Graph traversal
    paths = traverse_from_seeds(seeds, intent, world_model, max_hops=max_hops)
    logger.info(f"Traversal found {len(paths)} paths")
    
    # 4. Temporal filtering
    paths = filter_by_temporal_intent(paths, intent, world_model)
    
    # 5. Collect unique entities, ordered by best path score
    entity_scores: dict[str, float] = {}
    for path in paths:
        for eid in path.entity_ids:
            # Take max score across all paths containing this entity
            entity_scores[eid] = max(entity_scores.get(eid, 0), path.score)
    
    sorted_entities = sorted(entity_scores.items(), key=lambda x: x[1], reverse=True)
    top_entity_ids = [eid for eid, _ in sorted_entities[:max_entities]]
    
    return RetrievedSubgraph(
        entity_ids=top_entity_ids,
        paths=paths[:20],  # keep top paths for debugging
        intent=intent,
    )


# ── Optional: LLM-in-the-loop path scoring ────────────────────────────────────

PATH_RELEVANCE_PROMPT = """Given this query and a path through a knowledge graph, rate relevance 0-10.

Query: {query}

Path:
{path_description}

Just the number:"""


def score_path_with_llm(
    path: TraversalPath,
    query: str,
    world_model: WorldModel,
    llm: LLMClient,
) -> float:
    """
    Use LLM to score path relevance. Expensive but accurate.
    Use sparingly (e.g., for top-k paths from traversal).
    """
    # Build path description
    parts = []
    for i, eid in enumerate(path.entity_ids):
        entity = world_model.get_entity(eid)
        if entity:
            parts.append(f"{entity.name} ({entity.type.value})")
            if i < len(path.edge_types):
                parts.append(f"--[{path.edge_types[i]}]-->")
    
    path_desc = " ".join(parts)
    
    try:
        result = llm.chat(
            messages=[{"role": "user", "content": PATH_RELEVANCE_PROMPT.format(
                query=query,
                path_description=path_desc,
            )}],
            model="gpt-4o-mini",
            max_tokens=10,
            temperature=0,
        )
        score = float(result["content"].strip()) / 10.0
        return min(max(score, 0), 1)
    except:
        return path.score  # fallback to traversal score


# ── Test ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    
    # Load world model
    from pathlib import Path
    wm_path = Path("output/world_model.json")
    if not wm_path.exists():
        print("No world model found at output/world_model.json")
        sys.exit(1)
    
    wm = WorldModel(wm_path)
    llm = LLMClient()
    
    query = sys.argv[1] if len(sys.argv) > 1 else "How has the PIE project evolved?"
    
    print(f"\nQuery: {query}\n")
    
    result = retrieve_subgraph(query, wm, llm)
    
    print(f"Intent: {result.intent}")
    print(f"\nRetrieved {len(result.entity_ids)} entities:")
    for eid in result.entity_ids:
        entity = wm.get_entity(eid)
        if entity:
            print(f"  - {entity.name} ({entity.type.value})")
    
    print(f"\nTop paths:")
    for path in result.paths[:5]:
        names = [wm.get_entity(eid).name if wm.get_entity(eid) else eid[:8] for eid in path.entity_ids]
        print(f"  {' -> '.join(names)} (score: {path.score:.2f})")
