"""Hybrid Retriever for PIE — BM25 + Dense + RRF + Temporal + Graph Expansion.

Two retrieval modes:

  retrieve()       — focused retrieval for single-answer queries (top 10-20)
    1. BM25 sparse search
    2. Dense matmul search
    3. RRF fusion
    4. Temporal boost
    5. One-hop graph expansion

  broad_retrieve() — exhaustive scan for "tell me everything about X" queries
    1. LLM decomposes query into 10-15 specific sub-queries
    2. Each sub-query runs through focused retrieval (top 20)
    3. All results unioned via multi-source RRF
    4. Returns 60-80 entities covering every angle of the topic

Usage:
    retriever = HybridRetriever(world_model, llm_client)

    # Single-answer query
    ids = retriever.retrieve(query, top_k=10)

    # Full memory scan
    ids = retriever.broad_retrieve(query, top_k=60)
    context = retriever.compile_context(ids, query, max_transitions=25)
"""
from __future__ import annotations

import json
import math
from datetime import datetime
from typing import Optional, TYPE_CHECKING

from pie.retrieval.bm25 import BM25Index
from pie.retrieval.temporal_retriever import TemporalRefExtractor
from pie.retrieval.context_compiler import compile_subgraph
from pie.config import PIEConfig

if TYPE_CHECKING:
    from pie.core.world_model import WorldModel
    from pie.core.llm import LLMClient


# RRF constant — 60 is the standard value from Cormack et al. (2009)
_RRF_K = 60


def _rrf_score(rank: int) -> float:
    return 1.0 / (_RRF_K + rank + 1)


class HybridRetriever:
    """Combines BM25 + dense retrieval with temporal scoring and graph expansion."""

    def __init__(
        self,
        world_model: "WorldModel",
        llm: "LLMClient",
        config: Optional[PIEConfig] = None,
    ):
        self.world_model = world_model
        self.llm = llm
        self.config = config or PIEConfig()

        self._bm25 = BM25Index()
        self._bm25.build(world_model)

        self._temporal_extractor = TemporalRefExtractor(self.config.temporal)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        now: Optional[datetime] = None,
        oversample: int = 3,
    ) -> list[str]:
        """Return a list of entity_ids ranked by hybrid score.

        Args:
            query:      Natural language query string.
            top_k:      Final number of entities to return.
            now:        Reference timestamp for temporal scoring (defaults to now).
            oversample: Fetch oversample*top_k candidates before reranking.
        """
        now = now or datetime.now()
        self._temporal_extractor.reference_time = now
        temporal_ref = self._temporal_extractor.extract(query)

        pool_k = top_k * oversample

        # --- 1. BM25 sparse retrieval ---
        bm25_hits = self._bm25.query(query, top_k=pool_k)
        bm25_rank = {eid: rank for rank, (eid, _) in enumerate(bm25_hits)}

        # --- 2. Dense semantic retrieval ---
        query_emb = self.llm.embed_single(query)
        dense_hits = self.world_model.find_by_embedding(query_emb, top_k=pool_k)
        dense_rank = {entity.id: rank for rank, (entity, _) in enumerate(dense_hits)}

        # --- 3. RRF fusion ---
        all_ids = set(bm25_rank) | set(dense_rank)
        rrf_scores: dict[str, float] = {}
        for eid in all_ids:
            score = 0.0
            if eid in bm25_rank:
                score += _rrf_score(bm25_rank[eid])
            if eid in dense_rank:
                score += _rrf_score(dense_rank[eid])
            rrf_scores[eid] = score

        # --- 4. Temporal boost (no extra API call) ---
        if temporal_ref is not None:
            for eid, rrf in rrf_scores.items():
                entity = self.world_model.entities.get(eid)
                if entity is None:
                    continue
                ts = datetime.fromtimestamp(entity.last_seen) if entity.last_seen else None
                t_score = self._temporal_score(ts, temporal_ref, now)
                rrf_scores[eid] = rrf * (0.5 + 0.5 * t_score)

        # Sort candidates by fused score
        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        top_ids = [eid for eid, _ in ranked]

        # --- 5. One-hop graph expansion ---
        expanded = self._graph_expand(top_ids, top_k)

        return expanded

    def broad_retrieve(
        self,
        query: str,
        top_k: int = 60,
        now: Optional[datetime] = None,
        n_subqueries: int = 12,
    ) -> list[str]:
        """Exhaustive scan retrieval for "tell me everything about X" queries.

        The core failure of single-vector retrieval on broad prompts:
        embedding("search all health memories") ≠ embedding("ADHD diagnosis")
        or embedding("carnivore diet") or embedding("CJC-1295 peptide stack").
        By decomposing into specific sub-queries we recover precision recall on
        every facet of the topic.

        Pipeline:
          1. LLM generates n_subqueries specific keyword queries from the intent
          2. Each sub-query runs through _raw_retrieve (BM25+dense+RRF, top_k=20)
          3. All results are merged with multi-source RRF (each sub-query = one voter)
          4. Top top_k entity_ids returned (no graph expansion — already broad enough)

        One LLM call for sub-query generation, then N fast local retrieval rounds.
        """
        now = now or datetime.now()

        # Step 1: LLM decomposes query into specific sub-queries
        sub_queries = self._decompose_query(query, n_subqueries)

        # Step 2 & 3: run each sub-query, accumulate rank scores across voters
        # multi_rrf[eid][subquery_idx] = rank
        rank_votes: dict[str, dict[int, int]] = {}

        for q_idx, sub_q in enumerate(sub_queries):
            results = self._raw_retrieve(sub_q, top_k=20, now=now)
            for rank, eid in enumerate(results):
                rank_votes.setdefault(eid, {})[q_idx] = rank

        # Compute multi-source RRF score: sum 1/(k+rank) over all voters
        fused: dict[str, float] = {}
        for eid, votes in rank_votes.items():
            fused[eid] = sum(_rrf_score(r) for r in votes.values())

        ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)
        return [eid for eid, _ in ranked[:top_k]]

    def _decompose_query(self, query: str, n: int) -> list[str]:
        """Generate n targeted search sub-queries from a broad query.

        Does NOT sample entity names as context — that caused sub-query generation
        to anchor on whatever happened to be near the top of the entity dict
        (e.g. "MCP server", "YC application") instead of the actual topic.
        Sub-queries are derived purely from the query intent.
        """
        try:
            result = self.llm.chat(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            f"Decompose the user's intent into {n} SHORT (2-6 word) keyword search "
                            "queries for a personal knowledge graph that stores life events, projects, "
                            "people, tools, beliefs, and decisions as typed entities.\n"
                            "Rules:\n"
                            "- Each query must be SHORT and SPECIFIC — concrete nouns, named concepts\n"
                            "- Cover different facets of the topic\n"
                            "- Never use generic filler words (search, find, tell me, etc.)\n"
                            "- Think: what specific entity name or topic keyword would appear in the graph?\n"
                            f"Return JSON: {{\"queries\": [\"...\", ...]}}"
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Intent: {query[:600]}",
                    },
                ],
                model="gpt-5.4",
                json_mode=True,
            )
            _c = result["content"]
            sub_queries: list[str] = (_c if isinstance(_c, dict) else json.loads(_c)).get("queries", [])
            if sub_queries:
                return sub_queries[:n]
        except Exception:
            pass

        return [query[:200]]

    def compile_context(
        self,
        entity_ids: list[str],
        query: Optional[str] = None,
        now: Optional[datetime] = None,
        max_transitions: int = 10,
    ) -> str:
        """Compile retrieved entity list to LLM-ready markdown.

        Args:
            max_transitions: Max transitions to include per entity.
                             Use 25+ for broad-scan queries to preserve full history.
        """
        now = now or datetime.now()
        entities = []
        for eid in entity_ids:
            entity = self.world_model.entities.get(eid)
            if entity:
                entities.append(entity)

        transitions_map = {
            e.id: self.world_model.get_transitions(e.id)[:max_transitions] for e in entities
        }
        relationships_map = {
            e.id: [
                (rel, self.world_model.entities[
                    rel.target_id if rel.source_id == e.id else rel.source_id
                ])
                for rel in self.world_model.get_relationships(e.id)
                if (rel.target_id if rel.source_id == e.id else rel.source_id)
                in self.world_model.entities
            ][:8]
            for e in entities
        }

        result = compile_subgraph(entities, transitions_map, relationships_map, now, query)
        return result.markdown

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _raw_retrieve(self, query: str, top_k: int, now: datetime) -> list[str]:
        """BM25 + dense + RRF for a single query. No graph expansion, no temporal boost.

        Used as the inner loop in broad_retrieve where we call this many times.
        Kept fast: BM25 is O(1) after build, dense is one matmul.
        """
        pool_k = top_k * 2

        bm25_hits = self._bm25.query(query, top_k=pool_k)
        bm25_rank = {eid: rank for rank, (eid, _) in enumerate(bm25_hits)}

        query_emb = self.llm.embed_single(query)
        dense_hits = self.world_model.find_by_embedding(query_emb, top_k=pool_k)
        dense_rank = {entity.id: rank for rank, (entity, _) in enumerate(dense_hits)}

        all_ids = set(bm25_rank) | set(dense_rank)
        scores = {
            eid: (
                (_rrf_score(bm25_rank[eid]) if eid in bm25_rank else 0) +
                (_rrf_score(dense_rank[eid]) if eid in dense_rank else 0)
            )
            for eid in all_ids
        }
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [eid for eid, _ in ranked[:top_k]]

    def _graph_expand(self, top_ids: list[str], budget: int) -> list[str]:
        """Add direct neighbours for multi-hop queries, up to the budget."""
        seen = set(top_ids)
        expanded = list(top_ids)

        for eid in list(top_ids):
            if len(expanded) >= budget:
                break
            for rel in self.world_model.get_relationships(eid):
                neighbour_id = (
                    rel.target_id if rel.source_id == eid else rel.source_id
                )
                if neighbour_id not in seen and len(expanded) < budget:
                    expanded.append(neighbour_id)
                    seen.add(neighbour_id)

        return expanded

    def _temporal_score(self, ts: Optional[datetime], temporal_ref, now: datetime) -> float:
        """Replicate TemporalRetriever._compute_temporal_score without circular import."""
        if ts is None:
            return 0.5
        if temporal_ref is None:
            days_ago = (now - ts).total_seconds() / 86400
            return math.exp(-days_ago / 90)

        in_range = True
        if temporal_ref.start and ts < temporal_ref.start:
            in_range = False
        if temporal_ref.end and ts > temporal_ref.end:
            in_range = False

        if in_range:
            return temporal_ref.confidence

        if temporal_ref.start and ts < temporal_ref.start:
            dist = (temporal_ref.start - ts).total_seconds() / 86400
        elif temporal_ref.end and ts > temporal_ref.end:
            dist = (ts - temporal_ref.end).total_seconds() / 86400
        else:
            dist = 0

        return math.exp(-dist / 30) * 0.5

    def rebuild(self):
        """Rebuild BM25 index after world model is updated (e.g. post-ingestion)."""
        self._bm25.build(self.world_model)
