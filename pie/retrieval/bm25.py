"""Okapi BM25 sparse retrieval index over world model entities.

No third-party dependencies. k1=1.5, b=0.75 are the canonical defaults
from Robertson et al. (1994), validated across many IR benchmarks.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pie.core.world_model import WorldModel


def _tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if len(t) > 1]


class BM25Index:
    """Incremental Okapi BM25 index over entity text fields."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._entity_ids: list[str] = []
        self._doc_freqs: dict[str, int] = defaultdict(int)
        self._term_freqs: list[dict[str, int]] = []
        self._doc_lengths: list[int] = []
        self._avg_doc_length: float = 1.0
        self._n_docs: int = 0

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, world_model: "WorldModel") -> None:
        """Build index from all entities in the world model."""
        self._entity_ids = []
        self._term_freqs = []
        self._doc_lengths = []
        self._doc_freqs = defaultdict(int)

        for eid, entity in world_model.entities.items():
            doc_text = self._entity_to_text(entity)
            tokens = _tokenize(doc_text)

            tf: dict[str, int] = defaultdict(int)
            for tok in tokens:
                tf[tok] += 1

            self._entity_ids.append(eid)
            self._term_freqs.append(dict(tf))
            self._doc_lengths.append(len(tokens))
            for term in tf:
                self._doc_freqs[term] += 1

        self._n_docs = len(self._entity_ids)
        self._avg_doc_length = (
            sum(self._doc_lengths) / self._n_docs if self._n_docs > 0 else 1.0
        )

    @staticmethod
    def _entity_to_text(entity) -> str:
        """Concatenate the text fields we want to index."""
        parts = [entity.name]
        parts.extend(entity.aliases[:5])

        state = entity.current_state
        if isinstance(state, dict):
            desc = state.get("description", "")
            if desc:
                parts.append(str(desc)[:600])
            for k, v in list(state.items())[:8]:
                if k != "description" and isinstance(v, (str, int, float)):
                    parts.append(str(v)[:200])
        else:
            parts.append(str(state)[:400])

        return " ".join(parts)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(self, text: str, top_k: int = 20) -> list[tuple[str, float]]:
        """Return [(entity_id, bm25_score)] sorted descending."""
        if self._n_docs == 0:
            return []

        query_tokens = _tokenize(text)
        if not query_tokens:
            return []

        scores = [0.0] * self._n_docs

        for term in set(query_tokens):
            df = self._doc_freqs.get(term, 0)
            if df == 0:
                continue
            idf = math.log((self._n_docs - df + 0.5) / (df + 0.5) + 1.0)

            for i, tf_dict in enumerate(self._term_freqs):
                tf = tf_dict.get(term, 0)
                if tf == 0:
                    continue
                dl = self._doc_lengths[i]
                tf_norm = (tf * (self.k1 + 1)) / (
                    tf + self.k1 * (1 - self.b + self.b * dl / self._avg_doc_length)
                )
                scores[i] += idf * tf_norm

        indexed = [
            (self._entity_ids[i], s) for i, s in enumerate(scores) if s > 0
        ]
        indexed.sort(key=lambda x: x[1], reverse=True)
        return indexed[:top_k]

    @property
    def is_built(self) -> bool:
        return self._n_docs > 0
