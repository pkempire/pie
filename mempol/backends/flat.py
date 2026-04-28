"""Flat backend: in-memory dense + cheap BM25-ish lexical, hybrid via RRF.

Simplest possible backend — no folders, no graph. Used as a baseline floor and
as the substrate for the read policies during development.
"""
from __future__ import annotations
import math
import re
from collections import Counter, defaultdict

import numpy as np

from .. import llm
from .base import Backend, Hit, Unit


_TOK = re.compile(r"[a-zA-Z0-9']+")


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in _TOK.findall(text)]


class BM25Index:
    """Compact BM25 (k1=1.2, b=0.75)."""

    def __init__(self, k1: float = 1.2, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.docs: list[list[str]] = []
        self.df: Counter = Counter()
        self.tf: list[Counter] = []
        self.dl: list[int] = []
        self._idf: dict[str, float] = {}
        self._avgdl: float = 0.0
        self._dirty = True

    def add(self, tokens: list[str]) -> None:
        self.docs.append(tokens)
        c = Counter(tokens)
        self.tf.append(c)
        self.dl.append(len(tokens))
        for t in c:
            self.df[t] += 1
        self._dirty = True

    def _build(self) -> None:
        n = len(self.docs)
        self._avgdl = (sum(self.dl) / n) if n else 0.0
        self._idf = {
            t: math.log((n - df + 0.5) / (df + 0.5) + 1.0) for t, df in self.df.items()
        }
        self._dirty = False

    def score(self, q_tokens: list[str], i: int) -> float:
        if self._dirty:
            self._build()
        s = 0.0
        tf = self.tf[i]
        dl = self.dl[i] or 1
        for t in q_tokens:
            if t not in tf:
                continue
            idf = self._idf.get(t, 0.0)
            f = tf[t]
            denom = f + self.k1 * (1 - self.b + self.b * dl / (self._avgdl or 1))
            s += idf * (f * (self.k1 + 1)) / denom
        return s

    def topk(self, q: str, k: int) -> list[tuple[int, float]]:
        if self._dirty:
            self._build()
        q_tokens = _tokens(q)
        scored = [(i, self.score(q_tokens, i)) for i in range(len(self.docs))]
        scored = [s for s in scored if s[1] > 0]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]


def _rrf(rank_lists: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion."""
    score: dict[int, float] = defaultdict(float)
    for ranks in rank_lists:
        for rank, doc_id in enumerate(ranks):
            score[doc_id] += 1.0 / (k + rank + 1)
    return sorted(score.items(), key=lambda x: x[1], reverse=True)


class FlatBackend(Backend):
    name = "flat"

    def __init__(self) -> None:
        self.units: list[Unit] = []
        self._uid_to_idx: dict[str, int] = {}
        self.bm25 = BM25Index()
        self._emb: np.ndarray | None = None

    def ingest(self, units: list[Unit]) -> None:
        for u in units:
            self._uid_to_idx[u.uid] = len(self.units)
            self.units.append(u)
            self.bm25.add(_tokens(u.text))
        # Recompute embeddings (idempotent via cache).
        texts = [u.text for u in self.units]
        emb = llm.embed(texts)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._emb = emb / norms

    def _dense_topk(self, query: str, k: int) -> list[tuple[int, float]]:
        if self._emb is None or len(self.units) == 0:
            return []
        q = llm.embed([query])[0]
        q = q / (np.linalg.norm(q) or 1.0)
        sims = self._emb @ q
        idx = np.argsort(-sims)[:k]
        return [(int(i), float(sims[i])) for i in idx]

    def retrieve(self, query: str, k: int = 10, source: str = "hybrid") -> list[Hit]:
        if not self.units:
            return []
        if source == "dense":
            scored = self._dense_topk(query, k)
            return [Hit(self.units[i], s, "dense") for i, s in scored]
        if source == "bm25":
            scored = self.bm25.topk(query, k)
            return [Hit(self.units[i], s, "bm25") for i, s in scored]
        # Hybrid via RRF
        dense = [i for i, _ in self._dense_topk(query, k * 2)]
        sparse = [i for i, _ in self.bm25.topk(query, k * 2)]
        fused = _rrf([dense, sparse])[:k]
        return [Hit(self.units[i], s, "hybrid") for i, s in fused]

    def expand(self, seed_uids: list[str], k_per: int = 3) -> list[Hit]:
        """Adjacent-turn expansion: include the +-1 turn relative to seeds (same session)."""
        out: list[Hit] = []
        seen = set(seed_uids)
        for uid in seed_uids:
            i = self._uid_to_idx.get(uid)
            if i is None:
                continue
            seed = self.units[i]
            sess = seed.metadata.get("session")
            for j in (i - 1, i + 1):
                if 0 <= j < len(self.units):
                    nb = self.units[j]
                    if nb.metadata.get("session") == sess and nb.uid not in seen:
                        out.append(Hit(nb, 0.5, "expand"))
                        seen.add(nb.uid)
        return out[: k_per * len(seed_uids)]
