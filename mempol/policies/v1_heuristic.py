"""v1: multi-step heuristic teacher.

Op sequence per query:
  reformulate → retrieve(hybrid) → maybe filter_by_time → maybe expand → rerank → stop_and_answer

This is the teacher policy that v2 (an SFT'd small LM) will learn to imitate.
Every op + observation is logged into the Trace so traces double as SFT data.
"""
from __future__ import annotations
import json
import re
from typing import Any

from .. import llm, config
from ..backends.base import Backend
from .base import ReadPolicy, Step, Trace
from .v0_naive import answer_with_context


_TEMPORAL_HINTS = re.compile(
    r"\b(when|date|time|year|month|week|day|before|after|since|first|last|recent|earliest|latest)\b",
    re.IGNORECASE,
)
_MULTIHOP_HINTS = re.compile(
    r"\b(both|also|else|other|together|relate|connection|why|because|how does|fields|topics)\b",
    re.IGNORECASE,
)


_REFORMULATE_PROMPT = (
    "Rewrite the user's question into a short, dense search query optimised for "
    "lexical+semantic retrieval over a personal conversation log. "
    "Keep proper nouns. Drop filler. Output ONLY the query string."
)


def _reformulate(q: str) -> str:
    out = llm.chat(
        [
            {"role": "system", "content": _REFORMULATE_PROMPT},
            {"role": "user", "content": q},
        ],
        model=config.REFORMULATE_MODEL,
    ).strip().strip('"').strip()
    return out or q


def _is_temporal(q: str) -> bool:
    return bool(_TEMPORAL_HINTS.search(q))


def _is_multihop(q: str) -> bool:
    return bool(_MULTIHOP_HINTS.search(q))


class HeuristicPolicy(ReadPolicy):
    name = "v1_heuristic"

    def __init__(
        self,
        first_k: int = 12,
        final_k: int = 6,
        do_reformulate: bool = True,
        do_expand: bool = True,
    ) -> None:
        self.first_k = first_k
        self.final_k = final_k
        self.do_reformulate = do_reformulate
        self.do_expand = do_expand

    def run(self, question: str, backend: Backend) -> Trace:
        t = Trace(qid="", question=question, backend=backend.name, policy=self.name)

        # 1. Reformulate
        q_search = question
        if self.do_reformulate:
            q_search = _reformulate(question)
            t.steps.append(Step("reformulate", {}, q_search[:120]))

        # 2. First retrieval
        hits = backend.retrieve(q_search, k=self.first_k, source="hybrid")
        t.steps.append(Step("retrieve", {"k": self.first_k, "source": "hybrid"}, f"{len(hits)} hits"))
        t.n_retrievals += 1

        # 3. Optional 1-hop expand for multi-hop queries
        if self.do_expand and _is_multihop(question) and hits:
            seed_uids = [h.unit.uid for h in hits[:3]]
            extra = backend.expand(seed_uids, k_per=2)
            if extra:
                t.steps.append(Step("expand", {"seeds": seed_uids, "k_per": 2}, f"+{len(extra)}"))
            seen = {h.unit.uid for h in hits}
            for h in extra:
                if h.unit.uid not in seen:
                    hits.append(h)

        # 4. Light rerank: keep top-final_k by dense similarity to the original question
        rerank_hits = backend.retrieve(question, k=self.final_k, source="dense") if hits else []
        rerank_uids = [h.unit.uid for h in rerank_hits]
        # Combine: prefer reranked, then fill from hits
        seen, out = set(), []
        for h in rerank_hits + hits:
            if h.unit.uid in seen:
                continue
            out.append(h)
            seen.add(h.unit.uid)
            if len(out) >= self.final_k:
                break
        t.steps.append(Step("rerank", {"strategy": "dense", "k": self.final_k}, f"kept {len(out)}"))
        t.n_retrievals += 1

        t.final_hits = out
        t.answer = answer_with_context(question, out)
        t.steps.append(Step("stop_and_answer", {}, t.answer[:120]))
        return t
