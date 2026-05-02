"""v1: multi-step heuristic teacher.

Op sequence per query:
  reformulate → retrieve(hybrid) → maybe filter_by_time → maybe expand → rerank → stop_and_answer

This is the teacher policy that v2 (an SFT'd small LM) will learn to imitate.
Every op + observation is logged into the Trace so traces double as SFT data.
"""
from __future__ import annotations
import json
from typing import Any

from .. import llm, config
from ..backends.base import Backend
from .base import ReadPolicy, Step, Trace
from .v0_naive import answer_with_context


_REFORMULATE_PROMPT = (
    "Rewrite the user's question into a short, dense search query optimised for "
    "lexical+semantic retrieval over a personal conversation log. "
    "Keep proper nouns. Drop filler. Output ONLY the query string."
)
_ROUTE_PROMPT = (
    "Route this memory question for retrieval. Decide whether answering likely "
    "requires following relationships across multiple remembered entities or "
    "events, and whether explicit time reasoning matters. Return strict JSON "
    "with booleans: needs_expand, needs_time_reasoning, reason."
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


def _route_question(q: str) -> dict[str, Any]:
    try:
        raw = llm.chat(
            [
                {"role": "system", "content": _ROUTE_PROMPT},
                {"role": "user", "content": q},
            ],
            model=config.REFORMULATE_MODEL,
            json_mode=True,
        )
        obj = json.loads(raw)
        return {
            "needs_expand": bool(obj.get("needs_expand")),
            "needs_time_reasoning": bool(obj.get("needs_time_reasoning")),
            "reason": str(obj.get("reason", "")),
        }
    except Exception:
        return {"needs_expand": False, "needs_time_reasoning": False, "reason": "route_parse_failed"}


class HeuristicPolicy(ReadPolicy):
    name = "v1_heuristic"

    def __init__(
        self,
        first_k: int = 12,
        final_k: int = 6,
        do_reformulate: bool = True,
        do_expand: bool = True,
        do_route: bool = True,
    ) -> None:
        self.first_k = first_k
        self.final_k = final_k
        self.do_reformulate = do_reformulate
        self.do_expand = do_expand
        self.do_route = do_route

    def run(self, question: str, backend: Backend) -> Trace:
        t = Trace(qid="", question=question, backend=backend.name, policy=self.name)

        # 1. Reformulate
        q_search = question
        if self.do_reformulate:
            q_search = _reformulate(question)
            t.steps.append(Step("reformulate", {}, q_search[:120]))
        route = _route_question(question) if self.do_route else {
            "needs_expand": False,
            "needs_time_reasoning": False,
            "reason": "route_disabled",
        }
        t.steps.append(Step("route", {}, json.dumps(route)[:160]))

        # 2. First retrieval
        hits = backend.retrieve(q_search, k=self.first_k, source="hybrid")
        t.steps.append(Step("retrieve", {"k": self.first_k, "source": "hybrid"}, f"{len(hits)} hits"))
        t.n_retrievals += 1

        # 3. Optional 1-hop expand for multi-hop queries
        if self.do_expand and route.get("needs_expand") and hits:
            seed_uids = [h.unit.uid for h in hits[:3]]
            extra = backend.expand(seed_uids, k_per=2)
            if extra:
                t.steps.append(Step("expand", {"seeds": seed_uids, "k_per": 2}, f"+{len(extra)}"))
            seen = {h.unit.uid for h in hits}
            for h in extra:
                if h.unit.uid not in seen:
                    hits.append(h)

        # 4. Light rerank: keep top-final_k by dense similarity to the original question.
        # FIX (audit #5): if the backend is sparse (e.g. Phase B with 1-2
        # entities in a fresh PIE), don't ask for more hits than exist.
        # Using max(2, len(hits)) ensures we get whatever context is there
        # without forcing the answer LLM into "not in context" by default.
        target_k = min(self.final_k, max(2, len(hits)))
        rerank_hits = backend.retrieve(question, k=target_k, source="dense") if hits else []
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
