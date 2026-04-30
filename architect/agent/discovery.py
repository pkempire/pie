"""Active discovery loop — MCP-Zero applied to component selection.

Motivation
==========
The naive setup is to dump every known component into the planner's
context and let it pick. This has two well-documented failure modes:

  1. **Dunning-Kruger floor**: the LLM only picks from what's in its
     prior. Components that shipped after its training cutoff are
     invisible. Even with a curated index, if a component is missing
     from the prompt the planner can't reach for it.

  2. **Prompt pollution**: 200 component cards in context degrades
     reasoning quality on the actual planning task and makes the agent
     more likely to hallucinate combinations.

MCP-Zero's framing (Wang et al., 2024) is **active tool discovery**:
the agent emits a `tool_wish` in natural language, a separate retrieval
step matches the wish to actual tools (or finds new ones), and the
agent only sees the matches. We adapt this here, with one extension:
unmatched wishes don't fail silently — they trigger a focused web
search that ingests new components into the KG. A wish that *can't* be
matched even after live search is itself useful signal: it's either a
product gap (no tool exists for this capability) or a phrasing problem
(the planner described something that exists under a different name).

API
===

    wish = ToolWish(
        capability="run a headless browser at scale with anti-bot evasion",
        context="b2b scraper, 10k pages/day, no captchas in workflow",
        nice_to_have=["managed infrastructure", "session persistence"],
    )
    matches = discover(wish, top_k=3, allow_live_search=True)
    # → list[ComponentMatch] with score, evidence, and 'fresh' flag

Concretely:

    1. Embed the wish, retrieve top-N candidates from the KG.
    2. If the best match is below `confidence_floor`, escalate:
         a. Generate 2-3 focused search queries via LLM.
         b. Run them through Exa neural search + GitHub repo search.
         c. Feed the top hits into enrich.enrich_component() in the
            background; results land in the KG within ~30s.
         d. Re-query the KG; return the (possibly fresh) matches.
    3. If after live search nothing clears the floor, log the wish to
       `unmatched_wishes` for later product analysis and return the
       best-available matches with a low-confidence flag.

The discovery agent is an LLM tool the planner calls. It's also useful
standalone for ingestion seeding (give it a domain — "GTM enrichment
APIs" — and it'll surface candidates).
"""
from __future__ import annotations
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from .. import db
from ..ingestion import enrich, github_client
from mempol import llm, config

logger = logging.getLogger(__name__)


# ─── Data model ─────────────────────────────────────────────────────────────
@dataclass
class ToolWish:
    """A natural-language description of a tool the planner wishes existed."""
    capability: str                                   # core: what should it DO
    context: str = ""                                  # constraints: scale, language, hosting
    nice_to_have: list[str] = field(default_factory=list)
    must_avoid: list[str] = field(default_factory=list)

    def as_query_text(self) -> str:
        parts = [self.capability]
        if self.context:
            parts.append(f"Context: {self.context}")
        if self.nice_to_have:
            parts.append("Nice to have: " + "; ".join(self.nice_to_have))
        if self.must_avoid:
            parts.append("Must avoid: " + "; ".join(self.must_avoid))
        return "\n".join(parts)


@dataclass
class ComponentMatch:
    component_id: int
    slug: str
    name: str
    score: float                                       # 0..1
    rationale: str                                     # why this matches
    fresh: bool = False                                 # True if just discovered live
    confidence: str = "high"                           # high|medium|low


# ─── Wish-search query generation ───────────────────────────────────────────
_QUERY_GEN_SYSTEM = """You generate web/GitHub search queries to find an
AI software component matching a developer's natural-language wish. Output
JSON with 3 short queries: one for general web search (Exa-style), one
for GitHub repository search, one for GitHub code search (looking for an
actual import/usage line). Each ≤ 6 words. No filler."""

_QUERY_GEN_USER = """WISH

  capability:    {capability}
  context:       {context}
  nice_to_have:  {nice_to_have}
  must_avoid:    {must_avoid}

Return JSON:
{{
  "web_query":    "...",
  "repo_query":   "...",
  "code_query":   "..."
}}"""


def _generate_queries(wish: ToolWish) -> dict:
    msgs = [
        {"role": "system", "content": _QUERY_GEN_SYSTEM},
        {"role": "user",   "content": _QUERY_GEN_USER.format(
            capability=wish.capability,
            context=wish.context or "(none)",
            nice_to_have="; ".join(wish.nice_to_have) or "(none)",
            must_avoid="; ".join(wish.must_avoid) or "(none)",
        )},
    ]
    raw = llm.chat(msgs, model=config.OBSERVER_MODEL or "gpt-4o-mini",
                    json_mode=True)
    try:
        return json.loads(raw)
    except Exception as e:
        logger.warning("query-gen parse failed: %s; raw=%r", e, raw[:200])
        return {}


# ─── KG retrieval ───────────────────────────────────────────────────────────
def _embed(text: str) -> list[float]:
    try:
        return llm.embed([text])[0].tolist()
    except Exception as e:
        logger.warning("wish embedding failed: %s", e)
        return []


def _retrieve_from_kg(wish: ToolWish, top_k: int) -> list[dict]:
    """Pull top-K from the component KG for this wish."""
    emb = _embed(wish.as_query_text())
    if not emb:
        return []
    with db.connect() as conn:
        return db.search_components(conn, emb, top_k=top_k)


# ─── Live search escalation ─────────────────────────────────────────────────
_RANK_SYSTEM = """You score candidate AI components against a wish.
Return JSON list, same length as input, each entry:
{ "score": 0..1, "rationale": "≤ 1 sentence why this matches the wish" }
Use the wish's capability / context / nice_to_have / must_avoid.
Score 0 means irrelevant or contradicts must_avoid; 1.0 means perfect."""


def _rank_candidates(wish: ToolWish, candidates: list[dict]) -> list[dict]:
    """Have an LLM grade each candidate against the wish. Returns the
    same candidates with `_llm_score` and `_llm_rationale` fields added."""
    if not candidates:
        return []
    user = (
        "WISH\n  capability: " + wish.capability + "\n"
        "  context: " + (wish.context or "(none)") + "\n"
        "  nice_to_have: " + "; ".join(wish.nice_to_have) + "\n"
        "  must_avoid: " + "; ".join(wish.must_avoid) + "\n\n"
        "CANDIDATES\n"
    )
    for i, c in enumerate(candidates):
        user += f"  [{i}] name={c['name']!r} type={c['type']!r}\n"
        user += f"      one_liner={c.get('one_liner','')!r}\n"
        user += f"      summary={(c.get('summary') or '')[:240]!r}\n"
    user += "\nReturn JSON list of length " + str(len(candidates)) + "."
    msgs = [
        {"role": "system", "content": _RANK_SYSTEM},
        {"role": "user",   "content": user},
    ]
    raw = llm.chat(msgs, model=config.OBSERVER_MODEL or "gpt-4o-mini",
                    json_mode=True)
    try:
        scores = json.loads(raw)
        if isinstance(scores, dict) and "scores" in scores:
            scores = scores["scores"]
        if not isinstance(scores, list):
            return candidates
    except Exception as e:
        logger.warning("rank-candidates parse failed: %s", e)
        return candidates
    out = []
    for c, s in zip(candidates, scores):
        c2 = dict(c)
        c2["_llm_score"]     = float(s.get("score", 0.0))
        c2["_llm_rationale"] = (s.get("rationale", "") or "")[:200]
        out.append(c2)
    out.sort(key=lambda x: x["_llm_score"], reverse=True)
    return out


def _live_search(wish: ToolWish, max_to_ingest: int = 4) -> list[str]:
    """Run live search on the wish, ingest the top hits, return the slugs
    of the newly-enriched components.

    For now: GitHub repo search via the existing github_client. Exa neural
    search hooks here once the Exa key is wired. The function returns slug
    strings (or [] if nothing landed)."""
    queries = _generate_queries(wish)
    if not queries:
        return []
    repos: list[dict] = []
    repo_q = queries.get("repo_query", "").strip()
    if repo_q:
        for item in github_client.search_repos(
            f"{repo_q} stars:>20", per_page=10,
        ):
            repos.append(item)
            if len(repos) >= max_to_ingest:
                break
    new_slugs: list[str] = []
    for repo in repos:
        name = (repo.get("name") or "").strip()
        url  = repo.get("html_url") or ""
        if not name:
            continue
        try:
            card = enrich.enrich_component(
                name=name,
                github_url=url,
                context=f"discovered via wish: {wish.capability[:80]}",
            )
            if card.get("slug"):
                new_slugs.append(card["slug"])
        except Exception as e:
            logger.warning("live-discovery enrichment failed for %s: %s",
                           url, e)
            continue
    return new_slugs


# ─── Public API ─────────────────────────────────────────────────────────────
def discover(
    wish: ToolWish,
    top_k: int = 3,
    confidence_floor: float = 0.55,
    allow_live_search: bool = True,
) -> list[ComponentMatch]:
    """Resolve a tool wish to actual component matches.

    Flow:
      1. Embed wish, retrieve top-(top_k * 3) from KG.
      2. LLM-rank against the wish (semantic + capability fit + must_avoid).
      3. If best score < confidence_floor and allow_live_search, run a
         focused web search, enrich the top hits, re-query KG.
      4. Return top_k matches with rationales and a 'fresh' flag for
         components that landed during this call.
    """
    # Initial KG pull
    candidates = _retrieve_from_kg(wish, top_k=top_k * 3)
    ranked = _rank_candidates(wish, candidates)
    top = ranked[:top_k]
    best = top[0]["_llm_score"] if top else 0.0
    fresh_slugs: set[str] = set()

    # Escalate if needed
    if best < confidence_floor and allow_live_search:
        logger.info("wish below floor (%.2f < %.2f), escalating to live search",
                    best, confidence_floor)
        fresh_slugs = set(_live_search(wish))
        if fresh_slugs:
            candidates = _retrieve_from_kg(wish, top_k=top_k * 3)
            ranked = _rank_candidates(wish, candidates)
            top = ranked[:top_k]

    # Log unmatched wishes for product-gap analysis
    final_best = top[0]["_llm_score"] if top else 0.0
    if final_best < confidence_floor:
        with db.connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO ingestion_queue "
                "(url, source, priority) VALUES (?, ?, ?)",
                (f"wish:{hash(wish.as_query_text())}",
                 "unmatched_wish", -2),
            )

    out: list[ComponentMatch] = []
    for c in top:
        score = float(c.get("_llm_score", 0.0))
        out.append(ComponentMatch(
            component_id=c["id"],
            slug=c["slug"],
            name=c["name"],
            score=score,
            rationale=c.get("_llm_rationale", "") or c.get("one_liner", ""),
            fresh=(c["slug"] in fresh_slugs),
            confidence=("high" if score >= 0.75
                          else "medium" if score >= confidence_floor
                          else "low"),
        ))
    return out
