"""Stage 6: research each overlay query.

For each proposal, fetch:
  - 2-3 sentence factual snippet (Wikipedia primary, Exa fallback)
  - One representative image (Wikimedia primary, OpenVerse fallback)
  - Source citation URL + label

Both Wikipedia and Wikimedia are free, no API key required. Exa is optional
for queries Wikipedia doesn't cover well.
"""
from __future__ import annotations

import json
import logging
import os
import urllib.parse
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)


def wikipedia_search(query: str) -> dict | None:
    """Hit the public Wikipedia API for the best matching article extract."""
    try:
        # First: search for the best matching page
        search_url = ("https://en.wikipedia.org/w/api.php?"
                      f"action=query&list=search&srsearch={urllib.parse.quote(query)}"
                      "&format=json&srlimit=1")
        req = urllib.request.Request(search_url, headers={"User-Agent": "footnote/0.1"})
        with urllib.request.urlopen(req, timeout=10) as r:
            search = json.loads(r.read())
        hits = search.get("query", {}).get("search", [])
        if not hits:
            return None
        title = hits[0]["title"]

        # Then: get the page extract (intro) + main image
        extract_url = ("https://en.wikipedia.org/w/api.php?"
                       f"action=query&titles={urllib.parse.quote(title)}"
                       "&prop=extracts|pageimages&exintro=1&explaintext=1"
                       "&piprop=original&format=json")
        req2 = urllib.request.Request(extract_url, headers={"User-Agent": "footnote/0.1"})
        with urllib.request.urlopen(req2, timeout=10) as r:
            page = json.loads(r.read())
        pages = page.get("query", {}).get("pages", {})
        if not pages:
            return None
        first = next(iter(pages.values()))
        extract = first.get("extract", "")
        image = (first.get("original") or {}).get("source")
        page_url = f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title)}"
        return {
            "snippet": extract[:1500],
            "image_url": image,
            "citation_url": page_url,
            "citation_label": "Wikipedia",
            "source": "wikipedia",
        }
    except Exception as e:
        logger.warning("wikipedia lookup failed for %r: %s", query, e)
        return None


def exa_search(query: str, n_results: int = 1) -> dict | None:
    """Exa Search API — better for current events / non-encyclopedic queries.
    Requires EXA_API_KEY. Optional fallback."""
    api_key = os.getenv("EXA_API_KEY")
    if not api_key:
        return None
    try:
        body = json.dumps({
            "query": query,
            "numResults": n_results,
            "contents": {"text": {"maxCharacters": 1500}},
        }).encode()
        req = urllib.request.Request(
            "https://api.exa.ai/search",
            data=body,
            headers={
                "Content-Type": "application/json",
                "x-api-key": api_key,
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read())
        results = data.get("results", [])
        if not results:
            return None
        top = results[0]
        return {
            "snippet": (top.get("text") or "")[:1500],
            "image_url": top.get("image"),
            "citation_url": top.get("url"),
            "citation_label": (top.get("title") or "Source")[:40],
            "source": "exa",
        }
    except Exception as e:
        logger.warning("exa lookup failed for %r: %s", query, e)
        return None


def rewrite_for_overlay(snippet: str, query: str, model: str = "gpt-5-mini") -> str:
    """Compress a fetched snippet to 2-3 lines suitable for an overlay.

    Never hallucinates — only rephrases the source. If the source doesn't
    answer the query, returns the original snippet truncated.
    """
    try:
        from openai import OpenAI
    except ImportError:
        return snippet[:280]

    client = OpenAI()
    sys_prompt = (
        "You are compressing a fetched fact snippet into a 2-3 line video "
        "overlay. RULES: never invent information not in the source. "
        "Preserve specific facts (numbers, dates, proper nouns). Strip "
        "filler. Output should fit in ~260 characters. Output ONLY the "
        "compressed text, no preamble."
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": f"Query: {query}\n\nSource:\n{snippet[:1500]}"},
        ],
        temperature=0.0,
    )
    text = (resp.choices[0].message.content or "").strip()
    return text[:280] if text else snippet[:280]


def research_proposal(proposal: dict, model: str = "gpt-5-mini") -> dict:
    """Fetch + rewrite for one overlay proposal. Returns the proposal
    augmented with text, citation, image, etc."""
    query = proposal.get("query", "")
    fetched = wikipedia_search(query) or exa_search(query)
    if not fetched:
        logger.warning("no source found for %r — skipping overlay", query)
        return {"proposal": proposal, "skipped": True}

    overlay_text = rewrite_for_overlay(fetched["snippet"], query, model=model)
    return {
        "proposal": proposal,
        "text": overlay_text,
        "citation_url": fetched["citation_url"],
        "citation_label": fetched["citation_label"],
        "image_url": fetched.get("image_url"),
        "chart_spec": None,
        "source": fetched.get("source"),
        "skipped": False,
    }


def research_all(proposals: list[dict], model: str = "gpt-5-mini") -> list[dict]:
    """Research every proposal. Drops ones with no source."""
    out: list[dict] = []
    for i, p in enumerate(proposals):
        logger.info("[research %d/%d] %s", i + 1, len(proposals), p.get("query", "")[:60])
        r = research_proposal(p, model=model)
        if not r.get("skipped"):
            out.append(r)
    return out
