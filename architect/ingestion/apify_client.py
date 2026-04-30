"""Apify-based scrapers for things GitHub's API can't reach.

We use Apify because the user already has the Apify MCP available and
because Apify's "actors" handle the JS-heavy single-page-app product
sites that a plain `requests.get` returns empty for. Two actors that
are most useful for us:

  apify/website-content-crawler   — full-page text extraction for any URL
  apify/rag-web-browser           — RAG-friendly content + structured data

Both of these are accessible via the same call_actor pattern. The Apify
MCP server in the user's session exposes `mcp__Apify__call-actor`, but we
also support a plain HTTPS fallback against api.apify.com if APIFY_TOKEN
is set in the env. That keeps this module usable from a cron job that
isn't running inside an MCP session.

Module surface (used by enrich.py and architecture_miner.py):

  fetch_page_html(url)              -> raw HTML (best effort)
  fetch_page_markdown(url)          -> markdown / clean text
  search_show_hn(query, limit)      -> list of HN posts about AI tooling
  search_x_launches(keyword)        -> recent x.com launches mentioning the keyword
"""
from __future__ import annotations
import logging
import os
import time
from typing import Any

import requests

logger = logging.getLogger(__name__)

APIFY_API = "https://api.apify.com/v2"

# Actor IDs we use. These are stable Apify-published actors.
ACTOR_WEBSITE_CRAWLER = "apify~website-content-crawler"
ACTOR_RAG_BROWSER     = "apify~rag-web-browser"
ACTOR_HN              = "epctex~hackernews-scraper"
ACTOR_X_SEARCH        = "apidojo~tweet-scraper"


def _token() -> str | None:
    return os.environ.get("APIFY_TOKEN") or os.environ.get("APIFY_API_TOKEN")


def _call_actor_sync(actor_id: str, run_input: dict,
                      timeout_secs: int = 90) -> list[dict]:
    """Run an Apify actor synchronously and return its dataset items.

    Uses `run-sync-get-dataset-items` so we don't need to poll. Returns
    [] on any failure — call sites are expected to handle empty.
    """
    token = _token()
    if not token:
        logger.warning("APIFY_TOKEN not set; apify actor %s will be skipped",
                       actor_id)
        return []
    url = f"{APIFY_API}/acts/{actor_id}/run-sync-get-dataset-items"
    try:
        r = requests.post(
            url,
            params={"token": token, "timeout": timeout_secs,
                    "memory": 1024, "format": "json"},
            json=run_input,
            timeout=timeout_secs + 10,
        )
        if r.status_code != 200:
            logger.warning("apify %s -> %d: %s", actor_id, r.status_code,
                           r.text[:200])
            return []
        data = r.json()
        return data if isinstance(data, list) else []
    except Exception as e:
        logger.warning("apify call failed (%s): %s", actor_id, e)
        return []


# ─── 1. Page fetching ────────────────────────────────────────────────────────
def fetch_page_html(url: str) -> str:
    """Run website-content-crawler on a single URL; return raw HTML."""
    items = _call_actor_sync(ACTOR_WEBSITE_CRAWLER, {
        "startUrls": [{"url": url}],
        "maxCrawlDepth": 0,            # one page only
        "maxCrawlPages": 1,
        "saveHtml": True,
    })
    if not items:
        return ""
    return items[0].get("html") or items[0].get("body") or ""


def fetch_page_markdown(url: str) -> str:
    """RAG-friendly clean text. Better for LLM extraction than raw HTML."""
    items = _call_actor_sync(ACTOR_RAG_BROWSER, {
        "query": url,
        "maxResults": 1,
        "scrapingTool": "browser-playwright",
        "outputFormats": ["markdown"],
    })
    if not items:
        return ""
    return items[0].get("markdown") or items[0].get("text") or ""


# ─── 2. Show HN: surface AI-tooling launches ────────────────────────────────
def search_show_hn(query: str = "AI agent", limit: int = 30) -> list[dict]:
    """Find recent Show HN posts mentioning the query. Each item:
        { title, url, points, comments, story_url, posted_at }
    The story_url is the project URL (the thing actually being shipped),
    which we then enqueue for enrichment.
    """
    items = _call_actor_sync(ACTOR_HN, {
        "search": f"Show HN: {query}",
        "max_items": limit,
        "search_by_date": True,
    })
    out: list[dict] = []
    for it in items:
        if not it.get("title", "").lower().startswith("show hn"):
            continue
        out.append({
            "title": it.get("title", ""),
            "url": it.get("url") or it.get("story_url") or "",
            "story_url": it.get("story_url") or it.get("url") or "",
            "points": int(it.get("points") or 0),
            "comments": int(it.get("comments_count") or 0),
            "posted_at": it.get("posted_at") or it.get("time_iso"),
        })
    return out


# ─── 3. x.com launches via search ───────────────────────────────────────────
def search_x_launches(keyword: str = "just shipped",
                       limit: int = 30) -> list[dict]:
    """x.com search via Apify's tweet scraper. Returns recent tweets that
    plausibly announce a new AI tool."""
    items = _call_actor_sync(ACTOR_X_SEARCH, {
        "searchTerms": [keyword],
        "maxItems": limit,
        "sort": "Latest",
    })
    out: list[dict] = []
    for it in items:
        out.append({
            "text":     it.get("text") or it.get("full_text", ""),
            "author":   (it.get("author") or {}).get("userName", ""),
            "url":      it.get("url") or "",
            "posted_at": it.get("createdAt") or it.get("created_at"),
            "likes":    int(it.get("likeCount") or 0),
            "retweets": int(it.get("retweetCount") or 0),
        })
    return out
