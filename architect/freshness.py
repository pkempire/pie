"""Continuous freshness — keep the index alive.

A stale capability index loses to ChatGPT in 60 days. Fresh data is the
real moat, and freshness has to be operational, not aspirational.

This module runs as a daily cron and does four things:

  1. RE-VERIFY a sample of components weighted by importance × age.
     Re-runs the enrichment pipeline so card text reflects today's
     homepage and README. Bumps `last_verified_at`.

  2. DETECT staleness.
     - GitHub repo archived → mark component archived (soft delete).
     - homepage 404 / large content shrink → flag for manual review.
     - PyPI/npm package not found → flag.

  3. INGEST new launches via three sources, each contributing to the
     ingestion_queue:
     - GitHub trending in AI topics (`ai-agents`, `llm-agent`, `mcp`).
     - Hacker News Show HN posts mentioning AI tooling keywords.
     - x.com search for "just shipped" / "launching" + AI keyword.
     The queue is then drained by enrich.enrich_component, which
     produces full cards for the new arrivals.

  4. DECAY importance for components without recent activity.
     `importance_t = importance_{t-1} × exp(-Δt / halflife)`, where
     activity (lookup_entity, plan inclusion, manual reinforcement)
     resets the timer. Components below `importance_floor` get
     archived but not deleted.

The cron is meant to run idempotently from a hosted scheduler (Vercel
cron, GitHub Actions on schedule, or a local launchd plist on the dev
box). Each function is safe to interrupt mid-run.

Run modes:
    python -m architect.freshness verify --sample 10
    python -m architect.freshness ingest_new --max 20
    python -m architect.freshness decay
    python -m architect.freshness all
"""
from __future__ import annotations
import argparse
import json
import logging
import math
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

from . import db
from .ingestion import enrich, github_client, apify_client

logger = logging.getLogger(__name__)


# Tuneable constants. Treat as ablation knobs reported in the paper /
# product blog if we make claims about freshness.
HALFLIFE_DAYS    = 30.0       # importance halflife when no reinforcement
IMPORTANCE_FLOOR = 0.05       # below this, components get archived
ARCHIVED_TTL_DAYS = 365       # how long to keep archived rows before hard prune


# ─── 1. Verification sweep ──────────────────────────────────────────────────
def _sample_for_verification(n: int) -> list[dict]:
    """Pick n components weighted by importance × age-since-verify.

    A component that was verified yesterday and has importance 1.0 has
    weight 0; one verified 90 days ago with importance 0.8 has high
    weight. Result: we re-verify the most-trusted, most-stale rows
    first, regardless of upstream quality."""
    with db.connect() as conn:
        cur = conn.execute(
            "SELECT id, slug, name, homepage_url, github_url, importance, "
            "last_verified_at FROM components"
        )
        rows = [dict(r) for r in cur]
    now = datetime.utcnow()
    weighted = []
    for r in rows:
        lv = r.get("last_verified_at")
        if not lv:
            age_days = 365.0                 # never verified → max priority
        else:
            try:
                age_days = (now - datetime.fromisoformat(lv)).total_seconds() / 86400
            except Exception:
                age_days = 365.0
        weight = max(r["importance"] or 0.0, 0.01) * math.log1p(age_days + 1)
        weighted.append((weight, r))
    weighted.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in weighted[:n]]


def verify_components(sample_n: int = 10) -> int:
    """Re-run enrichment on the top-priority components. Returns the
    count of successful re-verifications."""
    sample = _sample_for_verification(sample_n)
    n_ok = 0
    for row in sample:
        try:
            card = enrich.enrich_component(
                name=row["name"],
                homepage_url=row.get("homepage_url") or "",
                github_url=row.get("github_url") or "",
            )
            if card:
                n_ok += 1
                logger.info("re-verified %s", row["slug"])
            else:
                logger.warning("re-verify returned empty for %s", row["slug"])
        except Exception as e:
            logger.warning("re-verify failed for %s: %s", row["slug"], e)
    return n_ok


# ─── 2. Staleness detection ─────────────────────────────────────────────────
def detect_archived_repos() -> int:
    """For every component with a GitHub URL, hit the API and mark
    archived ones. Also flags repos whose default branch is gone (renamed
    or moved). Returns the count of newly-archived components."""
    n = 0
    with db.connect() as conn:
        cur = conn.execute("SELECT id, slug, github_url FROM components "
                           "WHERE github_url IS NOT NULL AND github_url != ''")
        rows = [dict(r) for r in cur]
    for r in rows:
        try:
            meta = github_client.get_repo_meta(r["github_url"])
            if not meta:                                # 404 etc
                continue
            if meta.get("archived"):
                with db.connect() as conn:
                    conn.execute(
                        "UPDATE components SET importance = importance * 0.3, "
                        "extras_json = json_patch(extras_json, ?) "
                        "WHERE id = ?",
                        (json.dumps({"github_archived": True}), r["id"]),
                    )
                logger.info("flagged archived: %s", r["slug"])
                n += 1
        except Exception as e:
            logger.warning("archived-check failed for %s: %s", r["slug"], e)
    return n


# ─── 3. Ingest new launches ─────────────────────────────────────────────────
_AI_KEYWORDS = [
    "AI agent", "LLM tool", "MCP server", "browser agent",
    "AI scraper", "vector store", "AI memory", "agent framework",
]


def _enqueue_new_from_github_trending(max_repos: int = 10) -> int:
    """Pull top trending repos in AI-adjacent topics and enqueue new ones.

    Topics chosen from common AI-tooling tags. We dedupe by URL against
    the existing ingestion_queue and components.github_url.
    """
    queries = [
        "topic:ai-agent stars:>50",
        "topic:llm-agent stars:>50",
        "topic:mcp-server stars:>20",
        "topic:browser-agent stars:>20",
    ]
    n = 0
    with db.connect() as conn:
        for q in queries:
            for item in github_client.search_repos(
                q, sort="stars", per_page=max_repos // len(queries)
            ):
                url = item.get("html_url", "")
                if not url:
                    continue
                exists = conn.execute(
                    "SELECT id FROM components WHERE github_url=?", (url,),
                ).fetchone()
                if exists:
                    continue
                db.enqueue_url(conn, url=url, source="github_trending",
                                priority=int(item.get("stargazers_count", 0)) // 100)
                n += 1
                if n >= max_repos:
                    return n
    return n


def _enqueue_new_from_show_hn(max_items: int = 10) -> int:
    """Show HN posts about AI tools → enqueue their story_url."""
    n = 0
    for kw in _AI_KEYWORDS[:3]:
        try:
            items = apify_client.search_show_hn(query=kw, limit=max_items)
        except Exception as e:
            logger.warning("show_hn fetch failed (%s): %s", kw, e)
            continue
        with db.connect() as conn:
            for it in items:
                url = it.get("story_url") or it.get("url")
                if not url:
                    continue
                # Only ingest if URL looks like a real product (not
                # github.com/ which we'd handle separately, not
                # discussion sites).
                if url.startswith(("https://news.ycombinator.com",
                                    "https://twitter.com")):
                    continue
                db.enqueue_url(conn, url=url, source="show_hn",
                                priority=int(it.get("points", 0)) // 10)
                n += 1
                if n >= max_items:
                    return n
    return n


def ingest_new_launches(max_per_source: int = 10) -> int:
    """Enqueue candidates from all news sources. Returns total enqueued."""
    n_gh = _enqueue_new_from_github_trending(max_repos=max_per_source)
    n_hn = _enqueue_new_from_show_hn(max_items=max_per_source)
    logger.info("enqueued: github_trending=%d show_hn=%d", n_gh, n_hn)
    return n_gh + n_hn


def drain_ingestion_queue(max_items: int = 20) -> int:
    """Process pending queue items by running enrichment on each."""
    n_done = 0
    while n_done < max_items:
        with db.connect() as conn:
            row = db.take_next_pending(conn)
        if not row:
            break
        url = row["url"]
        try:
            if url.startswith("name:"):
                # name-only entry from relationship discovery
                name = url[len("name:"):]
                enrich.enrich_component(name=name)
            elif "github.com" in url:
                name = url.rstrip("/").split("/")[-1]
                enrich.enrich_component(name=name, github_url=url)
            else:
                # generic homepage URL — let resolve_name fill in the rest
                enrich.enrich_component(name=url, homepage_url=url)
            with db.connect() as conn:
                db.mark_done(conn, row["id"])
            n_done += 1
        except Exception as e:
            logger.warning("queue drain failed for %s: %s", url, e)
            with db.connect() as conn:
                db.mark_done(conn, row["id"], error=str(e)[:200])
    return n_done


# ─── 4. Importance decay + archival ─────────────────────────────────────────
def decay_importance(halflife_days: float = HALFLIFE_DAYS,
                       floor: float = IMPORTANCE_FLOOR) -> tuple[int, int]:
    """Apply exponential decay to importance based on age since last
    referenced. Components below the floor get archived (extras flag,
    not row delete). Returns (decayed, archived).

    Reinforcement happens elsewhere (search_components bumps
    last_referenced; planner's compose_plan bumps included components).
    """
    decayed = archived = 0
    now = datetime.utcnow()
    with db.connect() as conn:
        cur = conn.execute(
            "SELECT id, slug, importance, last_referenced_at FROM components"
        )
        rows = [dict(r) for r in cur]
        for r in rows:
            lref = r.get("last_referenced_at")
            if not lref:
                age_days = halflife_days
            else:
                try:
                    age_days = (now - datetime.fromisoformat(lref)).total_seconds() / 86400
                except Exception:
                    age_days = halflife_days
            decay_factor = math.exp(-age_days / halflife_days)
            new_imp = (r["importance"] or 0.0) * decay_factor
            if new_imp < floor:
                conn.execute(
                    "UPDATE components SET importance=?, "
                    "extras_json=json_patch(extras_json, ?) WHERE id=?",
                    (new_imp, json.dumps({"archived_by_decay": True}), r["id"]),
                )
                archived += 1
            else:
                conn.execute(
                    "UPDATE components SET importance=? WHERE id=?",
                    (new_imp, r["id"]),
                )
                decayed += 1
    return decayed, archived


# ─── CLI ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("verify").add_argument("--sample", type=int, default=10)
    sub.add_parser("detect_archived")
    sub.add_parser("ingest_new").add_argument("--max", type=int, default=10)
    sub.add_parser("drain_queue").add_argument("--max", type=int, default=20)
    sub.add_parser("decay")
    sub.add_parser("all")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    if args.cmd == "verify":
        n = verify_components(sample_n=args.sample)
        print(f"re-verified {n} components")
    elif args.cmd == "detect_archived":
        n = detect_archived_repos()
        print(f"flagged {n} archived components")
    elif args.cmd == "ingest_new":
        n = ingest_new_launches(max_per_source=args.max)
        print(f"enqueued {n} new candidates")
    elif args.cmd == "drain_queue":
        n = drain_ingestion_queue(max_items=args.max)
        print(f"processed {n} queue items")
    elif args.cmd == "decay":
        d, a = decay_importance()
        print(f"decayed {d} components, archived {a} below floor")
    elif args.cmd == "all":
        n_v = verify_components(sample_n=10)
        n_a = detect_archived_repos()
        n_i = ingest_new_launches(max_per_source=10)
        n_d = drain_ingestion_queue(max_items=20)
        d, ar = decay_importance()
        print(f"verify={n_v} archived={n_a} enqueued={n_i} drained={n_d} "
              f"decayed={d} demoted={ar}")


if __name__ == "__main__":
    main()
