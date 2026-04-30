"""Enrich a single component end-to-end.

Pipeline:
  1. Resolve the name → canonical URLs (LLM prior; cheap if you already
     pass URLs in).
  2. Fetch the homepage HTML, strip to plain text.
  3. Fetch the GitHub README (raw via api.github.com).
  4. LLM-extract the structured "card" (extractors.extract_card).
  5. Embed the (name + summary + capability_long) for vector search.
  6. Upsert into components, tags, component_tags.
  7. Best-effort upsert relationships from `integrates_with`,
     `alternative_to`, `depends_on` — only for targets that already exist
     in the DB (we don't auto-enrich unknown referenced components in
     this pass; they go on the ingestion_queue for later).

This is the workhorse function called from scripts/enrich_one.py and
from the daily ingestion loop.
"""
from __future__ import annotations
import json
import logging
import re
from typing import Any

from .. import db
from ..ingestion import extractors, github_client, apify_client
from mempol import llm

logger = logging.getLogger(__name__)


def _slugify(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (name or "").strip().lower())
    return re.sub(r"^-+|-+$", "", s) or "unknown"


def _strip_html_to_text(html: str, max_chars: int = 12000) -> str:
    """Crude HTML→text. Production uses readability/trafilatura; for the
    MVP a regex strip is enough and avoids new deps."""
    if not html:
        return ""
    # drop scripts and styles
    html = re.sub(r"(?is)<script[^>]*>.*?</script>", " ", html)
    html = re.sub(r"(?is)<style[^>]*>.*?</style>", " ", html)
    # drop tags, decode common entities, collapse whitespace
    text = re.sub(r"<[^>]+>", " ", html)
    text = (text.replace("&nbsp;", " ")
                 .replace("&amp;", "&")
                 .replace("&lt;", "<")
                 .replace("&gt;", ">")
                 .replace("&quot;", '"')
                 .replace("&#x27;", "'")
                 .replace("&#39;", "'"))
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]


def _fetch_homepage_text(url: str) -> str:
    """Try Apify-based fetcher first (handles JS-heavy sites); fall back
    to a plain requests GET."""
    try:
        html = apify_client.fetch_page_html(url)
        if html:
            return _strip_html_to_text(html)
    except Exception as e:
        logger.warning("apify fetch failed for %s: %s; falling back", url, e)
    try:
        import requests
        r = requests.get(url, timeout=20, headers={
            "User-Agent": "architect-bot/0.1 (+research)"
        })
        r.raise_for_status()
        return _strip_html_to_text(r.text)
    except Exception as e:
        logger.warning("plain-requests fetch failed for %s: %s", url, e)
        return ""


def _fetch_readme(github_url: str) -> str:
    if not github_url:
        return ""
    try:
        return github_client.fetch_readme(github_url) or ""
    except Exception as e:
        logger.warning("README fetch failed for %s: %s", github_url, e)
        return ""


def _embed_card(card: dict) -> list[float]:
    """Build the embedding text from the high-signal fields of the card."""
    parts = [
        card.get("canonical_name") or "",
        card.get("one_liner") or "",
        card.get("summary") or "",
        card.get("capability_long") or "",
        " ".join(card.get("tags") or []),
    ]
    text = "\n".join(p for p in parts if p)
    try:
        return llm.embed([text])[0].tolist()
    except Exception as e:
        logger.warning("embedding failed: %s", e)
        return []


def _link_relationships(conn, source_id: int, card: dict) -> int:
    """Upsert edges to existing components (by slug fuzzy match)."""
    n = 0
    for rel_type, key in [
        ("integrates_with", "integrates_with"),
        ("alternative_to",  "alternative_to"),
        ("depends_on",      "depends_on"),
    ]:
        for target_name in (card.get(key) or []):
            target_slug = _slugify(target_name)
            row = db.get_component(conn, target_slug)
            if not row:
                # Don't auto-create stubs in this pass; queue for later
                # ingestion so we get real cards rather than placeholders.
                # Using the homepage url field is unsafe for unresolved
                # names; we just stash on the queue with a name-only marker.
                db.enqueue_url(
                    conn,
                    url=f"name:{target_name}",
                    source="enrich_relationship_target",
                    priority=-1,
                )
                continue
            db.add_relationship(
                conn, source_id=source_id, target_id=row["id"],
                type=rel_type, confidence=0.7,
                evidence_url=card.get("homepage_url") or card.get("github_url"),
                note=f"asserted by {card.get('canonical_name')}'s own docs",
            )
            n += 1
    return n


# ─── Main entry point ────────────────────────────────────────────────────────
def enrich_component(name: str, homepage_url: str = "",
                      github_url: str = "", context: str = "") -> dict:
    """End-to-end enrichment of a single component.

    Returns the upserted card dict. Side effects: rows in components, tags,
    component_tags, possibly relationships and ingestion_queue.
    """
    # 1. Resolve URLs if not given.
    if not homepage_url or not github_url:
        resolved = extractors.resolve_name(name, context=context)
        homepage_url = homepage_url or resolved.get("homepage_url", "")
        github_url   = github_url   or resolved.get("github_url",   "")

    # 2. Fetch sources.
    homepage_text = _fetch_homepage_text(homepage_url) if homepage_url else ""
    readme_text   = _fetch_readme(github_url) if github_url else ""
    if not homepage_text and not readme_text:
        logger.warning("no usable text for %s (homepage=%s gh=%s)",
                       name, homepage_url, github_url)

    # 3. LLM extraction.
    card = extractors.extract_card(
        name=name, homepage_url=homepage_url, github_url=github_url,
        homepage_text=homepage_text, readme_text=readme_text,
    )
    if not card:
        return {}

    # Backfill from inputs if the LLM omitted them.
    card.setdefault("canonical_name", name)
    card.setdefault("homepage_url", homepage_url)
    card.setdefault("github_url",   github_url)
    card.setdefault("slug", _slugify(card["canonical_name"]))
    card.setdefault("type", "tool")
    card.setdefault("tags", [])
    card.setdefault("aliases", [])

    # 4. Embed.
    embedding = _embed_card(card)

    # 5. Upsert into DB.
    with db.connect() as conn:
        cid = db.upsert_component(
            conn,
            slug=card["slug"],
            name=card["canonical_name"],
            aliases_json=card.get("aliases") or [],
            type=card["type"],
            one_liner=card.get("one_liner", ""),
            summary=card.get("summary", ""),
            capability_long=card.get("capability_long", ""),
            homepage_url=card.get("homepage_url"),
            github_url=card.get("github_url"),
            docs_url=card.get("docs_url"),
            mcp_url=card.get("mcp_url"),
            pricing_model=card.get("pricing_model"),
            hosted_or_self=card.get("hosted_or_self"),
            license=card.get("license"),
            embedding_json=embedding,
            last_verified_at=db._now(),
            extras_json={
                "canonical_examples": card.get("canonical_examples", []),
            },
        )

        # Tags
        for tag_name in (card.get("tags") or []):
            tag_id = db.upsert_tag(conn, slug=_slugify(tag_name), name=tag_name)
            db.tag_component(conn, cid, tag_id, weight=1.0)

        # Relationships (only to already-known targets)
        n_rels = _link_relationships(conn, cid, card)
        logger.info("enriched %s (id=%d, %d tags, %d edges, embed=%s)",
                    card["canonical_name"], cid, len(card.get("tags") or []),
                    n_rels, "yes" if embedding else "no")
    return card
