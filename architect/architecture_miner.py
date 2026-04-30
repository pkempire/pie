"""Architecture miner — sample real production code that uses our components.

Why this matters
================
Knowing that Stagehand exists is half the value. The other half is
knowing *how* people use it in real systems. Co-occurrence tells the
planner that Stagehand + Browserbase + OpenAI SDK is the canonical
"managed-browser-agent" stack, which is the kind of structured opinion
that ChatGPT can't reproduce because it isn't in the training data
(and goes stale fast even when it is).

For each known component, we sample real-world repositories that use it,
extract their architecture cards, and write the (architecture, component)
edges into `architecture_components`. Every additional repo we ingest
strengthens or weakens our existing relationship edges via
reinforcement.

Sources, in priority order
==========================

1. **GitHub repo search** — query for the component's name in README +
   high-star repos. Highest signal: maintained, real apps.

2. **GitHub code search** — query for an unambiguous import string
   (e.g. `from stagehand import` / `BROWSERBASE_API_KEY`). Highest
   precision: forces the component to be actually imported, not just
   mentioned. Requires GITHUB_TOKEN.

3. **Awesome lists** — `awesome-mcp`, `awesome-langchain`,
   `awesome-llm-apps`. Each entry is a curated example of the component
   being used in a real or demo project.

4. **n8n community templates** — the community repo of public n8n
   workflows mentions specific tool integrations.

5. **Show HN with project URL** — `apify_client.search_show_hn` returns
   recent posts. Each story_url is a candidate project; if the project's
   GitHub or homepage references our component, we ingest.

6. **arxiv applied papers** — papers describing systems often cite the
   tools used in their methods section. (We don't auto-ingest arxiv yet
   in v0; flagged as future work.)

For each candidate repo we extract an "architecture card" via
extractors.extract_architecture and link it to ALL components mentioned.
This automatically grows the co-occurrence graph.

CLI:
    python -m architect.scripts.mine_architectures Stagehand --max_repos 30

Programmatic:
    from architect.architecture_miner import mine_for_component
    mine_for_component("Stagehand", max_repos=30)
"""
from __future__ import annotations
import logging
import re
from typing import Iterable

from . import db
from .ingestion import extractors, github_client

logger = logging.getLogger(__name__)


# ─── Repo gathering ──────────────────────────────────────────────────────────
def _component_to_search_queries(component_name: str) -> list[tuple[str, str]]:
    """Build the queries we'll try, paired with their source label.

    Returns [(query, source_label), ...]. We try multiple variants so we
    don't depend on any single phrasing.
    """
    name = component_name.strip()
    quoted = f'"{name}"'
    repo_query = f'{quoted} in:readme stars:>20'
    return [
        (repo_query,                                      "github_repo_readme"),
        # code-search variants are run separately; require gh token
    ]


def _component_to_code_queries(component_name: str) -> list[str]:
    """Code-search queries for this component. We try a few likely import
    forms — the LLM can later filter out false positives."""
    name = component_name.strip()
    queries = [
        f'"from {name.lower()} import"',
        f'"import {name.lower()}"',
        f'"@{name.lower()}/"',
        f'"{name.lower()}.com"',          # API/SDK references
        f'"{name.upper()}_API_KEY"',
    ]
    # de-dupe but preserve order
    seen, out = set(), []
    for q in queries:
        if q not in seen:
            seen.add(q); out.append(q)
    return out


def gather_candidate_repos(component_name: str,
                            max_repos: int = 30) -> list[dict]:
    """Search GitHub for repos that plausibly use this component.

    Returns a list of repo metadata dicts (the GitHub /search/repositories
    response items, with `_source` and `_evidence_query` annotated).
    """
    candidates: dict[str, dict] = {}      # repo_url -> meta

    # Repo readme search
    for query, src in _component_to_search_queries(component_name):
        for item in github_client.search_repos(query, per_page=min(max_repos, 30)):
            url = item.get("html_url") or ""
            if not url or url in candidates:
                continue
            candidates[url] = {
                **item,
                "_source": src,
                "_evidence_query": query,
            }
            if len(candidates) >= max_repos:
                break
        if len(candidates) >= max_repos:
            break

    # Code search — only if we have a token AND we still have budget
    import os
    if os.environ.get("GITHUB_TOKEN") and len(candidates) < max_repos:
        for query in _component_to_code_queries(component_name):
            for item in github_client.search_code(query, per_page=10):
                repo = item.get("repository") or {}
                url = repo.get("html_url") or ""
                if not url or url in candidates:
                    continue
                candidates[url] = {
                    **repo,
                    "_source": "github_code",
                    "_evidence_query": query,
                    "_evidence_path": item.get("path"),
                }
                if len(candidates) >= max_repos:
                    break
            if len(candidates) >= max_repos:
                break

    return list(candidates.values())


# ─── Architecture extraction ─────────────────────────────────────────────────
def ingest_architecture(repo_url: str) -> dict | None:
    """Pull README + dep files, run extract_architecture, upsert.

    Returns the architecture card dict, or None on failure.
    """
    meta = github_client.get_repo_meta(repo_url)
    if not meta:
        logger.info("skipping %s — couldn't fetch metadata", repo_url)
        return None
    readme = github_client.fetch_readme(repo_url) or ""
    deps   = github_client.fetch_dep_excerpt(repo_url)

    card = extractors.extract_architecture(
        repo_url=repo_url,
        stars=int(meta.get("stargazers_count", 0)),
        description=meta.get("description") or "",
        readme_text=readme,
        imports_text=deps,
    )
    if not card:
        return None

    # Upsert into architectures + architecture_components
    with db.connect() as conn:
        aid = db.upsert_architecture(
            conn,
            source="github",
            source_url=repo_url,
            name=card.get("name") or meta.get("name") or repo_url,
            description=meta.get("description") or "",
            summary=card.get("summary", ""),
            pattern=card.get("pattern", ""),
            quality_signal=float(card.get("quality_signal", 0.0))
                * (1.0 if not card.get("is_template_or_demo") else 0.5),
            raw_json={
                "stars": meta.get("stargazers_count"),
                "forks": meta.get("forks_count"),
                "language": meta.get("language"),
                "topics": meta.get("topics", []),
                "evidence_query": meta.get("_evidence_query"),
            },
            last_verified_at=db._now(),
        )

        for comp_ref in (card.get("components_used") or []):
            comp_name = (comp_ref.get("name") or "").strip()
            if not comp_name:
                continue
            slug = re.sub(r"[^a-z0-9]+", "-", comp_name.lower()).strip("-")
            row = db.get_component(conn, slug)
            if not row:
                # Component referenced but not yet in our DB — queue for
                # later enrichment.
                db.enqueue_url(
                    conn,
                    url=f"name:{comp_name}",
                    source="architecture_miner_unknown",
                    priority=0,
                )
                continue
            db.link_architecture_component(
                conn, architecture_id=aid, component_id=row["id"],
                role=comp_ref.get("role", ""),
                evidence=(comp_ref.get("evidence") or "")[:200],
            )

            # Reinforce co-occurrence as relationships in the component graph.
            # For each pair of known components in this architecture, bump
            # an `integrates_with` edge — confidence 0.3 per observation.
            # (Done in a second pass after we collect all known components.)

        # Second pass: pairwise co-occurrence reinforcement
        cur = conn.execute(
            "SELECT component_id FROM architecture_components WHERE architecture_id=?",
            (aid,),
        )
        comp_ids = [row["component_id"] for row in cur]
        for i, a in enumerate(comp_ids):
            for b in comp_ids[i + 1:]:
                db.add_relationship(
                    conn, source_id=a, target_id=b,
                    type="integrates_with",
                    confidence=0.3,
                    evidence_url=repo_url,
                    note=f"co-occurs in {repo_url.split('/')[-1]}",
                )
    return card


# ─── Top-level entry ─────────────────────────────────────────────────────────
def mine_for_component(component_name: str, max_repos: int = 30) -> int:
    """Discover and ingest architectures using this component.

    Returns the count of architectures successfully ingested.
    """
    logger.info("mining architectures for %s (max=%d)",
                component_name, max_repos)
    candidates = gather_candidate_repos(component_name, max_repos=max_repos)
    logger.info("  %d candidate repos gathered", len(candidates))
    n_ingested = 0
    for cand in candidates:
        url = cand.get("html_url") or ""
        if not url:
            continue
        try:
            card = ingest_architecture(url)
        except Exception as e:
            logger.warning("ingest_architecture failed on %s: %s", url, e)
            continue
        if card:
            n_ingested += 1
    logger.info("  ingested %d architectures for %s",
                n_ingested, component_name)
    return n_ingested
