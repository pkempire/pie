"""Thin GitHub API wrapper for enrichment + architecture mining.

We don't pull the full octokit; for our needs (fetch README, search code,
search repos, list dependents) the raw REST API + a User-Agent header is
sufficient. If the user has GITHUB_TOKEN set we use it (5000 req/hour
authenticated vs 60 unauthenticated).
"""
from __future__ import annotations
import logging
import os
import re
from typing import Iterator

import requests

logger = logging.getLogger(__name__)

API = "https://api.github.com"


def _headers() -> dict:
    h = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "architect-bot/0.1 (+research)",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def _parse_repo_url(url: str) -> tuple[str, str] | None:
    """github.com/owner/repo → (owner, repo). Robust to trailing slashes."""
    m = re.match(r"https?://github\.com/([^/]+)/([^/]+?)(?:\.git)?/?$",
                 (url or "").strip())
    return (m.group(1), m.group(2)) if m else None


# ─── Reads ───────────────────────────────────────────────────────────────────
def fetch_readme(repo_url: str) -> str:
    """Return the README text of a repo, or '' if not found."""
    parsed = _parse_repo_url(repo_url)
    if not parsed:
        return ""
    owner, repo = parsed
    r = requests.get(f"{API}/repos/{owner}/{repo}/readme",
                     headers={**_headers(), "Accept": "application/vnd.github.raw"},
                     timeout=20)
    if r.status_code == 200:
        return r.text
    logger.warning("README fetch %s/%s -> %d", owner, repo, r.status_code)
    return ""


def get_repo_meta(repo_url: str) -> dict:
    """Return the repo metadata blob (stars, forks, description, topics, …)."""
    parsed = _parse_repo_url(repo_url)
    if not parsed:
        return {}
    owner, repo = parsed
    r = requests.get(f"{API}/repos/{owner}/{repo}", headers=_headers(),
                     timeout=15)
    return r.json() if r.status_code == 200 else {}


def list_repo_files(repo_url: str, max_files: int = 80) -> list[str]:
    """Top-level + 1 nested level of file paths. Used for inferring
    package.json / pyproject / requirements presence."""
    parsed = _parse_repo_url(repo_url)
    if not parsed:
        return []
    owner, repo = parsed
    out: list[str] = []
    r = requests.get(f"{API}/repos/{owner}/{repo}/contents/", headers=_headers(),
                     timeout=15)
    if r.status_code != 200:
        return []
    for entry in r.json()[:30]:
        out.append(entry["path"])
        if entry.get("type") == "dir" and len(out) < max_files:
            r2 = requests.get(entry["url"], headers=_headers(), timeout=15)
            if r2.status_code == 200:
                for sub in r2.json()[:20]:
                    out.append(sub["path"])
    return out[:max_files]


def fetch_file(repo_url: str, path: str) -> str:
    parsed = _parse_repo_url(repo_url)
    if not parsed:
        return ""
    owner, repo = parsed
    r = requests.get(
        f"{API}/repos/{owner}/{repo}/contents/{path}",
        headers={**_headers(), "Accept": "application/vnd.github.raw"},
        timeout=20,
    )
    return r.text if r.status_code == 200 else ""


# ─── Search ──────────────────────────────────────────────────────────────────
def search_repos(query: str, sort: str = "stars",
                 per_page: int = 30) -> Iterator[dict]:
    """Yield repo dicts for a repo-search query.

    Useful queries:
      - 'topic:browser-agent stars:>50'
      - 'awesome AI agents'
      - 'stagehand in:readme'
    """
    url = f"{API}/search/repositories"
    r = requests.get(url, params={"q": query, "sort": sort,
                                    "per_page": per_page},
                     headers=_headers(), timeout=20)
    if r.status_code != 200:
        logger.warning("repo search %r -> %d", query, r.status_code)
        return
    for item in r.json().get("items", []):
        yield item


def search_code(query: str, per_page: int = 30) -> Iterator[dict]:
    """GitHub code search. Useful for finding USERS of a component:
        'from stagehand import' / 'BROWSERBASE_API_KEY'
    Code search requires authentication.
    """
    if not os.environ.get("GITHUB_TOKEN"):
        logger.warning("GITHUB_TOKEN not set — code search will fail")
    r = requests.get(f"{API}/search/code",
                     params={"q": query, "per_page": per_page},
                     headers=_headers(), timeout=20)
    if r.status_code != 200:
        logger.warning("code search %r -> %d", query, r.status_code)
        return
    for item in r.json().get("items", []):
        yield item


# ─── Dependency-file extraction ──────────────────────────────────────────────
_DEP_FILES = (
    "package.json",
    "requirements.txt",
    "pyproject.toml",
    "Cargo.toml",
    "go.mod",
    "Gemfile",
)


def fetch_dep_excerpt(repo_url: str, max_chars: int = 4000) -> str:
    """Return a concatenated excerpt of any dependency files at the repo
    root. Used as the `imports_text` input to extract_architecture."""
    paths = list_repo_files(repo_url, max_files=80)
    chunks: list[str] = []
    for p in paths:
        name = p.split("/")[-1]
        if name in _DEP_FILES:
            content = fetch_file(repo_url, p)
            if content:
                chunks.append(f"--- {p} ---\n{content[:1500]}")
    return "\n\n".join(chunks)[:max_chars]
