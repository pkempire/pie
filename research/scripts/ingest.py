"""Ingest a paper from arXiv into `research/papers/`.

Given an arxiv ID, this:
  1. Fetches abstract + intro + key sections via the arxiv API
  2. Asks an LLM to fill the schema fields
  3. Writes `research/papers/<id>-<slug>.md` with yaml frontmatter
  4. Refuses to overwrite an existing file unless --force is passed
     (so you can edit by hand and not lose work)

Usage:
    python -m research.scripts.ingest 2508.19828
    python -m research.scripts.ingest 2503.09516 --model gpt-5
    python -m research.scripts.ingest 2508.19828 --force          # overwrite
    python -m research.scripts.ingest --batch arxiv_ids.txt        # one per line

You can ingest non-arxiv sources too:
    python -m research.scripts.ingest --url https://raw.works/recursive-language-models-as-memory-systems/

The LLM call is the only part that needs an API key. If OPENAI_API_KEY isn't
set, the script will still fetch + write a stub file with the abstract; you
can fill the rest by hand.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[2]
PAPERS_DIR = REPO / "research" / "papers"
PAPERS_DIR.mkdir(parents=True, exist_ok=True)


# ─── Required schema fields (see research/SCHEMA.md) ─────────────────────────

APPROACH_CLASSES = {
    "write-time-compression", "read-time-decompression",
    "RL-for-memory", "RL-for-tool-use", "agent-orchestration",
    "temporal-reasoning", "benchmark", "substrate", "theory",
    "infrastructure", "survey",
}

REWARD_SHAPES = {
    "trajectory-level", "per-op-state-distance",
    "per-op-outcome-attribution", "verbal-reflection",
    "supervised", "none",
}


# ─── arXiv fetch ─────────────────────────────────────────────────────────────

def _arxiv_id_to_slug(arxiv_id: str, title: str) -> str:
    """Turn a title into a short URL-safe slug for the filename."""
    base = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    base = base[:48].rstrip("-") or "paper"
    return f"{arxiv_id}-{base}"


def fetch_arxiv_metadata(arxiv_id: str) -> dict[str, Any]:
    """Hit arxiv.org/abs API and parse out title, authors, abstract, date.

    Uses the public arxiv API:
      http://export.arxiv.org/api/query?id_list=<ID>
    """
    import urllib.request
    import xml.etree.ElementTree as ET

    url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
    req = urllib.request.Request(url, headers={"User-Agent": "mempol-research/0.1"})
    with urllib.request.urlopen(req, timeout=30) as r:
        body = r.read().decode("utf-8")

    ns = {"a": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(body)
    entry = root.find("a:entry", ns)
    if entry is None:
        raise ValueError(f"arxiv returned no entry for {arxiv_id}")

    title = (entry.findtext("a:title", default="", namespaces=ns) or "").strip()
    abstract = (entry.findtext("a:summary", default="", namespaces=ns) or "").strip()
    published = (entry.findtext("a:published", default="", namespaces=ns) or "").strip()
    authors = [
        (a.findtext("a:name", default="", namespaces=ns) or "").strip()
        for a in entry.findall("a:author", ns)
    ]
    year = int(published[:4]) if published else None
    return {
        "title": re.sub(r"\s+", " ", title),
        "abstract": re.sub(r"\s+", " ", abstract),
        "authors": [a for a in authors if a],
        "date_published": published[:10] if published else "",
        "year": year,
        "arxiv_id": arxiv_id,
    }


def fetch_url(url: str) -> dict[str, Any]:
    """Fetch a non-arxiv URL (raw.works blog, github README, etc.) and
    return a stub metadata dict. Body text is best-effort scrape."""
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "mempol-research/0.1"})
    with urllib.request.urlopen(req, timeout=30) as r:
        body = r.read().decode("utf-8", errors="ignore")
    # Crude: strip tags, keep first 8k chars.
    text = re.sub(r"<[^>]+>", " ", body)
    text = re.sub(r"\s+", " ", text)[:8000]
    netloc = urlparse(url).netloc
    return {
        "title": f"[{netloc}] {url}",
        "abstract": text,
        "authors": [],
        "date_published": "",
        "year": datetime.now().year,
        "arxiv_id": "",
        "source_url": url,
    }


# ─── LLM extraction ──────────────────────────────────────────────────────────

EXTRACT_PROMPT = """You are filling out a paper-summary schema from an abstract. Be precise. Use only what's in the abstract — if a field can't be determined, set it to null.

Schema to fill (JSON):

{{
  "approach_class": "<one of: write-time-compression, read-time-decompression, RL-for-memory, RL-for-tool-use, agent-orchestration, temporal-reasoning, benchmark, substrate, theory, infrastructure, survey>",
  "problem": "<one sentence: what problem does this paper address>",
  "approach": "<one sentence: what method does it use>",
  "benchmarks": ["<benchmark name>", ...],
  "results": ["<headline numerical claim verbatim from abstract>", ...],
  "reward_shape": "<one of: trajectory-level, per-op-state-distance, per-op-outcome-attribution, verbal-reflection, supervised, none — ONLY for papers that train a policy>",
  "base_model": "<base LLM if mentioned, else null>",
  "limitations": ["<limitation they admit in abstract>", ...],
  "tags": ["<free-form tags for grep>", ...]
}}

ABSTRACT:
{abstract}

Return ONLY the JSON, no preamble."""


def llm_extract(abstract: str, model: str = "gpt-5-mini") -> dict[str, Any]:
    """Call an LLM to fill the schema fields. If no API key, return stubs."""
    if not os.getenv("OPENAI_API_KEY"):
        logger.warning("OPENAI_API_KEY not set — returning stub schema; "
                       "fill in research/papers/ files by hand.")
        return {
            "approach_class": None, "problem": None, "approach": None,
            "benchmarks": [], "results": [], "reward_shape": None,
            "base_model": None, "limitations": [], "tags": [],
        }
    try:
        from openai import OpenAI
    except ImportError:
        logger.error("`pip install openai` to enable LLM extraction.")
        return {}

    client = OpenAI()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a precise research-paper summarizer that returns only valid JSON matching the schema you're given."},
            {"role": "user", "content": EXTRACT_PROMPT.format(abstract=abstract)},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    raw = resp.choices[0].message.content or "{}"
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.error("LLM returned non-JSON: %s", raw[:200])
        return {}


# ─── Writing the markdown file ───────────────────────────────────────────────

def _yaml_value(v: Any) -> str:
    """Tiny yaml emitter so we don't depend on PyYAML."""
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, list):
        if not v:
            return "[]"
        return "\n" + "\n".join(f"  - " + _yaml_inline(x) for x in v)
    return _yaml_inline(v)


def _yaml_inline(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, (int, float, bool)):
        return str(v).lower() if isinstance(v, bool) else str(v)
    s = str(v).replace('"', '\\"').replace("\n", " ").strip()
    return f'"{s}"'


def render_frontmatter(meta: dict[str, Any], extracted: dict[str, Any],
                        relevance: str = "?",
                        relevance_reason: str = "[fill in by hand]") -> str:
    """Build the yaml frontmatter block."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    lines = ["---"]
    lines.append(f'arxiv_id: "{meta.get("arxiv_id", "")}"')
    lines.append(f'title: "{meta.get("title", "").replace(chr(34), chr(39))}"')
    lines.append(f"authors: {_yaml_value(meta.get('authors', []))}")
    if meta.get("year"):
        lines.append(f"year: {meta['year']}")
    if meta.get("date_published"):
        lines.append(f'date_published: "{meta["date_published"]}"')
    lines.append(f'date_ingested: "{now}"')
    if meta.get("source_url"):
        lines.append(f'source_url: "{meta["source_url"]}"')
    lines.append("")

    # Extracted by LLM
    cls = extracted.get("approach_class")
    if cls and cls not in APPROACH_CLASSES:
        logger.warning("approach_class %r not in taxonomy; setting to null", cls)
        cls = None
    lines.append(f'approach_class: {_yaml_inline(cls)}')
    lines.append(f'problem: {_yaml_inline(extracted.get("problem"))}')
    lines.append(f'approach: {_yaml_inline(extracted.get("approach"))}')
    lines.append(f"benchmarks: {_yaml_value(extracted.get('benchmarks', []))}")
    lines.append(f"results: {_yaml_value(extracted.get('results', []))}")
    rs = extracted.get("reward_shape")
    if rs and rs not in REWARD_SHAPES:
        rs = None
    lines.append(f"reward_shape: {_yaml_inline(rs)}")
    lines.append(f"base_model: {_yaml_inline(extracted.get('base_model'))}")
    lines.append("")

    # Human-filled
    lines.append(f'relevance: "{relevance}"          # high | medium | low | ?')
    lines.append(f'relevance_reason: {_yaml_inline(relevance_reason)}')
    lines.append("steal: []  # things to potentially steal — fill by hand")
    lines.append(f"limitations: {_yaml_value(extracted.get('limitations', []))}")
    lines.append(f"tags: {_yaml_value(extracted.get('tags', []))}")
    lines.append("---")
    return "\n".join(lines)


def render_body(meta: dict[str, Any], extracted: dict[str, Any]) -> str:
    """Body markdown with section stubs ready for human editing."""
    title = meta.get("title", "(untitled)")
    abstract = meta.get("abstract", "")
    out = []
    out.append(f"# {title}\n")
    out.append("## Quick read\n")
    out.append(f"_LLM-extracted approach:_ {extracted.get('approach') or '_(not extracted)_'}\n")
    out.append("\n_(fill in by hand: a 2-sentence summary for someone who has read 5 other papers in this area)_\n")
    out.append("\n## Why it matters to us\n")
    out.append("_(fill in by hand: concrete connection to our project; what number on what benchmark we'd compete on)_\n")
    out.append("\n## Method in one paragraph\n")
    out.append("_(fill in by hand: op vocabulary, env shape, reward signal, training setup)_\n")
    out.append("\n## Results in numbers\n")
    results = extracted.get("results", []) or []
    if results:
        for r in results:
            out.append(f"- {r}")
    else:
        out.append("_(none extracted; fill by hand)_")
    out.append("\n\n## What they don't do\n")
    limits = extracted.get("limitations", []) or []
    if limits:
        for l in limits:
            out.append(f"- {l}")
    else:
        out.append("_(fill in by hand)_")
    out.append("\n\n## Open questions / followups\n")
    out.append("_(fill in by hand)_\n")
    out.append("\n## Abstract (verbatim, for reference)\n")
    out.append("> " + textwrap.fill(abstract, width=78).replace("\n", "\n> "))
    return "\n".join(out)


def file_path_for(meta: dict[str, Any]) -> Path:
    aid = meta.get("arxiv_id") or ""
    title = meta.get("title", "paper")
    if aid:
        slug = _arxiv_id_to_slug(aid, title)
    else:
        # for non-arxiv sources, use the netloc + a date
        netloc = urlparse(meta.get("source_url", "")).netloc or "url"
        slug = f"{datetime.now().strftime('%Y%m%d')}-{netloc}"
    return PAPERS_DIR / f"{slug}.md"


def ingest_one(arxiv_id: str | None, url: str | None,
                model: str, force: bool, relevance: str | None) -> Path:
    if arxiv_id:
        meta = fetch_arxiv_metadata(arxiv_id)
    elif url:
        meta = fetch_url(url)
    else:
        raise ValueError("need arxiv_id or url")

    path = file_path_for(meta)
    if path.exists() and not force:
        logger.info("skip: %s exists (pass --force to overwrite)", path)
        return path

    extracted = llm_extract(meta["abstract"], model=model)
    fm = render_frontmatter(
        meta, extracted,
        relevance=relevance or "?",
        relevance_reason="(fill in by hand)",
    )
    body = render_body(meta, extracted)
    path.write_text(fm + "\n\n" + body + "\n")
    logger.info("wrote %s", path)
    return path


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(level=logging.INFO,
                         format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("arxiv_id", nargs="?", help="arXiv ID (e.g. 2508.19828)")
    p.add_argument("--url", help="non-arxiv URL to ingest")
    p.add_argument("--batch", type=Path,
                    help="text file with one arxiv id per line")
    p.add_argument("--model", default="gpt-5-mini",
                    help="LLM model for extraction")
    p.add_argument("--force", action="store_true",
                    help="overwrite existing paper file")
    p.add_argument("--relevance", choices=["high", "medium", "low"],
                    help="pre-fill the relevance field")
    args = p.parse_args()

    if args.batch:
        ids = [line.strip() for line in args.batch.read_text().splitlines()
               if line.strip() and not line.startswith("#")]
        for aid in ids:
            try:
                ingest_one(aid, None, args.model, args.force, args.relevance)
            except Exception as e:
                logger.error("FAIL %s: %s", aid, e)
        return

    if not args.arxiv_id and not args.url:
        p.error("supply arxiv_id or --url or --batch")

    ingest_one(args.arxiv_id, args.url, args.model, args.force, args.relevance)


if __name__ == "__main__":
    main()
