"""Build a Karpathy-spartan static-site wiki from research/{papers,concepts,systems}/.

Reads markdown files with yaml frontmatter from three source directories:
  - research/papers/    — one .md per paper (already exists)
  - research/concepts/  — one .md per concept (e.g. sleep-consolidation.md)
  - research/systems/   — one .md per system (e.g. mem0.md)

Outputs research/wiki/_site/ — static HTML browsable locally or on GitHub Pages.

Backlinks: every page lists all pages that mention it (by slug or title).
Cross-links: anywhere a page mentions [[slug]] or [[Display|slug]], render
as a hyperlink.

Style: single column, max-width 720px, mono headings, plain links. Karpathy
LLM-wiki spartan. No JS.

Usage:
    python -m research.wiki.build
    python -m research.wiki.build --serve   # also runs a local http server on :8800
"""
from __future__ import annotations

import argparse
import collections
import http.server
import logging
import os
import re
import socketserver
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[2]
PAPERS = REPO / "research" / "papers"
CONCEPTS = REPO / "research" / "concepts"
SYSTEMS = REPO / "research" / "systems"
GOALS = REPO / "research" / "goals"
CONTENT = REPO / "research" / "content"
SITE = REPO / "research" / "wiki" / "_site"
SITE.mkdir(parents=True, exist_ok=True)

WIKILINK = re.compile(r"\[\[([^\]|]+?)(?:\|([^\]]+?))?\]\]")


@dataclass
class Page:
    kind: str               # "paper" | "concept" | "system"
    slug: str               # filename without .md
    title: str
    fm: dict[str, Any] = field(default_factory=dict)
    body: str = ""
    outbound: set[str] = field(default_factory=set)   # slugs this page links to

    @property
    def href(self) -> str:
        return f"{self.kind}s/{self.slug}.html"


def _parse_fm(text: str) -> tuple[dict[str, Any], str]:
    """Minimal yaml frontmatter parser (mirrors research/scripts/aggregate.py)."""
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end < 0:
        return {}, text
    raw_fm = text[3:end].lstrip("\n")
    body = text[end + 4:].lstrip("\n")

    fm: dict[str, Any] = {}
    pending_key: str | None = None
    for line in raw_fm.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if line.startswith("  - "):
            val = _strip_inline(line[4:])
            if pending_key:
                if fm.get(pending_key) is None:
                    fm[pending_key] = []
                fm[pending_key].append(val)
            continue
        m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*):\s*(.*?)(\s*#.*)?$", line)
        if not m:
            continue
        key, raw_val = m.group(1), m.group(2).strip()
        if raw_val in ("", "[]"):
            fm[key] = [] if raw_val == "[]" else None
            pending_key = key if raw_val == "" else None
            continue
        if raw_val == "null":
            fm[key] = None
            pending_key = None
            continue
        if raw_val.startswith("["):
            fm[key] = [_strip_inline(x) for x in raw_val.strip("[]").split(",") if x.strip()]
            pending_key = None
            continue
        fm[key] = _strip_inline(raw_val)
        pending_key = key
    return fm, body


def _strip_inline(s: str) -> Any:
    s = s.strip()
    if s.startswith('"') and s.endswith('"'):
        return s[1:-1]
    if s.startswith("'") and s.endswith("'"):
        return s[1:-1]
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return s


def load_pages() -> dict[str, Page]:
    """Load all pages from the four source dirs, keyed by slug."""
    pages: dict[str, Page] = {}
    for kind, dir_ in [("paper", PAPERS), ("concept", CONCEPTS),
                        ("system", SYSTEMS), ("goal", GOALS),
                        ("content", CONTENT)]:
        if not dir_.exists():
            continue
        for p in sorted(dir_.glob("*.md")):
            text = p.read_text()
            fm, body = _parse_fm(text)
            # Slug = filename stem. For goals, also accept short forms like
            # `goal-01-gepa-consolidator-on-locomo` in wiki links.
            slug = p.stem
            title = str(fm.get("title", slug))
            page = Page(kind=kind, slug=slug, title=title, fm=fm, body=body)
            pages[slug] = page
            # Add alias: drop the "NN-" numeric prefix so links can be shorter
            if kind == "goal":
                short = re.sub(r"^\d+-", "goal-", slug)
                if short != slug and short not in pages:
                    pages[short] = page
    return pages


def find_links(body: str, all_slugs: set[str]) -> set[str]:
    """Find all [[slug]] or [[slug|Display]] references in body that match a real slug.

    Convention: slug is first, optional display label is second.
    [[mem0]] or [[mem0|Mem0 system]] both link to slug "mem0".
    """
    found: set[str] = set()
    for m in WIKILINK.finditer(body):
        target = m.group(1).strip()
        if target in all_slugs:
            found.add(target)
    return found


def render_wikilinks(body: str, pages: dict[str, Page]) -> str:
    """Replace [[slug]] / [[slug|Display]] with <a href=...>Display</a>.

    Convention: slug first, optional display second. If display is omitted,
    use the target page's title.
    """
    def replace(m: re.Match) -> str:
        target = m.group(1).strip()
        display = (m.group(2) or "").strip()
        if target in pages:
            page = pages[target]
            if not display:
                display = page.title
            href = "../" + page.href
            return f'<a href="{href}">{display}</a>'
        return f'<span class="dead-link">{display or target}</span>'
    return WIKILINK.sub(replace, body)


def markdown_to_html(body: str) -> str:
    """Minimal markdown renderer. We don't need a full library for our use case."""
    try:
        import markdown   # type: ignore
        return markdown.markdown(body, extensions=["fenced_code", "tables"])
    except ImportError:
        # Fallback: pre-wrap everything. Loses headings/lists but readable.
        escaped = body.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        return f"<pre>{escaped}</pre>"


# ─── Templates ─────────────────────────────────────────────────────────────

CSS = """
* { box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
       max-width: 720px; margin: 0 auto; padding: 2rem 1.25rem 4rem;
       color: #1a1a1a; line-height: 1.55; font-size: 16px; }
h1, h2, h3 { font-family: ui-monospace, "SF Mono", Monaco, monospace;
             font-weight: 500; line-height: 1.3; margin-top: 1.5em; }
h1 { font-size: 22px; margin-top: 0; }
h2 { font-size: 18px; border-top: 1px solid #eee; padding-top: 1em; }
h3 { font-size: 16px; }
a { color: #0a58ca; text-decoration: none; border-bottom: 1px dotted #0a58ca; }
a:hover { background: #f0f4ff; }
.dead-link { color: #a00; text-decoration: line-through; }
nav { font-size: 13px; color: #666; margin-bottom: 1.5rem;
      padding-bottom: 0.75rem; border-bottom: 1px solid #eee; }
nav a { color: #666; border-bottom: none; margin-right: 1em; }
.meta { font-size: 13px; color: #666; margin-bottom: 1rem; }
.meta dt { font-weight: 500; color: #444; display: inline-block; min-width: 8em; }
.meta dd { display: inline; margin: 0 1.5em 0 0; }
.meta div { margin: 4px 0; }
code { font-family: ui-monospace, "SF Mono", Monaco, monospace;
       background: #f5f5f5; padding: 1px 5px; border-radius: 3px; font-size: 14px; }
pre { background: #f5f5f5; padding: 12px; border-radius: 6px; overflow-x: auto;
      font-family: ui-monospace, "SF Mono", Monaco, monospace; font-size: 13px;
      line-height: 1.5; }
pre code { background: none; padding: 0; }
table { border-collapse: collapse; margin: 1em 0; }
th, td { border: 1px solid #ddd; padding: 6px 12px; font-size: 14px; }
th { background: #f5f5f5; font-weight: 500; }
blockquote { border-left: 3px solid #ddd; margin: 1em 0; padding: 0.25em 1em;
             color: #555; font-style: italic; }
.backlinks { margin-top: 3rem; padding-top: 1rem; border-top: 1px solid #eee;
             font-size: 13px; color: #666; }
.backlinks h3 { font-size: 14px; margin: 0 0 0.5em; color: #444; }
.backlinks ul { list-style: none; padding-left: 0; }
.backlinks li { padding: 2px 0; }
.tag-list { font-size: 12px; color: #888; margin: 4px 0; }
.tag-list span { background: #f0f0f0; padding: 2px 8px; border-radius: 3px;
                 margin-right: 4px; display: inline-block; margin-bottom: 4px; }
ul, ol { padding-left: 1.5em; }
li { margin: 4px 0; }
"""

NAV_HTML = """<nav>
<a href="../index.html">home</a>
<a href="../goals.html">goals</a>
<a href="../papers.html">papers</a>
<a href="../concepts.html">concepts</a>
<a href="../systems.html">systems</a>
<a href="../contents.html">content</a>
<a href="https://github.com/USERNAME/REPO" target="_blank" rel="noopener">source</a>
</nav>"""


def render_page(page: Page, pages: dict[str, Page], backlinks: list[Page]) -> str:
    # Body with wiki-links rendered
    body_with_links = render_wikilinks(page.body, pages)
    body_html = markdown_to_html(body_with_links)

    # Meta block (frontmatter that's worth surfacing)
    meta_html = []
    if page.kind == "paper":
        fm = page.fm
        if fm.get("authors"):
            authors = ", ".join(fm["authors"][:4])
            if len(fm.get("authors", [])) > 4:
                authors += " et al."
            meta_html.append(f"<div><dt>Authors</dt><dd>{authors}</dd></div>")
        if fm.get("date_published"):
            meta_html.append(f"<div><dt>Published</dt><dd>{fm['date_published']}</dd></div>")
        if fm.get("arxiv_id"):
            meta_html.append(f'<div><dt>arXiv</dt><dd><a href="https://arxiv.org/abs/{fm["arxiv_id"]}">{fm["arxiv_id"]}</a></dd></div>')
        if fm.get("approach_class"):
            meta_html.append(f"<div><dt>Approach</dt><dd><code>{fm['approach_class']}</code></dd></div>")
        if fm.get("relevance"):
            meta_html.append(f"<div><dt>Relevance</dt><dd>{fm['relevance']}</dd></div>")
        if fm.get("benchmarks"):
            meta_html.append(f"<div><dt>Benchmarks</dt><dd>{', '.join(fm['benchmarks'])}</dd></div>")
    elif page.kind == "goal":
        fm = page.fm
        status = fm.get("status", "?")
        meta_html.append(f"<div><dt>Status</dt><dd><code>{status}</code></dd></div>")
        if fm.get("started"):
            meta_html.append(f"<div><dt>Started</dt><dd>{fm['started']}</dd></div>")
        if fm.get("priority"):
            meta_html.append(f"<div><dt>Priority</dt><dd>{fm['priority']}</dd></div>")
        if fm.get("budget"):
            meta_html.append(f"<div><dt>Budget</dt><dd>{fm['budget']}</dd></div>")
    elif page.kind == "concept":
        fm = page.fm
        if fm.get("our_status"):
            meta_html.append(f"<div><dt>Our status</dt><dd><code>{fm['our_status']}</code></dd></div>")
        if fm.get("category"):
            meta_html.append(f"<div><dt>Category</dt><dd>{fm['category']}</dd></div>")

    meta_block = ""
    if meta_html:
        meta_block = '<dl class="meta">' + "".join(meta_html) + "</dl>"

    # Tags
    tags = page.fm.get("tags") or []
    tags_block = ""
    if tags:
        tags_block = '<div class="tag-list">' + "".join(f"<span>{t}</span>" for t in tags) + "</div>"

    # Backlinks
    backlinks_block = ""
    if backlinks:
        items = "\n".join(
            f'  <li><a href="../{p.href}">{p.title}</a> <span style="color:#999">[{p.kind}]</span></li>'
            for p in sorted(backlinks, key=lambda x: x.title)
        )
        backlinks_block = f'<div class="backlinks"><h3>Referenced by</h3><ul>\n{items}\n</ul></div>'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{page.title}</title>
<style>{CSS}</style>
</head>
<body>
{NAV_HTML}
{meta_block}
{tags_block}
{body_html}
{backlinks_block}
</body>
</html>"""


def render_index_page(title: str, pages: list[Page], extra_intro: str = "") -> str:
    items = []
    for p in sorted(pages, key=lambda x: x.title):
        approach = p.fm.get("approach_class") or p.fm.get("category") or ""
        approach_html = f' <span style="color:#999;font-size:13px">[{approach}]</span>' if approach else ""
        year = p.fm.get("year")
        year_html = f' <span style="color:#999;font-size:13px">({year})</span>' if year else ""
        items.append(f'<li><a href="{p.href}">{p.title}</a>{year_html}{approach_html}</li>')
    items_html = "\n".join(items)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>{CSS}</style>
</head>
<body>
{NAV_HTML.replace('../', '')}
<h1>{title}</h1>
{extra_intro}
<ul>
{items_html}
</ul>
</body>
</html>"""


def render_home(by_kind: dict[str, list[Page]], goals: list[Page] | None = None) -> str:
    n_papers = len(by_kind.get("paper", []))
    n_concepts = len(by_kind.get("concept", []))
    n_systems = len(by_kind.get("system", []))
    n_content = len(by_kind.get("content", []))
    n_goals = len(goals) if goals else 0

    # Active-goal block — surface the top 3 actives on the home page
    actives = [g for g in (goals or []) if g.fm.get("status") == "active"][:3]
    actives_html = ""
    if actives:
        items = "\n".join(
            f'<li><a href="goals/{g.slug}.html">{g.title}</a> '
            f'<span style="color:#999;font-size:13px">— priority {g.fm.get("priority","?")}</span></li>'
            for g in actives
        )
        actives_html = f"<h2>Active goals</h2>\n<ul>\n{items}\n</ul>"

    # Video scripts (content) block
    content_pages = [c for c in by_kind.get("content", [])
                      if c.slug != "README"]
    content_html = ""
    if content_pages:
        items = "\n".join(
            f'<li><a href="contents/{c.slug}.html">{c.title}</a> '
            f'<span style="color:#999;font-size:13px">— {c.fm.get("status","?")}</span></li>'
            for c in content_pages
        )
        content_html = f'<h2>Working Memory — video scripts</h2>\n<ul>\n{items}\n</ul>'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>AI memory research wiki</title>
<style>{CSS}</style>
</head>
<body>
<nav>
<a href="index.html">home</a>
<a href="goals.html">goals</a>
<a href="papers.html">papers</a>
<a href="concepts.html">concepts</a>
<a href="systems.html">systems</a>
<a href="contents.html">content</a>
</nav>
<h1>AI memory research</h1>
<p>A working wiki of papers, concepts, systems, active research goals, and Working Memory video scripts on memory for LLM agents.</p>
<p>State as of build: <strong>{n_papers}</strong> papers, <strong>{n_concepts}</strong> concepts, <strong>{n_systems}</strong> systems, <strong>{n_goals}</strong> goals, <strong>{n_content}</strong> content pieces.</p>
{actives_html}
{content_html}
<h2>Start here</h2>
<ul>
<li><a href="concepts/sleep-consolidation.html">Sleep consolidation</a> — the asynchronous-write architecture</li>
<li><a href="concepts/write-time-vs-read-time.html">Write-time vs read-time compression</a> — the design-space axis</li>
<li><a href="concepts/noreplay-vs-retrieval.html">NoReplay vs retrieval-flavored memory</a> — Yang's framing</li>
<li><a href="concepts/gepa-vs-grpo.html">GEPA vs GRPO</a> — natural-language gradient vs scalar gradient</li>
<li><a href="papers/2605.20616-auto-dreamer.html">Auto-Dreamer</a> — current consolidator SOTA</li>
<li><a href="papers/2601.02163-evermemos.html">EverMemOS</a> — current LoCoMo leaderboard #1 at 93.05% (table-extracted)</li>
</ul>
<h2>All</h2>
<ul>
<li><a href="goals.html">All goals ({n_goals})</a></li>
<li><a href="papers.html">All papers ({n_papers})</a></li>
<li><a href="concepts.html">All concepts ({n_concepts})</a></li>
<li><a href="systems.html">All systems ({n_systems})</a></li>
<li><a href="contents.html">All content ({n_content})</a></li>
</ul>
</body>
</html>"""


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serve", action="store_true", help="serve _site/ at http://localhost:8800")
    args = parser.parse_args()

    pages = load_pages()
    if not pages:
        logger.warning("No pages found. Did you write any?")
        return

    all_slugs = set(pages.keys())

    # Compute outbound links per page
    for page in pages.values():
        page.outbound = find_links(page.body, all_slugs)

    # Compute backlinks (inverse of outbound)
    backlinks: dict[str, list[Page]] = collections.defaultdict(list)
    for page in pages.values():
        for tgt in page.outbound:
            backlinks[tgt].append(page)

    # Output dir layout
    for kind in ("paper", "concept", "system", "goal", "content"):
        (SITE / f"{kind}s").mkdir(parents=True, exist_ok=True)

    # Render every page
    for page in pages.values():
        out_path = SITE / page.href
        out_path.parent.mkdir(parents=True, exist_ok=True)
        html = render_page(page, pages, backlinks.get(page.slug, []))
        out_path.write_text(html)

    # Index pages
    by_kind: dict[str, list[Page]] = collections.defaultdict(list)
    for page in pages.values():
        by_kind[page.kind].append(page)

    (SITE / "papers.html").write_text(render_index_page(
        "Papers", by_kind.get("paper", []),
        '<p>One file per paper. yaml frontmatter (arxiv id, approach class, benchmarks, results, what to steal, limitations) plus prose body.</p>'
    ))
    (SITE / "concepts.html").write_text(render_index_page(
        "Concepts", by_kind.get("concept", []),
        '<p>One page per concept (architecture pattern, training technique, evaluation axis). Concepts with <code>our_status</code> set to <code>active-pursuit</code> are ones we are currently working on.</p>'
    ))
    (SITE / "systems.html").write_text(render_index_page(
        "Systems", by_kind.get("system", []),
        '<p>One page per memory system (production or research). Per-system summary, cross-linked to papers and concepts.</p>'
    ))
    (SITE / "contents.html").write_text(render_index_page(
        "Content / video scripts", by_kind.get("content", []),
        '<p>One page per Working Memory video script. Outlines, assets, production notes, citations.</p>'
    ))
    # Render goals — but sort by priority + status (active first)
    goals = by_kind.get("goal", [])
    # Deduplicate (the alias mechanism may produce two entries for the same goal)
    seen_paths = set()
    unique_goals = []
    for g in goals:
        if g.slug not in seen_paths:
            seen_paths.add(g.slug)
            unique_goals.append(g)
    goal_order = {"active": 0, "planned": 1, "parked": 2, "done": 3, "?": 4}
    unique_goals.sort(key=lambda g: (
        goal_order.get(g.fm.get("status", "?"), 99),
        int(g.fm.get("priority", 99)),
    ))
    (SITE / "goals.html").write_text(render_index_page(
        "Goals", unique_goals,
        '<p>What we are actively working on, sorted by status (active → planned → parked → done) then priority.</p>'
    ))
    (SITE / "index.html").write_text(render_home(by_kind, unique_goals))

    logger.info("Built %d pages → %s", len(pages) + 4, SITE)
    logger.info("  papers:   %d", len(by_kind.get("paper", [])))
    logger.info("  concepts: %d", len(by_kind.get("concept", [])))
    logger.info("  systems:  %d", len(by_kind.get("system", [])))

    if args.serve:
        os.chdir(SITE)
        port = 8800
        handler = http.server.SimpleHTTPRequestHandler
        with socketserver.TCPServer(("", port), handler) as httpd:
            logger.info("Serving %s at http://localhost:%d", SITE, port)
            httpd.serve_forever()


if __name__ == "__main__":
    main()
