"""Read all `research/papers/*.md`, regenerate the grouped views + STATUS.md.

Outputs:
  research/STATUS.md          — overview of every paper, completeness flags
  research/groups/by_approach.md
  research/groups/by_benchmark.md
  research/groups/by_year.md
  research/groups/by_relevance.md

Usage:
  python -m research.scripts.aggregate
"""
from __future__ import annotations

import argparse
import collections
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[2]
PAPERS_DIR = REPO / "research" / "papers"
GROUPS_DIR = REPO / "research" / "groups"
STATUS_FILE = REPO / "research" / "STATUS.md"
GROUPS_DIR.mkdir(parents=True, exist_ok=True)


REQUIRED_FIELDS = ("arxiv_id", "title", "year", "approach_class", "problem", "approach")
PREFERRED_FIELDS = ("benchmarks", "results")


@dataclass
class Paper:
    path: Path
    fm: dict[str, Any] = field(default_factory=dict)
    body: str = ""

    @property
    def title(self) -> str:
        return str(self.fm.get("title", self.path.stem))

    @property
    def arxiv_id(self) -> str:
        return str(self.fm.get("arxiv_id", ""))

    @property
    def year(self) -> int | None:
        y = self.fm.get("year")
        try:
            return int(y) if y is not None else None
        except (ValueError, TypeError):
            return None

    @property
    def approach_class(self) -> str:
        return str(self.fm.get("approach_class") or "uncategorized")

    @property
    def benchmarks(self) -> list[str]:
        b = self.fm.get("benchmarks") or []
        return list(b) if isinstance(b, list) else []

    @property
    def relevance(self) -> str:
        return str(self.fm.get("relevance") or "?")

    def missing_required(self) -> list[str]:
        return [f for f in REQUIRED_FIELDS
                if not self.fm.get(f) or self.fm.get(f) in ("null", "")]

    def missing_preferred(self) -> list[str]:
        return [f for f in PREFERRED_FIELDS
                if not self.fm.get(f) or self.fm.get(f) in ([], "null", "")]

    def link(self) -> str:
        """Markdown link for cross-referencing in group views."""
        rel = self.path.relative_to(REPO / "research").as_posix()
        return f"[{self.title}]({rel})"


# ─── YAML frontmatter parser (small, no PyYAML dep) ──────────────────────────

def parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Split a markdown file into (fm_dict, body). Supports the subset
    of yaml our schema emits — strings, ints, lists of strings."""
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
            # list item continuing previous key
            val = _strip_inline(line[4:])
            if pending_key:
                # If previous line set the key to None (empty value, list to follow),
                # promote it to a fresh list before appending.
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
            # inline list (best-effort)
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


# ─── Load all papers ─────────────────────────────────────────────────────────

def load_papers() -> list[Paper]:
    papers: list[Paper] = []
    for p in sorted(PAPERS_DIR.glob("*.md")):
        text = p.read_text()
        fm, body = parse_frontmatter(text)
        papers.append(Paper(path=p, fm=fm, body=body))
    return papers


# ─── Grouped views ───────────────────────────────────────────────────────────

def write_by_approach(papers: list[Paper]) -> None:
    by: dict[str, list[Paper]] = collections.defaultdict(list)
    for p in papers:
        by[p.approach_class].append(p)
    lines = ["# Papers by approach class\n",
              "_(auto-generated by `research.scripts.aggregate`. Edit papers/*.md to change.)_\n"]
    for cls in sorted(by):
        lines.append(f"\n## {cls}  ({len(by[cls])})\n")
        for p in sorted(by[cls], key=lambda x: -(x.year or 0)):
            year = p.year or "????"
            rel = p.relevance
            problem = p.fm.get("problem") or ""
            lines.append(f"- **{p.title}** ({year}, rel={rel}) — {problem}")
            lines.append(f"  → {p.link()}")
    (GROUPS_DIR / "by_approach.md").write_text("\n".join(lines) + "\n")


def write_by_benchmark(papers: list[Paper]) -> None:
    by: dict[str, list[Paper]] = collections.defaultdict(list)
    for p in papers:
        for b in p.benchmarks:
            by[b].append(p)
        if not p.benchmarks:
            by["(no benchmark listed)"].append(p)
    lines = ["# Papers by benchmark\n",
              "_(auto-generated. Edit papers/*.md to change.)_\n"]
    for bench in sorted(by):
        lines.append(f"\n## {bench}  ({len(by[bench])})\n")
        for p in sorted(by[bench], key=lambda x: -(x.year or 0)):
            results = p.fm.get("results") or []
            res_str = "; ".join(results) if isinstance(results, list) else str(results)
            lines.append(f"- **{p.title}** ({p.year or '?'}) — {res_str or '_(no results extracted)_'}")
            lines.append(f"  → {p.link()}")
    (GROUPS_DIR / "by_benchmark.md").write_text("\n".join(lines) + "\n")


def write_by_year(papers: list[Paper]) -> None:
    by: dict[int, list[Paper]] = collections.defaultdict(list)
    for p in papers:
        by[p.year or 0].append(p)
    lines = ["# Papers by year\n",
              "_(auto-generated. Edit papers/*.md to change.)_\n"]
    for y in sorted(by, reverse=True):
        lines.append(f"\n## {y or 'unknown'}  ({len(by[y])})\n")
        for p in by[y]:
            lines.append(f"- [{p.approach_class}] **{p.title}** — {p.link()}")
    (GROUPS_DIR / "by_year.md").write_text("\n".join(lines) + "\n")


def write_by_relevance(papers: list[Paper]) -> None:
    by: dict[str, list[Paper]] = collections.defaultdict(list)
    for p in papers:
        by[p.relevance].append(p)
    order = {"high": 0, "medium": 1, "low": 2, "?": 3}
    lines = ["# Papers by relevance to our project\n",
              "_(auto-generated. Edit papers/*.md to change.)_\n"]
    for rel in sorted(by, key=lambda r: order.get(r, 99)):
        lines.append(f"\n## {rel}  ({len(by[rel])})\n")
        for p in sorted(by[rel], key=lambda x: -(x.year or 0)):
            reason = p.fm.get("relevance_reason") or ""
            lines.append(f"- **{p.title}** ({p.year or '?'}, {p.approach_class}) — {reason}")
            lines.append(f"  → {p.link()}")
    (GROUPS_DIR / "by_relevance.md").write_text("\n".join(lines) + "\n")


# ─── STATUS.md ───────────────────────────────────────────────────────────────

def write_status(papers: list[Paper]) -> None:
    lines = ["# Research status\n",
              "_(auto-generated by `research.scripts.aggregate`. Do not edit by hand — edit `papers/*.md` then re-run.)_\n",
              f"\n**Total papers ingested:** {len(papers)}\n"]

    needs_review = [p for p in papers if p.missing_required() or p.missing_preferred()]
    if needs_review:
        lines.append(f"\n## Needs review ({len(needs_review)})\n")
        for p in needs_review:
            miss = []
            req = p.missing_required()
            pref = p.missing_preferred()
            if req:
                miss.append(f"missing required: {', '.join(req)}")
            if pref:
                miss.append(f"missing preferred: {', '.join(pref)}")
            lines.append(f"- {p.link()} — {'; '.join(miss)}")

    # By approach summary
    by_cls = collections.Counter(p.approach_class for p in papers)
    lines.append("\n## Papers by approach class\n")
    for cls, n in sorted(by_cls.items(), key=lambda kv: -kv[1]):
        lines.append(f"- **{cls}**: {n}")

    # Relevance ratings
    by_rel = collections.Counter(p.relevance for p in papers)
    lines.append("\n## Relevance distribution\n")
    for rel in ("high", "medium", "low", "?"):
        if rel in by_rel:
            lines.append(f"- **{rel}**: {by_rel[rel]}")

    # Benchmarks coverage
    bench_counts = collections.Counter()
    for p in papers:
        for b in p.benchmarks:
            bench_counts[b] += 1
    if bench_counts:
        lines.append("\n## Benchmarks referenced (count)\n")
        for b, n in bench_counts.most_common():
            lines.append(f"- **{b}**: {n}")

    lines.append("\n## All papers\n")
    for p in sorted(papers, key=lambda x: -(x.year or 0)):
        flag = ""
        if p.missing_required():
            flag = " ⚠️"
        lines.append(f"- {p.link()} ({p.year or '?'}, {p.approach_class}, rel={p.relevance}){flag}")

    STATUS_FILE.write_text("\n".join(lines) + "\n")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    args = p.parse_args()

    papers = load_papers()
    if not papers:
        logger.warning("No papers found in %s; nothing to aggregate.", PAPERS_DIR)
        return

    write_by_approach(papers)
    write_by_benchmark(papers)
    write_by_year(papers)
    write_by_relevance(papers)
    write_status(papers)

    logger.info("Aggregated %d papers → STATUS.md + groups/", len(papers))


if __name__ == "__main__":
    main()
