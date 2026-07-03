"""Deterministic grounding validator for the research wiki.

The wiki's failure mode at scale is hallucinated/paraphrased summaries: a paper
file claims "47% -> 11%" or an author list that never appears in the source.
This validator makes "is every entry grounded?" a deterministic, no-LLM check
that can run in pre-commit / CI. It NEVER calls a model — it only checks files
against rules and against each entry's stored source abstract.

Checks (each is pass/fail, no judgment calls):
  R1  required frontmatter fields present and non-empty
  R2  approach_class in taxonomy; reward_shape in taxonomy (if present)
  R3  arxiv_id well-formed (NNNN.NNNNN) OR explicitly a non-arxiv source_url
  R4  provenance present: a `## Source` section OR sidecar `<file>.source.txt`
      containing the verbatim fetched abstract  (this is what makes claims auditable)
  R5  every numeric token in `results:` appears in the stored source text,
      UNLESS the result line is tagged [table] / [third-party] / [derived]
      (mirrors the abstract-verified vs table-only vs third-party discipline)
  R6  no `(see arXiv)` / `Anonymous` / `[fill in by hand]` placeholders left in
      machine fields  (catches hand-authored stubs like the ones added 2026-06-02)

Exit code 0 = all green, 1 = any failure. `--strict` makes R4/R5 hard failures;
without it they are warnings so you can adopt incrementally.

Usage:
    python research/scripts/validate.py            # report, warn on R4/R5
    python research/scripts/validate.py --strict   # CI mode, fail on everything
    python research/scripts/validate.py --json      # machine-readable
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PAPERS = REPO / "research" / "papers"

REQUIRED = ["arxiv_id", "title", "approach_class", "problem", "approach",
            "benchmarks", "results", "relevance"]
APPROACH_CLASSES = {
    "write-time-compression", "read-time-decompression", "RL-for-memory",
    "RL-for-tool-use", "agent-orchestration", "temporal-reasoning", "benchmark",
    "substrate", "theory", "infrastructure", "survey",
}
REWARD_SHAPES = {
    "trajectory-level", "per-op-state-distance", "per-op-outcome-attribution",
    "verbal-reflection", "supervised", "none", None, "null",
}
PLACEHOLDERS = ["(see arxiv)", "anonymous", "[fill in by hand]", "unknown — fetch",
                "tbd", "xxx"]
RESULT_TAG_EXEMPT = ("[table]", "[third-party]", "[derived]", "[no-number]")
ARXIV_RE = re.compile(r"^\d{4}\.\d{4,5}$")
NUM_RE = re.compile(r"\d+(?:\.\d+)?")


def split_frontmatter(text: str) -> tuple[str, str]:
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            return text[3:end], text[end + 4:]
    return "", text


def parse_frontmatter(fm: str) -> dict:
    """Minimal YAML-ish parser sufficient for this schema (scalars + simple lists).
    Deterministic; avoids a PyYAML dependency. Good enough for validation."""
    out, key = {}, None
    for raw in fm.splitlines():
        line = raw.split(" #", 1)[0].rstrip() if not raw.strip().startswith("#") else ""
        if not line.strip():
            continue
        if re.match(r"^\s*-\s+", line) and key is not None:
            out.setdefault(key, [])
            if isinstance(out[key], list):
                out[key].append(line.strip()[2:].strip().strip('"'))
            continue
        m = re.match(r"^([A-Za-z_][\w]*):\s*(.*)$", line)
        if m:
            key, val = m.group(1), m.group(2).strip()
            if val == "":
                out[key] = []          # list follows (or empty)
            else:
                out[key] = val.strip().strip('"')
    return out


def source_text_for(path: Path, body: str) -> str:
    side = path.with_suffix(".source.txt")
    if side.exists():
        return side.read_text().lower()
    m = re.search(r"##\s*Source[^\n]*\n(.+?)(?:\n##\s|\Z)", body, re.S | re.I)
    return (m.group(1).lower() if m else "")


def check_file(path: Path) -> list[dict]:
    text = path.read_text()
    fm_raw, body = split_frontmatter(text)
    fm = parse_frontmatter(fm_raw)
    issues = []

    def fail(rule, msg):
        issues.append({"rule": rule, "msg": msg})

    # R1 required
    for k in REQUIRED:
        v = fm.get(k, None)
        if v is None or v == "" or v == [] or v == "null":
            fail("R1", f"missing/empty required field: {k}")

    # R2 taxonomy
    ac = fm.get("approach_class")
    if ac and ac not in APPROACH_CLASSES:
        fail("R2", f"approach_class '{ac}' not in taxonomy")
    rs = fm.get("reward_shape")
    if rs not in REWARD_SHAPES:
        fail("R2", f"reward_shape '{rs}' not in taxonomy")

    # R3 id well-formed
    aid = (fm.get("arxiv_id") or "").strip()
    if not ARXIV_RE.match(aid) and not fm.get("source_url"):
        fail("R3", f"arxiv_id '{aid}' malformed and no source_url given")

    # R6 placeholders in machine fields
    for field in ("title", "authors", "problem", "approach", "results"):
        blob = json.dumps(fm.get(field, "")).lower()
        for ph in PLACEHOLDERS:
            if ph in blob:
                fail("R6", f"placeholder '{ph}' left in field '{field}'")

    # R4 provenance
    src = source_text_for(path, body)
    if not src:
        fail("R4", "no stored source abstract (## Source section or .source.txt) — claims unauditable")

    # R5 numbers grounded
    results = fm.get("results", [])
    if isinstance(results, list) and src:
        for r in results:
            rl = str(r).lower()
            if any(tag in rl for tag in RESULT_TAG_EXEMPT):
                continue
            nums = NUM_RE.findall(rl)
            ungrounded = [n for n in nums if n not in src]
            if ungrounded:
                fail("R5", f"numbers {ungrounded} in results not found in source: {str(r)[:60]!r}")

    return issues


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true",
                    help="Treat R4/R5 (provenance/number-grounding) as hard failures.")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    SOFT = {"R4", "R5"}
    files = sorted(PAPERS.glob("*.md"))
    report, hard_fail, soft_fail = {}, 0, 0
    for f in files:
        issues = check_file(f)
        if issues:
            report[f.name] = issues
            for i in issues:
                if i["rule"] in SOFT and not args.strict:
                    soft_fail += 1
                else:
                    hard_fail += 1

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        if not report:
            print(f"✓ all {len(files)} paper entries pass grounding checks")
        for name, issues in report.items():
            print(f"\n{name}")
            for i in issues:
                sev = "warn" if (i["rule"] in SOFT and not args.strict) else "FAIL"
                print(f"  [{sev}] {i['rule']}: {i['msg']}")
        print(f"\n{len(files)} files | {hard_fail} hard failures | {soft_fail} warnings")
    sys.exit(1 if hard_fail else 0)


if __name__ == "__main__":
    main()
