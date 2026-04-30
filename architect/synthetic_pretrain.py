"""Synthetic (user_query → architecture) dataset builder.

Why this exists
===============
Naive embedding search over component descriptions has two failure modes
the planner inherits:

  1. Users don't phrase requests the way component docs are written.
     A user says "track when prices on competitor sites change"; the
     Stagehand README says "AI browser automation framework." The vector
     similarity is moderate. Worse, the user doesn't even know to look
     for "browser automation" — they're describing an *outcome*, not a
     mechanism.

  2. The planner has no examples of full architectures vs. component
     names. It can pick components but doesn't know how they snap
     together — that knowledge lives in real repos.

The fix is data: take every architecture we've mined from real repos,
generate plausible user queries that should map to that architecture,
and use those (query, architecture) pairs as either:

  - Few-shot examples retrieved at planning time (immediate use,
    works with any LLM), or
  - SFT data for fine-tuning a small planner model later (4-figure
    cost, ~5-10pt win on planning quality).

Both uses share the same dataset shape, so we build it once.

Pipeline
========

  for each architecture in `architectures` table (filtered to
   `quality_signal > 0.3` so we don't train on demos):
      1. Fetch the architecture card (summary, pattern, components).
      2. Prompt: "given this architecture, write 3 plausible user
         queries the developer might have asked that would lead a
         competent system designer to recommend exactly this stack."
      3. Verify each query: does an embedding-search over component
         summaries surface the architecture's components in the top-K?
         If not, the query is too narrow / too broad — skip.
      4. Persist as JSONL: {query, architecture_id, components,
         pattern, source_url, generation_model}.

Output: `architect/data/synthetic_queries.jsonl`. Treat as append-only
so re-runs incrementally extend the dataset.

Future: the same generator can produce pairs from clusters/patterns
(once Layer 2 has output), giving us (query → pattern) examples that
help the planner short-circuit pattern matching.
"""
from __future__ import annotations
import argparse
import json
import logging
from collections import Counter
from pathlib import Path

from . import db
from mempol import llm, config

logger = logging.getLogger(__name__)


_GEN_SYSTEM = """You produce plausible USER QUERIES that would lead a
competent staff engineer to recommend a specific software architecture.

You will receive:
  - a one-line architecture summary
  - the components used and their roles
  - the canonical example repo

Write 3 short user queries. Vary them:
  - Query A: outcome-focused ("I want to do X to my customers")
  - Query B: stack-focused ("I'm building Y; how should I assemble it?")
  - Query C: pain-focused ("I keep running into Z; what fixes it?")

Rules:
  - Each query 1-2 sentences. No more than ~30 words.
  - Don't use any component name in the query — the planner has to
    infer the components from the query's intent.
  - Phrase like a developer typing into a search bar, not like
    marketing copy.

Return JSON: {"queries": [{"style": "outcome|stack|pain", "text": "..."}]}"""


_GEN_USER_TEMPLATE = """ARCHITECTURE
  source:      {source_url}
  pattern:     {pattern}
  summary:     {summary}

COMPONENTS USED
{components_block}

Write 3 user queries (outcome / stack / pain). Return JSON only."""


def _format_components_block(arch_id: int) -> str:
    with db.connect() as conn:
        cur = conn.execute(
            """SELECT c.name, c.one_liner, ac.role
               FROM architecture_components ac
               JOIN components c ON c.id = ac.component_id
               WHERE ac.architecture_id = ?""",
            (arch_id,),
        )
        return "\n".join(
            f"  - {row['name']:24s} ({row['role'] or 'unspecified'}) — {row['one_liner']}"
            for row in cur
        )


def _component_set(arch_id: int) -> set[int]:
    with db.connect() as conn:
        cur = conn.execute(
            "SELECT component_id FROM architecture_components WHERE architecture_id=?",
            (arch_id,),
        )
        return {row["component_id"] for row in cur}


def _verify_via_retrieval(
    query: str, arch_components: set[int], top_k: int = 8,
) -> tuple[bool, float]:
    """Run an embedding search over the components and check that ≥ 60%
    of `arch_components` show up in the top-K. Acts as a sanity filter:
    if a query is too vague to surface its own architecture's components,
    it's not training data, it's noise."""
    try:
        emb = llm.embed([query])[0].tolist()
    except Exception as e:
        logger.warning("verify embedding failed: %s", e)
        return False, 0.0
    with db.connect() as conn:
        results = db.search_components(conn, emb, top_k=top_k)
    retrieved = {r["id"] for r in results}
    overlap = arch_components & retrieved
    frac = len(overlap) / max(len(arch_components), 1)
    return (frac >= 0.6, frac)


def _generate_queries_for_arch(arch_row: dict) -> list[dict]:
    """Generate 3 candidate queries; LLM returns JSON list."""
    msgs = [
        {"role": "system", "content": _GEN_SYSTEM},
        {"role": "user",   "content": _GEN_USER_TEMPLATE.format(
            source_url=arch_row["source_url"],
            pattern=arch_row.get("pattern", "") or "(no pattern label)",
            summary=arch_row.get("summary", "") or "(no summary)",
            components_block=_format_components_block(arch_row["id"]),
        )},
    ]
    raw = llm.chat(msgs, model=config.OBSERVER_MODEL or "gpt-4o-mini",
                    json_mode=True)
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj.get("queries", []) or []
        return obj if isinstance(obj, list) else []
    except Exception as e:
        logger.warning("query-gen parse fail (arch=%s): %s",
                       arch_row.get("id"), e)
        return []


# ─── Public entry ───────────────────────────────────────────────────────────
def build_dataset(
    out_path: Path,
    min_quality: float = 0.3,
    max_archs: int = 0,                                 # 0 = no cap
    verify: bool = True,
) -> int:
    """Iterate good-quality architectures and emit (query → arch) pairs.

    Append-only writes to `out_path` (JSONL). Returns the count of pairs
    written this run. Idempotent across runs at the architecture level
    (we skip arches whose source_url already appears in the file).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seen_urls: set[str] = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            try:
                seen_urls.add(json.loads(line).get("source_url", ""))
            except Exception:
                continue
    if seen_urls:
        logger.info("resume: skipping %d architectures already in dataset",
                    len(seen_urls))

    written = 0
    with db.connect() as conn:
        cur = conn.execute(
            "SELECT * FROM architectures WHERE quality_signal >= ? "
            "ORDER BY quality_signal DESC",
            (min_quality,),
        )
        archs = [dict(row) for row in cur]

    if max_archs:
        archs = archs[:max_archs]

    with out_path.open("a") as f:
        for arch in archs:
            if arch["source_url"] in seen_urls:
                continue
            comp_ids = _component_set(arch["id"])
            if len(comp_ids) < 2:
                continue                                # one-component arch is not a system

            queries = _generate_queries_for_arch(arch)
            for q in queries:
                text = (q.get("text") or "").strip()
                if not text:
                    continue
                style = (q.get("style") or "").strip()

                if verify:
                    ok, frac = _verify_via_retrieval(text, comp_ids)
                    if not ok:
                        logger.debug("dropping low-recall query (frac=%.2f): %s",
                                     frac, text[:80])
                        continue

                pair = {
                    "query":          text,
                    "style":          style,
                    "architecture_id": arch["id"],
                    "source_url":     arch["source_url"],
                    "pattern":        arch.get("pattern", ""),
                    "components":     sorted(comp_ids),
                    "verify_frac":    None,
                    "generation_model": config.OBSERVER_MODEL or "gpt-4o-mini",
                }
                f.write(json.dumps(pair) + "\n")
                written += 1
            logger.info("arch %s — %d queries written",
                        arch["source_url"][:60], len(queries))

    logger.info("wrote %d (query, arch) pairs to %s", written, out_path)
    return written


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path,
                        default=Path("architect/data/synthetic_queries.jsonl"))
    parser.add_argument("--min_quality", type=float, default=0.3)
    parser.add_argument("--max_archs", type=int, default=0)
    parser.add_argument("--no-verify", action="store_true",
                        help="Skip the retrieval-overlap verification pass.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    n = build_dataset(out_path=args.out,
                      min_quality=args.min_quality,
                      max_archs=args.max_archs,
                      verify=not args.no_verify)
    print(f"OK — {n} pairs written")


if __name__ == "__main__":
    main()
