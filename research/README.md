# research/

Single source of truth for the literature we've reviewed and what
we know about it. Solves the "restart from scratch every conversation"
problem by giving every paper a canonical file with a fixed schema.

## Layout

```
research/
  README.md                    # this file
  SCHEMA.md                    # the per-paper schema, with examples
  STATUS.md                    # auto-generated overview of what we have
  papers/                      # one .md per paper, yaml frontmatter + body
    2508.19828-memory-r1.md
    2503.09516-search-r1.md
    ...
  groups/                      # auto-generated grouped views
    by_approach.md
    by_benchmark.md
    by_year.md
    by_relevance.md
  scripts/
    ingest.py                  # arxiv_id → fetched + LLM-extracted .md file
    aggregate.py               # papers/ → groups/ regenerated
    show.py                    # query interface (`show --benchmark locomo`)
```

## Quick start

```bash
# Add a paper by arxiv id (auto-fetches abstract + key sections, LLM-fills schema)
python -m research.scripts.ingest 2508.19828

# Regenerate the grouped views and STATUS.md
python -m research.scripts.aggregate

# Query: every paper that uses LongMemEval
python -m research.scripts.show --benchmark longmemeval

# Query: papers grouped by approach
python -m research.scripts.show --groupby approach
```

## Design principles

1. **One file per paper.** Easy to inspect, easy to edit by hand, easy to diff in git.
2. **YAML frontmatter for structured fields.** Title, arxiv_id, authors, date, benchmarks, approach class, headline result. Machine-readable.
3. **Markdown body for everything else.** Notes, key insights, why-this-matters. Human-readable.
4. **Schema enforced by ingestion.** New papers must have every required field. If the LLM can't extract it, the file gets `?` and is flagged in STATUS.md.
5. **No silent drift.** Existing files don't auto-update. If you re-ingest, you see a diff. You commit the change.
