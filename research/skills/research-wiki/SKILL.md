---
name: research-wiki
description: Reproduce the Karpathy-spartan research-wiki workflow we built for AI memory research — works for any technical research area. Use when the user says "I want to track papers and concepts on X", "build me a research wiki for Y", "set up a lit-review system for Z", or wants to structure educational content over a body of research.
---

# research-wiki skill

Scaffolds a Karpathy-style markdown research wiki with structured paper ingestion, concept/system/goal pages, auto-generated cross-references, and a static HTML site. Originally built for AI-memory research; works for any technical area where you read many papers and want navigable structured notes.

## When to use

Trigger phrases: "research wiki for X", "lit review system for X", "track papers on X", "knowledge base for X", "navigate research on X", "obsidian-style notes for X", "Karpathy wiki for X", or any time the user wants to make educational content (videos, blog series) over a body of papers — the wiki becomes the show notes.

## What you produce

```
<repo>/research/
  README.md, SCHEMA.md, STATUS.md
  papers/        — one .md per paper, yaml frontmatter + body
  concepts/      — patterns, techniques, evaluation axes
  systems/       — products / research systems
  goals/         — active research goals (status: active|planned|parked|done)
  groups/        — auto-regenerated grouped views
  scripts/{ingest,aggregate,show}.py
  wiki/build.py + _site/ (rendered HTML)
```

## Steps

### 1. Define the taxonomy for THIS area

Ask the user for:
- One-sentence area definition
- 8–12 `approach_class` enum values specific to the area

Don't reuse the AI-memory taxonomy verbatim. Examples:
- Long-context: `attention-mechanism`, `retrieval-augmented`, `evaluation-benchmark`, `compression-method`, `infrastructure`
- Materials science: `synthesis-route`, `device-architecture`, `characterization-method`, `theory`, `benchmark`

This taxonomy is load-bearing — get it right at the start.

### 2. Stamp the boilerplate

Copy `/Users/parthkocheta/personal-intelligence-system/research/scripts/{ingest,aggregate,show}.py` and `wiki/build.py`. Substitute the area-specific taxonomy in `SCHEMA.md`.

### 3. Seed with 5–10 foundational papers

Ask user for arxiv IDs. For each:
```bash
python -m research.scripts.ingest <arxiv_id>
```

Then the user reviews — edits relevance / steal / limitations fields.

### 4. Write 3–5 concept pages

The minimum-viable set:
- One `design-space-axis` (main axis along which approaches differ)
- One `evaluation-axis` (how the area is currently scored)
- One `architecture-pattern` (recurring technique)
- One `capability-gap` (what the area hasn't solved)
- One `tooling` (what's missing as engineering)

### 5. Build and serve

```bash
python -m research.scripts.aggregate
python -m research.wiki.build --serve   # localhost:8800
```

Or static:
```bash
open research/wiki/_site/index.html
```

### 6. Optional: push to GitHub Pages

`.github/workflows/wiki.yml` that runs build on push to main; deploys `_site/`.

## Schema (each paper file)

```yaml
arxiv_id, title, authors, year, date_published, date_ingested
approach_class      # from the area's enum
problem             # one sentence
approach            # one sentence
benchmarks          # list
results             # verbatim headline numbers
reward_shape        # for RL papers
base_model
relevance: high | medium | low
relevance_reason
steal               # list of things to potentially adopt
limitations         # list of what they admit
tags
```

Body sections: Quick read, Why it matters to us, Method in one paragraph, Results in numbers, What they don't do, Open questions, Abstract (verbatim).

## Cross-reference convention

`[[slug]]` or `[[slug|Display]]`. Build script resolves to HTML hyperlinks and auto-populates backlinks on the destination page.

## Critical discipline — verify before claiming

Mark every numerical claim's verification status in the body:
- ✅ **abstract-verified** — exact number appears in the abstract
- ⚠️ **table-only** — number in paper's table but not abstract (LLM-extracted, trust medium)
- ⛔ **unverified** — claimed elsewhere, needs PDF read

Without this discipline, an LLM-generated wiki repeats unverified claims as facts. The `relevance_reason` field is also where to call out verification.

## Anti-patterns

- Don't invent metric numbers. If no verbatim quote, `results: []` + TODO.
- Don't generate generic concept pages — every concept page must reference ≥2 specific papers/systems by `[[slug]]`.
- Don't paraphrase the abstract in "Quick read" — that's the user's job; the verbatim Abstract section already exists.

## Reference

`/Users/parthkocheta/personal-intelligence-system/research/` is the canonical example — 27 papers, 9 concepts, 7 systems, 4 goals, full backlink graph, deployed-ready static site.
