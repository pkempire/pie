# Operating process — how work enters this repo (deterministic by default)

The wiki, the paper, and the channel are only as trustworthy as their weakest
entry. This doc defines the **deterministic, auditable pipeline** every artifact
must pass through. The rule of thumb: *a model may draft, but only a checked,
sourced, reproducible step may land.* Hand-authoring from memory is banned — it
is the exact mechanism that put `Anonymous` authors and unverifiable numbers
into the wiki.

## 1. Papers → wiki (grounded ingestion)

**One entry path. No exceptions.**

```
python research/scripts/ingest.py <arxiv_id>      # fetches the real abstract, fills schema verbatim
python research/scripts/validate.py --strict      # gate: provenance + grounded numbers + taxonomy
python research/scripts/aggregate.py              # regenerate STATUS.md + groups/
```

Rules the pipeline enforces:
- **Provenance is mandatory.** `ingest.py` stores the verbatim fetched abstract
  in the file (`## Source`) or a sidecar `<file>.source.txt`. If it isn't stored,
  the claim can't be audited → `validate.py` R4 fails it.
- **Numbers must be verbatim from the source.** `results:` entries are quotes
  from the abstract. Table-only / third-party / derived numbers must be tagged
  `[table]` / `[third-party]` / `[derived]` so they're not silently treated as
  abstract-verified (R5).
- **LLM extracts, never invents.** The extractor prompt is "use only what's in
  the abstract, else null," run at temperature 0 with a pinned model. Same
  abstract in → same fields out.
- **Machine fields vs human judgment are separated.** `problem/approach/results`
  come from the source; `relevance/relevance_reason/steal` are human opinion and
  are labeled as such. Never blend.
- **No placeholders in machine fields** (`(see arXiv)`, `Anonymous`,
  `[fill in by hand]`) — R6 hard-fails them.

If the arXiv API is down, the paper goes to `research/INGEST-QUEUE.md` with status
`NEW-QUEUED`. It does **not** get hand-authored as a shortcut.

## 2. Validation gate (runs with no API key, deterministic)

`validate.py` is the pre-commit / CI gate. It calls no model; it only checks
files against rules and against each entry's stored source. Green = every claim
is traceable. Wire it as a git pre-commit hook:

```
# .git/hooks/pre-commit
python research/scripts/validate.py --strict || exit 1
```

Drift control at scale: `validate.py` can be extended with `--refetch` to diff a
stored abstract against the live arXiv version (papers get revised; numbers move).

## 3. Experiments → results (reproducible by construction)

Every experiment writes a **run manifest** next to its outputs and is resumable:

- Pinned: model id, prompt file (by sha), dataset slice (conv id + question ids),
  seed, code commit. Recorded in `summary.json`.
- **Checkpoint per unit of work** (per question / per chunk) to JSONL so a killed
  run resumes instead of restarting — see `scripts/compare_pie_vs_gepa.py`.
- **No number is reported without its manifest.** A score in the paper/wiki links
  to the `summary.json` that produced it.
- Cost + wall-clock estimated before launch; logged after.

## 4. Honest status reporting

Status docs distinguish three states explicitly, and never conflate them:
- **built** — code exists and imports/constructs.
- **run** — produced a number, with a manifest.
- **verified** — number reproduced ≥2× / on held-out / by a second judge.

"Built" is not "run." "Run once" is not "verified." (Example: as of 2026-06-04
the GEPA consolidator is *run-once-on-a-smoke* — 60→80 on 5 questions — and the
PIE-vs-GEPA comparison is *built, not run*.)

## 5. What this buys at scale

When the wiki is 200 papers, "is it still grounded?" is a 2-second deterministic
check, not a manual re-read. When the paper cites a number, the manifest says
exactly which run produced it. When a model drafts a summary, the gate refuses it
unless the source backs it. The system stays trustworthy without trusting the
model.
