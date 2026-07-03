---
title: "Goal 05 — Footnote: AI video annotation product"
status: "planned"
priority: 5
started: null
owner: "us"
budget: "~$200 dev compute for MVP; $50/mo ongoing API costs"
tags: ["product", "video", "Remotion", "DSPy", "commercialization", "agent"]
---

# Goal 05 — Footnote: AI video annotation product

## Thesis

Talking-head educational videos are everywhere on YouTube — history explainers, finance, science, philosophy, language learning, AI commentary. They're high information density but low *visual* density: it's a person in front of a bookshelf for 12 minutes. Viewers retain more when each named entity / claim / fact gets a brief visual anchor with the source.

Currently this is hand-done in After Effects or Premiere. It's the single biggest production-quality multiplier that small creators don't do because it costs them 4-8 hours per video.

**The product**: drop a video in, get back the same video with researched, cited, well-placed contextual overlays at the right moments — plus a structured written guide derived from the same audio + frames.

## Reference: the Napoleon screenshot

The screenshot the user shared shows the *exact* product. A talking-head video about Napoleon's self-education. At 0:55 an overlay appears: "Napoleon was a voracious reader who carried a specially commissioned, miniaturized 'travelling library' on his military campaigns. Housed in custom mahogany and leather cases..." Cites "Wikipedia +3," includes an image of the actual travelling library, lands in a spot that doesn't cover the speaker.

That overlay is hand-made today. We automate it.

## Competition

| Tool | What it does | What it doesn't |
|---|---|---|
| Captions.ai, Submagic | Captions, basic B-roll | No researched facts, no citations |
| Opus Clips | Short-form repurposing from long video | Different problem entirely |
| Descript | Full editing platform with AI features | No automatic contextual research |
| Riverside | Recording / editing | No automatic enhancement |
| Manual editor | Real facts, real images, hand-placed | 4-8 hours per video |

**Nobody is doing AI-driven contextual factual overlays with real research and citations.** That's the wedge.

## Architecture

See the rendered pipeline diagram above and at this goal page.

Nine stages, each a DSPy module so GEPA can optimize the prompts later:

1. **Input** — video.mp4 or URL
2. **Whisper transcribe** — word-level timestamps + speaker diarization
3. **Frame sampler** — ffmpeg 1fps + scene-change detection
4. **Segmenter** (DSPy) — transcript + frames → segments with topic/entities/significance
5. **Overlay Proposer** (DSPy + vision) — should an overlay help here? if yes, what?
6. **Fact Fetcher** + **Image fetch** + **Snippet rewriter** — Exa/Wikipedia + Bing Images + LLM compress
7. **Layout Planner** (vision) — bbox placement that avoids speaker face
8. **Composer** — Remotion renders the final mp4
9. **Guide Writer** (DSPy) — same artifacts → structured markdown guide

The whole pipeline is data-driven. Templates parameterize fonts, colors, animation styles per creator brand.

## Human-in-the-loop checkpoint (the quality multiplier)

Before final render, show the creator a web UI:
- One card per proposed overlay
- Each card: timestamp + frame preview + suggested text + citation + placement preview
- Buttons: ✓ approve / ✗ reject / ✎ edit
- Approved overlays → Composer
- Rejections logged → next-pass training data for GEPA

This is the single feature that separates "AI fully automates" (frequently wrong) from "AI proposes, creator chooses" (high quality, fast). Pro-tier-only feature.

## Tech stack

- **Whisper** (large-v3) — transcription
- **GPT-5 / Claude Opus 4.6** — Segmenter + Overlay Proposer + Snippet rewriter
- **GPT-5 vision / Claude vision** — Layout Planner
- **Exa Search API** — fact lookup with citation
- **Wikipedia API + Wikimedia** — primary source for historical/factual content
- **Bing Image Search API** — image overlays
- **DSPy** — pipeline framework (so GEPA-optimizable)
- **Remotion** (remotion.dev) — programmatic video composition
- **Next.js + Vercel** — web app + render API
- **FastAPI** — render queue + job orchestration
- **R2 / S3** — video storage
- **Postgres** — jobs + accounts + creator brand templates

## Pricing

- **Self-serve $20/mo** — 50 min processed, 1080p, watermarked
- **Pro $100/mo** — 500 min, 4K, human-in-loop UI, Remotion source export
- **Enterprise** — custom, brand templates, API, SLA

## Cost / margin (10-min video)

- Whisper-large API: ~$0.06
- Vision API frames (~10 keyframes): ~$0.20
- LLM (Segmenter + Proposer + Rewriter + GuideWriter): ~$0.50
- Search APIs (Exa + Wikipedia + Bing Images): ~$0.05
- **Total cost: ~$0.85**
- Pro-tier effective price: ~$2 per 10-min video
- ~60% gross margin, improves as API prices drop

Render compute can run on user machine (Remotion CLI) or your render farm. We'd start with user-machine render for the MVP to keep COGS to zero.

## MVP timeline

| Day | Output |
|---|---|
| 1-2 | Whisper integration + transcript → segments LLM working on 1 test video |
| 3 | Overlay Proposer end-to-end (no fact-fetch yet) |
| 4 | Fact Fetcher + Snippet Rewriter (Exa + Wikipedia integration) |
| 5 | Layout Planner with vision |
| 6 | Remotion template + Composer |
| 7 | Guide Writer branch |
| 8-10 | Refine on 5 real videos; fix edge cases |
| 11-14 | Web UI + auth + Stripe + landing page |
| 15-20 | First 10 paying users from Twitter / Indie Hackers |

Three weeks from spec to first paying user, optimistically. Five weeks realistically.

## Target customers (initial)

- History / philosophy YouTubers (the screenshot was this exact category)
- Finance / business explainers (10-20 min talking-head format)
- Science / health communicators
- AI commentary channels
- Educational podcast video versions

All are people who already produce 4-20 min talking-head videos every week and would happily pay $100/mo to skip the After Effects research-and-overlay tax.

## Name candidates

- **Footnote** ← preferred. Describes function, easy to say, memorable.
- Marginalia
- Glossa
- Cite
- Annotate
- Contextly

## What we'd actually build first (this week)

1. `scripts/footnote/` — Python package with the DSPy pipeline
2. CLI: `footnote process video.mp4 --output enhanced.mp4 --guide guide.md`
3. One Remotion template
4. Test on three real talking-head videos from YouTube (download via yt-dlp)
5. Run the result by 5 friends who make educational videos; iterate

If all five say "I would pay for this," build the web app.

## Why now

- Whisper-large-v3 is reliable enough for word-level timestamps
- GPT-5 / Claude Opus 4.6 are good enough at vision + multi-step reasoning for the proposer
- Remotion is mature; programmatic video rendering is solved
- Creator economy is paying for tools more than ever
- No existing tool does this specific thing well

## Connection to our other work

- DSPy pipeline → [[gepa-vs-grpo|GEPA]] can optimize the Segmenter / Proposer / Layout Planner prompts as the system sees more videos. This is the killer feature: the system gets better as creators use it.
- The "live dashboard" pattern we built for GEPA transfers directly to a "live processing dashboard" for the video render queue.
- The research wiki / video script work in `research/content/` is on the receiving end of this: we use Footnote to produce our own Working Memory videos.

## Related

- Concept: [[multi-agent-delegation]] — the proposer + fact-fetcher + layout-planner is naturally multi-agent
- Goal: [[03-public-research-wiki|Goal 03 (wiki + content)]] — our content benefits from this tool
- Paper: [[2507.19457-gepa]] — the optimizer that makes pipeline prompts adaptive
