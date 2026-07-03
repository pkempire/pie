# research/content/

Video scripts and content plans for the **Working Memory** YouTube channel.

Each `.md` file is one video script with:
- Yaml frontmatter: target length, audience, status, production stage
- Outline with timestamps
- For each section: script notes + image/diagram ideas + code demos + papers to cite + screenshots + tweets to reference
- Production notes (what asset goes where)

## Production stack

Hand-drawn / whiteboard:
- **Excalidraw** (excalidraw.com) — Karpathy's choice. Hand-drawn aesthetic, exports SVG. Used by Anthropic interpretability papers.
- **tldraw** (tldraw.com) — similar, slightly more polished.

Programmatic video:
- **Remotion** (remotion.dev) — React-based; you literally write `<Sequence>` components. Best fit for code walkthroughs + animated diagrams + voiceover.
- **Motion Canvas** (motioncanvas.io) — TypeScript, Manim-like, easier curve.
- **Manim** (manim.community) — 3Blue1Brown's library. Steepest curve; best for math animations.

Screen recording / code walkthrough:
- **ScreenStudio** (Mac) — auto-zooms into cursor, pro-looking out of the box.
- **OBS** — free, cross-platform.
- **Loom** — quick recordings.

Editing:
- **DaVinci Resolve** — free, pro-grade.
- **CapCut** — easier, faster for short edits.

Audio:
- **Cleanvoice** / **Adobe Podcast** — denoise.
- **ElevenLabs** — voice cloning for retakes.
- **Whisper + Pyannote** — auto-captions.

Reference style:
- **Karpathy** (neural-networks-zero-to-hero) — Jupyter notebook live code, hand-drawn diagrams, narrated, ~1-2hr.
- **3Blue1Brown** — Manim animations, narrative arc per concept, ~15-30 min.
- **Yannic Kilcher** — paper walkthroughs, talking-head, critical takes, ~30-60 min.
- **AI Coffee Break** — short-form paper explainers, ~10 min, lots of graphics.
- **AI Explained** — current-events analysis, clean voiceover, lots of screen capture.

Our blend: Karpathy depth + 3B1B animations for concepts that need them + Yannic's "here's what's actually new" honesty.

## Video status pipeline

`outline` → `script` → `assets-listed` → `recording` → `editing` → `published`

## Files

- `01-ai-memory-deep-dive.md` — the introductory survey
- `02-noreplay-vs-retrieval.md`
- `03-gepa-from-scratch.md`
- `04-sleep-consolidation-live.md` — the live walkthrough of our actual experiment
- `05-temporally-blind-llms.md`
- `06-substrate-design-space.md`
