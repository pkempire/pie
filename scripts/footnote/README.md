# Footnote

Talking-head educational video in → enhanced video with researched, cited contextual overlays + structured written guide out.

See `research/goals/05-footnote-product.md` for the full spec.

## Status

**MVP shipped.** Every stage has a real implementation. End-to-end pipeline runs. Remotion project scaffolded. Smoke test script included.

## Structure

```
scripts/footnote/
├── pipeline.py          orchestrator + CLI + typed artifacts
├── smoke_test.py        end-to-end smoke runner
├── dspy_modules.py      DSPy signatures (used by stages, GEPA-ready)
└── stages/
    ├── transcribe.py    Whisper word-level timestamps
    ├── sample.py        ffmpeg fps + scene-change frame sampling
    ├── segment.py       LLM segments transcript into topical units
    ├── propose.py       LLM + vision proposes overlays per segment
    ├── research.py      Wikipedia + Exa + LLM compress to overlay text
    ├── layout.py        Face detection + deterministic placement
    ├── compose.py       Emit Remotion props + npx remotion render
    └── guide.py         LLM writes markdown article from artifacts

remotion/footnote/
├── package.json
├── tsconfig.json
├── remotion.config.ts
└── src/
    ├── index.tsx
    ├── Root.tsx
    ├── FootnoteComposition.tsx
    └── FactCard.tsx     the actual overlay component
```

## Install

```bash
# System dependencies
brew install ffmpeg

# Python dependencies
pip install openai opencv-python

# Optional: OpenCV-headless if you don't need GUI
pip install opencv-python-headless

# Optional: yt-dlp for grabbing test videos
pip install yt-dlp

# Remotion (Node 18+)
cd remotion/footnote
npm install   # or bun install
cd ../..
```

## API keys

```bash
export OPENAI_API_KEY=sk-...
# Optional — Wikipedia is the primary source so this is strictly fallback
export EXA_API_KEY=...
```

## Smoke test on a real video

```bash
# Grab a 12-minute YouTube video as test material
yt-dlp -f 'bv*+ba/b' --merge-output-format mp4 \
    'https://www.youtube.com/watch?v=YOUR_VIDEO_ID' \
    -o /tmp/footnote_test.mp4

# Run the smoke — processes first 60 seconds (~$0.10)
python -m scripts.footnote.smoke_test /tmp/footnote_test.mp4

# Look at the artifacts
ls footnote_smoke/clip/
# transcript.json segments.json proposals.json researched.json layouts.json
```

## Full pipeline

```bash
# Process the whole video
python -m scripts.footnote.pipeline /tmp/footnote_test.mp4 \
    --output-dir ./footnote_out

# Render the final video with Remotion (if installed)
cd remotion/footnote
npx remotion render src/index.tsx FootnoteComposition out.mp4 \
    --props=../../footnote_out/footnote_test/footnote_props.json
```

## How temporal consistency is solved

Key design choice: the LLM only produces **specs** (timing + text + position + image URL). Remotion's React rendering is deterministic — same props → identical pixels every render. No LLM in the per-frame render loop = no drift, no flicker, no inconsistency.

Animations interpolate between fixed keyframes using `interpolate` and `spring` from Remotion. Each overlay has explicit entry/exit animations that don't depend on neighboring overlays.

## How many overlays per video

Rule of thumb: **one per 30-90 seconds** depending on segment density. Hard floor: 8 seconds between accepted proposals (cognitive-load limit). For a 10-minute video: 8-15 overlays is typical. The Proposer rejects proposals when:
- significance < 0.4
- prior overlay covered the same topic
- speaker face dominates the frame

## Long videos

Handled in `stages/segment.py:chunk_transcript`. The transcript is chunked into 5-minute blocks; each block is segmented independently then concatenated. This keeps each LLM call inside its context window and produces consistent output for arbitrary-length videos.

## Cost per 10-min video

- Whisper API: ~$0.06
- Segmenter (gpt-5-mini): ~$0.05
- Proposer with vision: ~$0.30 (5-10 calls × $0.03)
- Research (Wikipedia + LLM rewrite): ~$0.10
- Layout (OpenCV face detection, no API): $0
- Guide writer: ~$0.20
- **Total: ~$0.71** per 10-minute video with gpt-5-mini

Remotion render runs locally; CPU only, no GPU needed.

## Tuning levers

- Overlay density: change `min_spacing_sec` in `propose_all` (default 8s)
- Segment granularity: change `chunk_minutes` in `segment_full` (default 5min)
- Frame sampling rate: change `fps` in `sample_frames` (default 1fps)
- Scene-change sensitivity: change `scene_threshold` (default 0.4)
- Overlay default duration: in `propose.py`, default is 8s
- Models: pass `--task-model`, `--reflection-model` (when wired)

## GEPA integration

Every LLM-driven stage is a DSPy module (`dspy_modules.py`). Once you have 100+ videos processed with creator approval/rejection data (from the human-in-loop UI you'd build next), GEPA can optimize each module's prompt against approval rate as the metric.

This is the same pattern as Goal 01 (GEPA consolidator on LoCoMo). Direct port.
