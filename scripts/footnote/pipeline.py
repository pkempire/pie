"""End-to-end pipeline orchestrator for Footnote.

Stages:
  1. Input        — load video file or URL
  2. Transcribe   — Whisper large-v3, word-level timestamps + diarization
  3. Sample       — ffmpeg 1 fps + scene-change keyframes
  4. Segment      — DSPy: transcript + frames → segments with topics
  5. Propose      — DSPy: per segment, should an overlay help? if so what?
  6. Research     — Exa + Wikipedia + Bing Images, returns snippet + image + citation
  7. Rewrite      — DSPy: compress snippet to overlay-length text
  8. Layout       — DSPy + vision: bbox per overlay timestamp
  9. Compose      — emit Remotion props JSON + run `npx remotion render`
 10. Guide        — DSPy: build structured markdown article from artifacts

Each stage produces typed artifacts persisted to a job dir so the pipeline
can resume mid-run. The whole thing is GEPA-optimizable because every LLM
step is a DSPy module with a metric.

Usage:
    python -m footnote.pipeline process /path/to/video.mp4 \\
        --output enhanced.mp4 --guide guide.md
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ─── Typed artifacts (the contract between stages) ──────────────────────────

@dataclass
class Word:
    text: str
    start: float          # seconds
    end: float
    speaker_id: int


@dataclass
class TranscriptArtifact:
    words: list[Word]
    duration_sec: float
    n_speakers: int


@dataclass
class Frame:
    path: Path
    timestamp: float
    is_keyframe: bool


@dataclass
class FramesArtifact:
    frames: list[Frame]
    fps_sampled: float


@dataclass
class Segment:
    """A logical unit of content with a single topic."""
    start: float
    end: float
    text: str             # transcript slice
    topic: str            # one-sentence description
    entities: list[str]   # named entities mentioned
    significance: float   # 0-1, how worth overlay-ing
    type: str             # claim | fact | definition | person | event | chartable


@dataclass
class SegmentsArtifact:
    segments: list[Segment]


@dataclass
class OverlayProposal:
    timestamp: float          # when overlay should appear
    duration: float           # how long it stays on screen
    type: str                 # fact-card | person-card | image | chart
    query: str                # what to research / fetch
    placement_intent: str     # "lower-third" | "upper-third" | "side"
    reason: str               # LLM rationale for this overlay


@dataclass
class OverlayProposalsArtifact:
    proposals: list[OverlayProposal]


@dataclass
class ResearchedOverlay:
    proposal: OverlayProposal
    text: str                 # final overlay text (2-3 lines)
    citation_url: str         # primary source
    citation_label: str       # "Wikipedia", "arXiv", etc.
    image_url: str | None     # accompanying image, if any
    chart_spec: dict | None   # vega-lite spec, if chartable


@dataclass
class ResearchedOverlaysArtifact:
    overlays: list[ResearchedOverlay]


@dataclass
class LayoutPlan:
    overlay_idx: int
    bbox: tuple[float, float, float, float]   # (x, y, w, h) in 0-1 normalized
    entry_animation: str      # "fade" | "slide-up" | "slide-left"
    exit_animation: str


@dataclass
class LayoutsArtifact:
    layouts: list[LayoutPlan]


@dataclass
class Job:
    """The full state of an in-progress or completed job."""
    job_id: str
    video_path: Path
    job_dir: Path
    transcript: TranscriptArtifact | None = None
    frames: FramesArtifact | None = None
    segments: SegmentsArtifact | None = None
    proposals: OverlayProposalsArtifact | None = None
    researched: ResearchedOverlaysArtifact | None = None
    layouts: LayoutsArtifact | None = None
    output_video: Path | None = None
    output_guide: Path | None = None


# ─── Stage stubs ────────────────────────────────────────────────────────────
# Each stage returns its artifact and persists it under job.job_dir.
# Real implementations live in scripts/footnote/stages/*.py

def stage_transcribe(job: Job) -> dict:
    """Whisper → word-level timestamps. Returns dict (TranscriptArtifact shape)."""
    from .stages.transcribe import transcribe
    return transcribe(job.video_path)


def stage_sample_frames(job: Job) -> list[dict]:
    """ffmpeg fps + scene-change. Returns list of {path, timestamp, is_keyframe}."""
    from .stages.sample import sample_frames
    out = job.job_dir / "frames"
    return sample_frames(job.video_path, out)


def stage_segment(job: Job) -> list[dict]:
    """LLM segments transcript into topical units."""
    from .stages.segment import segment_full
    return segment_full(job.transcript)


def stage_propose(job: Job) -> list[dict]:
    """LLM + vision: per segment, propose 0-1 overlays."""
    from .stages.propose import propose_all
    return propose_all(job.segments, job.frames)


def stage_research(job: Job) -> list[dict]:
    """Wikipedia + Exa + image search + LLM compress."""
    from .stages.research import research_all
    return research_all(job.proposals)


def stage_layout(job: Job) -> list[dict]:
    """Face detection + deterministic placement."""
    from .stages.layout import layout_all
    return layout_all(job.researched, job.frames)


def stage_compose(job: Job) -> Path | None:
    """Emit Remotion props + render."""
    from .stages.compose import compose
    output_path = job.job_dir / "output.mp4"
    return compose(job.researched, job.layouts, job.video_path, output_path)


def stage_guide(job: Job) -> Path:
    """LLM writes structured markdown article from artifacts."""
    from .stages.guide import write_guide
    article = write_guide(job.transcript, job.segments, job.researched)
    out = job.job_dir / "guide.md"
    out.write_text(article)
    return out


# ─── Orchestration ──────────────────────────────────────────────────────────

def run_pipeline(video_path: Path, output_dir: Path,
                  resume: bool = True) -> Job:
    """End-to-end orchestrator. Persists artifacts under output_dir/<job_id>/
    so the pipeline can resume on failure.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    job = Job(
        job_id=video_path.stem,
        video_path=video_path,
        job_dir=output_dir / video_path.stem,
    )
    job.job_dir.mkdir(parents=True, exist_ok=True)

    # Each stage checks for its persisted artifact and skips if present.
    job.transcript = _load_or_compute(job, "transcript.json", stage_transcribe)
    job.frames = _load_or_compute(job, "frames.json", stage_sample_frames)
    job.segments = _load_or_compute(job, "segments.json", stage_segment)
    job.proposals = _load_or_compute(job, "proposals.json", stage_propose)
    job.researched = _load_or_compute(job, "researched.json", stage_research)
    job.layouts = _load_or_compute(job, "layouts.json", stage_layout)
    job.output_video = stage_compose(job)
    try:
        job.output_guide = stage_guide(job)
    except Exception as e:
        logger.warning("guide stage failed: %s", e)
        job.output_guide = None
    return job


def _load_or_compute(job: Job, artifact_name: str, compute_fn) -> Any:
    """Resume helper. Persists JSON-serializable artifacts only."""
    path = job.job_dir / artifact_name
    if path.exists():
        logger.info("resume: loading %s", artifact_name)
        return json.loads(path.read_text())
    logger.info("compute: %s", artifact_name)
    result = compute_fn(job)
    # Coerce dataclasses to dicts
    if hasattr(result, "__dataclass_fields__"):
        serializable = asdict(result)
    else:
        serializable = result
    path.write_text(json.dumps(serializable, indent=2, default=str))
    return result


# ─── CLI ────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="input video file")
    parser.add_argument("--output-dir", type=Path, default=Path("./footnote_out"),
                         help="job artifacts directory")
    parser.add_argument("--no-resume", action="store_true",
                         help="ignore existing artifacts and recompute everything")
    args = parser.parse_args()

    if not args.video.exists():
        raise SystemExit(f"video not found: {args.video}")

    job = run_pipeline(args.video, args.output_dir, resume=not args.no_resume)
    print(f"✓ output video: {job.output_video}")
    print(f"✓ guide:        {job.output_guide}")
    print(f"  job dir:      {job.job_dir}")


if __name__ == "__main__":
    main()
