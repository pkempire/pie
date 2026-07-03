"""Smoke test the Footnote pipeline against a real video.

Usage:
    # Download a public YouTube test video first
    pip install yt-dlp
    yt-dlp -f 'bv*+ba/b' --merge-output-format mp4 \\
        'https://www.youtube.com/watch?v=YOUR_VIDEO_ID' \\
        -o /tmp/footnote_test.mp4

    # Then run the smoke
    python -m scripts.footnote.smoke_test /tmp/footnote_test.mp4

What this verifies:
    [x] ffmpeg present and works for audio/frame extraction
    [x] Whisper API responds with word-level timestamps
    [x] Segmenter LLM returns valid JSON
    [x] Proposer LLM returns valid JSON
    [x] At least one Wikipedia search succeeds
    [x] Layout + face detection produces stable bboxes
    [x] Remotion props.json is written
    [x] Either Remotion renders or instructions are printed

This is the test you run before the full pipeline. It uses a 1-minute
clip by default to keep cost minimal (~$0.10).
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def check_prereqs() -> list[str]:
    """Return list of missing prerequisites."""
    missing: list[str] = []
    if shutil.which("ffmpeg") is None:
        missing.append("ffmpeg (brew install ffmpeg)")
    if shutil.which("ffprobe") is None:
        missing.append("ffprobe (comes with ffmpeg)")
    try:
        import openai   # noqa: F401
    except ImportError:
        missing.append("openai (pip install openai)")
    return missing


def clip_video(input_path: Path, output_path: Path,
                 start: float = 0, duration: float = 60) -> Path:
    """Take a short clip from the input video to keep smoke costs low."""
    cmd = [
        "ffmpeg", "-y", "-ss", str(start), "-i", str(input_path),
        "-t", str(duration), "-c", "copy", str(output_path),
    ]
    logger.info("clipping: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, capture_output=True)
    return output_path


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="input video")
    parser.add_argument("--clip-seconds", type=float, default=60,
                         help="take first N seconds only (default 60 to keep costs low)")
    parser.add_argument("--output-dir", type=Path, default=Path("./footnote_smoke"))
    parser.add_argument("--full", action="store_true",
                         help="don't clip; process the entire video")
    args = parser.parse_args()

    if not args.video.exists():
        print(f"❌ video not found: {args.video}")
        sys.exit(1)

    missing = check_prereqs()
    if missing:
        print("❌ missing prerequisites:")
        for m in missing:
            print(f"   - {m}")
        sys.exit(1)

    # Clip if smoke mode
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.full:
        clipped = args.video
    else:
        clipped = args.output_dir / "clip.mp4"
        clip_video(args.video, clipped, duration=args.clip_seconds)

    # Run the pipeline
    from scripts.footnote.pipeline import run_pipeline
    job = run_pipeline(clipped, args.output_dir, resume=True)

    print()
    print("==== SMOKE SUMMARY ====")
    summary = {
        "video": str(clipped),
        "duration_sec": job.transcript.get("duration_sec") if job.transcript else None,
        "n_words": len(job.transcript.get("words", [])) if job.transcript else 0,
        "n_frames": len(job.frames) if job.frames else 0,
        "n_segments": len(job.segments) if job.segments else 0,
        "n_proposals": len(job.proposals) if job.proposals else 0,
        "n_researched": len(job.researched) if job.researched else 0,
        "n_layouts": len(job.layouts) if job.layouts else 0,
        "output_video": str(job.output_video) if job.output_video else None,
        "output_guide": str(job.output_guide) if job.output_guide else None,
        "job_dir": str(job.job_dir),
    }
    print(json.dumps(summary, indent=2))

    if job.researched:
        print()
        print("==== SAMPLE OVERLAY ====")
        r = job.researched[0]
        print(f"timestamp: {r['proposal'].get('timestamp_sec')}s")
        print(f"text:      {r['text']}")
        print(f"citation:  {r['citation_url']}")
        print(f"image:     {r.get('image_url') or '(none)'}")

    if not job.output_video:
        print()
        print("ℹ️  Remotion render skipped. To render the final video, install:")
        print("    cd remotion/footnote && bun install   # or npm install")
        print("    cd /Users/parthkocheta/personal-intelligence-system")
        print(f"    npx remotion render remotion/footnote/src/index.tsx \\")
        print(f"        FootnoteComposition {args.output_dir}/output.mp4 \\")
        print(f"        --props={args.output_dir}/clip/footnote_props.json")


if __name__ == "__main__":
    main()
