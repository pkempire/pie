"""Stage 3: sample frames from the video.

Two passes: scene-change keyframes (where the visual content actually
shifts) and a 1fps base sample. Combined and deduped by timestamp.

Why both:
  - Scene-change frames are *informative* (cuts, B-roll, slide changes)
  - 1fps frames give regular sampling so we never miss a long static span
  - Talking-head videos have few scene changes — the 1fps catches most
    moments where the speaker's expression/pose changes too
"""
from __future__ import annotations

import logging
import re
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


def _ffprobe_duration(video_path: Path) -> float:
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(video_path),
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    return float(out)


def sample_frames(video_path: Path, out_dir: Path,
                   fps: float = 1.0,
                   scene_threshold: float = 0.4) -> list[dict]:
    """Sample frames; returns list of {path, timestamp, is_keyframe}.

    fps=1.0 means one frame per second of base sample.
    scene_threshold is ffmpeg's `select='gt(scene,N)'` — 0.4 catches hard
    cuts and most B-roll inserts without false-positives from gestures.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    duration = _ffprobe_duration(video_path)
    frames: list[dict] = []

    # Pass 1: scene-change frames with their timestamps captured via showinfo.
    scene_log = Path(tempfile.mkstemp(suffix=".log")[1])
    cmd_scene = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vf", f"select='gt(scene,{scene_threshold})',showinfo",
        "-vsync", "vfr",
        str(out_dir / "scene_%04d.jpg"),
    ]
    logger.info("scene-change pass: %s", " ".join(cmd_scene))
    result = subprocess.run(cmd_scene, capture_output=True, text=True)
    # showinfo writes to stderr — parse for pts_time
    for match in re.finditer(r"pts_time:([\d.]+)", result.stderr):
        ts = float(match.group(1))
        # Find the corresponding output file by index
        idx = len(frames) + 1
        path = out_dir / f"scene_{idx:04d}.jpg"
        if path.exists():
            frames.append({
                "path": str(path),
                "timestamp": ts,
                "is_keyframe": True,
            })

    # Pass 2: regular 1fps sample.
    cmd_base = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vf", f"fps={fps}",
        str(out_dir / "base_%04d.jpg"),
    ]
    logger.info("base fps pass: %s", " ".join(cmd_base))
    subprocess.run(cmd_base, check=True, capture_output=True)
    base_files = sorted(out_dir.glob("base_*.jpg"))
    for i, path in enumerate(base_files):
        ts = i / fps
        frames.append({
            "path": str(path),
            "timestamp": ts,
            "is_keyframe": False,
        })

    # Sort by timestamp and dedupe close-together frames
    frames.sort(key=lambda f: f["timestamp"])
    deduped: list[dict] = []
    last_ts = -1.0
    for f in frames:
        if f["timestamp"] - last_ts >= 0.5:   # at most 2 frames/sec
            deduped.append(f)
            last_ts = f["timestamp"]

    logger.info("sampled %d frames over %.1fs", len(deduped), duration)
    return [{"frames": deduped, "fps_sampled": fps}][0] if False else deduped
