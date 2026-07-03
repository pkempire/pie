"""Stage 2: transcribe video audio with word-level timestamps.

Uses OpenAI Whisper API via the openai SDK. Word-level timestamps via the
`timestamp_granularities=["word", "segment"]` flag introduced in mid-2024.

Speaker diarization is intentionally OFF in the MVP — most talking-head
videos have one speaker; we add diarization in v2 via pyannote-audio.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


def extract_audio(video_path: Path, sample_rate: int = 16000) -> Path:
    """ffmpeg: extract a mono 16kHz WAV from any video container. Whisper
    accepts mp4 directly but converting first keeps file sizes small and
    decouples decoding from API upload."""
    out = Path(tempfile.mkstemp(suffix=".wav")[1])
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-ac", "1", "-ar", str(sample_rate),
        "-vn", "-f", "wav", str(out),
    ]
    logger.info("extract_audio: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, capture_output=True)
    return out


def transcribe(video_path: Path, model: str = "whisper-1") -> dict:
    """Transcribe with word-level timestamps. Returns a dict matching the
    TranscriptArtifact shape: words=[{text,start,end,speaker_id}, ...],
    duration_sec, n_speakers."""
    try:
        from openai import OpenAI
    except ImportError as e:
        raise ImportError("pip install openai") from e

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set")

    audio = extract_audio(video_path)
    client = OpenAI()

    logger.info("whisper: uploading %s (%d bytes)", audio, audio.stat().st_size)
    with audio.open("rb") as f:
        resp = client.audio.transcriptions.create(
            file=f,
            model=model,
            response_format="verbose_json",
            timestamp_granularities=["word", "segment"],
        )

    # OpenAI returns a Pydantic-like object; coerce to dict.
    data = resp.model_dump() if hasattr(resp, "model_dump") else dict(resp)
    words_raw = data.get("words", []) or []
    words = [
        {
            "text": w.get("word", "").strip(),
            "start": float(w.get("start", 0.0)),
            "end": float(w.get("end", 0.0)),
            "speaker_id": 0,
        }
        for w in words_raw
        if w.get("word", "").strip()
    ]

    duration = float(data.get("duration", 0.0)) or (
        words[-1]["end"] if words else 0.0
    )
    audio.unlink(missing_ok=True)

    logger.info("whisper: %d words, %.1fs duration", len(words), duration)
    return {
        "words": words,
        "duration_sec": duration,
        "n_speakers": 1,   # MVP: single-speaker assumption
    }


def transcript_to_text(transcript: dict, with_timestamps: bool = True) -> str:
    """Render a transcript as a single string for LLM consumption.
    With timestamps every ~10 words for context."""
    out: list[str] = []
    buf: list[str] = []
    last_ts = -10.0
    for w in transcript["words"]:
        if with_timestamps and (w["start"] - last_ts) >= 10.0:
            if buf:
                out.append(" ".join(buf))
                buf = []
            mm = int(w["start"] // 60)
            ss = int(w["start"] % 60)
            out.append(f"\n[{mm:02d}:{ss:02d}]")
            last_ts = w["start"]
        buf.append(w["text"])
    if buf:
        out.append(" ".join(buf))
    return " ".join(out).strip()
