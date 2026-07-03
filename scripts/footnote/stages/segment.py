"""Stage 4: segment the transcript into topical units.

The segmenter is a DSPy module with a typed signature. We pass the full
transcript with timestamp markers; the LLM returns a list of segment
objects.

For long videos (>10 min) we chunk the transcript and segment each chunk
independently — solves context-window issues and produces consistent output.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


SEGMENTER_INSTRUCTIONS = """\
You are segmenting a video transcript into topical units. The transcript
comes from a talking-head educational video. Your job: identify the
*atomic topics* the speaker covers and emit one segment per topic.

For each segment, output:
  - start_sec, end_sec (float seconds)
  - text: the transcript text in this segment (verbatim, no summary)
  - topic: ONE sentence describing the topic
  - entities: list of named entities (people, places, books, organizations,
    historical events, technical terms) the speaker references in this segment
  - significance: float 0-1, how much would a contextual overlay benefit
    a viewer here? 0 = filler (intro/outro/transitions), 1 = critical claim
    with specific named entity that benefits from a source
  - type: one of [claim, fact, definition, person, event, chartable]

CONSTRAINTS:
  - Segments should be 5-30 seconds long. Don't over-segment.
  - Skip pure filler (greetings, "and so", "as I was saying").
  - significance=0 segments WILL still be emitted; the downstream proposer
    decides whether to overlay.
  - Entities should be UNIQUE, NAMED things. Skip generic terms.

OUTPUT FORMAT: valid JSON list of segment objects. No prose, no markdown,
just the JSON array.
"""


def chunk_transcript(transcript: dict, chunk_minutes: float = 5.0) -> list[dict]:
    """Split a transcript into ~chunk_minutes blocks at sentence boundaries.

    For each chunk, return the same shape as the input transcript with the
    words filtered to that time range. Used to keep LLM context windows
    manageable on long videos.
    """
    words = transcript["words"]
    if not words:
        return []
    chunk_sec = chunk_minutes * 60
    chunks: list[dict] = []
    cur: list[dict] = []
    cur_start = words[0]["start"]
    for w in words:
        if (w["start"] - cur_start) >= chunk_sec and cur:
            # End at last word ending in "."!?
            chunks.append({
                "words": cur,
                "duration_sec": cur[-1]["end"] - cur_start,
                "n_speakers": transcript.get("n_speakers", 1),
                "offset_sec": cur_start,
            })
            cur = []
            cur_start = w["start"]
        cur.append(w)
    if cur:
        chunks.append({
            "words": cur,
            "duration_sec": cur[-1]["end"] - cur_start,
            "n_speakers": transcript.get("n_speakers", 1),
            "offset_sec": cur_start,
        })
    return chunks


def segment_chunk(transcript_chunk: dict, model: str = "gpt-5-mini") -> list[dict]:
    """Run the segmenter LLM on a single transcript chunk."""
    try:
        from openai import OpenAI
    except ImportError as e:
        raise ImportError("pip install openai") from e

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set")

    from .transcribe import transcript_to_text
    transcript_text = transcript_to_text(transcript_chunk)

    client = OpenAI()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SEGMENTER_INSTRUCTIONS},
            {"role": "user", "content": f"Transcript:\n\n{transcript_text}\n\nReturn ONLY the JSON segment array."},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    raw = resp.choices[0].message.content or "{}"
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        logger.error("segment LLM returned non-JSON: %s", raw[:200])
        return []

    segments = parsed if isinstance(parsed, list) else parsed.get("segments", [])
    # Adjust timestamps to absolute video time
    offset = transcript_chunk.get("offset_sec", 0.0)
    for s in segments:
        s["start"] = float(s.get("start_sec", s.get("start", 0))) + offset
        s["end"] = float(s.get("end_sec", s.get("end", 0))) + offset
        s.setdefault("significance", 0.5)
        s.setdefault("entities", [])
        s.setdefault("type", "claim")
    return segments


def segment_full(transcript: dict, chunk_minutes: float = 5.0,
                  model: str = "gpt-5-mini") -> list[dict]:
    """Segment a full transcript by chunking + concatenating segment lists."""
    chunks = chunk_transcript(transcript, chunk_minutes)
    logger.info("segmenting %d chunks", len(chunks))
    all_segments: list[dict] = []
    for i, chunk in enumerate(chunks):
        logger.info("  chunk %d/%d", i + 1, len(chunks))
        all_segments.extend(segment_chunk(chunk, model=model))
    return all_segments
