"""Stage 5: propose overlays.

For each segment, the LLM decides: would an overlay help here? If so,
what kind, when exactly, for how long, and what to research?

Two key constraints encoded in the prompt:
  - max one overlay per 8 seconds of video (cognitive load)
  - skip if speaker face dominates the frame (visual budget)

We pass a frame from each segment so the LLM can reason about visual
content. If gpt-5 vision is too expensive, fall back to caption text from
a separate vision pass.
"""
from __future__ import annotations

import base64
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


PROPOSER_INSTRUCTIONS = """\
You are deciding whether to overlay contextual information on a talking-
head educational video. You see one segment of the video (transcript text
+ topic + entities) and a frame from that segment.

RULES (hard):
  - Output AT MOST one overlay per segment.
  - Output NO overlays for segments where significance < 0.4.
  - Output NO overlays where the segment is mid-explanation of the same
    topic that a prior segment already covered.
  - Output NO overlays for filler (intros, outros, transitions, jokes).
  - When unsure, prefer NO overlay over a weak one. Quality > coverage.

WHEN AN OVERLAY HELPS (output yes):
  - Speaker references a named person, book, place, organization, or event
    WITHOUT explaining it (the audience would benefit from context)
  - Speaker makes a specific factual claim that would benefit from a citation
  - Speaker mentions a number or stat that an image / chart would clarify
  - Speaker references something visually interesting (an object, a portrait,
    an artifact) that an image could show

FOR EACH OVERLAY, output:
  - timestamp_sec: when on the timeline (use the segment's start + a little)
  - duration_sec: 6-12 seconds typically
  - type: fact-card | person-card | image | chart
  - query: precise search query for the fact / image (e.g., "Napoleon
    travelling library miniature books campaign")
  - placement_intent: lower-third | upper-third | side-left | side-right
  - reason: ONE sentence explaining why this overlay helps the viewer

OUTPUT FORMAT: valid JSON object {"overlays": [...]}. If no overlay is
warranted, return {"overlays": []}.
"""


def _b64_image(path: str | Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def _pick_frame_for_segment(segment: dict, frames: list[dict]) -> dict | None:
    """Pick the frame closest to segment midpoint."""
    if not frames:
        return None
    mid = (segment["start"] + segment["end"]) / 2
    best = min(frames, key=lambda f: abs(f["timestamp"] - mid))
    return best


def propose_for_segment(segment: dict, frame: dict | None,
                         prior_overlays: list[dict],
                         model: str = "gpt-5-mini") -> list[dict]:
    """LLM call with optional vision input. Returns 0-1 overlay proposals."""
    try:
        from openai import OpenAI
    except ImportError as e:
        raise ImportError("pip install openai") from e

    client = OpenAI()

    # Build user content. Include frame if available.
    user_content: list[dict] = [
        {"type": "text", "text": (
            f"Segment topic: {segment.get('topic', '')}\n"
            f"Significance: {segment.get('significance', 0)}\n"
            f"Entities: {segment.get('entities', [])}\n"
            f"Type: {segment.get('type', 'claim')}\n"
            f"Time: {segment['start']:.1f}-{segment['end']:.1f}s\n\n"
            f"Transcript text:\n{segment.get('text', '')}\n\n"
            f"Prior overlays placed (don't repeat topics): {json.dumps(prior_overlays[-3:], default=str)}"
        )}
    ]
    if frame and Path(frame["path"]).exists():
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{_b64_image(frame['path'])}"},
        })

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": PROPOSER_INSTRUCTIONS},
            {"role": "user", "content": user_content},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
    )
    raw = resp.choices[0].message.content or "{}"
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return []
    return parsed.get("overlays", [])


def propose_all(segments: list[dict], frames: list[dict],
                 min_spacing_sec: float = 8.0,
                 model: str = "gpt-5-mini") -> list[dict]:
    """Run the proposer over all segments. Enforces minimum spacing between
    accepted overlays."""
    proposals: list[dict] = []
    last_ts = -float("inf")
    for i, seg in enumerate(segments):
        if seg.get("start", 0) - last_ts < min_spacing_sec:
            continue   # spacing not met yet
        frame = _pick_frame_for_segment(seg, frames)
        new = propose_for_segment(seg, frame, proposals, model=model)
        if not new:
            continue
        # Keep only the first (we ask for at most 1)
        ov = new[0]
        ov.setdefault("timestamp_sec", seg["start"] + 1.0)
        ov.setdefault("duration_sec", 8.0)
        proposals.append(ov)
        last_ts = ov["timestamp_sec"]
        logger.info("[%d/%d] proposed overlay @ %.1fs: %s",
                     i + 1, len(segments), ov["timestamp_sec"],
                     ov.get("query", "")[:60])
    return proposals
