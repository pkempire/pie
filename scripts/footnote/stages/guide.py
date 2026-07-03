"""Stage 10: produce a structured written guide from the pipeline artifacts.

This is the second output of Footnote — a fully cited markdown article
generated from the transcript + segments + researched overlays.

The guide reads like a high-quality blog post: title, intro, H2s per
topic cluster, body text from transcript (lightly cleaned), inline
citations linking to the same sources as the video overlays.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


GUIDE_INSTRUCTIONS = """\
You are turning a video transcript + segment list + researched fact-overlays
into a long-form, well-cited markdown article. The article should:

  - Have a descriptive H1 title (drawn from the video's content)
  - Open with a 2-3 sentence intro that hooks the reader
  - Group segments into ~3-7 H2 sections, each covering a coherent topic
  - Write the BODY in your own words — do not just copy transcript verbatim.
    Smooth out filler, fix grammar, make it readable as an article.
  - Include inline citations using markdown links wherever a fact came
    from a researched overlay: e.g. "Napoleon carried a [travelling
    library](https://en.wikipedia.org/wiki/...)..."
  - Where the video has a chart or image overlay, embed it: ![alt](image_url)
  - End with a "Further reading" H2 listing all unique citation URLs

CRITICAL CONSTRAINTS:
  - Do NOT invent facts not in the transcript or researched overlays.
  - Do NOT add prose that the transcript doesn't support.
  - Preserve specific facts (numbers, dates, names) verbatim.
  - Keep the voice / style consistent throughout.

OUTPUT: a single markdown document. No JSON wrapper, no preamble. Start
with the H1 title directly.
"""


def _condense_transcript(transcript: dict, max_chars: int = 20000) -> str:
    """Get plain transcript text without timestamps, capped."""
    text = " ".join(w["text"] for w in transcript.get("words", []))
    if len(text) > max_chars:
        # Take first half + last half
        half = max_chars // 2
        text = text[:half] + "\n\n[...transcript continues...]\n\n" + text[-half:]
    return text


def write_guide(transcript: dict, segments: list[dict],
                 researched_overlays: list[dict],
                 model: str = "gpt-5-mini") -> str:
    """Generate the markdown article. Returns the full article string."""
    try:
        from openai import OpenAI
    except ImportError as e:
        raise ImportError("pip install openai") from e

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set")

    transcript_text = _condense_transcript(transcript)
    segments_summary = json.dumps(
        [{"start": s["start"], "topic": s.get("topic", ""), "entities": s.get("entities", [])}
         for s in segments],
        indent=2,
    )
    overlays_summary = json.dumps(
        [{
            "timestamp": ro["proposal"].get("timestamp_sec"),
            "text": ro["text"],
            "citation": ro["citation_url"],
            "label": ro["citation_label"],
            "image": ro.get("image_url"),
        } for ro in researched_overlays],
        indent=2,
    )

    user_msg = (
        f"## Transcript\n\n{transcript_text}\n\n"
        f"## Segments\n\n{segments_summary}\n\n"
        f"## Researched overlays (use these as your fact-citation source)\n\n"
        f"{overlays_summary}\n\n"
        "Write the article now."
    )

    client = OpenAI()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": GUIDE_INSTRUCTIONS},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.4,
    )
    return resp.choices[0].message.content or ""
