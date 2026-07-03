"""DSPy modules for the Footnote pipeline.

Every LLM-driven stage is a DSPy module with a typed signature. This is the
load-bearing design choice: it lets us:
  1. Swap the underlying LM without code changes
  2. Optimize prompts with GEPA later
  3. Test each stage in isolation

Implementations are stubbed for now — fill in by porting the consolidator
pattern from mempol/dspy_consolidator/.
"""
from __future__ import annotations

from typing import Literal

try:
    import dspy
except ImportError as e:
    raise ImportError(
        "Install: pip install dspy-ai\n"
        "Then export OPENAI_API_KEY and configure: dspy.configure(lm=dspy.LM(...))"
    ) from e


# ─── Signatures ─────────────────────────────────────────────────────────────

class SegmenterSignature(dspy.Signature):
    """Read a video transcript with timestamps. Split it into coherent
    segments — each segment is one topic, claim, definition, person, event,
    or chartable data point. Skip filler.

    Each segment must include:
      - start_sec, end_sec
      - one-sentence topic description
      - named entities (people, places, books, concepts)
      - significance 0-1: how much would a viewer benefit from an overlay here?
      - type: one of claim / fact / definition / person / event / chartable

    Aim for segment lengths between 5 and 30 seconds. Don't over-segment.
    """
    transcript: str = dspy.InputField(desc="full transcript with [HH:MM:SS] word-level timestamps")
    frame_captions: str = dspy.InputField(desc="optional: short captions for sampled frames")
    segments: list[dict] = dspy.OutputField(desc="list of segment objects matching the schema above")


class OverlayProposerSignature(dspy.Signature):
    """Given a segment and a frame from that segment, decide whether a
    contextual overlay would help a viewer. If yes, propose what.

    HARD CONSTRAINTS — do not violate:
      - max one overlay per 8 seconds of video
      - skip if speaker's face occupies > 60% of frame (intense moment)
      - skip if segment significance < 0.4
      - skip if segment is mid-explanation of the same topic the prior overlay covered

    SOFT GUIDANCE:
      - prefer overlays on named entities the speaker references without
        explaining (e.g., "Napoleon's travelling library")
      - prefer overlays that add receipts (citation + image) over rephrasings
      - prefer overlays on claims that benefit from a source

    For each overlay, output:
      - timestamp_sec: when it should appear
      - duration_sec: how long it stays on screen (typically 6-12s)
      - type: fact-card | person-card | image | chart
      - query: what to research / fetch
      - placement_intent: lower-third | upper-third | side
      - reason: 1-sentence rationale for why this helps the viewer
    """
    segment: dict = dspy.InputField(desc="segment object from Segmenter output")
    frame_image: str = dspy.InputField(desc="base64-encoded frame image or URL")
    prior_overlays: list[dict] = dspy.InputField(desc="overlays already placed earlier")
    overlays: list[dict] = dspy.OutputField(desc="0 or more overlay proposals")


class SnippetRewriterSignature(dspy.Signature):
    """Given a fetched fact snippet (from Wikipedia / Exa / arxiv), compress
    it into 2-3 lines suitable for a video overlay.

    RULES:
      - Never invent information not in the source.
      - Preserve specific facts (numbers, dates, proper nouns).
      - Strip filler ("Generally speaking", "It is widely known that").
      - Output should fit in ~250 characters.

    Also output highlight_spans — substring ranges of the most important
    phrases to visually highlight in the overlay.
    """
    source_text: str = dspy.InputField(desc="raw text from search / Wikipedia")
    overlay_query: str = dspy.InputField(desc="the original query that prompted this overlay")
    overlay_text: str = dspy.OutputField(desc="2-3 line condensed text for the overlay")
    highlight_spans: list[tuple[int, int]] = dspy.OutputField(desc="(start, end) char ranges to highlight")


class LayoutPlannerSignature(dspy.Signature):
    """Given a video frame and an overlay text, decide where on screen the
    overlay should land.

    RULES (in priority order):
      1. Never cover the speaker's face.
      2. Never cover the speaker's hands if they're gesturing meaningfully.
      3. Prefer the third opposite to the speaker's gaze.
      4. Prefer lower-third if face is in upper portion.
      5. Maintain ~20px margin from frame edges.

    Output:
      - bbox: (x, y, w, h) in 0-1 normalized coords
      - entry_animation: fade | slide-up | slide-left | slide-right
      - exit_animation: fade | slide-down | slide-right
    """
    frame_image: str = dspy.InputField(desc="base64-encoded frame image")
    overlay_text: str = dspy.InputField(desc="the text being overlaid")
    overlay_type: str = dspy.InputField(desc="fact-card | person-card | image | chart")
    face_bboxes: list[tuple[float, float, float, float]] = dspy.InputField(desc="detected face bboxes")
    bbox: tuple[float, float, float, float] = dspy.OutputField(desc="(x, y, w, h) 0-1 normalized")
    entry_animation: str = dspy.OutputField()
    exit_animation: str = dspy.OutputField()


class GuideWriterSignature(dspy.Signature):
    """Produce a structured markdown article from a video's transcript,
    segments, and researched overlays.

    Format:
      - Title (from video metadata or first segment)
      - Intro paragraph (2-3 sentences)
      - One H2 per major segment cluster
      - Body text from transcript, lightly edited for readability
      - Footnotes for citations
      - Embedded frame screenshots at segment boundaries
      - Cross-link to video timestamps via anchor links

    Output should be SEO-friendly and PDF-renderable.
    """
    transcript: str = dspy.InputField()
    segments: list[dict] = dspy.InputField()
    researched_overlays: list[dict] = dspy.InputField()
    title: str = dspy.InputField()
    guide_markdown: str = dspy.OutputField(desc="full markdown article")


# ─── Modules ────────────────────────────────────────────────────────────────

class Segmenter(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self.predict = dspy.ChainOfThought(SegmenterSignature)

    def forward(self, transcript: str, frame_captions: str = "") -> dspy.Prediction:
        return self.predict(transcript=transcript, frame_captions=frame_captions)


class OverlayProposer(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self.predict = dspy.ChainOfThought(OverlayProposerSignature)

    def forward(self, segment: dict, frame_image: str,
                 prior_overlays: list[dict]) -> dspy.Prediction:
        return self.predict(segment=segment, frame_image=frame_image,
                              prior_overlays=prior_overlays)


class SnippetRewriter(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self.predict = dspy.ChainOfThought(SnippetRewriterSignature)

    def forward(self, source_text: str, overlay_query: str) -> dspy.Prediction:
        return self.predict(source_text=source_text, overlay_query=overlay_query)


class LayoutPlanner(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self.predict = dspy.Predict(LayoutPlannerSignature)

    def forward(self, frame_image: str, overlay_text: str,
                 overlay_type: str, face_bboxes: list) -> dspy.Prediction:
        return self.predict(frame_image=frame_image, overlay_text=overlay_text,
                              overlay_type=overlay_type, face_bboxes=face_bboxes)


class GuideWriter(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self.predict = dspy.ChainOfThought(GuideWriterSignature)

    def forward(self, transcript: str, segments: list[dict],
                 researched_overlays: list[dict], title: str) -> dspy.Prediction:
        return self.predict(transcript=transcript, segments=segments,
                              researched_overlays=researched_overlays, title=title)


# ─── Footnote pipeline as one DSPy module (optimizable end-to-end) ──────────

class FootnotePipeline(dspy.Module):
    """The whole pipeline as one DSPy module. GEPA can optimize each stage's
    prompt with end-to-end downstream feedback (creator approval / rejection
    of proposed overlays via the human-in-loop UI).
    """
    def __init__(self) -> None:
        super().__init__()
        self.segmenter = Segmenter()
        self.proposer = OverlayProposer()
        self.rewriter = SnippetRewriter()
        self.layout = LayoutPlanner()
        self.guide_writer = GuideWriter()
