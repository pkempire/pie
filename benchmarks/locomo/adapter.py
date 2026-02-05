"""
LoCoMo → PIE Format Adapter

Converts LoCoMo's very long-term conversational memory format into PIE
Conversation objects, preserving timestamps and session structure.

LoCoMo format (from snap-research/locomo):
  - Each sample contains a conversation with multiple sessions
  - Sessions have timestamps (session_<n>_date_time)
  - Turns contain speaker, dia_id, text, and optional image data
  - QA annotations with 5 question types: single-hop, multi-hop, temporal,
    commonsense/world-knowledge, adversarial
  - Event summaries for each speaker per session

PIE format:
  Conversation objects with Turn objects, sorted chronologically.

Key differences from LongMemEval:
  - LoCoMo has 2 speakers (peer-to-peer chat, not user+assistant)
  - Much longer conversations (~300 turns, 9K tokens avg, up to 35 sessions)
  - Includes multimodal data (images with captions)
  - Event graphs with causal relationships
"""

from __future__ import annotations
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pie.core.models import Turn, Conversation


# ── Date Parsing ──────────────────────────────────────────────────────────────

# LoCoMo date formats:
# "18 May 2023 at 1:39 pm"
# "7 April 2023 at 4:01 pm"
_DATE_RE_12H = re.compile(
    r"(\d{1,2})\s+(\w+)\s+(\d{4})\s+at\s+(\d{1,2}):(\d{2})\s*(am|pm)",
    re.IGNORECASE
)

MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7,
    "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def parse_locomo_date(date_str: str) -> float:
    """
    Parse a LoCoMo date string into a Unix timestamp.

    Format: "18 May 2023 at 1:39 pm"
    Returns: float (Unix timestamp, UTC)
    """
    if not date_str:
        return 0.0

    m = _DATE_RE_12H.match(date_str.strip())
    if not m:
        raise ValueError(f"Cannot parse LoCoMo date: {date_str!r}")

    day = int(m.group(1))
    month_str = m.group(2).lower()
    year = int(m.group(3))
    hour = int(m.group(4))
    minute = int(m.group(5))
    ampm = m.group(6).lower()

    month = MONTH_MAP.get(month_str)
    if not month:
        raise ValueError(f"Unknown month: {month_str}")

    # Convert 12h to 24h
    if ampm == "pm" and hour != 12:
        hour += 12
    elif ampm == "am" and hour == 12:
        hour = 0

    dt = datetime(year, month, day, hour, minute, tzinfo=timezone.utc)
    return dt.timestamp()


def format_date_for_context(date_str: str) -> str:
    """Format a LoCoMo date into a human-readable string for context."""
    try:
        ts = parse_locomo_date(date_str)
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        return dt.strftime("%B %d, %Y at %I:%M %p")
    except Exception:
        return date_str


# ── Session → Conversation Conversion ────────────────────────────────────────


def session_to_conversation(
    session: list[dict[str, Any]],
    date_str: str,
    session_index: int,
    sample_id: str,
    speaker_a: str,
    speaker_b: str,
) -> Conversation:
    """
    Convert a single LoCoMo session into a PIE Conversation.

    Args:
        session: List of turn dicts with {name, dia_id, text, ...}.
        date_str: Session timestamp string.
        session_index: 0-based index in the conversation.
        sample_id: The parent sample's ID.
        speaker_a: Name of first speaker.
        speaker_b: Name of second speaker.

    Returns:
        A PIE Conversation object with properly timestamped turns.
    """
    timestamp = parse_locomo_date(date_str)

    turns = []
    for turn_idx, raw_turn in enumerate(session):
        name = raw_turn.get("name", "")
        text = raw_turn.get("text", "")

        # Skip empty turns
        if not text.strip():
            continue

        # Map speaker names to roles (user/assistant pattern)
        # In LoCoMo, both are users — we'll treat speaker_a as "user"
        # and speaker_b as "assistant" for compatibility
        if name == speaker_a:
            role = "user"
        elif name == speaker_b:
            role = "assistant"
        else:
            role = "user"

        # Include image captions if present
        if raw_turn.get("blip_caption"):
            text = f"{text}\n[Image: {raw_turn['blip_caption']}]"

        # Space turns a few seconds apart
        turn_ts = timestamp + (turn_idx * 5)

        turns.append(Turn(
            role=role,
            text=text.strip(),
            timestamp=turn_ts,
        ))

    convo_id = f"{sample_id}_session_{session_index}"

    # Title from first user message
    title = "Chat session"
    for t in turns:
        if t.role == "user":
            title = t.text[:80].replace("\n", " ")
            if len(t.text) > 80:
                title += "..."
            break

    return Conversation(
        id=convo_id,
        title=title,
        created_at=timestamp,
        updated_at=timestamp + (len(turns) * 5),
        model=None,
        turns=turns,
    )


def sample_to_conversations(sample: dict[str, Any]) -> list[Conversation]:
    """
    Convert an entire LoCoMo sample into a chronologically sorted
    list of PIE Conversations.

    Args:
        sample: A LoCoMo sample dict with conversation, qa, etc.

    Returns:
        Chronologically sorted list of Conversation objects.
    """
    convo = sample["conversation"]
    sample_id = sample["sample_id"]
    speaker_a = convo.get("speaker_a", "Speaker A")
    speaker_b = convo.get("speaker_b", "Speaker B")

    conversations = []

    # Find all session keys (session_1, session_2, etc.)
    session_keys = sorted(
        [k for k in convo.keys() if k.startswith("session_") and not k.endswith("_date_time")],
        key=lambda x: int(x.split("_")[1])
    )

    for i, session_key in enumerate(session_keys):
        session = convo[session_key]
        date_key = f"{session_key}_date_time"
        date_str = convo.get(date_key, "")

        c = session_to_conversation(
            session=session,
            date_str=date_str,
            session_index=i,
            sample_id=sample_id,
            speaker_a=speaker_a,
            speaker_b=speaker_b,
        )
        if c.turns:
            conversations.append(c)

    conversations.sort(key=lambda c: c.created_at)
    return conversations


# ── Dataset Loading ──────────────────────────────────────────────────────────


def load_dataset(
    path: str | Path = "benchmarks/locomo/data/locomo10.json",
) -> list[dict[str, Any]]:
    """
    Load the LoCoMo dataset.

    Returns list of sample dicts, each with:
      - sample_id
      - conversation (sessions, speakers, timestamps)
      - qa (question-answer annotations)
      - observation (session observations)
      - session_summary (session-level summaries)
      - event_summary (per-speaker event summaries)
    """
    path = Path(path)
    with open(path) as f:
        data = json.load(f)

    # LoCoMo10 is a dict with sample_id keys
    if isinstance(data, dict):
        return [{"sample_id": k, **v} for k, v in data.items()]
    return data


def flatten_qa(dataset: list[dict]) -> list[dict]:
    """
    Flatten the dataset to individual QA items.

    Each QA item includes:
      - question_id: unique ID
      - question: the question text
      - question_type: single_hop, multi_hop, temporal, adversarial, commonsense
      - answer: the ground truth answer
      - evidence: list of dia_ids with evidence
      - sample_id: parent conversation ID
      - conversation: full conversation data
    """
    items = []
    for sample in dataset:
        sample_id = sample["sample_id"]
        qa_list = sample.get("qa", [])
        conversation = sample.get("conversation", {})

        for i, qa in enumerate(qa_list):
            items.append({
                "question_id": f"{sample_id}_q{i}",
                "question": qa.get("question", ""),
                "question_type": qa.get("category", "unknown"),
                "answer": qa.get("answer", ""),
                "evidence": qa.get("evidence", []),
                "sample_id": sample_id,
                "conversation": conversation,
            })

    return items


def filter_by_category(
    items: list[dict],
    category: str,
) -> list[dict]:
    """Filter QA items to a single question_type category."""
    return [item for item in items if item["question_type"] == category]


def filter_by_ids(
    items: list[dict],
    question_ids: list[str],
) -> list[dict]:
    """Filter items to specific question IDs."""
    id_set = set(question_ids)
    return [item for item in items if item["question_id"] in id_set]


# ── Haystack Formatting ──────────────────────────────────────────────────────


def format_conversation_as_text(
    conversation: dict[str, Any],
    max_chars: int | None = None,
) -> str:
    """
    Format an entire LoCoMo conversation as a single text document.
    Used for the full-context baseline.

    Returns a formatted string like:
      [Session 1 — May 18, 2023 at 1:39 PM]
      Alice: ...
      Bob: ...
    """
    speaker_a = conversation.get("speaker_a", "Speaker A")
    speaker_b = conversation.get("speaker_b", "Speaker B")

    session_keys = sorted(
        [k for k in conversation.keys() if k.startswith("session_") and not k.endswith("_date_time")],
        key=lambda x: int(x.split("_")[1])
    )

    parts = []
    total_chars = 0

    for i, session_key in enumerate(session_keys):
        session = conversation[session_key]
        date_key = f"{session_key}_date_time"
        date_str = conversation.get(date_key, "")
        human_date = format_date_for_context(date_str)

        header = f"\n[Session {i + 1} — {human_date}]"
        parts.append(header)
        total_chars += len(header)

        for turn in session:
            name = turn.get("name", "Unknown")
            text = turn.get("text", "").strip()
            if not text:
                continue

            # Include image caption
            if turn.get("blip_caption"):
                text = f"{text} [Image: {turn['blip_caption']}]"

            line = f"{name}: {text}"

            if max_chars and total_chars + len(line) > max_chars:
                parts.append("\n[... remaining sessions truncated ...]")
                return "\n".join(parts)

            parts.append(line)
            total_chars += len(line)

        parts.append("")  # blank line between sessions

    return "\n".join(parts)


def format_session_as_text(
    session: list[dict],
    date_str: str,
    session_index: int,
) -> str:
    """Format a single session as text with a header."""
    human_date = format_date_for_context(date_str)
    lines = [f"[Session {session_index + 1} — {human_date}]"]
    for turn in session:
        name = turn.get("name", "Unknown")
        text = turn.get("text", "").strip()
        if text:
            if turn.get("blip_caption"):
                text = f"{text} [Image: {turn['blip_caption']}]"
            lines.append(f"{name}: {text}")
    return "\n".join(lines)


# ── Observation/Summary Extraction ───────────────────────────────────────────


def get_session_observations(sample: dict) -> list[str]:
    """Get pre-computed observations for each session."""
    observation = sample.get("observation", {})
    results = []
    for key in sorted(observation.keys()):
        if key.startswith("session_") and key.endswith("_observation"):
            results.append(observation[key])
    return results


def get_session_summaries(sample: dict) -> list[str]:
    """Get pre-computed session summaries."""
    summaries = sample.get("session_summary", {})
    results = []
    for key in sorted(summaries.keys()):
        if key.startswith("session_") and key.endswith("_summary"):
            results.append(summaries[key])
    return results


# ── Summary Stats ─────────────────────────────────────────────────────────────


def dataset_stats(dataset: list[dict]) -> dict:
    """Compute summary statistics for the LoCoMo dataset."""
    from collections import Counter

    qa_items = flatten_qa(dataset)
    type_counts = Counter(item["question_type"] for item in qa_items)

    session_counts = []
    turn_counts = []
    char_counts = []

    for sample in dataset:
        convo = sample.get("conversation", {})
        session_keys = [k for k in convo.keys() if k.startswith("session_") and not k.endswith("_date_time")]
        session_counts.append(len(session_keys))

        total_turns = 0
        total_chars = 0
        for key in session_keys:
            session = convo[key]
            total_turns += len(session)
            for turn in session:
                total_chars += len(turn.get("text", ""))

        turn_counts.append(total_turns)
        char_counts.append(total_chars)

    def safe_stats(lst):
        if not lst:
            return {"mean": 0, "min": 0, "max": 0}
        return {
            "mean": sum(lst) / len(lst),
            "min": min(lst),
            "max": max(lst),
        }

    return {
        "total_conversations": len(dataset),
        "total_questions": len(qa_items),
        "question_types": dict(type_counts),
        "sessions_per_conversation": safe_stats(session_counts),
        "turns_per_conversation": safe_stats(turn_counts),
        "chars_per_conversation": safe_stats(char_counts),
    }
