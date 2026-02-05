"""
LongMemEval → PIE Format Adapter

Converts LongMemEval's haystack format into PIE Conversation objects,
preserving timestamps and session structure.

LongMemEval format:
  haystack_sessions: list of sessions, each a list of [{role, content}] turns
  haystack_dates: parallel list of date strings ("2023/05/20 (Sat) 02:21")

PIE format:
  Conversation objects with Turn objects, sorted chronologically.

Each LongMemEval question has its OWN haystack (~53 sessions), representing
a synthetic user's chat history. We convert each haystack into a list of
PIE Conversations for ingestion into a fresh world model.
"""

from __future__ import annotations
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pie.core.models import Turn, Conversation


# ── Date Parsing ──────────────────────────────────────────────────────────────

# LongMemEval date format: "2023/05/20 (Sat) 02:21"
_DATE_RE = re.compile(
    r"(\d{4})/(\d{2})/(\d{2})\s+\([A-Za-z]+\)\s+(\d{2}):(\d{2})"
)


def parse_longmemeval_date(date_str: str) -> float:
    """
    Parse a LongMemEval date string into a Unix timestamp.

    Format: "2023/05/20 (Sat) 02:21"
    Returns: float (Unix timestamp, UTC)
    """
    m = _DATE_RE.match(date_str.strip())
    if not m:
        raise ValueError(f"Cannot parse LongMemEval date: {date_str!r}")
    year, month, day, hour, minute = (int(x) for x in m.groups())
    dt = datetime(year, month, day, hour, minute, tzinfo=timezone.utc)
    return dt.timestamp()


def format_date_for_context(date_str: str) -> str:
    """
    Format a LongMemEval date into a human-readable string for context.

    "2023/05/20 (Sat) 02:21" → "May 20, 2023 at 02:21"
    """
    m = _DATE_RE.match(date_str.strip())
    if not m:
        return date_str
    year, month, day, hour, minute = (int(x) for x in m.groups())
    dt = datetime(year, month, day, hour, minute)
    return dt.strftime("%B %d, %Y at %H:%M")


def parse_question_date(date_str: str) -> float:
    """Parse the question_date field (same format as haystack_dates)."""
    return parse_longmemeval_date(date_str)


# ── Session → Conversation Conversion ────────────────────────────────────────


def session_to_conversation(
    session: list[dict[str, str]],
    date_str: str,
    session_index: int,
    question_id: str,
) -> Conversation:
    """
    Convert a single LongMemEval session into a PIE Conversation.

    Args:
        session: List of {role, content} turn dicts.
        date_str: The corresponding haystack_dates entry.
        session_index: 0-based index in the haystack.
        question_id: The parent question's ID (for tracing).

    Returns:
        A PIE Conversation object with properly timestamped turns.
    """
    timestamp = parse_longmemeval_date(date_str)

    turns = []
    for turn_idx, raw_turn in enumerate(session):
        role = raw_turn.get("role", "user")
        content = raw_turn.get("content", "")
        if not content.strip():
            continue

        # Space turns a few seconds apart so they have distinct timestamps
        turn_ts = timestamp + (turn_idx * 5)

        turns.append(Turn(
            role=role,
            text=content.strip(),
            timestamp=turn_ts,
        ))

    # Generate a stable conversation ID from question + session index
    convo_id = f"{question_id}_session_{session_index}"

    # Generate a descriptive title from the first user message
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


def haystack_to_conversations(
    haystack_sessions: list[list[dict[str, str]]],
    haystack_dates: list[str],
    question_id: str,
) -> list[Conversation]:
    """
    Convert an entire LongMemEval haystack into a chronologically sorted
    list of PIE Conversations.

    Args:
        haystack_sessions: List of sessions, each a list of turns.
        haystack_dates: Parallel list of date strings.
        question_id: The parent question's ID.

    Returns:
        Chronologically sorted list of Conversation objects.
    """
    if len(haystack_sessions) != len(haystack_dates):
        raise ValueError(
            f"Mismatch: {len(haystack_sessions)} sessions vs "
            f"{len(haystack_dates)} dates for question {question_id}"
        )

    conversations = []
    for i, (session, date_str) in enumerate(zip(haystack_sessions, haystack_dates)):
        convo = session_to_conversation(
            session=session,
            date_str=date_str,
            session_index=i,
            question_id=question_id,
        )
        if convo.turns:  # skip empty sessions
            conversations.append(convo)

    # Sort chronologically (should already be, but be safe)
    conversations.sort(key=lambda c: c.created_at)
    return conversations


# ── Dataset Loading ──────────────────────────────────────────────────────────


def load_dataset(
    path: str | Path = "benchmarks/longmemeval/data/longmemeval_s_cleaned.json",
) -> list[dict[str, Any]]:
    """
    Load the full LongMemEval dataset.

    Returns list of question dicts, each with:
      - question_id, question_type, question, answer, question_date
      - haystack_sessions, haystack_dates
      - answer_session_ids, haystack_session_ids
    """
    path = Path(path)
    with open(path) as f:
        data = json.load(f)
    return data


def load_oracle_dataset(
    path: str | Path = "benchmarks/longmemeval/data/longmemeval_oracle.json",
) -> list[dict[str, Any]]:
    """Load the oracle version (minimal sessions, only evidence)."""
    path = Path(path)
    with open(path) as f:
        data = json.load(f)
    return data


def filter_by_category(
    dataset: list[dict],
    category: str,
) -> list[dict]:
    """Filter dataset to a single question_type category."""
    return [item for item in dataset if item["question_type"] == category]


def filter_by_ids(
    dataset: list[dict],
    question_ids: list[str],
) -> list[dict]:
    """Filter dataset to specific question IDs."""
    id_set = set(question_ids)
    return [item for item in dataset if item["question_id"] in id_set]


# ── Haystack Formatting (for full-context baseline) ─────────────────────────


def format_haystack_as_text(
    haystack_sessions: list[list[dict[str, str]]],
    haystack_dates: list[str],
    max_chars: int | None = None,
) -> str:
    """
    Format an entire haystack as a single text document.
    Used for the full-context baseline (stuff everything into the prompt).

    Returns a formatted string like:
      [Session 1 — May 20, 2023 at 02:21]
      User: ...
      Assistant: ...

      [Session 2 — May 22, 2023 at 14:30]
      ...
    """
    parts = []
    total_chars = 0

    for i, (session, date_str) in enumerate(zip(haystack_sessions, haystack_dates)):
        human_date = format_date_for_context(date_str)
        header = f"\n[Session {i + 1} — {human_date}]"
        parts.append(header)
        total_chars += len(header)

        for turn in session:
            role = turn.get("role", "user").capitalize()
            content = turn.get("content", "").strip()
            if not content:
                continue

            line = f"{role}: {content}"

            if max_chars and total_chars + len(line) > max_chars:
                parts.append("\n[... remaining sessions truncated ...]")
                return "\n".join(parts)

            parts.append(line)
            total_chars += len(line)

        parts.append("")  # blank line between sessions

    return "\n".join(parts)


def format_session_as_text(
    session: list[dict[str, str]],
    date_str: str,
    session_index: int,
) -> str:
    """Format a single session as text with a header."""
    human_date = format_date_for_context(date_str)
    lines = [f"[Session {session_index + 1} — {human_date}]"]
    for turn in session:
        role = turn.get("role", "user").capitalize()
        content = turn.get("content", "").strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


# ── Summary Stats ─────────────────────────────────────────────────────────────


def dataset_stats(dataset: list[dict]) -> dict:
    """Compute summary statistics for the dataset."""
    from collections import Counter

    type_counts = Counter(item["question_type"] for item in dataset)

    session_counts = [len(item["haystack_sessions"]) for item in dataset]
    turn_counts = [
        sum(len(s) for s in item["haystack_sessions"]) for item in dataset
    ]
    char_counts = [
        sum(
            len(t.get("content", ""))
            for s in item["haystack_sessions"]
            for t in s
        )
        for item in dataset
    ]

    return {
        "total_questions": len(dataset),
        "question_types": dict(type_counts),
        "sessions_per_question": {
            "mean": sum(session_counts) / len(session_counts),
            "min": min(session_counts),
            "max": max(session_counts),
        },
        "turns_per_question": {
            "mean": sum(turn_counts) / len(turn_counts),
            "min": min(turn_counts),
            "max": max(turn_counts),
        },
        "chars_per_question": {
            "mean": sum(char_counts) / len(char_counts),
            "min": min(char_counts),
            "max": max(char_counts),
        },
    }
