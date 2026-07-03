"""Egocentric worldline compiler — deterministic temporal grounding.

Timestamp injection hands the model raw coordinates ("8 May 2023") and forces
it to do four hard things implicitly in one forward pass: date arithmetic,
chronological ordering, validity resolution, and gap inference. Models are bad
at all four. This module does all four in Python and presents time the way an
agent should experience it:

  1. Egocentric offsets — every event is rendered relative to NOW
     ("5 months, 12 days ago"), not as an absolute coordinate the model must
     subtract in its head.
  2. Explicit passage-of-time markers — "( 6 weeks pass )" gap lines between
     consecutive events, so elapsed time is a token the model reads, not a
     subtraction it must perform. Long gaps carry a staleness warning.
  3. A time-arithmetic card — every date-to-date and date-to-NOW duration
     precomputed exactly, so the answer step never does date math.
  4. Question time-anchoring — deterministic classification of the temporal
     operator (locate / duration / point_in_time / order / current_state /
     change / frequency) plus resolution of partial references ("in May" →
     which May, inferred from the worldline span).

Everything here is pure Python, deterministic, and LLM-free. The LLM-facing
policy that consumes this lives in mempol/policies/temporal_ground.py.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

@dataclass
class Event:
    """One dated piece of evidence on the worldline."""

    ts: float                  # unix seconds; 0.0 = undated
    text: str
    source_id: str = ""
    speaker: str = ""
    session: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def dated(self) -> bool:
        return self.ts > 0.0


def _dt(ts: float) -> datetime:
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def fmt_date(ts: float) -> str:
    """'Mon 2023-05-08, 1:56 pm' — weekday included so the model never derives it."""
    d = _dt(ts)
    hour = d.hour % 12 or 12
    ampm = "am" if d.hour < 12 else "pm"
    return f"{d.strftime('%a %Y-%m-%d')}, {hour}:{d.minute:02d} {ampm}"


def fmt_day(ts: float) -> str:
    return _dt(ts).strftime("%a %Y-%m-%d")


# ---------------------------------------------------------------------------
# Calendar-exact deltas
# ---------------------------------------------------------------------------

def calendar_delta(a: datetime, b: datetime) -> tuple[int, int, int]:
    """Exact (years, months, days) from a to b, a <= b."""
    if a > b:
        a, b = b, a
    years = b.year - a.year
    months = b.month - a.month
    days = b.day - a.day
    if days < 0:
        months -= 1
        prev_month_end = b.replace(day=1) - timedelta(days=1)
        days += prev_month_end.day
    if months < 0:
        years -= 1
        months += 12
    return years, months, days


def humanize_delta(seconds: float, max_units: int = 2) -> str:
    """Coarse human phrase for an elapsed span: '3 weeks, 4 days', '1 year, 2 months'."""
    seconds = abs(seconds)
    if seconds < 60:
        return "moments"
    if seconds < 3600:
        m = int(seconds // 60)
        return f"{m} minute{'s' if m != 1 else ''}"
    if seconds < 86400:
        h = int(seconds // 3600)
        return f"{h} hour{'s' if h != 1 else ''}"
    days = seconds / 86400
    if days < 14:
        d = int(days)
        rest_h = int((days - d) * 24)
        parts = [f"{d} day{'s' if d != 1 else ''}"]
        if rest_h and max_units > 1:
            parts.append(f"{rest_h} hour{'s' if rest_h != 1 else ''}")
        return ", ".join(parts[:max_units])
    if days < 60:
        w = int(days // 7)
        rest_d = int(days - w * 7)
        parts = [f"{w} week{'s' if w != 1 else ''}"]
        if rest_d and max_units > 1:
            parts.append(f"{rest_d} day{'s' if rest_d != 1 else ''}")
        return ", ".join(parts[:max_units])
    if days < 365:
        mo = int(days // 30.44)
        rest_d = int(days - mo * 30.44)
        parts = [f"{mo} month{'s' if mo != 1 else ''}"]
        if rest_d and max_units > 1:
            parts.append(f"{rest_d} day{'s' if rest_d != 1 else ''}")
        return ", ".join(parts[:max_units])
    y = int(days // 365.25)
    rest_mo = int((days - y * 365.25) / 30.44)
    parts = [f"{y} year{'s' if y != 1 else ''}"]
    if rest_mo and max_units > 1:
        parts.append(f"{rest_mo} month{'s' if rest_mo != 1 else ''}")
    return ", ".join(parts[:max_units])


def exact_delta_phrase(a_ts: float, b_ts: float) -> str:
    """Calendar-exact phrase from earlier to later: '5 months, 12 days'."""
    y, m, d = calendar_delta(_dt(min(a_ts, b_ts)), _dt(max(a_ts, b_ts)))
    parts = []
    if y:
        parts.append(f"{y} year{'s' if y != 1 else ''}")
    if m:
        parts.append(f"{m} month{'s' if m != 1 else ''}")
    if d or not parts:
        parts.append(f"{d} day{'s' if d != 1 else ''}")
    return ", ".join(parts[:2])


def offset_phrase(ts: float, now_ts: float) -> str:
    """Egocentric offset from NOW: '5 months, 12 days ago', 'in 3 days', 'today'."""
    if abs(now_ts - ts) < 86400 and fmt_day(ts) == fmt_day(now_ts):
        return "today"
    phrase = exact_delta_phrase(ts, now_ts)
    return f"{phrase} ago" if ts <= now_ts else f"in {phrase}"


# ---------------------------------------------------------------------------
# Worldline compilation
# ---------------------------------------------------------------------------

def compile_worldline(
    events: list[Event],
    now_ts: float,
    gap_threshold_days: float = 1.0,
    volatile_gap_days: float = 30.0,
    max_text_chars: int = 800,
) -> str:
    """Render events as a chronological worldline with explicit time gaps.

    Gap lines make elapsed time something the model reads rather than computes.
    Gaps longer than `volatile_gap_days` warn that earlier plans/states may
    have changed — the textual analogue of memory decay.
    """
    dated = sorted((e for e in events if e.dated), key=lambda e: e.ts)
    undated = [e for e in events if not e.dated]

    lines = [f"# WORLDLINE (oldest -> newest; NOW = {fmt_date(now_ts)})", ""]
    prev_ts: float | None = None
    for e in dated:
        if prev_ts is not None:
            gap_s = e.ts - prev_ts
            if gap_s >= gap_threshold_days * 86400:
                gap = humanize_delta(gap_s)
                if gap_s >= volatile_gap_days * 86400:
                    lines.append(f"( {gap} pass -- long gap: earlier plans and states may have changed )")
                else:
                    lines.append(f"( {gap} pass )")
        head = f"({offset_phrase(e.ts, now_ts)} | {fmt_date(e.ts)}"
        if e.session:
            head += f" | S{e.session}"
        if e.source_id:
            head += f" | {e.source_id}"
        head += ")"
        speaker = f"{e.speaker}: " if e.speaker else ""
        lines.append(f"- {head} {speaker}{e.text[:max_text_chars]}")
        prev_ts = e.ts

    if undated:
        lines += ["", "## UNDATED EVIDENCE (could not be placed in time)"]
        for e in undated:
            src = f"({e.source_id}) " if e.source_id else ""
            speaker = f"{e.speaker}: " if e.speaker else ""
            lines.append(f"- {src}{speaker}{e.text[:max_text_chars]}")

    return "\n".join(lines)


def arithmetic_card(
    events: list[Event],
    now_ts: float,
    reference_ts: float | None = None,
    reference_label: str = "",
    max_days: int = 30,
) -> str:
    """Precomputed date arithmetic over the distinct days on the worldline.

    The answer step is instructed to read durations from this table and never
    subtract dates itself.
    """
    day_ts: dict[str, float] = {}
    for e in sorted((e for e in events if e.dated), key=lambda x: x.ts):
        day_ts.setdefault(fmt_day(e.ts), e.ts)
    days = list(day_ts.items())[:max_days]
    if not days:
        return "## TIME ARITHMETIC\n_(no dated events)_"

    lines = ["## TIME ARITHMETIC (precomputed -- read durations here, never compute date math yourself)", ""]
    lines.append(f"NOW = {fmt_date(now_ts)}")
    for day, ts in days:
        exact_days = int(round(abs(now_ts - ts) / 86400))
        rel = "before NOW" if ts <= now_ts else "after NOW"
        lines.append(f"- {day}: {exact_delta_phrase(ts, now_ts)} {rel} ({exact_days} days)")
    if len(days) > 1:
        lines += ["", "Consecutive gaps:"]
        for (d1, t1), (d2, t2) in zip(days, days[1:]):
            exact_days = int(round((t2 - t1) / 86400))
            lines.append(f"- {d1} -> {d2}: {exact_delta_phrase(t1, t2)} ({exact_days} days)")
    if reference_ts:
        label = f' ("{reference_label}")' if reference_label else ""
        lines += ["", f"REFERENCE TIME = {fmt_day(reference_ts)}{label}"]
        for day, ts in days:
            rel = "before" if ts <= reference_ts else "after"
            lines.append(f"- {day}: {exact_delta_phrase(ts, reference_ts)} {rel} the reference time")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Question time-anchoring
# ---------------------------------------------------------------------------

_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12,
}
_MONTH_RE = re.compile(
    r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b",
    re.IGNORECASE,
)
_YEAR_RE = re.compile(r"\b((?:19|20)\d{2})\b")
_DAY_RE = re.compile(
    r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(january|february|march|april|may|june|july|august|september|october|november|december)\b"
    r"|\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{1,2})(?:st|nd|rd|th)?\b",
    re.IGNORECASE,
)

# Operator patterns, checked in priority order. First hit wins.
_OPERATOR_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("duration", re.compile(
        r"\bhow long\b|\bhow much time\b|\bhow many (?:days|weeks|months|years)\b|\bhow old\b", re.IGNORECASE)),
    ("frequency", re.compile(
        r"\bhow many times\b|\bhow often\b|\bhow frequently\b", re.IGNORECASE)),
    ("locate", re.compile(
        r"\bwhen (?:did|was|were|does|is|had)\b|\bwhat (?:date|day|month|year)\b|\bon which (?:date|day)\b", re.IGNORECASE)),
    ("change", re.compile(
        r"\bchange[ds]?\b|\bused to\b|\bno longer\b|\banymore\b|\bany more\b|\bswitch(?:ed)?\b|\bquit\b|\bstopped\b", re.IGNORECASE)),
    ("current_state", re.compile(
        r"\bstill\b|\bcurrent(?:ly)?\b|\bright now\b|\bas of (?:now|today)\b|\bnowadays\b|\bthese days\b|\bmost recent(?:ly)?\b|\blatest\b|\blast time\b", re.IGNORECASE)),
    ("order", re.compile(
        r"\bbefore\b|\bafter\b|\bfirst\b|\bearlier\b|\bprior to\b|\bpreviously\b|\bsince\b|\buntil\b|\bfollowed\b", re.IGNORECASE)),
]


@dataclass
class TemporalFrame:
    """Deterministic temporal reading of a question."""

    is_temporal: bool
    operator: str = "none"        # locate|duration|point_in_time|order|current_state|change|frequency|none
    signals: list[str] = field(default_factory=list)
    reference_raw: str = ""
    reference_ts: float | None = None
    reference_granularity: str = ""   # day|month|year


def _resolve_reference(question: str, event_ts: list[float], now_ts: float) -> tuple[str, float | None, str]:
    """Resolve an explicit or partial date mention against the worldline span.

    'in May' with no year is resolved to the May that falls inside the
    worldline's span (latest such not after NOW) — the exact resolution step
    models routinely get wrong.
    """
    q = question.lower()
    day = None
    m_day = _DAY_RE.search(q)
    month = None
    if m_day:
        if m_day.group(1):
            day = int(m_day.group(1))
            month = _MONTHS[m_day.group(2).lower()]
        else:
            month = _MONTHS[m_day.group(3).lower()]
            day = int(m_day.group(4))
        raw = m_day.group(0)
    else:
        m_month = _MONTH_RE.search(q)
        if m_month:
            month = _MONTHS[m_month.group(1).lower()]
            raw = m_month.group(1)
        else:
            raw = ""
    m_year = _YEAR_RE.search(question)
    year = int(m_year.group(1)) if m_year else None
    if m_year and not raw:
        raw = m_year.group(1)

    if month is None and year is None:
        return "", None, ""

    if year is None:
        # Pick the year in which this month falls within the worldline span,
        # preferring the most recent occurrence not after NOW.
        candidates = sorted({_dt(t).year for t in event_ts if t > 0} | {_dt(now_ts).year})
        year_pick = None
        for y in candidates:
            try:
                ref = datetime(y, month, day or 1, tzinfo=timezone.utc)
            except ValueError:
                continue
            if ref.timestamp() <= now_ts + 31 * 86400:
                year_pick = y
        if year_pick is None:
            return raw, None, ""
        year = year_pick
    if year and month is None:
        return raw, datetime(year, 1, 1, tzinfo=timezone.utc).timestamp(), "year"
    try:
        ref = datetime(year, month, day or 1, tzinfo=timezone.utc)
    except ValueError:
        return raw, None, ""
    return raw, ref.timestamp(), ("day" if day else "month")


def classify_question(question: str, event_ts: list[float] | None = None, now_ts: float | None = None) -> TemporalFrame:
    """Deterministic temporal-operator classification + reference resolution."""
    signals: list[str] = []
    operator = "none"
    for op, pat in _OPERATOR_PATTERNS:
        m = pat.search(question)
        if m:
            signals.append(m.group(0))
            if operator == "none":
                operator = op

    raw, ref_ts, gran = ("", None, "")
    if event_ts and now_ts:
        raw, ref_ts, gran = _resolve_reference(question, event_ts, now_ts)
    if ref_ts is not None and operator in ("none", "order"):
        operator = "point_in_time"
        signals.append(raw)

    return TemporalFrame(
        is_temporal=operator != "none",
        operator=operator,
        signals=signals,
        reference_raw=raw,
        reference_ts=ref_ts,
        reference_granularity=gran,
    )
