"""
Gap Analyzer — tracks interaction timestamps and characterizes absences.

Simple module: records when the MCP server gets queried (proxy for user
interaction), computes gap characteristics for the briefing.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict


@dataclass
class GapAnalysis:
    """Result of analyzing the current interaction gap."""
    hours_since_last: float
    days_since_last: float
    gap_characterization: str  # "continuing" | "short_break" | "normal_gap" | "extended_absence" | "long_absence"
    avg_gap_hours: float | None
    gap_ratio: float | None  # current gap / avg gap
    total_interactions: int
    last_interaction_timestamp: float | None
    last_interaction_date: str | None


class GapAnalyzer:
    """Track interaction gaps and characterize absences."""

    def __init__(self, persist_path: str | Path):
        self.persist_path = Path(persist_path)
        self.interactions: list[float] = []
        self._load()

    def _load(self):
        if self.persist_path.exists():
            try:
                data = json.loads(self.persist_path.read_text())
                self.interactions = data.get("interactions", [])
            except (json.JSONDecodeError, KeyError):
                self.interactions = []

    def _save(self):
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        self.persist_path.write_text(json.dumps({
            "interactions": self.interactions,
        }, indent=2))

    def record_interaction(self, timestamp: float | None = None):
        """Record a new interaction. Call on every briefing request."""
        ts = timestamp or time.time()
        # Don't record if less than 5 minutes since last (avoid spam)
        if self.interactions and (ts - self.interactions[-1]) < 300:
            return
        self.interactions.append(ts)
        # Keep last 1000 interactions
        if len(self.interactions) > 1000:
            self.interactions = self.interactions[-1000:]
        self._save()

    def analyze(self, now: float | None = None) -> GapAnalysis:
        """Analyze the current gap since last interaction."""
        now = now or time.time()

        if not self.interactions:
            return GapAnalysis(
                hours_since_last=0,
                days_since_last=0,
                gap_characterization="first_interaction",
                avg_gap_hours=None,
                gap_ratio=None,
                total_interactions=0,
                last_interaction_timestamp=None,
                last_interaction_date=None,
            )

        last = self.interactions[-1]
        gap_seconds = now - last
        gap_hours = gap_seconds / 3600
        gap_days = gap_hours / 24

        # Compute avg gap from history
        avg_gap_hours = None
        gap_ratio = None
        if len(self.interactions) >= 2:
            gaps = []
            for i in range(1, len(self.interactions)):
                g = (self.interactions[i] - self.interactions[i - 1]) / 3600
                gaps.append(g)
            avg_gap_hours = sum(gaps) / len(gaps)
            if avg_gap_hours > 0:
                gap_ratio = round(gap_hours / avg_gap_hours, 1)

        # Characterize
        if gap_hours < 1:
            characterization = "continuing"
        elif gap_hours < 8:
            characterization = "short_break"
        elif gap_days < 2:
            characterization = "normal_gap"
        elif gap_days < 7:
            characterization = "extended_absence"
        else:
            characterization = "long_absence"

        from datetime import datetime
        last_date = datetime.fromtimestamp(last).strftime("%Y-%m-%d %H:%M")

        return GapAnalysis(
            hours_since_last=round(gap_hours, 1),
            days_since_last=round(gap_days, 1),
            gap_characterization=characterization,
            avg_gap_hours=round(avg_gap_hours, 1) if avg_gap_hours else None,
            gap_ratio=gap_ratio,
            total_interactions=len(self.interactions),
            last_interaction_timestamp=last,
            last_interaction_date=last_date,
        )

    @property
    def last_interaction_time(self) -> float | None:
        """Timestamp of last interaction, or None if first time."""
        return self.interactions[-1] if self.interactions else None
