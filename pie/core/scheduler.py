"""
PIE Scheduler — self-messaging and wake-up system.

The simplest useful version: PIE can schedule future check-ins for itself.
Each wake-up is a message PIE writes to its future self, with context about
what to check, what to predict, and what to do.

Wake-ups are stored as JSON. A cron job or manual trigger reads pending
wake-ups and generates an action prompt.

This is NOT a complex task queue. It's a note-to-self system:
- "Check if sponsorFind outreach got replies in 3 days"
- "Follow up on Lucid Labs lead in 1 week"
- "Review if PIE prompt engine is being used after 2 weeks"
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


@dataclass
class WakeUp:
    """A scheduled self-message."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    created_at: float = field(default_factory=time.time)
    trigger_at: float = 0.0  # when to fire
    entity_id: str | None = None  # related entity
    entity_name: str | None = None  # for display
    message: str = ""  # what future-self should do
    context: str = ""  # why this was scheduled
    priority: str = "normal"  # low, normal, high, urgent
    status: str = "pending"  # pending, fired, dismissed, expired
    fired_at: float | None = None
    result: str | None = None  # what happened when it fired

    @property
    def trigger_dt(self) -> datetime:
        return datetime.fromtimestamp(self.trigger_at)

    @property
    def created_dt(self) -> datetime:
        return datetime.fromtimestamp(self.created_at)

    @property
    def is_due(self) -> bool:
        return self.status == "pending" and time.time() >= self.trigger_at

    def time_until(self) -> str:
        delta = self.trigger_at - time.time()
        if delta <= 0:
            return "NOW"
        if delta < 3600:
            return f"{delta/60:.0f}min"
        if delta < 86400:
            return f"{delta/3600:.1f}h"
        return f"{delta/86400:.1f}d"


@dataclass
class PredictedStateChange:
    """A state change PIE predicts should happen."""
    entity_id: str
    entity_name: str
    prediction: str  # what should change
    expected_by: float  # timestamp
    confidence: float  # 0-1
    reasoning: str
    based_on: str  # pattern name or data source
    status: str = "predicted"  # predicted, confirmed, wrong, expired


class Scheduler:
    """Manages wake-ups and predicted state changes."""

    def __init__(self, storage_path: str = "output/schedule.json"):
        self.storage_path = Path(storage_path)
        self.wakeups: list[WakeUp] = []
        self.predictions: list[PredictedStateChange] = []
        self._load()

    def _load(self):
        if self.storage_path.exists():
            data = json.loads(self.storage_path.read_text())
            self.wakeups = [WakeUp(**w) for w in data.get("wakeups", [])]
            self.predictions = [PredictedStateChange(**p) for p in data.get("predictions", [])]

    def save(self):
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "wakeups": [asdict(w) for w in self.wakeups],
            "predictions": [asdict(p) for p in self.predictions],
            "last_saved": time.time(),
        }
        self.storage_path.write_text(json.dumps(data, indent=2))

    # ── Wake-ups ──

    def schedule(
        self,
        message: str,
        delay_hours: float | None = None,
        trigger_at: float | None = None,
        entity_id: str | None = None,
        entity_name: str | None = None,
        context: str = "",
        priority: str = "normal",
    ) -> WakeUp:
        """Schedule a wake-up (self-message to future)."""
        if trigger_at is None:
            if delay_hours is None:
                delay_hours = 24  # default: tomorrow
            trigger_at = time.time() + delay_hours * 3600

        wakeup = WakeUp(
            trigger_at=trigger_at,
            entity_id=entity_id,
            entity_name=entity_name,
            message=message,
            context=context,
            priority=priority,
        )
        self.wakeups.append(wakeup)
        self.save()
        return wakeup

    def schedule_in(self, message: str, days: float = 1, **kwargs) -> WakeUp:
        """Convenience: schedule X days from now."""
        return self.schedule(message, delay_hours=days * 24, **kwargs)

    def get_due(self) -> list[WakeUp]:
        """Get all wake-ups that should fire now."""
        return [w for w in self.wakeups if w.is_due]

    def get_pending(self) -> list[WakeUp]:
        """Get all pending wake-ups, sorted by trigger time."""
        pending = [w for w in self.wakeups if w.status == "pending"]
        pending.sort(key=lambda w: w.trigger_at)
        return pending

    def fire(self, wakeup_id: str, result: str = "") -> WakeUp | None:
        """Mark a wake-up as fired."""
        for w in self.wakeups:
            if w.id == wakeup_id:
                w.status = "fired"
                w.fired_at = time.time()
                w.result = result
                self.save()
                return w
        return None

    def dismiss(self, wakeup_id: str) -> WakeUp | None:
        """Dismiss a wake-up without acting on it."""
        for w in self.wakeups:
            if w.id == wakeup_id:
                w.status = "dismissed"
                self.save()
                return w
        return None

    # ── Predictions ──

    def predict(
        self,
        entity_id: str,
        entity_name: str,
        prediction: str,
        expected_in_days: float,
        confidence: float = 0.5,
        reasoning: str = "",
        based_on: str = "",
    ) -> PredictedStateChange:
        """Record a predicted state change."""
        p = PredictedStateChange(
            entity_id=entity_id,
            entity_name=entity_name,
            prediction=prediction,
            expected_by=time.time() + expected_in_days * 86400,
            confidence=confidence,
            reasoning=reasoning,
            based_on=based_on,
        )
        self.predictions.append(p)
        self.save()
        return p

    def get_active_predictions(self) -> list[PredictedStateChange]:
        """Get predictions that haven't expired or been resolved."""
        now = time.time()
        active = [
            p for p in self.predictions
            if p.status == "predicted" and p.expected_by > now
        ]
        active.sort(key=lambda p: p.expected_by)
        return active

    def get_expired_predictions(self) -> list[PredictedStateChange]:
        """Get predictions that passed their deadline without resolution."""
        now = time.time()
        return [
            p for p in self.predictions
            if p.status == "predicted" and p.expected_by <= now
        ]

    # ── Briefing generation ──

    def generate_briefing(self) -> str:
        """Generate a text briefing of what needs attention right now."""
        lines = []
        now = time.time()

        # Due wake-ups
        due = self.get_due()
        if due:
            lines.append("## Wake-ups Due NOW\n")
            for w in due:
                entity_str = f" [{w.entity_name}]" if w.entity_name else ""
                lines.append(f"- [{w.priority.upper()}]{entity_str} {w.message}")
                if w.context:
                    lines.append(f"  Context: {w.context}")
            lines.append("")

        # Upcoming wake-ups (next 48h)
        upcoming = [
            w for w in self.get_pending()
            if w.trigger_at <= now + 48 * 3600 and not w.is_due
        ]
        if upcoming:
            lines.append("## Coming Up (next 48h)\n")
            for w in upcoming:
                entity_str = f" [{w.entity_name}]" if w.entity_name else ""
                lines.append(f"- [{w.time_until()}]{entity_str} {w.message}")
            lines.append("")

        # Active predictions
        predictions = self.get_active_predictions()
        if predictions:
            lines.append("## Active Predictions\n")
            for p in predictions[:6]:
                days_left = (p.expected_by - now) / 86400
                lines.append(
                    f"- **{p.entity_name}**: {p.prediction} "
                    f"(expected in {days_left:.0f}d, confidence: {p.confidence:.0%})"
                )
            lines.append("")

        # Expired predictions (need review)
        expired = self.get_expired_predictions()
        if expired:
            lines.append("## Predictions to Review (expired)\n")
            for p in expired[:4]:
                lines.append(f"- **{p.entity_name}**: {p.prediction} — did this happen?")
            lines.append("")

        if not lines:
            lines.append("No wake-ups or predictions pending. Schedule some with `schedule()`.")

        return "\n".join(lines)

    # ── Summary ──

    def summary(self) -> dict:
        """Quick stats."""
        return {
            "pending_wakeups": len(self.get_pending()),
            "due_now": len(self.get_due()),
            "active_predictions": len(self.get_active_predictions()),
            "expired_predictions": len(self.get_expired_predictions()),
            "total_wakeups": len(self.wakeups),
            "total_predictions": len(self.predictions),
        }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PIE Scheduler")
    parser.add_argument("--briefing", action="store_true", help="Show current briefing")
    parser.add_argument("--summary", action="store_true", help="Show stats")
    parser.add_argument("--schedule", nargs=2, metavar=("MESSAGE", "HOURS"),
                        help="Schedule a wake-up")
    parser.add_argument("--predict", nargs=3, metavar=("ENTITY", "PREDICTION", "DAYS"),
                        help="Record a prediction")
    parser.add_argument("--storage", default="output/schedule.json")
    args = parser.parse_args()

    sched = Scheduler(args.storage)

    if args.schedule:
        msg, hours = args.schedule
        w = sched.schedule(msg, delay_hours=float(hours))
        print(f"Scheduled: {w.message} → fires in {w.time_until()} (id: {w.id})")

    elif args.predict:
        entity, prediction, days = args.predict
        p = sched.predict("", entity, prediction, float(days))
        print(f"Predicted: {entity} → {prediction} (in {days}d)")

    elif args.summary:
        print(json.dumps(sched.summary(), indent=2))

    else:
        print(sched.generate_briefing())
