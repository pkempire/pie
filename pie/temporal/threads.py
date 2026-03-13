"""
Thread Tracker — tracks active conversation threads, deadlines, and commitments.

Persisted as JSON alongside the world model. Updated after each conversation
via the update_world MCP tool.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from dataclasses import dataclass, field, asdict
from datetime import datetime


@dataclass
class Commitment:
    """A tracked commitment (something the user said they'd do)."""
    what: str
    who: str  # who committed (usually "user" or entity name)
    due_date: str | None  # human-readable date string
    due_timestamp: float | None  # epoch if parseable
    created_at: float = field(default_factory=time.time)
    completed: bool = False


@dataclass
class Thread:
    """An active conversation thread."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    topic: str = ""
    entity_id: str | None = None  # linked to world model entity
    entity_name: str | None = None
    opened: float = field(default_factory=time.time)
    last_mentioned: float = field(default_factory=time.time)
    status: str = "active"  # active | waiting | completed | abandoned
    deadline: str | None = None  # human-readable
    deadline_timestamp: float | None = None
    commitments: list[dict] = field(default_factory=list)
    notes: str = ""


class ThreadTracker:
    """Track conversation threads, deadlines, commitments across sessions."""

    def __init__(self, persist_path: str | Path):
        self.persist_path = Path(persist_path)
        self.threads: dict[str, Thread] = {}
        self._load()

    def _load(self):
        if self.persist_path.exists():
            try:
                data = json.loads(self.persist_path.read_text())
                for tid, tdata in data.get("threads", {}).items():
                    self.threads[tid] = Thread(**{
                        k: v for k, v in tdata.items()
                        if k in Thread.__dataclass_fields__
                    })
            except (json.JSONDecodeError, KeyError, TypeError):
                self.threads = {}

    def _save(self):
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "threads": {
                tid: asdict(t) for tid, t in self.threads.items()
            }
        }
        self.persist_path.write_text(json.dumps(data, indent=2, default=str))

    # ── CRUD ──

    def open_thread(
        self,
        topic: str,
        entity_id: str | None = None,
        entity_name: str | None = None,
        deadline: str | None = None,
        deadline_timestamp: float | None = None,
    ) -> Thread:
        """Open a new thread."""
        thread = Thread(
            topic=topic,
            entity_id=entity_id,
            entity_name=entity_name,
            deadline=deadline,
            deadline_timestamp=deadline_timestamp,
        )
        self.threads[thread.id] = thread
        self._save()
        return thread

    def add_commitment(
        self,
        thread_id: str | None,
        what: str,
        who: str = "user",
        due_date: str | None = None,
        due_timestamp: float | None = None,
    ):
        """Add a commitment to a thread (or create a standalone one)."""
        commitment = {
            "what": what,
            "who": who,
            "due_date": due_date,
            "due_timestamp": due_timestamp,
            "created_at": time.time(),
            "completed": False,
        }

        if thread_id and thread_id in self.threads:
            self.threads[thread_id].commitments.append(commitment)
        else:
            # Create standalone thread for this commitment
            thread = Thread(
                topic=what,
                deadline=due_date,
                deadline_timestamp=due_timestamp,
                commitments=[commitment],
            )
            self.threads[thread.id] = thread

        self._save()

    def touch_thread(self, thread_id: str):
        """Update last_mentioned timestamp."""
        if thread_id in self.threads:
            self.threads[thread_id].last_mentioned = time.time()
            self._save()

    def complete_thread(self, thread_id: str):
        """Mark a thread as completed."""
        if thread_id in self.threads:
            self.threads[thread_id].status = "completed"
            self._save()

    # ── Queries ──

    def get_active_threads(self) -> list[Thread]:
        """Get all non-completed, non-abandoned threads."""
        return [
            t for t in self.threads.values()
            if t.status in ("active", "waiting")
        ]

    def get_approaching_deadlines(self, window_days: int = 7) -> list[dict]:
        """Get deadlines approaching in the next N days."""
        now = time.time()
        cutoff = now + (window_days * 86400)
        results = []

        for t in self.threads.values():
            if t.status in ("completed", "abandoned"):
                continue
            if t.deadline_timestamp and t.deadline_timestamp <= cutoff:
                days_left = (t.deadline_timestamp - now) / 86400
                results.append({
                    "thread_id": t.id,
                    "topic": t.topic,
                    "entity_name": t.entity_name,
                    "due_date": t.deadline,
                    "days_remaining": round(days_left, 1),
                    "is_overdue": days_left < 0,
                    "description": t.notes[:100] if t.notes else "",
                })

        results.sort(key=lambda r: r["days_remaining"])
        return results

    def get_overdue_commitments(self) -> list[dict]:
        """Get commitments past their due date."""
        now = time.time()
        results = []

        for t in self.threads.values():
            if t.status in ("completed", "abandoned"):
                continue
            for c in t.commitments:
                if c.get("completed"):
                    continue
                due_ts = c.get("due_timestamp")
                if due_ts and due_ts < now:
                    days_overdue = (now - due_ts) / 86400
                    results.append({
                        "thread_id": t.id,
                        "thread_topic": t.topic,
                        "what": c.get("what", ""),
                        "who": c.get("who", ""),
                        "due_date": c.get("due_date", ""),
                        "days_overdue": round(days_overdue, 1),
                    })

        results.sort(key=lambda r: -r["days_overdue"])
        return results

    def get_stale_threads(self, threshold_days: int = 14) -> list[dict]:
        """Get active threads that haven't been mentioned in a while."""
        now = time.time()
        threshold = threshold_days * 86400
        results = []

        for t in self.threads.values():
            if t.status != "active":
                continue
            silence = now - t.last_mentioned
            if silence > threshold:
                results.append({
                    "thread_id": t.id,
                    "topic": t.topic,
                    "entity_name": t.entity_name,
                    "days_silent": round(silence / 86400, 1),
                    "has_commitments": bool(t.commitments),
                    "has_deadline": bool(t.deadline),
                })

        results.sort(key=lambda r: -r["days_silent"])
        return results

    def update_from_conversation(
        self,
        mentioned_entities: list[str] | None = None,
        new_deadlines: list[dict] | None = None,
        new_commitments: list[dict] | None = None,
    ):
        """Update threads based on a conversation. Called after update_world."""
        now = time.time()

        # Touch threads whose entities were mentioned
        if mentioned_entities:
            for t in self.threads.values():
                if t.entity_name and t.entity_name.lower() in [
                    e.lower() for e in mentioned_entities
                ]:
                    t.last_mentioned = now

        # Add new deadlines
        if new_deadlines:
            for d in new_deadlines:
                self.open_thread(
                    topic=d.get("topic", d.get("entity", "Unknown")),
                    entity_name=d.get("entity"),
                    deadline=d.get("due_date"),
                    deadline_timestamp=d.get("due_timestamp"),
                )

        # Add new commitments
        if new_commitments:
            for c in new_commitments:
                self.add_commitment(
                    thread_id=None,  # standalone
                    what=c.get("what", ""),
                    who=c.get("who", "user"),
                    due_date=c.get("due_date"),
                    due_timestamp=c.get("due_timestamp"),
                )

        self._save()
