"""SQLite store for temporal agent state."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from mempol import config
from mempol.core.store import now_iso, stable_id

from .schema import ActiveProcess, ContextDecision, OutcomeEvent, StateTransition, TemporalState


def _json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


def _loads(raw: str | None, default: Any) -> Any:
    if not raw:
        return default
    try:
        return json.loads(raw)
    except Exception:
        return default


def _lt_or_empty(lhs: str, rhs: str) -> bool:
    return bool(lhs) and lhs < rhs


class TemporalMemoryStore:
    """Temporal state, process, decision, and outcome persistence."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def close(self) -> None:
        self.conn.close()

    def commit(self) -> None:
        self.conn.commit()

    def _init_schema(self) -> None:
        self.conn.executescript(
            """
            PRAGMA journal_mode = WAL;

            CREATE TABLE IF NOT EXISTS temporal_states (
                id TEXT PRIMARY KEY,
                scope_id TEXT NOT NULL,
                key TEXT NOT NULL,
                content TEXT NOT NULL,
                state_type TEXT NOT NULL DEFAULT 'state',
                valid_from TEXT NOT NULL DEFAULT '',
                valid_until TEXT NOT NULL DEFAULT '',
                observed_at TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'active',
                confidence REAL NOT NULL DEFAULT 1.0,
                volatility_seconds REAL,
                source_span_ids_json TEXT NOT NULL DEFAULT '[]',
                supersedes_state_ids_json TEXT NOT NULL DEFAULT '[]',
                metadata_json TEXT NOT NULL DEFAULT '{}'
            );
            CREATE INDEX IF NOT EXISTS idx_temporal_scope_key
                ON temporal_states(scope_id, key, status, valid_from);

            CREATE TABLE IF NOT EXISTS state_transitions (
                id TEXT PRIMARY KEY,
                scope_id TEXT NOT NULL,
                transition_type TEXT NOT NULL,
                old_state_ids_json TEXT NOT NULL DEFAULT '[]',
                new_state_id TEXT NOT NULL DEFAULT '',
                reason TEXT NOT NULL DEFAULT '',
                observed_at TEXT NOT NULL DEFAULT '',
                source_span_ids_json TEXT NOT NULL DEFAULT '[]',
                trace_event_ids_json TEXT NOT NULL DEFAULT '[]',
                metadata_json TEXT NOT NULL DEFAULT '{}'
            );
            CREATE INDEX IF NOT EXISTS idx_transitions_scope_time
                ON state_transitions(scope_id, observed_at);

            CREATE TABLE IF NOT EXISTS active_processes (
                id TEXT PRIMARY KEY,
                scope_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                description TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'active',
                started_at TEXT NOT NULL DEFAULT '',
                expected_at TEXT NOT NULL DEFAULT '',
                deadline_at TEXT NOT NULL DEFAULT '',
                last_checked_at TEXT NOT NULL DEFAULT '',
                source_span_ids_json TEXT NOT NULL DEFAULT '[]',
                metadata_json TEXT NOT NULL DEFAULT '{}'
            );
            CREATE INDEX IF NOT EXISTS idx_process_scope_status
                ON active_processes(scope_id, status, deadline_at);

            CREATE TABLE IF NOT EXISTS context_decisions (
                id TEXT PRIMARY KEY,
                scope_id TEXT NOT NULL,
                task TEXT NOT NULL,
                now TEXT NOT NULL,
                action TEXT NOT NULL DEFAULT 'answer',
                candidate_state_ids_json TEXT NOT NULL DEFAULT '[]',
                selected_state_ids_json TEXT NOT NULL DEFAULT '[]',
                selected_span_ids_json TEXT NOT NULL DEFAULT '[]',
                selected_process_ids_json TEXT NOT NULL DEFAULT '[]',
                token_budget INTEGER NOT NULL DEFAULT 0,
                token_estimate INTEGER NOT NULL DEFAULT 0,
                rationale TEXT NOT NULL DEFAULT '',
                metrics_json TEXT NOT NULL DEFAULT '{}',
                metadata_json TEXT NOT NULL DEFAULT '{}'
            );
            CREATE INDEX IF NOT EXISTS idx_context_decisions_scope_time
                ON context_decisions(scope_id, now);

            CREATE TABLE IF NOT EXISTS outcome_events (
                id TEXT PRIMARY KEY,
                decision_id TEXT NOT NULL,
                scope_id TEXT NOT NULL,
                score REAL NOT NULL DEFAULT 0.0,
                outcome_type TEXT NOT NULL DEFAULT 'unknown',
                feedback TEXT NOT NULL DEFAULT '',
                metrics_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL DEFAULT '',
                metadata_json TEXT NOT NULL DEFAULT '{}'
            );
            CREATE INDEX IF NOT EXISTS idx_outcome_decision ON outcome_events(decision_id);
            """
        )
        self.conn.commit()

    # ─── State ────────────────────────────────────────────────────────────
    def upsert_state(self, state: TemporalState) -> None:
        observed_at = state.observed_at or now_iso()
        valid_from = state.valid_from or observed_at
        self.conn.execute(
            """
            INSERT INTO temporal_states
              (id, scope_id, key, content, state_type, valid_from, valid_until,
               observed_at, status, confidence, volatility_seconds,
               source_span_ids_json, supersedes_state_ids_json, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
              scope_id=excluded.scope_id,
              key=excluded.key,
              content=excluded.content,
              state_type=excluded.state_type,
              valid_from=excluded.valid_from,
              valid_until=excluded.valid_until,
              observed_at=excluded.observed_at,
              status=excluded.status,
              confidence=excluded.confidence,
              volatility_seconds=excluded.volatility_seconds,
              source_span_ids_json=excluded.source_span_ids_json,
              supersedes_state_ids_json=excluded.supersedes_state_ids_json,
              metadata_json=excluded.metadata_json
            """,
            (
                state.id,
                state.scope_id,
                state.key,
                state.content,
                state.state_type,
                valid_from,
                state.valid_until,
                observed_at,
                state.status,
                state.confidence,
                state.volatility_seconds,
                _json(state.source_span_ids),
                _json(state.supersedes_state_ids),
                _json(state.metadata),
            ),
        )

    def apply_transition(self, transition: StateTransition, new_state: TemporalState | None = None) -> None:
        """Apply a transition and mark superseded/archived states when needed."""
        observed_at = transition.observed_at or now_iso()
        if new_state is not None:
            self.upsert_state(new_state)
        terminal_status = "archived" if transition.transition_type == "archive" else "superseded"
        for old_id in transition.old_state_ids:
            self.conn.execute(
                """
                UPDATE temporal_states
                SET status=?, valid_until=CASE WHEN valid_until='' THEN ? ELSE valid_until END
                WHERE id=?
                """,
                (terminal_status, observed_at, old_id),
            )
        self.conn.execute(
            """
            INSERT OR REPLACE INTO state_transitions
              (id, scope_id, transition_type, old_state_ids_json, new_state_id,
               reason, observed_at, source_span_ids_json, trace_event_ids_json, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                transition.id,
                transition.scope_id,
                transition.transition_type,
                _json(transition.old_state_ids),
                transition.new_state_id,
                transition.reason,
                observed_at,
                _json(transition.source_span_ids),
                _json(transition.trace_event_ids),
                _json(transition.metadata),
            ),
        )

    def current_states(
        self,
        scope_id: str,
        *,
        at: str | None = None,
        key_prefix: str = "",
        include_stale: bool = False,
        limit: int = 500,
    ) -> list[TemporalState]:
        at = at or now_iso()
        status_clause = "" if include_stale else "AND status='active'"
        key_clause = "AND key LIKE ?" if key_prefix else ""
        params: list[Any] = [scope_id, at, at]
        if key_prefix:
            params.append(f"{key_prefix}%")
        params.append(limit)
        rows = self.conn.execute(
            f"""
            SELECT * FROM temporal_states
            WHERE scope_id=?
              AND (valid_from='' OR valid_from<=?)
              AND (valid_until='' OR valid_until>?)
              {status_clause}
              {key_clause}
            ORDER BY confidence DESC, valid_from DESC
            LIMIT ?
            """,
            params,
        ).fetchall()
        return [self._state_from_row(r) for r in rows]

    def state_history(self, scope_id: str, key: str, limit: int = 100) -> list[TemporalState]:
        rows = self.conn.execute(
            """
            SELECT * FROM temporal_states
            WHERE scope_id=? AND key=?
            ORDER BY valid_from DESC, observed_at DESC
            LIMIT ?
            """,
            (scope_id, key, limit),
        ).fetchall()
        return [self._state_from_row(r) for r in rows]

    def transitions(self, scope_id: str, limit: int = 200) -> list[StateTransition]:
        rows = self.conn.execute(
            """
            SELECT * FROM state_transitions
            WHERE scope_id=?
            ORDER BY observed_at DESC
            LIMIT ?
            """,
            (scope_id, limit),
        ).fetchall()
        return [self._transition_from_row(r) for r in rows]

    # ─── Active Processes ─────────────────────────────────────────────────
    def upsert_process(self, process: ActiveProcess) -> None:
        self.conn.execute(
            """
            INSERT INTO active_processes
              (id, scope_id, kind, description, status, started_at, expected_at,
               deadline_at, last_checked_at, source_span_ids_json, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
              kind=excluded.kind,
              description=excluded.description,
              status=excluded.status,
              started_at=excluded.started_at,
              expected_at=excluded.expected_at,
              deadline_at=excluded.deadline_at,
              last_checked_at=excluded.last_checked_at,
              source_span_ids_json=excluded.source_span_ids_json,
              metadata_json=excluded.metadata_json
            """,
            (
                process.id,
                process.scope_id,
                process.kind,
                process.description,
                process.status,
                process.started_at,
                process.expected_at,
                process.deadline_at,
                process.last_checked_at,
                _json(process.source_span_ids),
                _json(process.metadata),
            ),
        )

    def due_processes(self, scope_id: str, *, now: str | None = None, limit: int = 100) -> list[ActiveProcess]:
        now = now or now_iso()
        rows = self.conn.execute(
            """
            SELECT * FROM active_processes
            WHERE scope_id=?
              AND status IN ('active', 'waiting', 'blocked')
              AND (
                (deadline_at!='' AND deadline_at<=?)
                OR (expected_at!='' AND expected_at<=?)
              )
            ORDER BY COALESCE(NULLIF(deadline_at, ''), expected_at) ASC
            LIMIT ?
            """,
            (scope_id, now, now, limit),
        ).fetchall()
        return [self._process_from_row(r) for r in rows]

    # ─── Decisions / Outcomes ─────────────────────────────────────────────
    def log_context_decision(self, decision: ContextDecision) -> None:
        self.conn.execute(
            """
            INSERT OR REPLACE INTO context_decisions
              (id, scope_id, task, now, action, candidate_state_ids_json,
               selected_state_ids_json, selected_span_ids_json, selected_process_ids_json,
               token_budget, token_estimate, rationale, metrics_json, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                decision.id,
                decision.scope_id,
                decision.task,
                decision.now or now_iso(),
                decision.action,
                _json(decision.candidate_state_ids),
                _json(decision.selected_state_ids),
                _json(decision.selected_span_ids),
                _json(decision.selected_process_ids),
                decision.token_budget,
                decision.token_estimate,
                decision.rationale,
                _json(decision.metrics),
                _json(decision.metadata),
            ),
        )

    def log_outcome(self, outcome: OutcomeEvent) -> None:
        self.conn.execute(
            """
            INSERT OR REPLACE INTO outcome_events
              (id, decision_id, scope_id, score, outcome_type, feedback,
               metrics_json, created_at, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                outcome.id,
                outcome.decision_id,
                outcome.scope_id,
                outcome.score,
                outcome.outcome_type,
                outcome.feedback,
                _json(outcome.metrics),
                outcome.created_at or now_iso(),
                _json(outcome.metadata),
            ),
        )

    def decision_training_rows(self, scope_id: str, limit: int = 1000) -> list[dict[str, Any]]:
        """Join context decisions with outcomes for offline policy learning."""
        rows = self.conn.execute(
            """
            SELECT
              d.*,
              o.id AS outcome_id,
              o.score AS outcome_score,
              o.outcome_type AS outcome_type,
              o.feedback AS feedback,
              o.metrics_json AS outcome_metrics_json
            FROM context_decisions d
            LEFT JOIN outcome_events o ON o.decision_id=d.id
            WHERE d.scope_id=?
            ORDER BY d.now DESC
            LIMIT ?
            """,
            (scope_id, limit),
        ).fetchall()
        return [
            {
                "decision_id": r["id"],
                "scope_id": r["scope_id"],
                "task": r["task"],
                "now": r["now"],
                "action": r["action"],
                "candidate_state_ids": _loads(r["candidate_state_ids_json"], []),
                "selected_state_ids": _loads(r["selected_state_ids_json"], []),
                "selected_span_ids": _loads(r["selected_span_ids_json"], []),
                "selected_process_ids": _loads(r["selected_process_ids_json"], []),
                "token_budget": r["token_budget"],
                "token_estimate": r["token_estimate"],
                "rationale": r["rationale"],
                "metrics": _loads(r["metrics_json"], {}),
                "outcome_id": r["outcome_id"],
                "outcome_score": r["outcome_score"],
                "outcome_type": r["outcome_type"],
                "feedback": r["feedback"],
                "outcome_metrics": _loads(r["outcome_metrics_json"], {}),
            }
            for r in rows
        ]

    def stats(self) -> dict[str, Any]:
        def count(table: str) -> int:
            return int(self.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])

        return {
            "path": str(self.path),
            "temporal_states": count("temporal_states"),
            "state_transitions": count("state_transitions"),
            "active_processes": count("active_processes"),
            "context_decisions": count("context_decisions"),
            "outcome_events": count("outcome_events"),
        }

    # ─── Row mappers ──────────────────────────────────────────────────────
    def _state_from_row(self, r: sqlite3.Row) -> TemporalState:
        return TemporalState(
            id=r["id"],
            scope_id=r["scope_id"],
            key=r["key"],
            content=r["content"],
            state_type=r["state_type"],
            valid_from=r["valid_from"],
            valid_until=r["valid_until"],
            observed_at=r["observed_at"],
            status=r["status"],
            confidence=float(r["confidence"]),
            volatility_seconds=r["volatility_seconds"],
            source_span_ids=_loads(r["source_span_ids_json"], []),
            supersedes_state_ids=_loads(r["supersedes_state_ids_json"], []),
            metadata=_loads(r["metadata_json"], {}),
        )

    def _transition_from_row(self, r: sqlite3.Row) -> StateTransition:
        return StateTransition(
            id=r["id"],
            scope_id=r["scope_id"],
            transition_type=r["transition_type"],
            old_state_ids=_loads(r["old_state_ids_json"], []),
            new_state_id=r["new_state_id"],
            reason=r["reason"],
            observed_at=r["observed_at"],
            source_span_ids=_loads(r["source_span_ids_json"], []),
            trace_event_ids=_loads(r["trace_event_ids_json"], []),
            metadata=_loads(r["metadata_json"], {}),
        )

    def _process_from_row(self, r: sqlite3.Row) -> ActiveProcess:
        return ActiveProcess(
            id=r["id"],
            scope_id=r["scope_id"],
            kind=r["kind"],
            description=r["description"],
            status=r["status"],
            started_at=r["started_at"],
            expected_at=r["expected_at"],
            deadline_at=r["deadline_at"],
            last_checked_at=r["last_checked_at"],
            source_span_ids=_loads(r["source_span_ids_json"], []),
            metadata=_loads(r["metadata_json"], {}),
        )


def temporal_store_for_run(run_name: str) -> TemporalMemoryStore:
    return TemporalMemoryStore(config.RESULTS_DIR / run_name / "temporal_memory.sqlite")


def temporal_id(prefix: str, *parts: object) -> str:
    return stable_id(prefix, *parts)
