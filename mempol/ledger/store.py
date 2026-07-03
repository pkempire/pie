"""SQLite store for the research-ledger layer."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from mempol import config
from mempol.core.store import now_iso, stable_id

from .schema import ContextPack, Membership, Project, ResearchObject, RunRecord, Thread


def _json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


def _loads(raw: str | None, default: Any) -> Any:
    if not raw:
        return default
    try:
        return json.loads(raw)
    except Exception:
        return default


class LedgerStore:
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

            CREATE TABLE IF NOT EXISTS projects (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                objective TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'active',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS threads (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                title TEXT NOT NULL,
                summary TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'active',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}'
            );
            CREATE INDEX IF NOT EXISTS idx_threads_project ON threads(project_id);

            CREATE TABLE IF NOT EXISTS memberships (
                id TEXT PRIMARY KEY,
                target_type TEXT NOT NULL,
                target_id TEXT NOT NULL,
                project_id TEXT NOT NULL,
                thread_id TEXT NOT NULL DEFAULT '',
                confidence REAL NOT NULL DEFAULT 1.0,
                assigned_by TEXT NOT NULL DEFAULT 'rule',
                rationale TEXT NOT NULL DEFAULT '',
                valid_from TEXT NOT NULL DEFAULT '',
                valid_until TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}'
            );
            CREATE INDEX IF NOT EXISTS idx_membership_target ON memberships(target_type, target_id);
            CREATE INDEX IF NOT EXISTS idx_membership_project ON memberships(project_id, thread_id);

            CREATE TABLE IF NOT EXISTS research_objects (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                thread_id TEXT NOT NULL DEFAULT '',
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                source_span_ids_json TEXT NOT NULL DEFAULT '[]',
                parent_ids_json TEXT NOT NULL DEFAULT '[]',
                status TEXT NOT NULL DEFAULT 'open',
                confidence REAL NOT NULL DEFAULT 1.0,
                novelty_score REAL,
                utility_score REAL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}'
            );
            CREATE INDEX IF NOT EXISTS idx_objects_project ON research_objects(project_id, thread_id, role);

            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                thread_id TEXT NOT NULL DEFAULT '',
                title TEXT NOT NULL,
                started_at TEXT NOT NULL DEFAULT '',
                ended_at TEXT NOT NULL DEFAULT '',
                actor TEXT NOT NULL DEFAULT '',
                command TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'unknown',
                metrics_json TEXT NOT NULL DEFAULT '{}',
                artifact_ids_json TEXT NOT NULL DEFAULT '[]',
                metadata_json TEXT NOT NULL DEFAULT '{}'
            );
            CREATE INDEX IF NOT EXISTS idx_runs_project ON runs(project_id, thread_id, started_at);

            CREATE TABLE IF NOT EXISTS context_packs (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                thread_id TEXT NOT NULL DEFAULT '',
                task TEXT NOT NULL,
                markdown TEXT NOT NULL,
                source_span_ids_json TEXT NOT NULL DEFAULT '[]',
                research_object_ids_json TEXT NOT NULL DEFAULT '[]',
                token_budget INTEGER NOT NULL DEFAULT 0,
                token_estimate INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                metrics_json TEXT NOT NULL DEFAULT '{}',
                metadata_json TEXT NOT NULL DEFAULT '{}'
            );
            CREATE INDEX IF NOT EXISTS idx_context_project ON context_packs(project_id, thread_id, created_at);
            """
        )
        self.conn.commit()

    def upsert_project(self, project: Project) -> None:
        ts = now_iso()
        created_at = project.created_at or ts
        updated_at = project.updated_at or ts
        self.conn.execute(
            """
            INSERT INTO projects (id, title, objective, status, created_at, updated_at, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
              title=excluded.title,
              objective=excluded.objective,
              status=excluded.status,
              updated_at=excluded.updated_at,
              metadata_json=excluded.metadata_json
            """,
            (project.id, project.title, project.objective, project.status, created_at, updated_at, _json(project.metadata)),
        )

    def upsert_thread(self, thread: Thread) -> None:
        ts = now_iso()
        created_at = thread.created_at or ts
        updated_at = thread.updated_at or ts
        self.conn.execute(
            """
            INSERT INTO threads (id, project_id, title, summary, status, created_at, updated_at, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
              project_id=excluded.project_id,
              title=excluded.title,
              summary=excluded.summary,
              status=excluded.status,
              updated_at=excluded.updated_at,
              metadata_json=excluded.metadata_json
            """,
            (thread.id, thread.project_id, thread.title, thread.summary, thread.status, created_at, updated_at, _json(thread.metadata)),
        )

    def upsert_membership(self, membership: Membership) -> None:
        created_at = membership.created_at or now_iso()
        self.conn.execute(
            """
            INSERT INTO memberships
              (id, target_type, target_id, project_id, thread_id, confidence, assigned_by,
               rationale, valid_from, valid_until, created_at, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
              project_id=excluded.project_id,
              thread_id=excluded.thread_id,
              confidence=excluded.confidence,
              assigned_by=excluded.assigned_by,
              rationale=excluded.rationale,
              valid_from=excluded.valid_from,
              valid_until=excluded.valid_until,
              metadata_json=excluded.metadata_json
            """,
            (
                membership.id,
                membership.target_type,
                membership.target_id,
                membership.project_id,
                membership.thread_id,
                membership.confidence,
                membership.assigned_by,
                membership.rationale,
                membership.valid_from,
                membership.valid_until,
                created_at,
                _json(membership.metadata),
            ),
        )

    def upsert_research_object(self, obj: ResearchObject) -> None:
        ts = now_iso()
        created_at = obj.created_at or ts
        updated_at = obj.updated_at or ts
        self.conn.execute(
            """
            INSERT INTO research_objects
              (id, project_id, thread_id, role, content, source_span_ids_json, parent_ids_json,
               status, confidence, novelty_score, utility_score, created_at, updated_at, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
              project_id=excluded.project_id,
              thread_id=excluded.thread_id,
              role=excluded.role,
              content=excluded.content,
              source_span_ids_json=excluded.source_span_ids_json,
              parent_ids_json=excluded.parent_ids_json,
              status=excluded.status,
              confidence=excluded.confidence,
              novelty_score=excluded.novelty_score,
              utility_score=excluded.utility_score,
              updated_at=excluded.updated_at,
              metadata_json=excluded.metadata_json
            """,
            (
                obj.id,
                obj.project_id,
                obj.thread_id,
                obj.role,
                obj.content,
                _json(obj.source_span_ids),
                _json(obj.parent_ids),
                obj.status,
                obj.confidence,
                obj.novelty_score,
                obj.utility_score,
                created_at,
                updated_at,
                _json(obj.metadata),
            ),
        )

    def upsert_run(self, run: RunRecord) -> None:
        self.conn.execute(
            """
            INSERT INTO runs
              (id, project_id, thread_id, title, started_at, ended_at, actor, command,
               status, metrics_json, artifact_ids_json, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
              project_id=excluded.project_id,
              thread_id=excluded.thread_id,
              title=excluded.title,
              started_at=excluded.started_at,
              ended_at=excluded.ended_at,
              actor=excluded.actor,
              command=excluded.command,
              status=excluded.status,
              metrics_json=excluded.metrics_json,
              artifact_ids_json=excluded.artifact_ids_json,
              metadata_json=excluded.metadata_json
            """,
            (
                run.id,
                run.project_id,
                run.thread_id,
                run.title,
                run.started_at,
                run.ended_at,
                run.actor,
                run.command,
                run.status,
                _json(run.metrics),
                _json(run.artifact_ids),
                _json(run.metadata),
            ),
        )

    def upsert_context_pack(self, pack: ContextPack) -> None:
        created_at = pack.created_at or now_iso()
        self.conn.execute(
            """
            INSERT INTO context_packs
              (id, project_id, thread_id, task, markdown, source_span_ids_json,
               research_object_ids_json, token_budget, token_estimate, created_at,
               metrics_json, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
              markdown=excluded.markdown,
              source_span_ids_json=excluded.source_span_ids_json,
              research_object_ids_json=excluded.research_object_ids_json,
              token_budget=excluded.token_budget,
              token_estimate=excluded.token_estimate,
              metrics_json=excluded.metrics_json,
              metadata_json=excluded.metadata_json
            """,
            (
                pack.id,
                pack.project_id,
                pack.thread_id,
                pack.task,
                pack.markdown,
                _json(pack.source_span_ids),
                _json(pack.research_object_ids),
                pack.token_budget,
                pack.token_estimate,
                created_at,
                _json(pack.metrics),
                _json(pack.metadata),
            ),
        )

    def list_projects(self) -> list[dict[str, Any]]:
        return [dict(r) for r in self.conn.execute("SELECT * FROM projects ORDER BY updated_at DESC").fetchall()]

    def list_threads(self, project_id: str = "") -> list[dict[str, Any]]:
        if project_id:
            rows = self.conn.execute("SELECT * FROM threads WHERE project_id=? ORDER BY updated_at DESC", (project_id,)).fetchall()
        else:
            rows = self.conn.execute("SELECT * FROM threads ORDER BY updated_at DESC").fetchall()
        return [dict(r) for r in rows]

    def list_memberships(self, project_id: str = "", thread_id: str = "", limit: int = 500) -> list[dict[str, Any]]:
        clauses = []
        params: list[Any] = []
        if project_id:
            clauses.append("project_id=?")
            params.append(project_id)
        if thread_id:
            clauses.append("thread_id=?")
            params.append(thread_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self.conn.execute(
            f"SELECT * FROM memberships {where} ORDER BY created_at DESC LIMIT ?",
            (*params, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def available_days(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT substr(valid_from, 1, 10) AS day, project_id, thread_id, COUNT(*) AS n
            FROM memberships
            WHERE valid_from != ''
            GROUP BY day, project_id, thread_id
            ORDER BY day DESC, n DESC
            """
        ).fetchall()
        return [dict(r) for r in rows]

    def stats(self) -> dict[str, Any]:
        def count(table: str) -> int:
            return int(self.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])

        return {
            "path": str(self.path),
            "projects": count("projects"),
            "threads": count("threads"),
            "memberships": count("memberships"),
            "research_objects": count("research_objects"),
            "runs": count("runs"),
            "context_packs": count("context_packs"),
        }


def ledger_for_run(run_name: str) -> LedgerStore:
    return LedgerStore(config.RESULTS_DIR / run_name / "ledger.sqlite")


def membership_id(target_type: str, target_id: str, project_id: str, thread_id: str = "") -> str:
    return stable_id("membership", target_type, target_id, project_id, thread_id)
