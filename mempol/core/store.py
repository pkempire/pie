"""SQLite-backed universal memory store.

The store is deliberately simple: raw artifacts, evidence spans, freeform
memory states, and trace events. Retrieval is hybrid-ready but starts with a
transparent lexical scorer so demos work without an API key or extra services.
"""
from __future__ import annotations

import json
import math
import re
import sqlite3
import time
import uuid
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

from mempol import config

from .schema import Artifact, MemoryState, Span, TraceEvent


_TOK_RE = re.compile(r"[A-Za-z0-9_./+-]+")


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def stable_id(prefix: str, *parts: object) -> str:
    import hashlib

    raw = "\n".join(str(p) for p in parts)
    return f"{prefix}_{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:16]}"


def estimate_tokens(text: str) -> int:
    # Good enough for budget dashboards without adding tiktoken as a hard dep.
    return max(1, math.ceil(len(text) / 4))


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in _TOK_RE.findall(text or "")]


def _json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


def _loads(raw: str | None, default: Any) -> Any:
    if not raw:
        return default
    try:
        return json.loads(raw)
    except Exception:
        return default


class SQLiteMemoryStore:
    """Persistence and lexical retrieval for the universal memory core."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def close(self) -> None:
        self.conn.close()

    def _init_schema(self) -> None:
        self.conn.executescript(
            """
            PRAGMA journal_mode = WAL;

            CREATE TABLE IF NOT EXISTS artifacts (
                id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                kind TEXT NOT NULL,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                uri TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS spans (
                id TEXT PRIMARY KEY,
                artifact_id TEXT NOT NULL REFERENCES artifacts(id) ON DELETE CASCADE,
                text TEXT NOT NULL,
                locator TEXT NOT NULL DEFAULT '',
                start INTEGER,
                end INTEGER,
                metadata_json TEXT NOT NULL DEFAULT '{}'
            );
            CREATE INDEX IF NOT EXISTS idx_spans_artifact ON spans(artifact_id);

            CREATE TABLE IF NOT EXISTS memory_states (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                source_span_ids_json TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                archived INTEGER NOT NULL DEFAULT 0,
                embedding_ref TEXT NOT NULL DEFAULT '',
                utility_json TEXT NOT NULL DEFAULT '{}',
                metadata_json TEXT NOT NULL DEFAULT '{}'
            );
            CREATE INDEX IF NOT EXISTS idx_memory_archived ON memory_states(archived);

            CREATE TABLE IF NOT EXISTS trace_events (
                id TEXT PRIMARY KEY,
                run_name TEXT NOT NULL,
                op TEXT NOT NULL,
                input_json TEXT NOT NULL DEFAULT '{}',
                output_json TEXT NOT NULL DEFAULT '{}',
                metrics_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_trace_created ON trace_events(created_at);
            """
        )
        self.conn.commit()

    # ─── Upserts ──────────────────────────────────────────────────────────
    def upsert_artifact(self, artifact: Artifact) -> None:
        created_at = artifact.created_at or now_iso()
        self.conn.execute(
            """
            INSERT INTO artifacts
              (id, source, kind, title, content, uri, created_at, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
              source=excluded.source,
              kind=excluded.kind,
              title=excluded.title,
              content=excluded.content,
              uri=excluded.uri,
              metadata_json=excluded.metadata_json
            """,
            (
                artifact.id,
                artifact.source,
                artifact.kind,
                artifact.title,
                artifact.content,
                artifact.uri,
                created_at,
                _json(artifact.metadata),
            ),
        )

    def upsert_span(self, span: Span) -> None:
        self.conn.execute(
            """
            INSERT INTO spans
              (id, artifact_id, text, locator, start, end, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
              artifact_id=excluded.artifact_id,
              text=excluded.text,
              locator=excluded.locator,
              start=excluded.start,
              end=excluded.end,
              metadata_json=excluded.metadata_json
            """,
            (
                span.id,
                span.artifact_id,
                span.text,
                span.locator,
                span.start,
                span.end,
                _json(span.metadata),
            ),
        )

    def upsert_memory_state(self, state: MemoryState) -> None:
        ts = now_iso()
        created_at = state.created_at or ts
        updated_at = state.updated_at or ts
        self.conn.execute(
            """
            INSERT INTO memory_states
              (id, content, source_span_ids_json, created_at, updated_at,
               archived, embedding_ref, utility_json, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
              content=excluded.content,
              source_span_ids_json=excluded.source_span_ids_json,
              updated_at=excluded.updated_at,
              archived=excluded.archived,
              embedding_ref=excluded.embedding_ref,
              utility_json=excluded.utility_json,
              metadata_json=excluded.metadata_json
            """,
            (
                state.id,
                state.content,
                _json(state.source_span_ids),
                created_at,
                updated_at,
                1 if state.archived else 0,
                state.embedding_ref,
                _json(state.utility),
                _json(state.metadata),
            ),
        )

    def log_trace(self, event: TraceEvent) -> None:
        created_at = event.created_at or now_iso()
        self.conn.execute(
            """
            INSERT OR REPLACE INTO trace_events
              (id, run_name, op, input_json, output_json, metrics_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.id or f"trace_{uuid.uuid4().hex}",
                event.run_name,
                event.op,
                _json(event.input),
                _json(event.output),
                _json(event.metrics),
                created_at,
            ),
        )

    def commit(self) -> None:
        self.conn.commit()

    # ─── Reads ────────────────────────────────────────────────────────────
    def get_artifact(self, artifact_id: str) -> Artifact | None:
        row = self.conn.execute("SELECT * FROM artifacts WHERE id=?", (artifact_id,)).fetchone()
        if not row:
            return None
        return Artifact(
            id=row["id"],
            source=row["source"],
            kind=row["kind"],
            title=row["title"],
            content=row["content"],
            uri=row["uri"],
            created_at=row["created_at"],
            metadata=_loads(row["metadata_json"], {}),
        )

    def get_span(self, span_id: str) -> Span | None:
        row = self.conn.execute("SELECT * FROM spans WHERE id=?", (span_id,)).fetchone()
        if not row:
            return None
        return Span(
            id=row["id"],
            artifact_id=row["artifact_id"],
            text=row["text"],
            locator=row["locator"],
            start=row["start"],
            end=row["end"],
            metadata=_loads(row["metadata_json"], {}),
        )

    def get_memory_state(self, state_id: str) -> MemoryState | None:
        row = self.conn.execute("SELECT * FROM memory_states WHERE id=?", (state_id,)).fetchone()
        if not row:
            return None
        return self._memory_from_row(row)

    def list_memory_states(self, limit: int = 500, include_archived: bool = False) -> list[MemoryState]:
        where = "" if include_archived else "WHERE archived=0"
        rows = self.conn.execute(
            f"SELECT * FROM memory_states {where} ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [self._memory_from_row(r) for r in rows]

    def list_spans(self, limit: int = 500) -> list[Span]:
        rows = self.conn.execute("SELECT * FROM spans LIMIT ?", (limit,)).fetchall()
        return [
            Span(
                id=r["id"],
                artifact_id=r["artifact_id"],
                text=r["text"],
                locator=r["locator"],
                start=r["start"],
                end=r["end"],
                metadata=_loads(r["metadata_json"], {}),
            )
            for r in rows
        ]

    def latest_traces(self, limit: int = 100) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM trace_events ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            {
                "id": r["id"],
                "run_name": r["run_name"],
                "op": r["op"],
                "input": _loads(r["input_json"], {}),
                "output": _loads(r["output_json"], {}),
                "metrics": _loads(r["metrics_json"], {}),
                "created_at": r["created_at"],
            }
            for r in rows
        ]

    def stats(self) -> dict[str, Any]:
        def count(table: str) -> int:
            return int(self.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])

        by_source = [
            dict(row)
            for row in self.conn.execute(
                "SELECT source, COUNT(*) AS n FROM artifacts GROUP BY source ORDER BY n DESC"
            )
        ]
        return {
            "path": str(self.path),
            "artifacts": count("artifacts"),
            "spans": count("spans"),
            "memory_states": count("memory_states"),
            "trace_events": count("trace_events"),
            "artifacts_by_source": by_source,
        }

    # ─── Retrieval ────────────────────────────────────────────────────────
    def retrieve(self, query: str, k: int = 8, include_spans: bool = True) -> list[dict[str, Any]]:
        q_tokens = _tokens(query)
        q_counts = Counter(q_tokens)
        candidates: list[dict[str, Any]] = []

        for state in self.list_memory_states(limit=5000):
            score = self._lexical_score(q_counts, state.content)
            metadata_blob = _json(state.metadata)
            score += 0.25 * self._lexical_score(q_counts, metadata_blob)
            if score > 0:
                candidates.append({
                    "kind": "memory_state",
                    "id": state.id,
                    "source": state.metadata.get("adapter") or "memory_state",
                    "score": score,
                    "text": state.content,
                    "source_span_ids": state.source_span_ids,
                    "metadata": state.metadata,
                    "token_estimate": estimate_tokens(state.content),
                })

        if include_spans:
            rows = self.conn.execute("SELECT * FROM spans LIMIT 10000").fetchall()
            for r in rows:
                score = self._lexical_score(q_counts, r["text"])
                if score > 0:
                    metadata = _loads(r["metadata_json"], {})
                    artifact = self.get_artifact(r["artifact_id"])
                    candidates.append({
                        "kind": "span",
                        "id": r["id"],
                        "source": artifact.source if artifact else metadata.get("adapter", "span"),
                        "score": score * 0.85,
                        "text": r["text"],
                        "artifact_id": r["artifact_id"],
                        "locator": r["locator"],
                        "metadata": metadata,
                        "token_estimate": estimate_tokens(r["text"]),
                    })

        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:k]

    def provenance_for_state(self, state: MemoryState, limit: int = 6) -> list[dict[str, Any]]:
        out = []
        for sid in state.source_span_ids[:limit]:
            span = self.get_span(sid)
            if not span:
                continue
            artifact = self.get_artifact(span.artifact_id)
            out.append({
                "span_id": span.id,
                "text": span.text,
                "locator": span.locator,
                "artifact_id": span.artifact_id,
                "artifact_title": artifact.title if artifact else "",
                "artifact_source": artifact.source if artifact else "",
                "artifact_uri": artifact.uri if artifact else "",
            })
        return out

    def _memory_from_row(self, row: sqlite3.Row) -> MemoryState:
        return MemoryState(
            id=row["id"],
            content=row["content"],
            source_span_ids=_loads(row["source_span_ids_json"], []),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            archived=bool(row["archived"]),
            embedding_ref=row["embedding_ref"],
            utility=_loads(row["utility_json"], {}),
            metadata=_loads(row["metadata_json"], {}),
        )

    @staticmethod
    def _lexical_score(q_counts: Counter, text: str) -> float:
        if not q_counts or not text:
            return 0.0
        d_counts = Counter(_tokens(text))
        if not d_counts:
            return 0.0
        score = 0.0
        for tok, qn in q_counts.items():
            dn = d_counts.get(tok, 0)
            if dn:
                score += (1.0 + math.log(1 + dn)) * qn
        # Mild length normalization to avoid giant states always winning.
        return score / math.sqrt(20 + sum(d_counts.values()))


def store_for_run(run_name: str) -> SQLiteMemoryStore:
    return SQLiteMemoryStore(config.RESULTS_DIR / run_name / "core_memory.sqlite")


def trace_id(op: str, *parts: object) -> str:
    return stable_id(f"trace_{op}", now_iso(), uuid.uuid4().hex, *parts)
