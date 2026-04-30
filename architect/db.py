"""SQLite wrapper for the architect KG.

Why not Postgres / pgvector?
  - Single-machine MVP. No hosted dependency, no connection pool, no auth.
  - At < 10k components, in-Python cosine similarity over the embedding
    column beats the engineering complexity of a vector extension.
  - Migration to Postgres later is a `pg_dump | psql` away because we're
    using vanilla ANSI SQL.

Vector search lives here in `search()`. We load the embeddings once into
a numpy matrix at query time (cached for the lifetime of the connection
on a small DB), then do a single cosine multiply.
"""
from __future__ import annotations
import json
import logging
import math
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).resolve().parent / "data" / "architect.db"
SCHEMA_PATH = Path(__file__).resolve().parent / "db" / "schema.sql"


# ─── Connection ──────────────────────────────────────────────────────────────
def init_db(db_path: Path = DB_PATH) -> None:
    """Create the DB file from schema.sql if it doesn't exist."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(SCHEMA_PATH.read_text())
        conn.commit()
    finally:
        conn.close()


@contextmanager
def connect(db_path: Path = DB_PATH) -> Iterable[sqlite3.Connection]:
    """Yield a row-factory-equipped connection. Init the DB if missing."""
    if not db_path.exists():
        init_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")


# ─── Component upsert ────────────────────────────────────────────────────────
def upsert_component(conn: sqlite3.Connection, **fields: Any) -> int:
    """Insert or update a component by slug. Returns its id.

    Required: slug, name, type. Everything else is optional and overrides
    only when supplied (so partial enrichment doesn't blow away existing
    fields).
    """
    if "slug" not in fields or "name" not in fields or "type" not in fields:
        raise ValueError("upsert_component requires slug, name, type")

    fields = dict(fields)
    fields["updated_at"] = _now()

    # JSON-serialise list/dict columns
    for json_col in ("aliases_json", "extras_json"):
        if json_col in fields and not isinstance(fields[json_col], str):
            fields[json_col] = json.dumps(fields[json_col])
    if "embedding_json" in fields and isinstance(fields["embedding_json"], list):
        fields["embedding_json"] = json.dumps(fields["embedding_json"])

    cur = conn.execute("SELECT id FROM components WHERE slug = ?", (fields["slug"],))
    row = cur.fetchone()
    if row is None:
        cols = ",".join(fields.keys())
        ph = ",".join("?" * len(fields))
        conn.execute(
            f"INSERT INTO components ({cols}) VALUES ({ph})",
            tuple(fields.values()),
        )
        return conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    cid = row["id"]
    set_clause = ",".join(f"{k}=?" for k in fields if k != "slug")
    vals = [v for k, v in fields.items() if k != "slug"] + [cid]
    conn.execute(f"UPDATE components SET {set_clause} WHERE id = ?", tuple(vals))
    return cid


def get_component(conn: sqlite3.Connection, slug: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM components WHERE slug = ?", (slug,)).fetchone()


# ─── Tags ────────────────────────────────────────────────────────────────────
def upsert_tag(conn: sqlite3.Connection, slug: str, name: str,
                definition: str = "") -> int:
    cur = conn.execute("SELECT id FROM tags WHERE slug = ?", (slug,))
    row = cur.fetchone()
    if row:
        if definition:
            conn.execute("UPDATE tags SET name=?, definition=? WHERE id=?",
                         (name, definition, row["id"]))
        return row["id"]
    conn.execute("INSERT INTO tags (slug, name, definition) VALUES (?, ?, ?)",
                 (slug, name, definition))
    return conn.execute("SELECT last_insert_rowid()").fetchone()[0]


def tag_component(conn: sqlite3.Connection, component_id: int, tag_id: int,
                   weight: float = 1.0) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO component_tags (component_id, tag_id, weight) "
        "VALUES (?, ?, ?)",
        (component_id, tag_id, weight),
    )


# ─── Relationships ───────────────────────────────────────────────────────────
def add_relationship(
    conn: sqlite3.Connection,
    source_id: int,
    target_id: int,
    type: str,
    confidence: float = 1.0,
    evidence_url: str | None = None,
    note: str = "",
) -> int:
    """Idempotent (source, target, type) upsert; bumps last_seen and confidence."""
    cur = conn.execute(
        "SELECT id, confidence FROM relationships "
        "WHERE source_id=? AND target_id=? AND type=?",
        (source_id, target_id, type),
    )
    row = cur.fetchone()
    if row:
        # Reinforce: average new evidence with existing
        new_conf = (row["confidence"] + confidence) / 2.0
        conn.execute(
            "UPDATE relationships SET confidence=?, last_seen_at=?, evidence_url=COALESCE(?, evidence_url), note=COALESCE(NULLIF(?,''), note) "
            "WHERE id=?",
            (new_conf, _now(), evidence_url, note, row["id"]),
        )
        return row["id"]
    conn.execute(
        "INSERT INTO relationships (source_id, target_id, type, confidence, evidence_url, note) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (source_id, target_id, type, confidence, evidence_url, note),
    )
    return conn.execute("SELECT last_insert_rowid()").fetchone()[0]


# ─── Architecture upsert ─────────────────────────────────────────────────────
def upsert_architecture(conn: sqlite3.Connection, **fields: Any) -> int:
    """Upsert by source_url. Returns id."""
    if "source" not in fields or "source_url" not in fields or "name" not in fields:
        raise ValueError("upsert_architecture requires source, source_url, name")
    if "raw_json" in fields and not isinstance(fields["raw_json"], str):
        fields["raw_json"] = json.dumps(fields["raw_json"])
    cur = conn.execute("SELECT id FROM architectures WHERE source_url=?",
                       (fields["source_url"],))
    row = cur.fetchone()
    if row is None:
        cols = ",".join(fields.keys())
        ph = ",".join("?" * len(fields))
        conn.execute(
            f"INSERT INTO architectures ({cols}) VALUES ({ph})",
            tuple(fields.values()),
        )
        return conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    aid = row["id"]
    set_clause = ",".join(f"{k}=?" for k in fields if k != "source_url")
    vals = [v for k, v in fields.items() if k != "source_url"] + [aid]
    conn.execute(f"UPDATE architectures SET {set_clause} WHERE id=?", tuple(vals))
    return aid


def link_architecture_component(
    conn: sqlite3.Connection, architecture_id: int, component_id: int,
    role: str = "", evidence: str = "",
) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO architecture_components "
        "(architecture_id, component_id, role, evidence) VALUES (?, ?, ?, ?)",
        (architecture_id, component_id, role, evidence),
    )


# ─── Search ──────────────────────────────────────────────────────────────────
def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (na * nb)


def search_components(
    conn: sqlite3.Connection,
    query_embedding: list[float],
    type: str | None = None,
    top_k: int = 10,
) -> list[dict]:
    """Cosine-similarity search over the components.embedding_json column.

    For < 10k components this finishes in ms; we don't need an ANN index yet.
    """
    where = ""
    params: tuple = ()
    if type:
        where = "WHERE type = ?"
        params = (type,)
    cur = conn.execute(
        f"SELECT id, slug, name, type, one_liner, summary, embedding_json, importance "
        f"FROM components {where}",
        params,
    )
    scored = []
    for row in cur:
        emb_str = row["embedding_json"]
        if not emb_str:
            continue
        try:
            emb = json.loads(emb_str)
        except Exception:
            continue
        score = _cosine(query_embedding, emb)
        # Slight bump for high-importance components (recency / usage)
        score += 0.05 * (row["importance"] or 0.0)
        scored.append((score, dict(row)))
    scored.sort(key=lambda x: x[0], reverse=True)
    out = []
    for score, row in scored[:top_k]:
        row["score"] = round(score, 4)
        row.pop("embedding_json", None)
        out.append(row)
    return out


def find_relationships(
    conn: sqlite3.Connection, component_id: int, type: str | None = None,
) -> list[dict]:
    where = "(source_id = ? OR target_id = ?)"
    params: list = [component_id, component_id]
    if type:
        where += " AND type = ?"
        params.append(type)
    cur = conn.execute(
        f"""
        SELECT r.*, c_src.slug AS source_slug, c_src.name AS source_name,
                    c_tgt.slug AS target_slug, c_tgt.name AS target_name
        FROM relationships r
        JOIN components c_src ON c_src.id = r.source_id
        JOIN components c_tgt ON c_tgt.id = r.target_id
        WHERE {where}
        ORDER BY r.confidence DESC, r.last_seen_at DESC
        """,
        tuple(params),
    )
    return [dict(row) for row in cur]


def find_co_occurring_components(
    conn: sqlite3.Connection, component_id: int, min_count: int = 1,
) -> list[dict]:
    """Components that appear in real architectures alongside this one.

    Returns rows with `co_count` (architectures shared) and `total` (this
    component's total architecture count) so callers can compute lift.
    """
    cur = conn.execute(
        """
        SELECT c.id, c.slug, c.name, COUNT(*) AS co_count
        FROM architecture_components ac1
        JOIN architecture_components ac2
          ON ac1.architecture_id = ac2.architecture_id
         AND ac2.component_id != ac1.component_id
        JOIN components c ON c.id = ac2.component_id
        WHERE ac1.component_id = ?
        GROUP BY c.id
        HAVING co_count >= ?
        ORDER BY co_count DESC
        """,
        (component_id, min_count),
    )
    return [dict(row) for row in cur]


# ─── Ingestion queue ─────────────────────────────────────────────────────────
def enqueue_url(conn: sqlite3.Connection, url: str, source: str,
                 priority: int = 0) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO ingestion_queue (url, source, priority) "
        "VALUES (?, ?, ?)",
        (url, source, priority),
    )


def take_next_pending(conn: sqlite3.Connection) -> sqlite3.Row | None:
    row = conn.execute(
        "SELECT * FROM ingestion_queue WHERE status='pending' "
        "ORDER BY priority DESC, enqueued_at ASC LIMIT 1"
    ).fetchone()
    if row:
        conn.execute(
            "UPDATE ingestion_queue SET status='in_progress' WHERE id=?",
            (row["id"],),
        )
    return row


def mark_done(conn: sqlite3.Connection, queue_id: int,
              error: str | None = None) -> None:
    conn.execute(
        "UPDATE ingestion_queue SET status=?, error_message=?, processed_at=? "
        "WHERE id=?",
        ("failed" if error else "done", error, _now(), queue_id),
    )
