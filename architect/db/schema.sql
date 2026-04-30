-- architect schema
-- ---------------------------------------------------------------------------
-- Local SQLite. Vector similarity is computed in Python over the embedding
-- column at query time. At < 10k components this is faster than the
-- engineering complexity of sqlite-vss, and keeps the dependency surface to
-- "Python stdlib + openai + requests".

PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;


-- ─── Components ─────────────────────────────────────────────────────────────
-- One row per atomic AI component (tool / library / API / MCP server / model
-- provider / framework / template / SDK / infra service).
CREATE TABLE IF NOT EXISTS components (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    slug                TEXT    UNIQUE NOT NULL,           -- "browserbase", "stagehand"
    name                TEXT    NOT NULL,                  -- canonical display name
    aliases_json        TEXT    NOT NULL DEFAULT '[]',      -- alternative names
    type                TEXT    NOT NULL,                  -- tool|library|api|mcp_server|model_api|framework|template|infra|sdk
    one_liner           TEXT    NOT NULL DEFAULT '',       -- one sentence (< 140 chars)
    summary             TEXT    NOT NULL DEFAULT '',       -- 2-3 sentences
    capability_long     TEXT    NOT NULL DEFAULT '',       -- 1-2 paragraphs
    homepage_url        TEXT,
    github_url          TEXT,
    docs_url            TEXT,
    mcp_url             TEXT,                              -- if it's an MCP server
    pricing_model       TEXT,                              -- free|freemium|paid|oss
    hosted_or_self      TEXT,                              -- hosted|self_hosted|both
    license             TEXT,                              -- MIT|Apache-2.0|proprietary|...
    embedding_json      TEXT,                              -- JSON-encoded float[1536]
    importance          REAL    NOT NULL DEFAULT 0.0,      -- decays without reinforcement
    last_verified_at    TEXT,                              -- ISO-8601 of last successful enrichment
    last_referenced_at  TEXT,                              -- ISO-8601 of last query/plan touch
    created_at          TEXT    NOT NULL DEFAULT (datetime('now')),
    updated_at          TEXT    NOT NULL DEFAULT (datetime('now')),
    -- Free-form structured extras (latest version, github stars, etc.)
    extras_json         TEXT    NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_components_type ON components(type);
CREATE INDEX IF NOT EXISTS idx_components_importance ON components(importance);


-- ─── Tags / capability concepts ──────────────────────────────────────────────
-- A component can be tagged with ≥1 capability concept. Tags are themselves
-- entities (so we can query "what tags exist?") and are joined to components
-- via component_tags.
CREATE TABLE IF NOT EXISTS tags (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    slug        TEXT    UNIQUE NOT NULL,                  -- "browser-agent"
    name        TEXT    NOT NULL,                         -- "Browser Agent"
    definition  TEXT    NOT NULL DEFAULT ''               -- 1-2 sentences
);

CREATE TABLE IF NOT EXISTS component_tags (
    component_id INTEGER NOT NULL REFERENCES components(id) ON DELETE CASCADE,
    tag_id       INTEGER NOT NULL REFERENCES tags(id)       ON DELETE CASCADE,
    weight       REAL    NOT NULL DEFAULT 1.0,             -- how central this tag is for this component
    PRIMARY KEY (component_id, tag_id)
);


-- ─── Relationships between components ────────────────────────────────────────
-- Typed edges. Mirrors KGmem's RelationshipType taxonomy.
CREATE TABLE IF NOT EXISTS relationships (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id       INTEGER NOT NULL REFERENCES components(id) ON DELETE CASCADE,
    target_id       INTEGER NOT NULL REFERENCES components(id) ON DELETE CASCADE,
    type            TEXT    NOT NULL,                       -- integrates_with|replaces|alternative_to|depends_on|part_of|uses
    confidence      REAL    NOT NULL DEFAULT 1.0,           -- [0..1]; reinforcement-learnable
    evidence_url    TEXT,                                   -- where we got this edge from (PR, README, blog post)
    note            TEXT    NOT NULL DEFAULT '',            -- human-readable detail
    created_at      TEXT    NOT NULL DEFAULT (datetime('now')),
    last_seen_at    TEXT    NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_rel_src ON relationships(source_id);
CREATE INDEX IF NOT EXISTS idx_rel_tgt ON relationships(target_id);
CREATE INDEX IF NOT EXISTS idx_rel_type ON relationships(type);


-- ─── Architectures (real-world systems) ──────────────────────────────────────
-- One row per discovered system that uses ≥2 of our components. The
-- architecture_components junction tells us which components co-occur in
-- the wild. This is the data that powers "X is commonly used with Y."
CREATE TABLE IF NOT EXISTS architectures (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    source          TEXT    NOT NULL,                       -- "github" | "awesome-list" | "n8n-template" | "show-hn" | "arxiv"
    source_url      TEXT    UNIQUE NOT NULL,                -- canonical link
    name            TEXT    NOT NULL,
    description     TEXT    NOT NULL DEFAULT '',
    summary         TEXT    NOT NULL DEFAULT '',            -- LLM-extracted "what does this app do"
    pattern         TEXT    NOT NULL DEFAULT '',            -- e.g. "browser-agent + scheduler + slack-alerting"
    quality_signal  REAL    NOT NULL DEFAULT 0.0,           -- normalised stars / forks / age signal
    raw_json        TEXT    NOT NULL DEFAULT '{}',          -- repo/readme/etc snapshot for reference
    discovered_at   TEXT    NOT NULL DEFAULT (datetime('now')),
    last_verified_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_arch_source ON architectures(source);

CREATE TABLE IF NOT EXISTS architecture_components (
    architecture_id INTEGER NOT NULL REFERENCES architectures(id) ON DELETE CASCADE,
    component_id    INTEGER NOT NULL REFERENCES components(id)    ON DELETE CASCADE,
    role            TEXT    NOT NULL DEFAULT '',            -- "browser_runtime" | "memory_layer" | etc.
    evidence        TEXT    NOT NULL DEFAULT '',            -- snippet of code/import line that proves it
    PRIMARY KEY (architecture_id, component_id)
);
CREATE INDEX IF NOT EXISTS idx_archcomp_comp ON architecture_components(component_id);


-- ─── User queries (analytics + future RL signal) ─────────────────────────────
CREATE TABLE IF NOT EXISTS user_queries (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    query               TEXT    NOT NULL,
    plan_components_json TEXT   NOT NULL DEFAULT '[]',      -- list of component_ids the planner emitted
    plan_format         TEXT    NOT NULL DEFAULT 'markdown',
    user_email          TEXT,
    follow_up_outcome   TEXT,                                -- free-text response a week later
    created_at          TEXT    NOT NULL DEFAULT (datetime('now'))
);


-- ─── Ingestion queue (work-in-progress URLs to enrich) ───────────────────────
CREATE TABLE IF NOT EXISTS ingestion_queue (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    url             TEXT    UNIQUE NOT NULL,
    source          TEXT    NOT NULL,                       -- "github_trending" | "show_hn" | "manual"
    priority        INTEGER NOT NULL DEFAULT 0,             -- higher = sooner
    status          TEXT    NOT NULL DEFAULT 'pending',     -- pending|in_progress|done|failed
    error_message   TEXT,
    enqueued_at     TEXT    NOT NULL DEFAULT (datetime('now')),
    processed_at    TEXT
);
CREATE INDEX IF NOT EXISTS idx_iq_status ON ingestion_queue(status);
