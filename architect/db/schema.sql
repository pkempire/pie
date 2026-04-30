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

    -- ── Multi-axis taxonomy ────────────────────────────────────────────────
    -- A flat `type` was too coarse. Four axes cover the questions a planner
    -- actually wants to ask:
    --   kind        what kind of thing it is
    --   runtime     how you run it
    --   deployment  where it lives
    --   stack_layer where in the architecture stack
    -- The legacy `type` column is kept (deprecated) for compatibility with
    -- existing queries; it shadows `kind`.
    type                TEXT    NOT NULL,                  -- DEPRECATED — mirrors `kind`
    kind                TEXT    NOT NULL DEFAULT 'tool',
       -- model | sdk | framework | api | mcp_server | infra | template | tool | dataset | application
    runtime             TEXT    NOT NULL DEFAULT 'mixed',
       -- python | typescript | cross | hosted | mcp | mixed
    deployment          TEXT    NOT NULL DEFAULT 'both',
       -- local | self_hosted | hosted_only | both
    stack_layer         TEXT    NOT NULL DEFAULT 'orchestration',
       -- foundation_model | inference_proxy | runtime_infra | client_library
       -- | orchestration | application | data

    one_liner           TEXT    NOT NULL DEFAULT '',
    summary             TEXT    NOT NULL DEFAULT '',
    capability_long     TEXT    NOT NULL DEFAULT '',
    homepage_url        TEXT,
    github_url          TEXT,
    docs_url            TEXT,
    mcp_url             TEXT,
    pricing_model       TEXT,                              -- free|freemium|paid|oss|usage_based
    hosted_or_self      TEXT,                              -- hosted|self_hosted|both
    license             TEXT,
    embedding_json      TEXT,
    importance          REAL    NOT NULL DEFAULT 0.0,
    last_verified_at    TEXT,
    last_referenced_at  TEXT,
    created_at          TEXT    NOT NULL DEFAULT (datetime('now')),
    updated_at          TEXT    NOT NULL DEFAULT (datetime('now')),
    extras_json         TEXT    NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_components_kind        ON components(kind);
CREATE INDEX IF NOT EXISTS idx_components_runtime     ON components(runtime);
CREATE INDEX IF NOT EXISTS idx_components_stack_layer ON components(stack_layer);
CREATE INDEX IF NOT EXISTS idx_components_importance  ON components(importance);


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
