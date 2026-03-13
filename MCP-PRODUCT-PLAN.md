# PIE → Temporal MCP: Full Build Plan
## From Research Project to Working Product

---

## What We're Building

An MCP server that gives any LLM client (Claude Desktop, Cursor, etc.) temporal awareness — knowledge of your world that includes *when* things happened, what's stale, what's approaching, and what you haven't touched in a while. Backed by PIE's existing world model (3,998 entities, 6,706 transitions, 3,237 relationships spanning Jan 2025 → Jan 2026).

Two LLM calls per conversation:
1. **Before** the user talks: generate a temporal briefing (what's active, what's stale, what deadlines are approaching, how long since last interaction)
2. **After** the conversation: extract any new entities/transitions/relationships and update the world model

Plus the TemporalBench benchmark to prove it works and measure it against every other approach.

---

## PIE Code Audit: What Exists Today

### Codebase Stats
- **26 Python files**, ~6,500 LOC
- **3,998 entities** (1,602 events, 602 concepts, 472 tools, 428 projects, 305 decisions, 251 orgs, 141 goals, 122 beliefs, 68 people, 7 periods)
- **6,706 state transitions** spanning Jan 3 2025 → Jan 31 2026
- **3,237 relationships**
- **20MB** world model JSON
- **Zero MCP code** — pure library, no server

### What We Keep (Direct Reuse)

| Module | LOC | What It Does | Role in MCP |
|--------|-----|-------------|-------------|
| `core/models.py` | 200 | Entity, StateTransition, Relationship dataclasses + enums | **Data layer** — unchanged |
| `core/world_model.py` | 700 | In-memory graph store with JSON persistence, 3-tier entity resolution (exact → fuzzy → embedding), alias management | **Data store** — load on startup, query on every request |
| `core/prompt_engine.py` | 510 | Generates structured briefing from world model: project snapshots, goals, predictions, attention flags | **THE CORE** — this IS the temporal briefing. Adapt to MCP tool output |
| `core/dynamics.py` | 600 | State transition analysis: volatility, staleness, co-occurrence, prediction | **Enrichment** — feeds into briefing quality |
| `core/llm.py` | 180 | OpenAI wrapper with JSON mode + retry | **LLM calls** — extraction + resolution |
| `core/parser.py` | 200 | ChatGPT JSON export → conversation objects | **Ingestion** — for batch import of historical data |
| `ingestion/pipeline.py` | 500 | End-to-end: parse → extract → resolve → update world model | **After-conversation updater** — adapt for single-conversation ingestion |
| `ingestion/prompts.py` | 280 | Extraction system prompt + message builders | **Extraction quality** — proven prompts, reuse directly |
| `resolution/resolver.py` | 350 | 3-tier entity resolution (string → embedding → LLM) | **Dedup** — prevents entity sprawl |
| `retrieval/context_compiler.py` | 230 | Subgraph → LLM-ready markdown with temporal annotations | **Query responses** — format entity lookups |
| `config.py` | 200 | Centralized config with documented thresholds | **Config** — unchanged |
| `export_viewer.py` | 410 | Self-contained HTML viewer export | **Demo/debug** — keep for explorer.html |
| `browse.py` | 460 | CLI world model browser | **Debug** — useful during development |

**Total reusable: ~4,820 LOC (74% of codebase)**

### What We Drop

| Module | LOC | Why Drop |
|--------|-----|----------|
| `core/temporal.py` | 600 | Survival functions, Hawkes process, rhythm detection — hardcoded math. Replace with LLM reasoning over raw temporal metadata |
| `core/scheduler.py` | 300 | Self-messaging/wake-up system — not integrated, not needed for MCP (client handles scheduling) |
| `retrieval/temporal_retriever.py` | 470 | Embedding-based temporal retrieval with weighted scoring — name match + LLM works fine, proved by existing query interface |
| `resolution/llm_resolver.py` | 240 | Alternative resolver — unused, `resolver.py` is the active one |
| `resolution/web_grounder.py` | 150 | Web verification — disabled in config, unreliable |
| `analysis/procedural_memory.py` | 280 | Standalone analysis — not integrated into pipeline |
| `grounding/web_enrichment.py` | ? | Web enrichment — disabled |

**Total dropped: ~2,040 LOC (but temporal.py's SURVIVAL DATA is valuable — we keep the learned rhythms as raw data for the LLM to reason over)**

### What We Write New

| Module | Est LOC | Purpose |
|--------|---------|---------|
| `mcp_server.py` | 400 | MCP server entry point — tool definitions, lifecycle |
| `temporal_briefing.py` | 300 | Pre-conversation briefing generator (replaces temporal.py's math with LLM reasoning) |
| `live_ingestion.py` | 250 | After-conversation extractor (thin wrapper around pipeline.py for single conversations) |
| `thread_tracker.py` | 200 | Track active threads, deadlines, commitments across conversations |
| `gap_analyzer.py` | 150 | Compute gap duration, characterize absence, flag staleness |

**Total new code: ~1,300 LOC**

---

## Architecture

```
┌────────────────────────────────────────────────┐
│               Claude Desktop / Cursor          │
│            (or any MCP-compatible client)       │
└───────────────────┬────────────────────────────┘
                    │ MCP Protocol (stdio)
                    │
┌───────────────────▼────────────────────────────┐
│              mcp_server.py                      │
│                                                 │
│  TOOLS:                                         │
│  ├── get_temporal_briefing()     ← BEFORE conv  │
│  ├── update_world(conversation)  ← AFTER conv   │
│  ├── query_entity(name)                         │
│  ├── query_world(question)                      │
│  ├── get_timeline(entity)                       │
│  ├── get_stale_threads()                        │
│  ├── get_approaching_deadlines()                │
│  └── get_world_snapshot()                       │
│                                                 │
│  RESOURCES:                                     │
│  └── pie://briefing  (auto-injected context)    │
│                                                 │
│  PROMPTS:                                       │
│  └── temporal-partner  (system prompt template) │
└───────────┬────────────┬───────────────────────┘
            │            │
    ┌───────▼──────┐  ┌──▼──────────────┐
    │ temporal_    │  │ live_           │
    │ briefing.py  │  │ ingestion.py    │
    │              │  │                 │
    │ gap_analyzer │  │ pipeline.py     │
    │ thread_tracker│ │ resolver.py     │
    └───────┬──────┘  └──┬──────────────┘
            │            │
    ┌───────▼────────────▼───────────────┐
    │         world_model.py              │
    │    (in-memory graph + JSON persist) │
    │                                     │
    │  3,998 entities                     │
    │  6,706 transitions                  │
    │  3,237 relationships                │
    │  Temporal metadata on every node    │
    └─────────────────────────────────────┘
```

---

## The 8 MCP Tools (Detailed Specs)

### Tool 1: `get_temporal_briefing`
**The most important tool. This IS the product.**

```python
@mcp.tool()
async def get_temporal_briefing(
    focus_project: str | None = None,
    current_context: str | None = None
) -> str:
    """
    Generate a temporal briefing for the current conversation.
    Call this at the START of every conversation to give the LLM
    awareness of the user's world state and temporal context.

    Args:
        focus_project: Optional project name to expand context for.
        current_context: Optional brief description of what the user
                        is working on right now (helps relevance).

    Returns:
        A structured temporal briefing (3-4K tokens) containing:
        - Active projects with temporal health (last touched, rhythm, staleness)
        - Approaching deadlines and commitments
        - Gap analysis (time since last interaction, what changed)
        - Attention flags (overdue items, stale threads)
        - Goals and their progress
        - Predictions (what should happen next based on patterns)
    """
```

**Implementation:** This is prompt_engine.py's `generate()` with two changes:
1. Replace survival function calls with raw temporal metadata (last_seen, transition count, gap since last) formatted for the LLM to reason over
2. Add gap analysis: time since the MCP server's last recorded interaction, what happened in between
3. Add thread tracking: active conversation threads with deadlines

**Output format (example):**
```
## Temporal Briefing — 2026-03-13 14:30

### Time Context
Last interaction: 2 days ago (Tuesday 3/11, working on PIE research).
Current gap: 2 days. This is within your normal rhythm (~1.5 days avg).

### Active Projects (by urgency)
1. **PIE / Personal Intelligence Engine** [ACTIVE]
   Last touched: 2d ago | Rhythm: ~1.5d | 289 state transitions
   Status: Research phase — temporal awareness thesis, benchmark design
   Next steps: Build TemporalBench, implement MCP server
   ⚠️ This has been your primary focus for 3 weeks straight

2. **sponsorFind** [DORMANT — 45d]
   Last touched: 45d ago | Rhythm: ~7d | 65 transitions
   Status: Built, positioned as quick cashflow option
   ⚠️ Silent 6.4x its rhythm — may need attention or explicit shelving

3. **Lucid Academy** [DORMANT — 30d]
   Last touched: 30d ago | Rhythm: ~14d | 38 transitions
   Status: Content platform, 180 students
   Note: Holiday gap may explain some silence

### Approaching Deadlines
(none tracked)

### Commitments
(none tracked)

### Attention Flags
- sponsorFind has defined next_steps but 45d of silence
- Lucid Labs hasn't been discussed in 60d — still relevant?

### World Model Health
3,998 entities tracked. 428 projects, 141 goals.
Active: 12 | Expected: 45 | Dormant: 180 | Fading: 150 | Dead: 41
```

### Tool 2: `update_world`
**Call AFTER each conversation to keep the world model current.**

```python
@mcp.tool()
async def update_world(
    conversation_summary: str,
    entities_mentioned: list[str] | None = None,
    deadlines_mentioned: list[dict] | None = None,
    commitments_made: list[dict] | None = None
) -> str:
    """
    Update the world model after a conversation.

    Args:
        conversation_summary: Brief summary of what was discussed.
        entities_mentioned: Names of projects/people/tools discussed.
        deadlines_mentioned: [{entity, deadline_date, description}]
        commitments_made: [{entity, commitment, due_date}]

    Returns:
        Summary of what was updated.
    """
```

**Implementation:** Thin wrapper around pipeline.py's extraction logic, adapted for single-conversation ingestion instead of batch. Uses the existing extraction prompts + 3-tier resolver.

### Tool 3: `query_entity`
```python
@mcp.tool()
async def query_entity(name: str) -> str:
    """Get full context for a specific entity (project, person, tool, etc.)."""
```
**Implementation:** world_model.find_by_name() → context_compiler.compile_entity_context()

### Tool 4: `query_world`
```python
@mcp.tool()
async def query_world(question: str) -> str:
    """Ask a natural language question about the user's world."""
```
**Implementation:** The existing query interface from ARCHITECTURE-FINAL — classify intent → retrieve subgraph → compile context. Uses the LLM to reason over retrieved entities rather than complex retrieval scoring.

### Tool 5: `get_timeline`
```python
@mcp.tool()
async def get_timeline(entity_name: str, limit: int = 20) -> str:
    """Get the chronological evolution of an entity."""
```
**Implementation:** Get transitions for entity, format chronologically with context_compiler.

### Tool 6: `get_stale_threads`
```python
@mcp.tool()
async def get_stale_threads(threshold_days: int = 14) -> str:
    """Get entities that haven't been touched in a while but have open next_steps."""
```
**Implementation:** Filter entities where last_seen > threshold AND current_state has next_steps. Sort by staleness. This is the "attention flags" section extracted as a standalone tool.

### Tool 7: `get_approaching_deadlines`
```python
@mcp.tool()
async def get_approaching_deadlines(window_days: int = 7) -> str:
    """Get deadlines approaching in the next N days."""
```
**Implementation:** New — scans thread_tracker's deadline store. Returns deadlines sorted by proximity.

### Tool 8: `get_world_snapshot`
```python
@mcp.tool()
async def get_world_snapshot(entity_type: str | None = None) -> str:
    """Get a high-level snapshot of the entire world model."""
```
**Implementation:** Entity counts by type + status, temporal health distribution, most active/most stale entities. Top-level overview.

---

## New Modules to Write

### `temporal_briefing.py` — Replaces temporal.py

The key philosophical change: **instead of computing survival probabilities and classifying entities via math, we give the LLM raw temporal metadata and let it reason.**

What we compute (simple arithmetic, not statistical models):
- `days_since_last_seen` = (now - entity.last_seen) / 86400
- `days_since_first_seen` = (now - entity.first_seen) / 86400
- `total_transitions` = len(transitions for entity)
- `avg_gap_between_transitions` = total_span / (n_transitions - 1)
- `gap_ratio` = days_since_last_seen / avg_gap
- `last_3_gaps` = actual gap durations of last 3 transitions

What we DON'T compute (let the LLM handle):
- Survival probability — just give it the gap_ratio and let it reason about "is this entity likely dead?"
- Rhythm classification — give it the gaps and let it say "this updates about weekly"
- Anomaly detection — give it the gap_ratio and let it say "this is unusually quiet"
- Predictions — give it the pattern and let it reason about what should happen next

```python
class TemporalBriefing:
    """Generate temporal briefings using raw metadata + LLM reasoning."""

    def __init__(self, world_model: WorldModel):
        self.wm = world_model

    def compute_temporal_metadata(self, entity_id: str, ref_time: float) -> dict:
        """Compute raw temporal stats for an entity. Pure arithmetic."""
        entity = self.wm.entities[entity_id]
        transitions = self._get_transitions(entity_id)

        gaps = self._compute_gaps(transitions)

        return {
            "name": entity.name,
            "type": entity.type.value,
            "first_seen": self._humanize(entity.first_seen, ref_time),
            "last_seen": self._humanize(entity.last_seen, ref_time),
            "days_silent": round((ref_time - entity.last_seen) / 86400, 1),
            "total_updates": len(transitions),
            "avg_gap_days": round(sum(gaps) / len(gaps), 1) if gaps else None,
            "last_3_gaps_days": [round(g, 1) for g in gaps[-3:]],
            "gap_ratio": round((ref_time - entity.last_seen) / (sum(gaps) / len(gaps)), 1) if gaps else None,
            "current_state_summary": self._summarize_state(entity.current_state),
            "has_next_steps": bool(entity.current_state.get("next_steps")),
            "next_steps": entity.current_state.get("next_steps", [])[:3],
        }

    def generate_briefing(self, ref_time: float, focus: str = None) -> str:
        """Generate the full temporal briefing."""
        # 1. Compute metadata for all relevant entities
        # 2. Sort by urgency (gap_ratio descending = most overdue first)
        # 3. Format as structured text
        # 4. Add gap analysis (time since last MCP interaction)
        # 5. Add thread/deadline/commitment tracking
        ...
```

### `thread_tracker.py` — New

Tracks active conversation threads, deadlines, and commitments. Persisted as a simple JSON file alongside the world model.

```python
@dataclass
class Thread:
    id: str
    entity_id: str | None  # linked to world model entity
    topic: str
    opened: float  # timestamp
    last_mentioned: float
    status: str  # "active" | "waiting" | "completed" | "abandoned"
    deadline: float | None
    commitments: list[dict]  # [{what, who, due}]

class ThreadTracker:
    """Track conversation threads, deadlines, commitments."""

    def __init__(self, persist_path: str):
        self.threads: dict[str, Thread] = {}
        self.load(persist_path)

    def open_thread(self, topic, entity_id=None, deadline=None): ...
    def add_commitment(self, thread_id, what, who, due): ...
    def get_approaching_deadlines(self, window_days=7): ...
    def get_overdue_commitments(self): ...
    def update_from_conversation(self, mentioned_entities, new_deadlines, new_commitments): ...
```

### `gap_analyzer.py` — New

Simple module that tracks when the MCP server was last queried (= proxy for "last user interaction") and computes gap characteristics.

```python
class GapAnalyzer:
    """Track interaction gaps and characterize absences."""

    def __init__(self, persist_path: str):
        self.interaction_log: list[float] = []  # timestamps
        self.load(persist_path)

    def record_interaction(self, timestamp: float = None): ...

    def analyze_current_gap(self, now: float) -> dict:
        """Returns gap duration, characterization, what might have changed."""
        last = self.interaction_log[-1] if self.interaction_log else None
        if not last:
            return {"status": "first_interaction"}

        gap_hours = (now - last) / 3600
        gap_days = gap_hours / 24

        # Compute avg gap from history
        avg_gap = self._avg_gap()

        return {
            "hours_since_last": round(gap_hours, 1),
            "days_since_last": round(gap_days, 1),
            "gap_ratio": round(gap_hours / (avg_gap * 24), 1) if avg_gap else None,
            "characterization": self._characterize(gap_days, avg_gap),
            # "normal" | "short_absence" | "extended_absence" | "long_absence"
        }
```

### `live_ingestion.py` — Adapter for single-conversation ingestion

```python
class LiveIngestion:
    """Update world model from a single conversation (not batch)."""

    def __init__(self, world_model: WorldModel, llm: LLMClient, resolver: EntityResolver):
        self.wm = world_model
        self.llm = llm
        self.resolver = resolver

    async def ingest_conversation(self, summary: str, timestamp: float) -> dict:
        """
        Extract entities/transitions from a conversation summary,
        resolve against existing world model, update.

        Returns: {entities_updated: [...], entities_created: [...], transitions: [...]}
        """
        # 1. Extract using existing prompts.py
        # 2. Resolve using existing resolver.py (3-tier)
        # 3. Apply to world model
        # 4. Persist
```

### `mcp_server.py` — Entry Point

```python
from mcp.server import Server
from mcp.server.stdio import stdio_server

app = Server("pie-temporal")

# Load world model on startup
wm = WorldModel(persist_path="output/world_model.json")
briefing = TemporalBriefing(wm)
tracker = ThreadTracker(persist_path="output/threads.json")
gap = GapAnalyzer(persist_path="output/interactions.json")
llm = LLMClient(...)
resolver = EntityResolver(wm, llm)
ingestion = LiveIngestion(wm, llm, resolver)

@app.tool()
async def get_temporal_briefing(focus_project: str | None = None) -> str:
    now = time.time()
    gap.record_interaction(now)

    gap_info = gap.analyze_current_gap(now)
    deadlines = tracker.get_approaching_deadlines()
    commitments = tracker.get_overdue_commitments()
    entity_briefing = briefing.generate_briefing(now, focus=focus_project)

    return format_full_briefing(gap_info, deadlines, commitments, entity_briefing)

@app.tool()
async def update_world(conversation_summary: str, ...) -> str:
    result = await ingestion.ingest_conversation(conversation_summary, time.time())
    tracker.update_from_conversation(...)
    return format_update_result(result)

# ... other tools ...

async def main():
    async with stdio_server() as (read, write):
        await app.run(read, write, app.create_initialization_options())
```

---

## Build Plan (Week by Week)

### Week 1: MCP Server Shell + Briefing (Core Product)

**Day 1-2: MCP server scaffold**
- [ ] Set up project structure (pyproject.toml, dependencies)
- [ ] `mcp_server.py` — basic MCP server that starts, connects, lists tools
- [ ] Verify it connects to Claude Desktop via config
- [ ] `get_world_snapshot` tool — simplest tool, just load world model and return stats

**Day 3-4: Temporal briefing (the money feature)**
- [ ] `temporal_briefing.py` — compute_temporal_metadata for all entities
- [ ] `gap_analyzer.py` — track interaction timestamps, characterize gaps
- [ ] Wire into `get_temporal_briefing` tool
- [ ] Test with Claude Desktop — does the briefing actually appear? Is it useful?

**Day 5-7: Iteration on briefing quality**
- [ ] Compare output to prompt_engine.py's existing output — is the new version better?
- [ ] Tune what metadata gets included (too much = noise, too little = useless)
- [ ] Add focus_project support
- [ ] Test with real conversations — does it help? Does the LLM actually USE the briefing?

**End of Week 1 deliverable:** A working MCP server that you can connect to Claude Desktop. When you start a conversation, you call `get_temporal_briefing` and the LLM knows your world. It feels like talking to someone who's been paying attention.

### Week 2: Live Ingestion + Thread Tracking

**Day 1-3: Live ingestion**
- [ ] `live_ingestion.py` — adapt pipeline.py for single-conversation extraction
- [ ] Wire into `update_world` tool
- [ ] Test: have a conversation about a project, call update_world, verify entity updated
- [ ] Handle edge cases: new entities, state contradictions, entity resolution with existing

**Day 4-5: Thread tracking**
- [ ] `thread_tracker.py` — Thread dataclass, persistence, CRUD
- [ ] Wire into `get_approaching_deadlines` and `get_stale_threads` tools
- [ ] Auto-extract threads/deadlines from update_world calls
- [ ] Test: mention a deadline, verify it shows up in next briefing

**Day 6-7: Query tools**
- [ ] `query_entity` — entity lookup + context compilation
- [ ] `query_world` — natural language query over world model
- [ ] `get_timeline` — chronological entity history

**End of Week 2 deliverable:** Full read-write MCP server. Conversations update the world model. Deadlines get tracked. The briefing gets better over time because the world model is being actively maintained.

### Week 3: TemporalBench (The Research Contribution)

**Day 1-3: Scenario authoring**
- [ ] Write 50 scenarios (10 per category) — hand-crafted, diverse
- [ ] Categories: deadline_tracking, proactive_recall, gap_awareness, staleness_detection, rhythm_recognition
- [ ] Each scenario: 3-6 sessions with real timestamps and evaluation points

**Day 4-5: Benchmark infrastructure**
- [ ] `runner.py` — feed scenarios to any agent, capture responses
- [ ] `judge.py` — LLM-as-judge scoring with rubrics
- [ ] `scoring.py` — aggregate, visualize, generate radar charts

**Day 6-7: Baselines**
- [ ] Implement 5 baselines (naked, timestamp, full_history, memory_system, temporal_briefing)
- [ ] Run against GPT-4o, Claude 3.5 Sonnet, Gemini
- [ ] Collect results, generate comparison charts

**End of Week 3 deliverable:** TemporalBench v1 — 50 scenarios, 5 baselines, results across 3+ models. Ready for GitHub release.

### Week 4: Polish + Release

**Day 1-2: PIE world model as eval data**
- [ ] Generate anonymized scenarios from real PIE data (replace names, preserve temporal structure)
- [ ] Add as "real-world" scenario set in TemporalBench

**Day 3-4: Documentation + README**
- [ ] README with narrative (the 0.4% stat → the problem → the benchmark → the results)
- [ ] Installation guide for MCP server
- [ ] API docs for all 8 tools
- [ ] Result visualizations (radar charts, comparison tables)

**Day 5: Video assets**
- [ ] Generate explorer.html with latest world model
- [ ] Record demo of MCP server in Claude Desktop
- [ ] Capture benchmark results as visual assets

**Day 6-7: Release**
- [ ] GitHub repo: `temporalbench` (benchmark) or `pie-temporal` (combined)
- [ ] PyPI package for MCP server
- [ ] Short paper draft (4-6 pages, workshop format)

---

## File Structure (Final)

```
pie-temporal/
├── pyproject.toml
├── README.md
├── pie/
│   ├── core/
│   │   ├── models.py          ← UNCHANGED
│   │   ├── world_model.py     ← UNCHANGED
│   │   ├── dynamics.py        ← UNCHANGED
│   │   ├── llm.py             ← UNCHANGED
│   │   ├── parser.py          ← UNCHANGED
│   │   ├── prompt_engine.py   ← KEEP as reference, briefing.py replaces for MCP
│   │   └── config.py          ← UNCHANGED
│   ├── ingestion/
│   │   ├── pipeline.py        ← UNCHANGED (batch ingestion)
│   │   ├── prompts.py         ← UNCHANGED
│   │   └── live.py            ← NEW: single-conversation ingestion
│   ├── resolution/
│   │   └── resolver.py        ← UNCHANGED
│   ├── retrieval/
│   │   └── context_compiler.py ← UNCHANGED
│   ├── temporal/               ← NEW: replaces core/temporal.py
│   │   ├── briefing.py        ← NEW: temporal briefing generator
│   │   ├── threads.py         ← NEW: thread/deadline/commitment tracker
│   │   └── gaps.py            ← NEW: interaction gap analyzer
│   ├── mcp/
│   │   ├── server.py          ← NEW: MCP server entry point
│   │   └── tools.py           ← NEW: tool implementations
│   ├── browse.py              ← UNCHANGED
│   └── export_viewer.py       ← UNCHANGED
├── temporalbench/
│   ├── scenarios/
│   │   ├── core/              ← 50 hand-crafted scenarios
│   │   └── real_world/        ← PIE-derived scenarios
│   ├── runner.py
│   ├── judge.py
│   ├── scoring.py
│   ├── baselines/
│   │   ├── naked.py
│   │   ├── timestamp.py
│   │   ├── full_history.py
│   │   ├── memory_system.py
│   │   └── temporal_briefing.py
│   └── visualize.py
├── explorer.html
└── output/
    ├── world_model.json
    ├── threads.json
    └── interactions.json
```

---

## Claude Desktop Config (What the User Adds)

```json
{
  "mcpServers": {
    "pie-temporal": {
      "command": "python",
      "args": ["-m", "pie.mcp.server"],
      "cwd": "/path/to/pie-temporal",
      "env": {
        "OPENAI_API_KEY": "sk-...",
        "PIE_WORLD_MODEL": "output/world_model.json"
      }
    }
  }
}
```

Then in Claude Desktop, the user can say "give me a temporal briefing" or the system prompt can auto-call `get_temporal_briefing` on conversation start.

---

## What Makes This Useful (Not Just Research)

Here's the test: if I (Claude) had access to this MCP server right now, what would change?

1. **I'd know your projects.** Not because you told me — because the world model has 428 projects with state histories. When you say "sponsorFind," I'd know it's a Streamlit app processing 29M YouTube videos, hasn't been touched in 45 days, and has defined next_steps that aren't happening.

2. **I'd know what's stale.** Instead of treating every conversation as if it exists in a vacuum, I'd know that Lucid Academy hasn't been mentioned in a month and Lucid Labs hasn't been mentioned in two months. I might ask if those are still active.

3. **I'd track deadlines across conversations.** If you mention "I need to submit the paper by April 1" today, and you come back in 2 weeks talking about something else, I'd bring it up: "By the way, your paper deadline is in 5 days."

4. **I'd notice gaps.** If you disappear for a week and come back, I wouldn't act like nothing happened. I'd say "been a week — what's the latest?"

5. **I'd learn your rhythms.** After enough interactions, the world model's temporal metadata reveals that you work on PIE in bursts, check sponsorFind weekly, and think about Lucid Academy monthly. I'd calibrate my attention accordingly.

That's the product. Not a research prototype. A thing that makes every conversation better because the AI has context about your life that persists and ages and stays relevant.

---

## Dependencies

```toml
[project]
name = "pie-temporal"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "mcp>=1.0.0",
    "openai>=1.0.0",
    "numpy>=1.24.0",         # cosine similarity in world_model.py
]

[project.optional-dependencies]
bench = [
    "matplotlib>=3.7.0",     # radar charts
    "pandas>=2.0.0",         # result aggregation
]
```

Deliberately minimal. No vector DB. No graph DB. No embeddings service. Just JSON files and OpenAI calls. The world model is ~20MB — fits in memory trivially.
