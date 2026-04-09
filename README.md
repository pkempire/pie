# PIE — Personal Intelligence Engine

PIE ingests your entire ChatGPT conversation history and builds a structured **temporal knowledge graph** of your life: every person, project, tool, belief, and decision — tracked across time with full state history.

Once built, you can query it conversationally ("what was I working on in January?"), browse it visually, get a daily briefing, or connect it to Claude via MCP so your AI has real memory of who you are.

---

## What you get

- **A knowledge graph of your life** — entities extracted from your conversations: people, projects, tools, organizations, beliefs, decisions, concepts. Each with a full history of how it changed over time.
- **Temporal queries** — ask natural language questions grounded in your actual data. "How has project X evolved?" "What did I decide about Y last year?"
- **Visual explorer** — browse entities, timelines, and relationship graphs in the browser. No backend needed.
- **Daily briefing** — a prioritized summary of what's active, stale, overdue, or coming up based on your world model.
- **Claude MCP integration** — plug PIE into Claude Desktop so every conversation starts with real context about you.

---

## Setup

**Requirements:** Python 3.10+, an OpenAI API key.

```bash
git clone https://github.com/parthkocheta/pie
cd pie
pip install -r requirements.txt
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

**Get your ChatGPT export:**  
Go to [chatgpt.com](https://chatgpt.com) → Settings → Data controls → Export data. You'll get a `conversations.json` in your Downloads folder.

---

## Run the ingestion pipeline

This reads your conversations, extracts entities and state changes via LLM, and writes `output/world_model.json`.

```bash
# Test run first (5 daily batches, ~$0.10)
python run.py --test --no-web

# Full run (all conversations from 2024 onward)
python run.py --no-web --year 2024

# Resume if interrupted
python run.py --no-web --start-date 2024-06-01

# Check what's in your world model
python run.py --stats
```

**Cost:** roughly $1–5 to process a full year of conversations using `gpt-4o-mini`. The `--test` flag limits to 5 batches so you can verify it works before committing.

---

## Browse visually

Start a local HTTP server in the project root:

```bash
python3 -m http.server 8000
```

Then open in your browser:

| URL | What it shows |
|-----|---------------|
| `http://localhost:8000/explorer.html` | Main dashboard — entity search, timeline, relationships, insights |
| `http://localhost:8000/graph_viz.html` | Force-directed knowledge graph — people, projects, orgs as nodes |
| `http://localhost:8000/eval_viewer.html` | Entity detail view with full transition history |
| `http://localhost:8000/board.html` | Command center — goals, active projects, decisions |

---

## Query your world model

Interactive conversational interface over your world model:

```bash
python3 -m pie.eval.query_interface --world-model output/world_model.json
```

Ask things like:
- *"What projects was I working on in mid-2024?"*
- *"How has my thinking on [topic] changed?"*
- *"Who are the people I've collaborated with most?"*

It retrieves relevant entities via embedding search, compiles their full state history into context, and answers with grounded citations.

---

## Daily briefing

```bash
python briefing.py
```

Prints a prioritized executive summary: what's active and moving, what's gone stale, what deadlines are approaching, what has the highest recent activity.

---

## Connect to Claude (MCP)

Add PIE as an MCP server so Claude Desktop has live access to your world model:

```json
{
  "mcpServers": {
    "pie": {
      "command": "python3",
      "args": ["/path/to/pie/mcp_server.py"],
      "env": { "OPENAI_API_KEY": "sk-..." }
    }
  }
}
```

See `claude_desktop_config.json` for the full template. Once connected, Claude can call tools like `get_temporal_briefing`, `search_entities`, `get_timeline`, and `get_commitments` before and after each conversation.

---

## Repo structure

```
pie/                     # Core Python module
  ingestion/
    pipeline.py          # Main orchestrator — parses conversations, runs extraction, saves world model
    prompts.py           # LLM prompts for entity/relationship extraction
  core/
    world_model.py       # In-memory + JSON-persisted graph store
    models.py            # Entity, StateTransition, Relationship data models
    llm.py               # OpenAI client wrapper (chat + embeddings)
    dynamics.py          # Importance scoring, staleness, volatility
    parser.py            # conversations.json → Conversation objects
  resolution/
    resolver.py          # 3-tier entity resolution (string → embedding → LLM)
  retrieval/
    context_compiler.py  # Subgraph → LLM-ready markdown with temporal context
    temporal_retriever.py
  eval/
    query_interface.py   # Interactive query CLI
    extraction_quality.py
  config.py              # All thresholds and settings

run.py                   # CLI entry point for ingestion pipeline
briefing.py              # Daily briefing generator
mcp_server.py            # MCP server for Claude Desktop integration

explorer.html            # Main visual dashboard
graph_viz.html           # Force-directed knowledge graph
eval_viewer.html         # Entity detail + transition history
board.html               # Command center view
lib/                     # Vendored vis.js for graph rendering

output/                  # Generated by pipeline (gitignored)
  world_model.json       # Your knowledge graph (not committed — personal data)
```

---

## How it works

**Ingestion pipeline** (`run.py` → `pie/ingestion/pipeline.py`):

1. Parse `conversations.json` → group into daily batches
2. For each batch: build a context preamble from existing world model state, format conversation text, call LLM with extraction prompt
3. LLM returns JSON: entities (name, type, current state), state changes, relationships
4. Entity resolution: for each extracted entity, run 3-tier matching against existing world model — fuzzy string match → embedding cosine similarity → LLM verification for ambiguous cases
5. Write creates/updates/relationships to world model, save checkpoint every 5 batches
6. After all batches: compute importance scores (transition count × recency decay), save final `output/world_model.json`

**Entity types:** person, project, tool, organization, belief, decision, concept, period, event

**Resolution tiers:**
- Tier 1 (free): fuzzy string match ≥ 0.95 → auto-accept, 0.90 + same type → auto-accept
- Tier 2 (cheap): embedding cosine similarity — accept > 0.85, reject < 0.70, ambiguous zone → Tier 3
- Tier 3 (expensive): LLM yes/no prompt, defaults to "no" on ambiguity to avoid bad merges

---

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | Used for extraction (gpt-4o-mini) and embeddings (text-embedding-3-large) |
| `BRAVE_API_KEY` | No | Web grounding for new tool/org entities. Free tier at brave.com/search/api |

---

## Cost estimate

| Operation | Model | Approx cost |
|-----------|-------|-------------|
| Full ingestion (1 year) | gpt-4o-mini | ~$2–5 |
| Entity resolution LLM calls | gpt-4o-mini | ~$0.50 |
| Embeddings (text-embedding-3-large) | — | ~$0.50 |
| Single query | gpt-4o-mini | ~$0.01 |
