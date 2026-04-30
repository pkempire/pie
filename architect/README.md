# architect — AI for system design

A semantic capability index of AI software components, plus an agent that
turns a user's spec into a concrete implementation plan.

## What this is

A planner agent over a curated, web-grounded knowledge graph of:

- **components**: tools, libraries, APIs, MCP servers, frameworks, model
  providers (Stagehand, Browserbase, Exa, Apify, LangGraph, Mem0,
  ElevenLabs, …)
- **relationships**: how components compose (`integrates_with`,
  `replaces`, `alternative_to`, `depends_on`)
- **architectures**: real production systems we've sampled from GitHub,
  HN, awesome lists. Each architecture references its components, giving
  us co-occurrence data and architectural patterns.

The MVP doesn't need GRPO or any RL. It's prompt-engineered planning over
a hand-curated graph that we keep fresh via daily ingestion.

## Layout

```
architect/
├── README.md                        — this file
├── db/
│   ├── schema.sql                   — SQLite schema (components, relationships, architectures)
│   └── seed_components.json         — 15 hand-curated AI components
├── db.py                            — sqlite3 wrapper + Python-side vector search
├── ingestion/
│   ├── enrich.py                    — fetch homepage + README, LLM-extract fields
│   ├── extractors.py                — extraction prompts
│   ├── github_client.py             — GitHub API wrapper (search code, repos, READMEs)
│   └── apify_client.py              — Apify-based scrapers (firecrawl-style)
├── architecture_miner.py            — find real-world systems using components
└── scripts/
    ├── seed.py                      — load seed_components.json into DB
    ├── enrich_one.py                — CLI: enrich a single component by URL
    └── mine_architectures.py        — CLI: discover architectures using a component
```

## Why local SQLite

For the MVP we don't want a hosted Postgres dependency. SQLite gives us
relational tables for free, and pgvector-equivalent vector search is
cheap to do in Python at the < 10k entity scale. When the index grows
past that we move to Postgres with pgvector.

## Setup

```bash
pip install openai sqlite-utils requests beautifulsoup4 markdown-it-py

# Initialise the DB
python -m architect.scripts.seed

# Enrich a single component
python -m architect.scripts.enrich_one "Browserbase" \
  --url https://browserbase.com

# Mine real-world architectures using a component
python -m architect.scripts.mine_architectures "Stagehand" --max_repos 30
```

## Data sources we care about

For component enrichment:
- **Official homepage** (the canonical "what does this do" source)
- **GitHub README** (the canonical "how do you use it" source)
- **Docs site** (deeper capability description)
- **Pricing page** (cost / hosting model)

For architecture mining:
- **GitHub code search** — `import { stagehand }` finds users
- **Awesome lists** — `awesome-mcp`, `awesome-langchain`, `awesome-llm-apps`
- **n8n community templates** — public workflow JSONs reference tools
- **Show HN** — real launches with stack disclosure
- **arxiv applied papers** — system papers describe the component stack
- **Vercel template gallery** — public starter projects

For freshness:
- **GitHub trending** — daily, AI-related topics
- **Hacker News new** — Show HN with AI keywords
- **x.com search** via Exa — `"just shipped"` + AI tooling keywords

## Rough flow

```
User query: "I want a long-running browser agent that scrapes
              competitor pricing and alerts me on changes"
        ↓
Planner agent decomposes:
   capability=[browser-automation, scheduled-scraping, change-detection,
                notification-channel, anti-bot-evasion]
        ↓
search_components for each capability →
   browser-automation:    Stagehand, Browser Use, Playwright
   scheduled-scraping:    n8n cron, Apify schedules
   change-detection:      diff against last run, store snapshots
   notification:          Slack webhook, Resend, Twilio
   anti-bot-evasion:      Browserbase (managed) vs raw Playwright
        ↓
agent picks: Browserbase + Stagehand + Apify cron + Postgres
              snapshot + Slack webhook
        ↓
compose_plan(format="markdown" | "n8n_json" | "cursor_spec")
        ↓
output ready to paste into Cursor or Claude Code
```
