"""
PIE Personal Wiki — Streamlit app.

Run:
    streamlit run pie/ui/app.py -- --world-model output/world_model.json

Or the shortcut defined in run.py:
    python3 run.py wiki
"""
from __future__ import annotations

import json
import sys
import argparse
from datetime import datetime
from pathlib import Path

import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="PIE — Personal Wiki",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  /* Shrink default padding */
  .block-container { padding: 1.5rem 2rem 2rem !important; max-width: 1100px; }
  /* Sidebar entity buttons */
  div[data-testid="stButton"] > button {
    text-align: left !important;
    font-size: 13px !important;
    padding: 4px 10px !important;
    height: auto !important;
    border: none !important;
    background: transparent !important;
    color: #bbb !important;
    white-space: normal !important;
    word-wrap: break-word !important;
  }
  div[data-testid="stButton"] > button:hover { color: #e8e8e8 !important; background: rgba(124,106,247,0.12) !important; }
  /* Active entity */
  div[data-testid="stButton"] > button.active { color: #a29af8 !important; border-left: 2px solid #7c6af7 !important; }
  /* Timeline dots */
  .tl-dot { display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:8px; vertical-align:middle; }
  .dot-creation  { background:#4caf72; }
  .dot-update    { background:#5b9cf6; }
  .dot-contradiction { background:#e05555; }
  .dot-resolution{ background:#f0c040; }
  .dot-archival  { background:#666; }
  /* Pill badges */
  .pill { display:inline-block; padding:2px 10px; border-radius:12px; font-size:11px; font-weight:600; letter-spacing:.4px; margin:2px; }
  .pill-project { background:#1e1e35; color:#a29af8; }
  .pill-person  { background:#1a2030; color:#5b9cf6; }
  .pill-org     { background:#2a1e10; color:#f0a060; }
  .pill-tool    { background:#0e2210; color:#4caf72; }
  .pill-belief  { background:#2a2408; color:#f0c040; }
  .pill-goal    { background:#2a1018; color:#e06080; }
  .pill-event   { background:#0e2428; color:#90d0c0; }
  .pill-decision{ background:#2a2010; color:#d0a020; }
  .pill-concept { background:#1a1a1a; color:#a0a0a0; }
  /* Status colors */
  .status-active    { color:#4caf72; font-weight:600; }
  .status-paused    { color:#f0c040; font-weight:600; }
  .status-completed { color:#888; font-weight:600; }
  .status-abandoned { color:#e05555; font-weight:600; }
  /* Intro box */
  .intro-card {
    background: linear-gradient(135deg, #1c1a30 0%, #1a1c28 100%);
    border: 1px solid #3a3560;
    border-radius: 8px;
    padding: 16px 20px;
    font-size: 15px;
    line-height: 1.7;
    margin-bottom: 1rem;
  }
  /* Search result card */
  .result-card {
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 6px;
    padding: 10px 14px;
    margin-bottom: 8px;
    cursor: pointer;
  }
  h1 { letter-spacing: -0.5px; }
  /* Hide streamlit branding */
  #MainMenu { visibility: hidden; }
  footer { visibility: hidden; }
  header[data-testid="stHeader"] { background: transparent; }
</style>
""", unsafe_allow_html=True)

# ── CLI args (passed after `--` in streamlit run) ──────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--world-model", type=Path, default=Path("output/world_model.json"))
    args, _ = parser.parse_known_args(sys.argv[1:])
    return args

_args = _parse_args()

# ── Load world model + retriever (cached across all rerenders) ─────────────

@st.cache_resource(show_spinner="Loading world model…")
def _load(world_model_path: str):
    from pie.core.world_model import WorldModel
    from pie.core.llm import LLMClient
    from pie.retrieval.hybrid_retriever import HybridRetriever
    from pie.config import PIEConfig

    wm = WorldModel(persist_path=Path(world_model_path))
    llm = LLMClient()
    retriever = HybridRetriever(wm, llm, PIEConfig())
    return wm, llm, retriever

try:
    wm, llm, retriever = _load(str(_args.world_model))
except Exception as e:
    st.error(f"**Failed to load world model:** {e}\n\nMake sure `OPENAI_API_KEY` is set and run ingestion first.")
    st.code("python3 run.py ingest")
    st.stop()

# ── Session state defaults ─────────────────────────────────────────────────

_SESSIONS_DIR = Path("output/sessions")
_WIKI_SESSION = _SESSIONS_DIR / "wiki.json"

def _load_session() -> list:
    """Load persisted chat history from disk."""
    try:
        _SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        if _WIKI_SESSION.exists():
            return json.loads(_WIKI_SESSION.read_text())
    except Exception:
        pass
    return []

def _save_session(history: list):
    """Persist chat history to disk."""
    try:
        _SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        _WIKI_SESSION.write_text(json.dumps(history, ensure_ascii=False, indent=2))
    except Exception:
        pass

_defaults = {
    "view": "home",             # home | daily | entity | ask | search
    "entity_id": None,
    "chat_history": None,       # None = not yet loaded
    "generated_pages": {},      # entity_id → dict from LLM
    "search_results": [],
    "last_search_query": "",
    "daily_agents": {},         # agent_id → {"content": str, "ts": float}
    "edit_entity_id": None,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Restore persisted chat history on cold start
if st.session_state.chat_history is None:
    st.session_state.chat_history = _load_session()

# ── Type metadata ──────────────────────────────────────────────────────────

TYPE_ORDER = ["project","person","organization","tool","belief","goal","event","decision","concept","period"]
TYPE_META = {
    "project":      ("⬡", "Projects",       "pill-project"),
    "person":       ("◎", "People",         "pill-person"),
    "organization": ("⬢", "Organizations",  "pill-org"),
    "tool":         ("⊛", "Tools",          "pill-tool"),
    "belief":       ("◈", "Beliefs",        "pill-belief"),
    "goal":         ("◆", "Goals",          "pill-goal"),
    "event":        ("◇", "Events",         "pill-event"),
    "decision":     ("◑", "Decisions",      "pill-decision"),
    "concept":      ("○", "Concepts",       "pill-concept"),
    "period":       ("▦", "Periods",        "pill-concept"),
}
def tmeta(t): return TYPE_META.get(t, ("●", t.title(), "pill-concept"))

def recency_dot(days: float) -> str:
    if days < 7:   return "🟢"
    if days < 30:  return "🟣"
    if days < 90:  return "🟡"
    if days < 365: return "⚪"
    return "·"

# ── Sidebar ────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🧠 PIE / personal wiki")

    # Stats line
    stats = wm.stats
    st.caption(
        f"{stats.get('entities',0):,} entities · "
        f"{stats.get('transitions',0):,} transitions · "
        f"{stats.get('relationships',0):,} relationships"
    )
    st.divider()

    # Top nav
    nav_col1, nav_col2, nav_col3, nav_col4 = st.columns(4)
    with nav_col1:
        if st.button("📅 Daily", use_container_width=True):
            st.session_state.view = "daily"
    with nav_col2:
        if st.button("🏠 Home", use_container_width=True):
            st.session_state.view = "home"
    with nav_col3:
        if st.button("✦ Ask", use_container_width=True):
            st.session_state.view = "ask"
    with nav_col4:
        if st.button("🔍 Search", use_container_width=True):
            st.session_state.view = "search"

    st.divider()

    # Quick search to filter sidebar
    sidebar_filter = st.text_input("Filter entities", placeholder="type to filter…", label_visibility="collapsed")

    # Group entities by type
    grouped: dict[str, list] = {}
    now_ts = datetime.now().timestamp()
    for eid, entity in wm.entities.items():
        etype = entity.type.value
        grouped.setdefault(etype, [])
        grouped[etype].append(entity)

    for etype in TYPE_ORDER:
        entities = grouped.get(etype, [])
        if not entities:
            continue

        # Apply filter
        if sidebar_filter:
            entities = [e for e in entities if sidebar_filter.lower() in e.name.lower()]
            if not entities:
                continue

        icon, label, _ = tmeta(etype)
        count = len(entities)
        # Sort by recency
        entities_sorted = sorted(entities, key=lambda e: -(e.last_seen or 0))[:60]

        with st.expander(f"{icon} {label}  `{count}`"):
            for entity in entities_sorted:
                days = (now_ts - (entity.last_seen or 0)) / 86400
                dot = recency_dot(days)
                label_text = f"{dot} {entity.name}"
                if st.button(label_text, key=f"e_{entity.id}", use_container_width=True):
                    st.session_state.entity_id = entity.id
                    st.session_state.view = "entity"

# ── Helpers ────────────────────────────────────────────────────────────────

def _fmt_ts(ts: float) -> str:
    if not ts:
        return "unknown"
    return datetime.fromtimestamp(ts).strftime("%b %d, %Y")

def _humanize(ts: float) -> str:
    if not ts:
        return ""
    days = (datetime.now().timestamp() - ts) / 86400
    if days < 1:   return "today"
    if days < 2:   return "yesterday"
    if days < 7:   return f"{int(days)}d ago"
    if days < 30:  return f"{int(days/7)}w ago"
    if days < 365: return f"{int(days/30)}mo ago"
    return f"{days/365:.1f}y ago"

def _dot(ttype: str) -> str:
    dots = {"creation":"🟢","update":"🔵","contradiction":"🔴","resolution":"🟡","archival":"⚫"}
    return dots.get(ttype, "⚪")

# ── Daily agents definition ────────────────────────────────────────────────

DAILY_AGENTS = [
    {
        "id": "priorities",
        "title": "🎯 Top Priorities",
        "query": "active goals urgent priorities next actions blocked waiting on high priority",
        "system": (
            "You are a ruthless executive assistant. From the context below, identify the "
            "TOP 5 most important things the user should work on RIGHT NOW. "
            "For each item: (1) entity name, (2) why it's highest priority, (3) single next concrete action. "
            "Be specific, direct, no filler. Numbered list. If something is blocked say what's blocking it."
        ),
        "top_k": 25,
        "n_subqueries": 8,
    },
    {
        "id": "projects",
        "title": "📋 Active Projects",
        "query": "active projects in progress status next step planning implementation",
        "system": (
            "List all ACTIVE projects from the context. For each use this format:\n"
            "**[Project Name]** — [status]\n"
            "_Last activity: [date]_ · _Next step: [specific action]_\n"
            "> [1 sentence of key context or blocker]\n\n"
            "Skip completed, abandoned, or vague concept-only entries. Group by domain if natural."
        ),
        "top_k": 30,
        "n_subqueries": 6,
    },
    {
        "id": "decisions",
        "title": "⏳ Open Decisions",
        "query": "pending decisions unresolved choices evaluating considering options planning",
        "system": (
            "List decisions that are still OPEN or PENDING — not yet made or not confirmed resolved. "
            "For each: **Decision**, context (1 sentence), what's needed to close it. "
            "Skip decisions that are clearly already made and executed. "
            "Flag any that have been open a long time without movement."
        ),
        "top_k": 20,
        "n_subqueries": 5,
    },
    {
        "id": "writing",
        "title": "✍️ Writing & Content",
        "query": "writing content documentation copy blog posts drafts pending applications essays",
        "system": (
            "Identify any WRITING, CONTENT, or DOCUMENTATION tasks that are in progress or pending. "
            "Include: blog posts, applications, essays, product copy, docs, emails, scripts, READMEs. "
            "For each: **[Task]** — status — [next action]. Be specific about what's still needed."
        ),
        "top_k": 15,
        "n_subqueries": 5,
    },
    {
        "id": "pulse",
        "title": "📡 Recent Pulse",
        "query": "recent updates progress changes events last week completed started new",
        "system": (
            "Summarize what has happened RECENTLY (focus on last 1-8 weeks). "
            "Key developments, progress made, things that started or completed, any shifts. "
            "Format as a punchy bulleted list grouped by rough theme. "
            "Lead each bullet with the date range if available."
        ),
        "top_k": 20,
        "n_subqueries": 7,
    },
]


def _run_agent(agent: dict) -> str:
    """Run a single daily agent: retrieve context + LLM synthesis. Returns markdown string."""
    entity_ids = retriever.broad_retrieve(
        agent["query"],
        top_k=agent["top_k"],
        n_subqueries=agent.get("n_subqueries", 8),
    )
    if not entity_ids:
        return "_No relevant context found._"
    context = retriever.compile_context(entity_ids, query=agent["query"], max_transitions=10)
    if len(context) > 50_000:
        context = context[:50_000] + "\n\n[...truncated...]"
    result = llm.chat(
        messages=[
            {"role": "system", "content": agent["system"]},
            {"role": "user", "content": f"Context from my knowledge base:\n\n{context}"},
        ],
        model="gpt-5.4",
        max_tokens=1200,
    )
    return (result.get("content") or "").strip()


@st.cache_data(show_spinner=False, ttl=3600)
def _generate_entity_page(entity_id: str, _model: str = "gpt-5.4") -> dict:
    """Generate LLM wiki page for an entity. Cached per entity_id for the session."""
    entity = wm.entities.get(entity_id)
    if not entity:
        return {}

    now_ts = datetime.now().timestamp()
    transitions = wm.get_transitions(entity_id)
    relationships = wm.get_relationships(entity_id)

    context = [
        f"Entity: {entity.name}",
        f"Type: {entity.type.value}",
        f"Aliases: {', '.join(entity.aliases) or 'none'}",
        f"First seen: {_fmt_ts(entity.first_seen)}",
        f"Last seen: {_fmt_ts(entity.last_seen)}",
        f"\nCurrent state:\n{json.dumps(entity.current_state, indent=2, default=str)}",
    ]
    if transitions:
        context.append(f"\nState history ({len(transitions)} transitions):")
        for t in transitions[-25:]:
            dt = _fmt_ts(t.timestamp)
            flag = " ⚠ CONTRADICTION" if t.transition_type.value == "contradiction" else ""
            context.append(f"  [{dt}] {t.transition_type.value}{flag}: {t.trigger_summary}")
    if relationships:
        context.append(f"\nRelationships:")
        for r in relationships[:15]:
            other_id = r.target_id if r.source_id == entity_id else r.source_id
            other = wm.entities.get(other_id)
            other_name = other.name if other else "unknown"
            arrow = "→" if r.source_id == entity_id else "←"
            context.append(f"  {arrow} {r.type.value}: {other_name} — {r.description}")

    prompt = "\n".join(context)
    system = (
        'You are writing a personal Wikipedia page entry for someone\'s life knowledge graph. '
        'Return a JSON object with these exact keys:\n'
        '{\n'
        '  "intro": "2-3 sentence paragraph — current situation and most important thing to know",\n'
        '  "summary": "markdown text — full history, key milestones, important context (use ## subheadings)",\n'
        '  "recent": "markdown bullet list — what changed in the last 90 days, with dates",\n'
        '  "todos": ["list of specific action items or open questions from the history"],\n'
        '  "status": "one of: active | paused | completed | abandoned | unclear",\n'
        '  "tags": ["3-5 thematic tags"]\n'
        '}\n'
        'Write from the owner\'s perspective. Be concrete. Highlight contradictions in history if any.'
    )
    result = llm.chat(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"Write a wiki page for:\n\n{prompt}"},
        ],
        model=_model,
        json_mode=True,
    )
    content = result["content"]
    # llm.chat(json_mode=True) already returns a parsed dict — don't double-parse
    return content if isinstance(content, dict) else json.loads(content)

# ── Views ──────────────────────────────────────────────────────────────────

view = st.session_state.view

# ────────────────────────────────────────────────────────────────────────────
# DAILY BRIEFING
# ────────────────────────────────────────────────────────────────────────────
if view == "daily":
    import time as _time

    st.title("📅 Daily Briefing")
    st.caption("Executive assistant view — hardwired agents that pull priorities, projects, decisions, and pulse from your knowledge base.")

    now_ts = _time.time()
    agents_state = st.session_state.daily_agents

    # Refresh controls
    hdr_col, refresh_col = st.columns([5, 1])
    with refresh_col:
        if st.button("🔄 Refresh all", use_container_width=True):
            st.session_state.daily_agents = {}
            st.rerun()

    oldest_ts = min((v.get("ts", 0) for v in agents_state.values()), default=0)
    if oldest_ts:
        age_min = int((now_ts - oldest_ts) / 60)
        hdr_col.caption(f"Last generated {age_min}m ago · click 🔄 to regenerate")

    st.divider()

    # Run agents in two columns
    col_left, col_right = st.columns(2)
    col_map = {
        "priorities": col_left,
        "projects": col_right,
        "decisions": col_left,
        "writing": col_right,
        "pulse": col_left,
    }

    for agent in DAILY_AGENTS:
        aid = agent["id"]
        target_col = col_map.get(aid, col_left)

        with target_col:
            with st.container(border=True):
                agent_hdr, agent_btn = st.columns([4, 1])
                agent_hdr.markdown(f"#### {agent['title']}")

                if aid in agents_state:
                    ts_str = _humanize(agents_state[aid]["ts"])
                    if agent_btn.button("↻", key=f"refresh_{aid}", help=f"Re-run (generated {ts_str})"):
                        del st.session_state.daily_agents[aid]
                        st.rerun()

                if aid not in agents_state:
                    with st.spinner(f"Running {agent['title']}…"):
                        try:
                            content = _run_agent(agent)
                        except Exception as e:
                            content = f"_Error: {e}_"
                        st.session_state.daily_agents[aid] = {"content": content, "ts": _time.time()}
                        st.rerun()

                st.markdown(agents_state.get(aid, {}).get("content", "_Pending…_"))

# ────────────────────────────────────────────────────────────────────────────
# HOME
# ────────────────────────────────────────────────────────────────────────────
elif view == "home":
    st.title("🧠 Personal Wikipedia")
    st.caption("Your world model, organized as a knowledge base. Click any entity in the sidebar or use Search / Ask.")

    col1, col2, col3, col4 = st.columns(4)
    type_counts = stats.get("entity_types", {})
    col1.metric("Total Entities", f"{stats.get('entities',0):,}")
    col2.metric("State Changes", f"{stats.get('transitions',0):,}")
    col3.metric("Relationships", f"{stats.get('relationships',0):,}")
    col4.metric("Entity Types", len(type_counts))

    st.divider()

    # Type breakdown
    st.subheader("Entity breakdown")
    cols = st.columns(5)
    for i, etype in enumerate([t for t in TYPE_ORDER if t in type_counts]):
        icon, label, _ = tmeta(etype)
        cols[i % 5].metric(f"{icon} {label}", type_counts[etype])

    st.divider()

    # Recently updated entities
    st.subheader("Recently updated")
    recent = sorted(wm.entities.values(), key=lambda e: -(e.last_seen or 0))[:12]
    cols = st.columns(3)
    for i, entity in enumerate(recent):
        with cols[i % 3]:
            icon, _, pill = tmeta(entity.type.value)
            with st.container(border=True):
                st.markdown(
                    f'<span class="pill {pill}">{icon} {entity.type.value}</span>',
                    unsafe_allow_html=True,
                )
                if st.button(entity.name, key=f"home_{entity.id}", use_container_width=True):
                    st.session_state.entity_id = entity.id
                    st.session_state.view = "entity"
                    st.rerun()
                st.caption(_humanize(entity.last_seen))

# ────────────────────────────────────────────────────────────────────────────
# ENTITY PAGE
# ────────────────────────────────────────────────────────────────────────────
elif view == "entity":
    entity_id = st.session_state.entity_id
    entity = wm.entities.get(entity_id) if entity_id else None

    if not entity:
        st.warning("Select an entity from the sidebar.")
        st.stop()

    now_ts = datetime.now().timestamp()
    transitions = wm.get_transitions(entity_id)
    relationships = wm.get_relationships(entity_id)
    icon, _, pill = tmeta(entity.type.value)

    # Header
    st.markdown(
        f'<span class="pill {pill}">{icon} {entity.type.value}</span>',
        unsafe_allow_html=True,
    )
    st.title(entity.name)

    meta_parts = [f"First seen {_fmt_ts(entity.first_seen)}", f"Last seen {_fmt_ts(entity.last_seen)}"]
    if entity.aliases:
        meta_parts.append(f"Also known as: {', '.join(entity.aliases)}")
    meta_parts.append(f"{len(transitions)} state changes · {len(relationships)} relationships")
    st.caption(" · ".join(meta_parts))

    # Regenerate button
    regen_col, _ = st.columns([1, 5])
    with regen_col:
        if st.button("↻ Regenerate AI summary"):
            _generate_entity_page.clear()

    # ── Generate LLM page (cached) ──
    with st.spinner("Generating AI summary…"):
        try:
            page = _generate_entity_page(entity_id)
        except Exception as e:
            page = {}
            st.warning(f"AI summary failed: {e}")

    # Intro card
    if page.get("intro"):
        status = page.get("status", "unclear")
        status_color = {"active":"🟢","paused":"🟡","completed":"🔵","abandoned":"🔴"}.get(status, "⚪")
        st.markdown(
            f'<div class="intro-card">'
            f'<strong>{status_color} {status.upper()}</strong> — {page["intro"]}'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Tags
    if page.get("tags"):
        tag_html = " ".join(f'<span class="pill pill-concept">{t}</span>' for t in page["tags"])
        st.markdown(tag_html, unsafe_allow_html=True)

    st.divider()

    # Tabs
    tab_overview, tab_history, tab_rels, tab_ai, tab_edit = st.tabs(
        ["📋 Current State", "🕐 History", "🔗 Relationships", "✦ AI Analysis", "✏️ Edit"]
    )

    with tab_overview:
        state = entity.current_state or {}
        if isinstance(state, dict) and state:
            for k, v in state.items():
                if k not in ("embedding", "id"):
                    st.markdown(f"**{k}:** {str(v)[:500]}")
        else:
            st.caption("No structured state recorded.")

    with tab_history:
        if transitions:
            for t in reversed(transitions):
                dot = _dot(t.transition_type.value)
                date_str = _fmt_ts(t.timestamp)
                relative = _humanize(t.timestamp)
                contradiction = " — ⚠️ **CONTRADICTION**" if t.transition_type.value == "contradiction" else ""
                st.markdown(
                    f"{dot} **{date_str}** `{relative}` — {t.trigger_summary}{contradiction}"
                )
        else:
            st.caption("No state transitions recorded.")

    with tab_rels:
        if relationships:
            for r in relationships:
                other_id = r.target_id if r.source_id == entity_id else r.source_id
                other = wm.entities.get(other_id)
                other_name = other.name if other else other_id
                arrow = "→" if r.source_id == entity_id else "←"
                other_icon, _, _ = tmeta(other.type.value if other else "concept")
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.markdown(f"{arrow} **`{r.type.value}`** {other_icon} {other_name}")
                    if r.description:
                        st.caption(r.description[:120])
                with col_b:
                    if other and st.button("Open", key=f"rel_{r.id}"):
                        st.session_state.entity_id = other_id
                        st.rerun()
        else:
            st.caption("No relationships recorded.")

    with tab_ai:
        if page.get("todos"):
            st.subheader("✅ Action Items")
            for item in page["todos"]:
                st.checkbox(item, value=False, key=f"todo_{hash(item)}")
            st.divider()

        if page.get("recent"):
            st.subheader("⚡ Recent Changes")
            st.markdown(page["recent"])
            st.divider()

        if page.get("summary"):
            st.subheader("📖 Full Summary")
            st.markdown(page["summary"])

    with tab_edit:
        st.caption("Edit the entity's description directly. Changes are saved to the world model as a new state transition.")

        state = entity.current_state or {}
        current_desc = state.get("description", "") if isinstance(state, dict) else str(state)
        current_notes = state.get("notes", "") if isinstance(state, dict) else ""

        new_desc = st.text_area(
            "Description",
            value=current_desc,
            height=200,
            key=f"edit_desc_{entity_id}",
        )
        new_notes = st.text_area(
            "Notes (markdown, freeform)",
            value=current_notes,
            height=150,
            placeholder="Add your own notes, context, next steps…",
            key=f"edit_notes_{entity_id}",
        )

        save_col, _ = st.columns([1, 5])
        with save_col:
            if st.button("💾 Save", key=f"save_{entity_id}", type="primary"):
                import time as _time2
                updates: dict = {}
                if new_desc != current_desc:
                    updates["description"] = new_desc
                if new_notes != current_notes:
                    updates["notes"] = new_notes
                if updates:
                    wm.update_entity_state(
                        entity_id=entity_id,
                        new_state=updates,
                        source_conversation_id="manual_edit",
                        timestamp=_time2.time(),
                        trigger_summary="Manual edit via wiki UI",
                    )
                    wm.save()
                    _generate_entity_page.clear()
                    st.success("Saved.")
                    st.rerun()
                else:
                    st.info("No changes detected.")

# ────────────────────────────────────────────────────────────────────────────
# SEARCH
# ────────────────────────────────────────────────────────────────────────────
elif view == "search":
    st.title("🔍 Search")

    col_q, col_btn = st.columns([5, 1])
    with col_q:
        query = st.text_input("Search query", placeholder="projects, people, tools…", label_visibility="collapsed")
    with col_btn:
        do_search = st.button("Search", use_container_width=True)

    if query and (do_search or query != st.session_state.last_search_query):
        st.session_state.last_search_query = query
        with st.spinner("Searching…"):
            entity_ids = retriever.retrieve(query, top_k=20)
            st.session_state.search_results = entity_ids

    results = st.session_state.search_results
    if results:
        st.caption(f"{len(results)} results")
        for eid in results:
            entity = wm.entities.get(eid)
            if not entity:
                continue
            icon, _, pill = tmeta(entity.type.value)
            desc = ""
            if isinstance(entity.current_state, dict):
                desc = entity.current_state.get("description", "")
            with st.container(border=True):
                c1, c2 = st.columns([6, 1])
                with c1:
                    st.markdown(
                        f'<span class="pill {pill}">{icon} {entity.type.value}</span> **{entity.name}**',
                        unsafe_allow_html=True,
                    )
                    if desc:
                        st.caption(desc[:200])
                with c2:
                    if st.button("Open →", key=f"sr_{eid}"):
                        st.session_state.entity_id = eid
                        st.session_state.view = "entity"
                        st.rerun()

# ────────────────────────────────────────────────────────────────────────────
# ASK
# ────────────────────────────────────────────────────────────────────────────
elif view == "ask":
    ask_title_col, ask_clear_col = st.columns([6, 1])
    ask_title_col.title("✦ Ask your knowledge base")
    if ask_clear_col.button("🗑 Clear", help="Clear conversation history"):
        st.session_state.chat_history = []
        _save_session([])
        st.rerun()

    if st.session_state.chat_history:
        ask_title_col.caption(f"{len(st.session_state.chat_history)} messages · session persisted to disk")

    # Chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("entities"):
                st.caption("Sources: " + ", ".join(msg["entities"]))

    # Model selector
    answer_model = st.selectbox(
        "Model",
        ["gpt-5.4", "gpt-5.4-mini", "gpt-4o", "gpt-4o-mini"],
        label_visibility="collapsed",
    )

    # Chat input — always uses smart LLM sub-query decomposition
    if prompt := st.chat_input("Ask anything about your world…"):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching your knowledge base…"):
                entity_ids = retriever.broad_retrieve(prompt, top_k=30)
                context = retriever.compile_context(
                    entity_ids, query=prompt, max_transitions=25
                )
                if len(context) > 80_000:
                    context = context[:80_000] + "\n\n[...truncated...]"

            entity_names = [
                wm.entities[eid].name for eid in entity_ids if eid in wm.entities
            ]

            SYSTEM = (
                "You are a personal knowledge assistant with access to the user's structured "
                "world model — their projects, people, decisions, beliefs, and how things "
                "have changed over time.\n\n"
                "Answer using ONLY the provided context. Make full use of temporal "
                "information (dates, change history, contradictions). "
                "Be exhaustive and structured — use the category headings from the question if provided."
            )

            def _stream():
                token_kwarg = (
                    {"max_completion_tokens": 4000}
                    if llm._uses_completion_tokens(answer_model)
                    else {"max_tokens": 4000}
                )
                stream = llm.client.chat.completions.create(
                    model=answer_model,
                    messages=[
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": f"Context:\n\n{context}\n\n---\n\nQuestion: {prompt}"},
                    ],
                    stream=True,
                    **token_kwarg,
                )
                for chunk in stream:
                    yield chunk.choices[0].delta.content or ""

            full_answer = st.write_stream(_stream())
            st.caption(f"{len(entity_ids)} entities · Sources: {', '.join(entity_names[:10])}")

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": full_answer,
            "entities": entity_names,
        })
        _save_session(st.session_state.chat_history)

        try:
            fu = llm.chat(
                messages=[
                    {"role": "system", "content": "Suggest 3 short follow-up questions. JSON: {\"questions\":[...]}"},
                    {"role": "user", "content": f"Q: {prompt}\nA: {full_answer[:300]}"},
                ],
                model="gpt-5.4",
                json_mode=True,
            )
            _fu_content = fu["content"]
            followups = (_fu_content if isinstance(_fu_content, dict) else json.loads(_fu_content)).get("questions", [])
            if followups:
                st.markdown("**Follow-up suggestions:**")
                for fq in followups:
                    if st.button(fq, key=f"fu_{hash(fq)}"):
                        st.session_state.chat_history.append({"role": "user", "content": fq})
                        st.rerun()
        except Exception:
            pass
