"""Streamlit dashboard for the architect KG.

Run:
    streamlit run architect/dashboard.py

Tabs:
  Components   browse, filter (kind / runtime / stack_layer / tag), search
  Graph        force-directed visualisation of relationships between components
  Architectures   real-world systems we mined; click to see their component stack
  Tags         capability tags + components attached to each
  Query        natural-language search over the component embeddings
  Stats        index health: counts, freshness, decay distribution

Designed for two audiences:
  1. Operators (us) — see what's in the index, query it, debug bad
     extractions.
  2. End users — eventually a stripped-down public version where the
     "Query" and "Architectures" tabs are surfaced as the planner UI.

For the MVP the same Streamlit app serves both; we'll fork to a Next.js
+ Vercel AI SDK frontend for the public product when we're ready to ship.
"""
from __future__ import annotations
import json
import math
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

import streamlit as st

from architect import db
from architect.agent.discovery import ToolWish, discover

# Optional deps for graph viz: use what's available, gracefully degrade.
try:
    import networkx as nx
    HAS_NX = True
except Exception:
    HAS_NX = False

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False


REFRESH_SEC = 30


# ─── Helpers ────────────────────────────────────────────────────────────────
@st.cache_data(ttl=REFRESH_SEC)
def _load_components() -> list[dict]:
    with db.connect() as conn:
        cur = conn.execute(
            "SELECT id, slug, name, kind, runtime, deployment, stack_layer, "
            "one_liner, summary, homepage_url, github_url, docs_url, mcp_url, "
            "pricing_model, license, importance, last_verified_at "
            "FROM components ORDER BY name"
        )
        return [dict(r) for r in cur]


@st.cache_data(ttl=REFRESH_SEC)
def _load_tags() -> list[dict]:
    with db.connect() as conn:
        cur = conn.execute(
            "SELECT t.id, t.slug, t.name, t.definition, "
            "       COUNT(ct.component_id) AS n_components "
            "FROM tags t LEFT JOIN component_tags ct ON ct.tag_id = t.id "
            "GROUP BY t.id ORDER BY n_components DESC, t.name"
        )
        return [dict(r) for r in cur]


@st.cache_data(ttl=REFRESH_SEC)
def _load_relationships() -> list[dict]:
    with db.connect() as conn:
        cur = conn.execute(
            "SELECT r.id, r.type, r.confidence, r.evidence_url, r.note, "
            "       sc.slug AS source_slug, sc.name AS source_name, "
            "       tc.slug AS target_slug, tc.name AS target_name "
            "FROM relationships r "
            "JOIN components sc ON sc.id = r.source_id "
            "JOIN components tc ON tc.id = r.target_id"
        )
        return [dict(r) for r in cur]


@st.cache_data(ttl=REFRESH_SEC)
def _load_architectures() -> list[dict]:
    with db.connect() as conn:
        cur = conn.execute(
            "SELECT * FROM architectures ORDER BY quality_signal DESC, discovered_at DESC"
        )
        return [dict(r) for r in cur]


@st.cache_data(ttl=REFRESH_SEC)
def _load_tags_for_component(component_id: int) -> list[str]:
    with db.connect() as conn:
        cur = conn.execute(
            "SELECT t.slug FROM tags t "
            "JOIN component_tags ct ON ct.tag_id = t.id "
            "WHERE ct.component_id=?", (component_id,))
        return [r["slug"] for r in cur]


# ─── Components tab ────────────────────────────────────────────────────────
def _tab_components():
    st.header("Components")
    components = _load_components()
    if not components:
        st.info("No components yet. Run `python -m architect.scripts.seed`.")
        return

    cols = st.columns(4)
    kinds = sorted({c["kind"] for c in components})
    runtimes = sorted({c["runtime"] for c in components})
    layers = sorted({c["stack_layer"] for c in components})

    sel_kind   = cols[0].multiselect("kind",        options=kinds)
    sel_rt     = cols[1].multiselect("runtime",     options=runtimes)
    sel_layer  = cols[2].multiselect("stack_layer", options=layers)
    text_q     = cols[3].text_input("name / one-liner contains", "")

    filtered = []
    for c in components:
        if sel_kind  and c["kind"]        not in sel_kind:  continue
        if sel_rt    and c["runtime"]     not in sel_rt:    continue
        if sel_layer and c["stack_layer"] not in sel_layer: continue
        if text_q and text_q.lower() not in (
            (c["name"] or "") + " " + (c["one_liner"] or "")
        ).lower():
            continue
        filtered.append(c)

    st.caption(f"showing {len(filtered)} of {len(components)} components")
    rows = [{
        "name":  c["name"],
        "kind":  c["kind"],
        "runtime": c["runtime"],
        "stack_layer": c["stack_layer"],
        "one_liner": c["one_liner"][:140],
        "importance": round(c["importance"] or 0, 2),
        "verified": (c["last_verified_at"] or "")[:10],
    } for c in filtered]
    st.dataframe(rows, use_container_width=True, height=380)

    # Detail view
    st.markdown("---")
    pick_options = {f"{c['name']} ({c['slug']})": c for c in filtered}
    if pick_options:
        chosen_label = st.selectbox("Open a component", list(pick_options.keys()))
        c = pick_options[chosen_label]
        _render_component_detail(c)


def _render_component_detail(c: dict):
    st.subheader(c["name"])
    cs = st.columns(4)
    cs[0].metric("kind",        c["kind"])
    cs[1].metric("runtime",     c["runtime"])
    cs[2].metric("deployment",  c["deployment"])
    cs[3].metric("stack_layer", c["stack_layer"])

    st.markdown(f"**One-liner.** {c['one_liner']}")
    if c.get("summary"):
        st.markdown(f"**Summary.** {c['summary']}")
    cols = st.columns(3)
    if c.get("homepage_url"): cols[0].markdown(f"[Homepage]({c['homepage_url']})")
    if c.get("github_url"):   cols[1].markdown(f"[GitHub]({c['github_url']})")
    if c.get("docs_url"):     cols[2].markdown(f"[Docs]({c['docs_url']})")

    tags = _load_tags_for_component(c["id"])
    if tags:
        st.caption("**Tags:** " + " · ".join(f"`{t}`" for t in tags))

    # Relationships
    with db.connect() as conn:
        rels = db.find_relationships(conn, c["id"])
    if rels:
        st.markdown("**Relationships**")
        rel_rows = [{
            "direction": "→" if r["source_slug"] == c["slug"] else "←",
            "type":      r["type"],
            "other":     r["target_name"] if r["source_slug"] == c["slug"] else r["source_name"],
            "confidence": round(r["confidence"], 2),
            "note":      (r["note"] or "")[:120],
        } for r in rels]
        st.dataframe(rel_rows, use_container_width=True, height=160)

    # Architectures using this
    with db.connect() as conn:
        cur = conn.execute(
            "SELECT a.name, a.source_url, a.summary, a.quality_signal "
            "FROM architecture_components ac "
            "JOIN architectures a ON a.id = ac.architecture_id "
            "WHERE ac.component_id = ? "
            "ORDER BY a.quality_signal DESC LIMIT 10",
            (c["id"],),
        )
        archs = [dict(r) for r in cur]
    if archs:
        st.markdown("**Used in (mined real-world systems)**")
        st.dataframe(archs, use_container_width=True, height=160)


# ─── Graph tab ─────────────────────────────────────────────────────────────
def _tab_graph():
    st.header("Component graph")
    if not (HAS_NX and HAS_MPL):
        st.warning("`networkx` and `matplotlib` required for graph viz: "
                   "`pip install networkx matplotlib`")
        return

    components = _load_components()
    rels = _load_relationships()
    if not components:
        st.info("No components yet.")
        return

    cs = st.columns(3)
    sel_layer = cs[0].multiselect("stack_layer filter", sorted({c["stack_layer"] for c in components}))
    show_archs = cs[1].checkbox("draw mined-co-occurrence edges", value=True)
    min_conf = cs[2].slider("min edge confidence", 0.0, 1.0, 0.5, 0.05)

    G = nx.Graph()
    keep = {c["slug"] for c in components if not sel_layer or c["stack_layer"] in sel_layer}
    color = []
    layer_palette = {
        "foundation_model":  "#e11d48",
        "client_library":    "#2563eb",
        "orchestration":     "#f59e0b",
        "runtime_infra":     "#16a34a",
        "data":              "#a855f7",
        "application":       "#0ea5e9",
        "inference_proxy":   "#64748b",
    }
    for c in components:
        if c["slug"] not in keep:
            continue
        G.add_node(c["slug"],
                    label=c["name"],
                    kind=c["kind"],
                    layer=c["stack_layer"])
    for r in rels:
        if r["source_slug"] not in keep or r["target_slug"] not in keep:
            continue
        if r["confidence"] < min_conf:
            continue
        if r["type"] == "integrates_with" and not show_archs:
            continue
        G.add_edge(r["source_slug"], r["target_slug"],
                    type=r["type"], conf=r["confidence"])

    if G.number_of_nodes() == 0:
        st.info("No nodes after filters.")
        return

    pos = nx.spring_layout(G, seed=42, k=1.4 / math.sqrt(max(G.number_of_nodes(), 2)))
    fig, ax = plt.subplots(figsize=(11, 7))
    node_colors = [layer_palette.get(G.nodes[n]["layer"], "#94a3b8")
                    for n in G.nodes]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=380,
                            edgecolors="#1f2937", linewidths=0.6, ax=ax)
    edge_widths = [G[u][v]["conf"] * 2.0 for u, v in G.edges]
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.6,
                            edge_color="#475569", ax=ax)
    nx.draw_networkx_labels(G, pos,
                             labels={n: G.nodes[n]["label"] for n in G.nodes},
                             font_size=8, ax=ax)
    ax.set_axis_off()
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    st.caption(f"{G.number_of_nodes()} nodes, {G.number_of_edges()} edges. "
                "Node colour = stack_layer.")


# ─── Architectures tab ─────────────────────────────────────────────────────
def _tab_architectures():
    st.header("Mined architectures")
    archs = _load_architectures()
    if not archs:
        st.info("No architectures mined yet. "
                "Run `python -m architect.scripts.mine_architectures Stagehand`.")
        return
    rows = [{
        "name":     a["name"],
        "source":   a["source"],
        "pattern":  a["pattern"][:60] if a["pattern"] else "",
        "quality":  round(a["quality_signal"], 2),
        "summary":  (a["summary"] or "")[:120],
        "url":      a["source_url"],
    } for a in archs]
    st.dataframe(rows, use_container_width=True, height=400)

    pick = {f"{a['name']} ({a['source_url'][-50:]})": a for a in archs}
    if pick:
        chosen = st.selectbox("Open an architecture", list(pick.keys()))
        a = pick[chosen]
        st.subheader(a["name"])
        st.markdown(f"**Source:** [{a['source_url']}]({a['source_url']})")
        st.markdown(f"**Pattern:** `{a['pattern'] or 'unlabeled'}`")
        st.markdown(f"**Summary:** {a['summary']}")

        with db.connect() as conn:
            cur = conn.execute(
                "SELECT c.name, c.kind, ac.role, ac.evidence "
                "FROM architecture_components ac "
                "JOIN components c ON c.id = ac.component_id "
                "WHERE ac.architecture_id = ? ORDER BY ac.role",
                (a["id"],),
            )
            comp_rows = [dict(r) for r in cur]
        if comp_rows:
            st.markdown("**Components used**")
            st.dataframe(comp_rows, use_container_width=True, height=200)


# ─── Tags tab ──────────────────────────────────────────────────────────────
def _tab_tags():
    st.header("Capability tags")
    tags = _load_tags()
    if not tags:
        st.info("No tags yet.")
        return
    st.dataframe([
        {"slug": t["slug"], "name": t["name"], "n_components": t["n_components"]}
        for t in tags
    ], use_container_width=True, height=300)

    pick = {t["name"]: t for t in tags}
    chosen = st.selectbox("Open a tag", list(pick.keys()))
    if chosen:
        t = pick[chosen]
        with db.connect() as conn:
            cur = conn.execute(
                "SELECT c.name, c.kind, c.one_liner FROM components c "
                "JOIN component_tags ct ON ct.component_id = c.id "
                "WHERE ct.tag_id=? ORDER BY c.importance DESC, c.name",
                (t["id"],),
            )
            rows = [dict(r) for r in cur]
        st.dataframe(rows, use_container_width=True, height=300)


# ─── Query playground ──────────────────────────────────────────────────────
def _tab_query():
    st.header("Capability search")
    st.caption("Embedding search over component summaries. Type a "
               "capability you want — e.g. 'detect bot traffic on a webhook'.")
    q = st.text_input("describe what you want", "")
    use_discovery = st.checkbox("Allow live discovery if KG match is weak (slower, costs API calls)",
                                  value=False)
    if not q:
        return
    if st.button("search"):
        wish = ToolWish(capability=q, context="", nice_to_have=[], must_avoid=[])
        with st.spinner("running discovery..."):
            try:
                matches = discover(wish, top_k=5,
                                    allow_live_search=use_discovery)
            except Exception as e:
                st.error(f"discovery failed: {e}")
                return
        if not matches:
            st.warning("No matches found.")
            return
        for m in matches:
            badge = "🆕 fresh" if m.fresh else m.confidence
            st.markdown(f"**{m.name}** · `{m.slug}` · {badge} · score {m.score:.2f}")
            st.caption(m.rationale)


# ─── Stats tab ─────────────────────────────────────────────────────────────
def _tab_stats():
    st.header("Index health")
    components = _load_components()
    rels = _load_relationships()
    archs = _load_architectures()
    tags = _load_tags()

    cs = st.columns(4)
    cs[0].metric("components", len(components))
    cs[1].metric("relationships", len(rels))
    cs[2].metric("architectures", len(archs))
    cs[3].metric("tags", len(tags))

    st.markdown("**By kind**")
    kind_counts = Counter(c["kind"] for c in components)
    st.bar_chart(dict(kind_counts.most_common()))

    st.markdown("**By stack_layer**")
    layer_counts = Counter(c["stack_layer"] for c in components)
    st.bar_chart(dict(layer_counts.most_common()))

    st.markdown("**By runtime**")
    rt_counts = Counter(c["runtime"] for c in components)
    st.bar_chart(dict(rt_counts.most_common()))

    st.markdown("**Importance distribution**")
    imp_buckets: dict[str, int] = defaultdict(int)
    for c in components:
        i = c["importance"] or 0.0
        if i < 0.05:    imp_buckets["<0.05 (archived)"] += 1
        elif i < 0.3:   imp_buckets["0.05-0.3 (cold)"]  += 1
        elif i < 0.7:   imp_buckets["0.3-0.7 (warm)"]   += 1
        else:           imp_buckets["≥0.7 (hot)"]       += 1
    st.bar_chart(dict(imp_buckets))


# ─── Main ──────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="architect", layout="wide")
    st.title("architect — AI for system design")
    st.caption(f"DB: `{db.DB_PATH}` · cache TTL {REFRESH_SEC}s "
                "· `streamlit run architect/dashboard.py`")
    tabs = st.tabs(["Components", "Graph", "Architectures", "Tags",
                     "Query", "Stats"])
    with tabs[0]: _tab_components()
    with tabs[1]: _tab_graph()
    with tabs[2]: _tab_architectures()
    with tabs[3]: _tab_tags()
    with tabs[4]: _tab_query()
    with tabs[5]: _tab_stats()


if __name__ == "__main__":
    main()
