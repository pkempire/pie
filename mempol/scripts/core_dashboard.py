"""Streamlit dashboard for the universal memory core.

Run:
  streamlit run mempol/scripts/core_dashboard.py -- --watch mempol/results/universal_smoke
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mempol.core.retrieval import retrieve_budgeted
from mempol.core.store import SQLiteMemoryStore


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", type=Path, default=Path("mempol/results/universal_smoke"))
    return ap.parse_args()


def _load_latest_query(path: Path) -> dict:
    p = path / "latest_core_query.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def main() -> None:
    args = _parse_args()
    db_path = args.watch / "core_memory.sqlite"
    st.set_page_config(page_title="Universal Memory Core", layout="wide")
    st.title("Universal Memory Core")
    st.caption(f"watching: {args.watch}")

    if not db_path.exists():
        st.error(f"No core memory DB found at {db_path}")
        st.stop()

    store = SQLiteMemoryStore(db_path)
    stats = store.stats()
    c = st.columns(4)
    c[0].metric("artifacts", stats["artifacts"])
    c[1].metric("spans", stats["spans"])
    c[2].metric("memory states", stats["memory_states"])
    c[3].metric("trace events", stats["trace_events"])

    st.subheader("Sources")
    st.dataframe(stats["artifacts_by_source"], use_container_width=True)

    st.subheader("Search")
    query = st.text_input("Query", value="What should I build next for the memory system?")
    k = st.slider("k", min_value=3, max_value=20, value=8)
    budget = st.slider("token budget", min_value=500, max_value=8000, value=3000, step=500)
    if query:
        hits, metrics = retrieve_budgeted(store, query, k=k, token_budget=budget)
        st.json(metrics)
        for hit in hits:
            with st.expander(f"{hit['kind']} · {hit['id']} · score={hit['score']:.3f}", expanded=False):
                st.write(hit["text"])
                st.json({k: v for k, v in hit.items() if k not in {"text"}})

    st.subheader("Latest Query")
    latest = _load_latest_query(args.watch)
    if latest:
        st.markdown("**Question**")
        st.write(latest.get("query", ""))
        st.markdown("**Answer**")
        st.write(latest.get("answer", ""))
        st.markdown("**Metrics**")
        st.json(latest.get("metrics", {}))
    else:
        st.info("No query has been run yet.")

    st.subheader("Recent Trace Events")
    st.dataframe(store.latest_traces(limit=50), use_container_width=True)
    store.close()


if __name__ == "__main__":
    main()
