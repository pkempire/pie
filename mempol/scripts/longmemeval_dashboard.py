"""Streamlit dashboard for LongMemEval matrix runs.

Run:
  streamlit run mempol/scripts/longmemeval_dashboard.py -- \
    --results mempol/results/lme_core_shards_merged \
    --variant longmemeval_s
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import streamlit as st

from mempol.data.longmemeval import load as load_lme
from mempol.scripts.longmemeval_matrix import (
    _canonical_cell_name,
    _cell_label,
    _dedupe_result_rows,
)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", required=True, help="Directory containing rows.jsonl/summary.json.")
    ap.add_argument("--variant", default="longmemeval_s", choices=["longmemeval_s", "longmemeval_oracle", "longmemeval_m"])
    ap.add_argument("--max-raw-turns", type=int, default=80)
    return ap.parse_args()


def _compact(text: str, n: int = 900) -> str:
    text = (text or "").replace("\n", " ").strip()
    return text if len(text) <= n else text[:n] + " ..."


@st.cache_data(show_spinner=False)
def _load_matrix(results_dir: str) -> tuple[list[dict], dict]:
    root = Path(results_dir)
    rows_path = root / "rows.jsonl"
    summary_path = root / "summary.json"
    rows: list[dict] = []
    if rows_path.exists():
        rows = [json.loads(l) for l in rows_path.read_text().splitlines() if l.strip()]
        rows = _dedupe_result_rows(rows)
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    return rows, summary


@st.cache_data(show_spinner=False)
def _load_lme_rows(variant: str) -> dict[str, dict[str, Any]]:
    out = {}
    for conv, qas in load_lme(variant=variant, n_convs=None, download=False):
        qa = qas[0]
        out[conv.sample_id] = {"conv": conv, "qa": qa}
    return out


def _by_question(rows: list[dict]) -> dict[str, dict[str, dict]]:
    out: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in rows:
        out[r["question_id"]][_canonical_cell_name(r["cell"])] = r
    return out


def _summary_table(summary: dict) -> list[dict[str, Any]]:
    table = []
    for cell, s in (summary.get("by_cell") or {}).items():
        table.append({
            "strategy": _cell_label(cell),
            "cell": cell,
            "acc": round(float(s.get("acc", 0.0)), 3),
            "n": s.get("n"),
            "errors": s.get("errors", 0),
            "avg_ctx_tokens": round(float(s.get("avg_retrieved_tokens_est", 0.0)), 1),
            "avg_stored_tokens": round(float(s.get("avg_stored_tokens_est", 0.0)), 1),
            "avg_units": round(float(s.get("avg_stored_units", 0.0)), 1),
            "avg_raw_tokens": round(float(s.get("avg_raw_tokens_est", 0.0)), 1),
            "avg_retrieval/storage": round(float(s.get("avg_retrieval_to_storage_ratio", 0.0)), 4),
        })
    return table


def _category_table(summary: dict) -> list[dict[str, Any]]:
    rows = []
    for cell, s in (summary.get("by_cell") or {}).items():
        for cat, cs in (s.get("by_category") or {}).items():
            rows.append({
                "strategy": _cell_label(cell),
                "cell": cell,
                "category": cat,
                "n": cs.get("n"),
                "acc": round(float(cs.get("acc", 0.0)), 3),
            })
    return rows


def main() -> None:
    args = _parse_args()
    st.set_page_config(page_title="LongMemEval Matrix", layout="wide")
    st.title("LongMemEval Matrix")
    st.caption(f"results: `{args.results}` · variant: `{args.variant}`")

    rows, summary = _load_matrix(args.results)
    lme = _load_lme_rows(args.variant)
    by_q = _by_question(rows)

    if not rows:
        st.error("No rows found. Expected rows.jsonl in the results directory.")
        return

    cells = sorted({_canonical_cell_name(r["cell"]) for r in rows})
    qids = sorted(by_q.keys())
    cats = sorted({r.get("category_name", "") for r in rows})

    top = st.columns(5)
    top[0].metric("questions", len(qids))
    top[1].metric("strategy rows", len(rows))
    top[2].metric("strategies", len(cells))
    top[3].metric("categories", len(cats))
    top[4].metric("errors", sum(1 for r in rows if r.get("error")))

    tab_summary, tab_question, tab_raw, tab_schema = st.tabs([
        "Summary",
        "Question Trace",
        "Raw Data",
        "Definitions",
    ])

    with tab_summary:
        st.subheader("Strategy Summary")
        st.dataframe(_summary_table(summary), use_container_width=True, hide_index=True)
        st.subheader("By Category")
        st.dataframe(_category_table(summary), use_container_width=True, hide_index=True)

    with tab_question:
        left, right = st.columns([1, 2])
        with left:
            cat_filter = st.selectbox("category", ["all"] + cats)
            visible_qids = [
                qid for qid in qids
                if cat_filter == "all" or next(iter(by_q[qid].values())).get("category_name") == cat_filter
            ]
            qid = st.selectbox("question", visible_qids, format_func=lambda x: f"{x[:8]} · {next(iter(by_q[x].values())).get('category_name')}")
            selected_cells = st.multiselect("strategies", cells, default=cells)
        first = next(iter(by_q[qid].values()))
        with right:
            st.markdown(f"**Q:** {first.get('question')}")
            st.markdown(f"**Gold:** {first.get('gold')}")
            st.caption(f"category: `{first.get('category_name')}` · turns: `{first.get('raw_turns') or first.get('n_turns')}` · sessions: `{first.get('raw_sessions') or first.get('n_sessions')}`")

        for cell in selected_cells:
            r = by_q[qid].get(cell)
            if not r:
                continue
            with st.expander(f"{_cell_label(cell)} · score={r.get('score')} · ctx={r.get('context_chars')} chars", expanded=True):
                st.markdown(f"**Answer:** {_compact(r.get('answer', ''), 1400)}")
                cols = st.columns(5)
                cols[0].metric("retrieved", r.get("retrieval_count", 0))
                cols[1].metric("ctx tokens", r.get("retrieved_tokens_est", 0))
                cols[2].metric("stored units", r.get("stored_units", 0))
                cols[3].metric("stored tokens", r.get("stored_tokens_est", 0))
                ratio = r.get("retrieval_to_storage_ratio")
                cols[4].metric("ctx/store", f"{ratio:.3f}" if isinstance(ratio, (int, float)) else "n/a")
                if r.get("error"):
                    st.error(r["error"])
                trace = r.get("trace") or {}
                steps = trace.get("steps") or []
                if steps:
                    st.markdown("**Trace steps**")
                    st.dataframe([
                        {
                            "op": s.get("op"),
                            "args": json.dumps(s.get("args") or {}, ensure_ascii=False),
                            "obs": s.get("obs_summary", ""),
                        }
                        for s in steps
                    ], use_container_width=True, hide_index=True)
                retrieved = trace.get("retrieved") or []
                if retrieved:
                    st.markdown("**Retrieved evidence**")
                    for hit in retrieved:
                        md = hit.get("metadata") or {}
                        st.code(
                            f"{hit.get('uid')} | {hit.get('source')} | score={hit.get('score')}\n"
                            f"{md.get('session_date') or ''} | {md.get('speaker') or md.get('name') or ''}\n"
                            f"{hit.get('text')}",
                            language="text",
                        )

    with tab_raw:
        raw_qid = st.selectbox("raw question", qids, key="raw_qid")
        bundle = lme.get(raw_qid)
        if not bundle:
            st.warning("Raw row not available locally.")
        else:
            conv = bundle["conv"]
            qa = bundle["qa"]
            st.markdown(f"**Q:** {qa.question}")
            st.markdown(f"**Gold:** {qa.answer}")
            query = st.text_input("filter raw turns containing", "")
            sessions = defaultdict(list)
            for t in conv.turns:
                if query and query.lower() not in (t.text or "").lower():
                    continue
                sessions[t.session].append(t)
            session_ids = list(sessions.keys())
            session = st.selectbox("session", session_ids)
            for t in sessions[session][: args.max_raw_turns]:
                st.code(f"{t.dia_id} | {t.session_date} | {t.speaker}\n{t.text}", language="text")

    with tab_schema:
        st.markdown(
            """
**What is a doc/unit here?**

- `turn_rag`, `hybrid_search`, `rerank_search`, `expand_search`, and `timeline_synthesis`: one retrieval unit is one LongMemEval turn, meaning one message from either user or assistant.
- `session_rag`: one retrieval unit is one full session, containing many turns.
- `full_context`: no retrieval units are selected; the whole haystack is formatted into the prompt up to the configured character cap.
- `cached_pie`, `fresh_pie`, `build_pie`: the retrieval unit is a PIE entity text representation from a session-by-session extracted world model.

**Key budgets shown**

- `raw_tokens`: estimated source haystack tokens, using chars / 4.
- `stored_tokens`: estimated text stored in the backend before retrieval.
- `ctx tokens`: estimated answer-context tokens actually passed to the answer model.
- `ctx/store`: answer context size divided by stored memory size. Smaller means targeted retrieval; too small can miss evidence.
            """
        )


if __name__ == "__main__":
    main()
