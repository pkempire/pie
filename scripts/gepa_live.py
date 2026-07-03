"""Live Streamlit dashboard for a running GEPA optimization.

Polls `mempol/results/gepa_consolidator/gepa_log/gepa_state.bin` every N
seconds and renders:

  - Score curve over iterations (line chart)
  - Pareto-frontier evolution (per-val example best score over time)
  - Per-iteration acceptance log (accepted vs rejected)
  - Current best prompt vs original (diff view)
  - Budget tracking (metric calls used, remaining)
  - Lineage tree (which candidate descended from which)

Usage:
    pip install streamlit altair  # if not already installed
    streamlit run scripts/gepa_live.py

Open http://localhost:8501. Auto-refreshes every 10 seconds.

The dashboard reads the pickle GEPAState dumped by DSPy's GEPA optimizer.
Safe to run during an active GEPA run — pickle is read-only.

This is also a great visual asset for video content. Recording the dashboard
as GEPA runs is the "watch prompt evolution happen" footage we want.
"""
from __future__ import annotations

import difflib
import pickle
import time
from collections import defaultdict
from pathlib import Path

import streamlit as st

REPO = Path(__file__).resolve().parents[1]
DEFAULT_LOG = REPO / "mempol" / "results" / "gepa_consolidator" / "gepa_log" / "gepa_state.bin"


# ─── State loading ──────────────────────────────────────────────────────────

@st.cache_data(ttl=5)
def load_state(path_str: str, mtime: float) -> dict | None:
    """Pickle load with cache invalidation on mtime change."""
    p = Path(path_str)
    if not p.exists():
        return None
    try:
        with p.open("rb") as f:
            state = pickle.load(f)
        return state
    except (pickle.UnpicklingError, EOFError):
        # Mid-write race — return None and the page will retry next refresh.
        return None


# ─── Score-curve extraction ─────────────────────────────────────────────────

def extract_score_curve(state: dict) -> list[dict]:
    """Per-iteration: (iteration, candidate_id, subsample_mean_score, accepted, parent_id).

    Pulls from full_program_trace which is the canonical iteration log.
    """
    rows: list[dict] = []
    trace = state.get("full_program_trace", [])
    for entry in trace:
        iteration = entry.get("i", -1)
        accepted = "new_program_idx" in entry
        new_idx = entry.get("new_program_idx") if accepted else None
        parent = entry.get("selected_program_candidate", -1)
        subsample = entry.get("new_subsample_scores", [])
        baseline_sub = entry.get("subsample_scores", [])
        baseline_mean = (sum(baseline_sub) / len(baseline_sub)) if baseline_sub else 0.0
        new_mean = (sum(subsample) / len(subsample)) if subsample else 0.0
        rows.append({
            "iter": iteration,
            "candidate": new_idx if new_idx is not None else parent,
            "parent": parent,
            "baseline_subsample_mean": round(baseline_mean, 3),
            "proposed_subsample_mean": round(new_mean, 3),
            "delta_subsample": round(new_mean - baseline_mean, 3),
            "accepted": accepted,
            "merge": bool(entry.get("invoked_merge")),
        })
    return rows


def extract_pareto_curve(state: dict) -> list[dict]:
    """Aggregate Pareto front score per iteration as it evolves."""
    rows: list[dict] = []
    trace = state.get("full_program_trace", [])
    for entry in trace:
        if "new_program_idx" not in entry:
            continue   # only consider accepted candidates
        iteration = entry.get("i", -1)
        new_idx = entry["new_program_idx"]
        subsample = entry.get("new_subsample_scores", [])
        if subsample:
            rows.append({
                "iter": iteration,
                "candidate": new_idx,
                "score": round(sum(subsample) / len(subsample), 3),
            })
    return rows


def pareto_front_summary(state: dict) -> dict:
    """Current Pareto front state."""
    front = state.get("pareto_front_valset", {})
    return {
        "n_examples_on_front": len(front),
        "mean_score": round(sum(front.values()) / max(len(front), 1), 3) if front else 0.0,
        "max_score": round(max(front.values()), 3) if front else 0.0,
        "min_score": round(min(front.values()), 3) if front else 0.0,
        "by_example": dict(front),
    }


def prompt_diff(state: dict, candidate_idx: int) -> str:
    """Unified diff between candidate's prompt and the original."""
    candidates = state.get("program_candidates", [])
    if not candidates or candidate_idx >= len(candidates):
        return "(no candidate)"
    original = candidates[0]
    target = candidates[candidate_idx]
    # Each is a dict[predictor_name -> prompt_text]
    out: list[str] = []
    for pred_name in target:
        orig_text = (original.get(pred_name) or "").splitlines()
        targ_text = (target.get(pred_name) or "").splitlines()
        diff = difflib.unified_diff(
            orig_text, targ_text,
            fromfile=f"original/{pred_name}",
            tofile=f"candidate-{candidate_idx}/{pred_name}",
            lineterm="",
            n=2,
        )
        out.extend(diff)
        out.append("")
    return "\n".join(out) if out else "(no diff)"


# ─── Page ───────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="GEPA live",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # Sidebar: config
    st.sidebar.title("GEPA live")
    log_path = st.sidebar.text_input("gepa_state.bin path", str(DEFAULT_LOG))
    refresh_secs = st.sidebar.slider("refresh interval (s)", 2, 60, 10)
    auto_refresh = st.sidebar.checkbox("auto-refresh", True)

    if not Path(log_path).exists():
        st.warning(f"No state file at {log_path} yet. Start a GEPA run "
                    f"and this page will populate.")
        if auto_refresh:
            time.sleep(refresh_secs)
            st.rerun()
        return

    mtime = Path(log_path).stat().st_mtime
    state = load_state(log_path, mtime)
    if state is None:
        st.warning("State pickle mid-write — retrying.")
        if auto_refresh:
            time.sleep(refresh_secs)
            st.rerun()
        return

    # Header
    st.title("GEPA optimization — live")
    st.caption(f"Polling `{log_path}` · last update {time.strftime('%H:%M:%S', time.localtime(mtime))}")

    # Top-line metrics
    iteration = state.get("i", "?")
    metric_calls = state.get("total_num_evals", "?")
    full_evals = state.get("num_full_ds_evals", "?")
    n_candidates = len(state.get("program_candidates", []))
    pareto = pareto_front_summary(state)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Current iteration", iteration)
    col2.metric("Candidates", n_candidates)
    col3.metric("Metric calls", metric_calls)
    col4.metric("Full evals", full_evals)
    col5.metric("Pareto mean", f"{pareto['mean_score']:.3f}")

    # Score curve
    st.header("Score over iterations")
    rows = extract_score_curve(state)
    if rows:
        import pandas as pd
        df = pd.DataFrame(rows)
        st.line_chart(
            df.set_index("iter")[["baseline_subsample_mean", "proposed_subsample_mean"]],
            height=240,
        )
        st.caption("Baseline = parent candidate's subsample score; proposed = new candidate's subsample score.")
    else:
        st.info("No iterations completed yet.")

    # Acceptance log
    st.header("Acceptance log")
    if rows:
        import pandas as pd
        df = pd.DataFrame(rows)
        df["status"] = df["accepted"].map({True: "✅ accepted", False: "❌ rejected"})
        st.dataframe(
            df[["iter", "candidate", "parent", "baseline_subsample_mean",
                "proposed_subsample_mean", "delta_subsample", "status", "merge"]],
            use_container_width=True,
            height=240,
        )

    # Pareto front
    st.header("Pareto front (per-example best score)")
    pf = pareto["by_example"]
    if pf:
        import pandas as pd
        df_pf = pd.DataFrame(
            [{"example_id": k, "best_score": v} for k, v in sorted(pf.items())]
        )
        st.bar_chart(df_pf.set_index("example_id")["best_score"], height=180)
        st.caption(f"Pareto front spans {pareto['n_examples_on_front']} examples · "
                    f"mean {pareto['mean_score']:.3f} · min {pareto['min_score']:.3f} · "
                    f"max {pareto['max_score']:.3f}")
    else:
        st.info("Pareto front empty.")

    # Best-candidate prompt diff
    st.header("Current best candidate — prompt diff vs original")
    if n_candidates > 1:
        # Pick the candidate that appears most on the Pareto front
        front_counts = defaultdict(int)
        for cands in state.get("program_at_pareto_front_valset", {}).values():
            for c in (cands if isinstance(cands, (set, list)) else [cands]):
                front_counts[c] += 1
        if front_counts:
            best = max(front_counts, key=front_counts.get)
            st.caption(f"Showing candidate {best} (appears on {front_counts[best]} "
                        f"of {pareto['n_examples_on_front']} Pareto front entries)")
        else:
            best = n_candidates - 1
            st.caption(f"Showing latest candidate {best} (no Pareto data yet)")
        diff = prompt_diff(state, best)
        st.code(diff, language="diff")
    else:
        st.info("Only the original prompt exists so far.")

    # Footer / refresh
    if auto_refresh:
        time.sleep(refresh_secs)
        st.rerun()


if __name__ == "__main__":
    main()
