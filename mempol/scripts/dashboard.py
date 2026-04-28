"""Live training dashboard for mempol Phase B / Phase A runs.

Reads two artifacts the Tinker cookbook writes to disk:

  <log_path>/metrics.jsonl         per-step metrics (one JSON object per step)
  <log_path>/rollouts/             per-step trajectory dumps
                                    (governed by Config.num_groups_to_log)

Usage:
    streamlit run mempol/scripts/dashboard.py -- --log_path /tmp/mempol/...

Multiple log paths (e.g. for sweep comparison):
    streamlit run mempol/scripts/dashboard.py -- \\
        --log_paths /tmp/mempol/cov_sweep/w0.0,/tmp/mempol/cov_sweep/w0.6

The dashboard auto-refreshes every 5 s and exposes:
  - Reward curve (total + coverage + qa_mean)
  - Op distribution (n_mutations, n_lookups, n_noops, n_entities)
  - Health indicators (entropy, frac_mixed, frac_all_good, frac_all_bad)
  - Wall clock per step
  - Most recent rollout text (rendered messages + tool calls)

Designed for two people watching the same shared filesystem (or NFS / iCloud
mirror) — one starts the run, both `streamlit run` against the same path.
For remote co-monitoring use W&B (set MEMPOL_WANDB_PROJECT).
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import streamlit as st


REFRESH_SEC = 5


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def _flatten_metrics(rows: list[dict]) -> dict[str, list[float]]:
    """Turn list of {step, metric_name: value, ...} dicts into per-metric lists."""
    series: dict[str, list[float]] = {}
    for r in rows:
        for k, v in r.items():
            if isinstance(v, (int, float)):
                series.setdefault(k, []).append(float(v))
    return series


def _list_recent_rollouts(rollout_dir: Path, n: int = 4) -> list[Path]:
    if not rollout_dir.exists():
        return []
    files = sorted(rollout_dir.glob("**/*.json"), key=lambda p: p.stat().st_mtime,
                   reverse=True)
    return files[:n]


def _render_one_rollout(path: Path) -> None:
    try:
        data = json.loads(path.read_text(errors="replace"))
    except Exception as e:
        st.warning(f"could not parse {path.name}: {e}")
        return
    msgs = data.get("messages") or data.get("trajectory") or []
    metrics = data.get("metrics") or {}
    reward = data.get("reward") or metrics.get("reward")

    cols = st.columns([1, 1, 1, 1])
    cols[0].metric("reward", f"{reward:.3f}" if isinstance(reward, (int, float)) else "—")
    cols[1].metric("coverage", f"{metrics.get('coverage_mean', 0):.3f}")
    cols[2].metric("qa_mean",  f"{metrics.get('qa_mean', 0):.3f}")
    cols[3].metric("n_ops",    f"{metrics.get('n_ops', 0):.0f}")

    for m in msgs:
        role = (m.get("role") if isinstance(m, dict) else "?") or "?"
        content = (m.get("content") if isinstance(m, dict) else str(m)) or ""
        tcs = m.get("tool_calls") if isinstance(m, dict) else None
        if role == "system":
            with st.expander(f"system ({len(content)} chars)", expanded=False):
                st.code(content[:4000], language="markdown")
        elif role == "user":
            st.markdown(f"**user**")
            st.code(content[:2000], language="markdown")
        elif role == "assistant":
            st.markdown("**assistant**")
            if content:
                st.code(content[:2000], language="markdown")
            if tcs:
                for tc in tcs:
                    name = tc.get("name") or (tc.get("function") or {}).get("name", "?")
                    args = tc.get("arguments") or (tc.get("function") or {}).get("arguments", {})
                    st.code(f"{name}({json.dumps(args, indent=2)[:1500]})", language="json")
        elif role == "tool":
            st.markdown("**tool result**")
            st.code(content[:1500], language="json")


def _render_run(log_path: Path, run_name: str) -> None:
    metrics_path = log_path / "metrics.jsonl"
    rollouts_dir = log_path / "rollouts"
    rows = _read_jsonl(metrics_path)
    if not rows:
        st.info(f"{run_name}: no metrics yet (waiting for {metrics_path})")
        return

    series = _flatten_metrics(rows)
    n_steps = len(rows)

    # Headline cards
    last = rows[-1]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("step", n_steps,
              delta=None if n_steps < 2 else "+1")
    c2.metric("reward",   f"{last.get('env/all/reward/total', float('nan')):.3f}")
    c3.metric("coverage", f"{last.get('env/all/coverage_mean', float('nan')):.3f}")
    c4.metric("entropy",  f"{last.get('optim/entropy', float('nan')):.4f}")

    # Curves
    def _line(metric_keys: list[str], title: str):
        chart_data = {}
        for k in metric_keys:
            if k in series:
                chart_data[k.split("/")[-1]] = series[k]
        if chart_data:
            st.subheader(title)
            st.line_chart(chart_data)

    _line(["env/all/reward/total", "env/all/coverage_mean", "env/all/qa_mean"],
          "Reward / coverage / qa over training")
    _line(["env/all/n_mutations", "env/all/n_lookups", "env/all/n_noops",
           "env/all/n_entities"], "Op distribution per episode")
    _line(["optim/entropy"], "Policy entropy (watch for collapse)")
    _line(["env/all/by_group/frac_mixed",
           "env/all/by_group/frac_all_good",
           "env/all/by_group/frac_all_bad"], "Group reward variance health")
    _line(["time/total", "time/env_step:mean", "time/policy_sample:mean"],
          "Wall clock per step (s)")

    # Trajectory viewer
    st.subheader("Recent trajectories")
    recent = _list_recent_rollouts(rollouts_dir, n=4)
    if not recent:
        st.caption("(no rollout dumps written yet — needs num_groups_to_log > 0 "
                   "and rollout_json_export=True in train Config)")
    else:
        tabs = st.tabs([p.stem for p in recent])
        for tab, p in zip(tabs, recent):
            with tab:
                _render_one_rollout(p)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_path", default=None,
                        help="Single run log dir (the one passed to train_write).")
    parser.add_argument("--log_paths", default=None,
                        help="Comma-separated list of run dirs for sweep comparison.")
    args, _ = parser.parse_known_args()

    paths: list[Path] = []
    if args.log_paths:
        paths = [Path(p.strip()).expanduser() for p in args.log_paths.split(",")]
    elif args.log_path:
        paths = [Path(args.log_path).expanduser()]
    else:
        st.error("Pass --log_path or --log_paths.")
        return

    st.set_page_config(page_title="mempol live", layout="wide")
    st.title("mempol live training monitor")
    st.caption(f"Auto-refresh every {REFRESH_SEC}s. Watching: "
               + ", ".join(str(p) for p in paths))

    if len(paths) == 1:
        _render_run(paths[0], paths[0].name)
    else:
        tabs = st.tabs([p.name for p in paths])
        for tab, p in zip(tabs, paths):
            with tab:
                _render_run(p, p.name)

    time.sleep(REFRESH_SEC)
    st.rerun()


if __name__ == "__main__":
    main()
