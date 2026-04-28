"""Live training dashboard for mempol Phase B / Phase A runs.

Reads what the Tinker cookbook writes to disk:

  <log_path>/metrics.jsonl         per-step metrics (one JSON per step)
  <log_path>/rollouts/             per-step rollout dumps
                                    (governed by Config.num_groups_to_log)

Run:
    streamlit run mempol/scripts/dashboard.py -- --log_path /tmp/mempol/...

For sweeps:
    streamlit run mempol/scripts/dashboard.py -- \\
        --log_paths /tmp/mempol/cov_sweep/w0.0,/tmp/mempol/cov_sweep/w0.6

The dashboard auto-refreshes every 5 s and shows:

  Top strip   reward / coverage / qa_mean / entropy headline cards
  Curves      reward + coverage + qa_mean over training
              op-mix evolution (mutations / lookups / noops over steps)
              entropy + KL-to-base over training
              group-variance health
  Trajectories
              best / worst rollout side-by-side from the most recent step
              per-question coverage scores
              KG snapshot summary (entities, types, dia_ids stored)
  Rollout list
              full text of any rollout you click

Designed for two people watching the same shared filesystem (or a NFS /
iCloud mirror). For remote viewing, also enable W&B
(`wandb_project=mempol` on the train_write CLI).
"""
from __future__ import annotations
import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import streamlit as st


REFRESH_SEC = 5


# ─────────────────────────────────────────────────────────────────────────────
# I/O
# ─────────────────────────────────────────────────────────────────────────────
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
    series: dict[str, list[float]] = {}
    for r in rows:
        for k, v in r.items():
            if isinstance(v, (int, float)):
                series.setdefault(k, []).append(float(v))
    return series


def _list_rollouts(rollout_dir: Path) -> list[Path]:
    if not rollout_dir.exists():
        return []
    return sorted(rollout_dir.glob("**/*.json"),
                  key=lambda p: p.stat().st_mtime, reverse=True)


def _load_rollout(path: Path) -> dict:
    try:
        return json.loads(path.read_text(errors="replace"))
    except Exception:
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Trajectory rendering
# ─────────────────────────────────────────────────────────────────────────────
def _extract_messages(rollout: dict) -> list[dict]:
    return rollout.get("messages") or rollout.get("trajectory") or []


def _extract_metrics(rollout: dict) -> dict:
    m = rollout.get("metrics") or {}
    if not m and "reward" in rollout:
        m = {"reward": rollout["reward"]}
    return m


def _summarise_ops(messages: list[dict]) -> Counter:
    """Count tool calls by name across all assistant messages."""
    c: Counter = Counter()
    for m in messages:
        if (m.get("role") if isinstance(m, dict) else None) != "assistant":
            continue
        tcs = m.get("tool_calls") if isinstance(m, dict) else None
        if tcs:
            for tc in tcs:
                name = tc.get("name") or (tc.get("function") or {}).get("name", "?")
                c[name] += 1
        elif isinstance(m.get("content"), str) and "<tool_call>" in m["content"]:
            import re
            for nm in re.findall(r'"name"\s*:\s*"([^"]+)"', m["content"]):
                c[nm] += 1
    return c


def _render_trajectory(rollout: dict, key_prefix: str = "") -> None:
    msgs = _extract_messages(rollout)
    metrics = _extract_metrics(rollout)
    ops = _summarise_ops(msgs)

    # Headline cards for this rollout
    cs = st.columns(5)
    cs[0].metric("reward", f"{rollout.get('reward', 0):.3f}"
                 if isinstance(rollout.get('reward'), (int, float)) else "—")
    cs[1].metric("coverage", f"{metrics.get('coverage_mean', 0):.2f}")
    cs[2].metric("qa_mean",  f"{metrics.get('qa_mean', 0):.2f}")
    cs[3].metric("ops",      str(int(metrics.get('n_ops', sum(ops.values())))))
    cs[4].metric("entities", str(int(metrics.get('n_entities', 0))))

    # Op breakdown
    if ops:
        op_str = " · ".join(f"{n}×{name}" for name, n in ops.most_common())
        st.caption(f"**Ops emitted:** {op_str}")

    # Per-question coverage if available
    pq = metrics.get("per_question_coverage") or rollout.get("per_question_coverage")
    if pq:
        with st.expander("per-question coverage", expanded=False):
            st.dataframe([
                {"question": q[:80], "coverage": f"{s:.2f}"}
                for q, s in pq
            ], use_container_width=True)

    # Render messages
    for i, m in enumerate(msgs):
        role = (m.get("role") if isinstance(m, dict) else "?") or "?"
        content = (m.get("content") if isinstance(m, dict) else str(m)) or ""
        tcs = m.get("tool_calls") if isinstance(m, dict) else None

        if role == "system":
            with st.expander(f"system ({len(content)} chars)", expanded=False):
                st.code(content[:6000], language="markdown")
        elif role == "user":
            st.markdown("**user**")
            st.code(content[:2500], language="markdown")
        elif role == "assistant":
            st.markdown("**assistant**")
            if content:
                with st.expander("text", expanded=False):
                    st.code(content[:2000], language="markdown")
            if tcs:
                for tc in tcs:
                    name = tc.get("name") or (tc.get("function") or {}).get("name", "?")
                    args = tc.get("arguments") or (tc.get("function") or {}).get("arguments", {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except Exception:
                            pass
                    st.code(
                        f"{name}({json.dumps(args, indent=2, ensure_ascii=False)[:1200]})",
                        language="json",
                    )
        elif role == "tool":
            st.markdown("**tool result**")
            st.code(content[:1200], language="json")


def _render_kg_summary(rollout: dict) -> None:
    """If the rollout dump includes a kg_snapshot block, render it.

    Expected shape (see TrajectoryDumper in write_env.py for the writer):
      kg_snapshot: {
        n_entities: int,
        entities: [{name, type, current_state, n_transitions, source_dia_ids}],
        stored_dia_ids: list[str],
      }
    """
    snap = rollout.get("kg_snapshot") or {}
    if not snap:
        return
    cs = st.columns(3)
    cs[0].metric("entities", snap.get("n_entities", 0))
    cs[1].metric("dia_ids stored", len(snap.get("stored_dia_ids") or []))
    type_counts = Counter()
    for e in snap.get("entities") or []:
        type_counts[e.get("type", "?")] += 1
    cs[2].metric("entity types", len(type_counts))
    if type_counts:
        st.caption("**By type: **" + " · ".join(
            f"{n}×{t}" for t, n in type_counts.most_common()))
    if snap.get("entities"):
        st.dataframe([
            {
                "name": e.get("name", "?"),
                "type": e.get("type", "?"),
                "transitions": e.get("n_transitions", 0),
                "state": json.dumps(e.get("current_state", {}))[:80],
            }
            for e in snap["entities"][:30]
        ], use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Per-step op-mix history (reads metrics.jsonl over time)
# ─────────────────────────────────────────────────────────────────────────────
def _op_mix_chart_data(series: dict[str, list[float]]) -> dict[str, list[float]]:
    keys = ["env/all/n_mutations", "env/all/n_lookups", "env/all/n_noops"]
    out = {}
    for k in keys:
        if k in series:
            out[k.split("/")[-1]] = series[k]
    return out


def _kl_chart_data(series: dict[str, list[float]]) -> dict[str, list[float]]:
    out = {}
    for k in ["optim/entropy", "kl_policy_base", "optim/kl_sample_train_v2"]:
        if k in series:
            out[k.split("/")[-1] if "/" in k else k] = series[k]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Per-run rendering
# ─────────────────────────────────────────────────────────────────────────────
def _render_run(log_path: Path, run_name: str) -> None:
    metrics_path = log_path / "metrics.jsonl"
    rollouts_dir = log_path / "rollouts"
    rows = _read_jsonl(metrics_path)
    if not rows:
        st.info(f"{run_name}: no metrics yet (waiting for {metrics_path})")
        return

    series = _flatten_metrics(rows)
    n_steps = len(rows)
    last = rows[-1]

    # ── Headline cards ─────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("step", n_steps)
    c2.metric("reward",   f"{last.get('env/all/reward/total', float('nan')):.3f}")
    c3.metric("coverage", f"{last.get('env/all/coverage_mean', float('nan')):.3f}")
    c4.metric("qa_mean",  f"{last.get('env/all/qa_mean', float('nan')):.3f}")
    c5.metric("entropy",  f"{last.get('optim/entropy', float('nan')):.4f}")

    # ── Curves (4 plots) ────────────────────────────────────────────────
    st.subheader("Reward / coverage / qa over training")
    chart = {}
    for k in ["env/all/reward/total", "env/all/coverage_mean", "env/all/qa_mean"]:
        if k in series:
            chart[k.split("/")[-1]] = series[k]
    if chart:
        st.line_chart(chart)

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Op mix per episode")
        op_chart = _op_mix_chart_data(series)
        if op_chart:
            st.line_chart(op_chart)
        st.caption("Watch n_lookups rise relative to n_mutations — that's the "
                   "policy learning to dedupe before writing.")
    with col_b:
        st.subheader("Entropy + KL")
        kl_chart = _kl_chart_data(series)
        if kl_chart:
            st.line_chart(kl_chart)
        st.caption("Entropy in 0.15–0.30 is healthy. Below 0.10 = collapsing.")

    col_c, col_d = st.columns(2)
    with col_c:
        st.subheader("Group variance health")
        gh = {}
        for k in ["env/all/by_group/frac_mixed",
                  "env/all/by_group/frac_all_good",
                  "env/all/by_group/frac_all_bad"]:
            if k in series:
                gh[k.split("/")[-1]] = series[k]
        if gh:
            st.line_chart(gh)
        st.caption("frac_mixed should stay ≥ 0.5 — that's where GRPO gradient "
                   "comes from.")
    with col_d:
        st.subheader("Wall clock per step (s)")
        tc = {}
        for k in ["time/total", "time/env_step:mean", "time/policy_sample:mean"]:
            if k in series:
                tc[k.split("/")[-1]] = series[k]
        if tc:
            st.line_chart(tc)

    # ── Trajectory comparison ───────────────────────────────────────────
    st.markdown("---")
    st.subheader("Trajectory comparison — most recent step")

    rollouts = _list_rollouts(rollouts_dir)
    if not rollouts:
        st.caption(f"(no rollout dumps found in {rollouts_dir} — set "
                   "`num_groups_to_log > 0` and `rollout_json_export=True` "
                   "in train Config)")
        time.sleep(REFRESH_SEC)
        st.rerun()
        return

    # Take the most recent group (top N by mtime)
    recent = rollouts[:32]
    rated = []
    for p in recent:
        r = _load_rollout(p)
        rew = r.get("reward")
        if isinstance(rew, (int, float)):
            rated.append((rew, p, r))
    if not rated:
        st.caption("(rollouts present but no reward field detected)")
    else:
        rated.sort(key=lambda x: x[0])
        worst_r, worst_p, worst = rated[0]
        best_r,  best_p,  best  = rated[-1]
        st.caption(f"Spread across {len(rated)} recent rollouts: "
                   f"min={worst_r:.3f}  max={best_r:.3f}  "
                   f"Δ={best_r - worst_r:.3f}")

        col_best, col_worst = st.columns(2)
        with col_best:
            st.markdown(f"### Best rollout (reward {best_r:.3f})")
            st.caption(f"`{best_p.name}`")
            _render_trajectory(best, key_prefix="best")
            with st.expander("KG snapshot", expanded=True):
                _render_kg_summary(best)
        with col_worst:
            st.markdown(f"### Worst rollout (reward {worst_r:.3f})")
            st.caption(f"`{worst_p.name}`")
            _render_trajectory(worst, key_prefix="worst")
            with st.expander("KG snapshot", expanded=True):
                _render_kg_summary(worst)

    # ── Browseable rollout list ─────────────────────────────────────────
    st.markdown("---")
    st.subheader("All recent rollouts")
    pick = st.selectbox(
        "open one",
        options=[p.relative_to(log_path) for p in rollouts[:50]],
        format_func=str,
    )
    if pick:
        _render_trajectory(_load_rollout(log_path / pick), key_prefix="picked")
        with st.expander("KG snapshot", expanded=False):
            _render_kg_summary(_load_rollout(log_path / pick))


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_path", default=None,
                        help="Single run log dir.")
    parser.add_argument("--log_paths", default=None,
                        help="Comma-separated list of run dirs to compare.")
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
    st.title("mempol training monitor")
    st.caption(f"Auto-refresh every {REFRESH_SEC}s · watching: "
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
