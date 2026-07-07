"""ctxpack dashboard — observe the pack-evolution process.

Run: streamlit run ctxpack/dashboard.py
Reads ctxpack/results/evolution/ (history.json + pack_r*.md) and the baseline run JSONs.
Views: score trajectory · per-question blame matrix · round-to-round pack diff · pack browser.
"""
from __future__ import annotations
import difflib, json
from pathlib import Path

import streamlit as st

REPO = Path(__file__).resolve().parent.parent
EVO = REPO / "ctxpack" / "results" / "evolution"

st.set_page_config(page_title="ctxpack — observability", layout="wide")
st.title("ctxpack — observability")

tab_lme, tab_evo = st.tabs(["LongMemEval run (live)", "mempol pack evolution"])

# ================= LongMemEval live tab =================
with tab_lme:
    st.caption("Experiment: question-BLIND timeline-structured pack (compiled from each question's "
               "~115k-token haystack before seeing the question) vs query-adaptive lexical RAG, at "
               "matched 4k-token budget. LLM-judged (free-form answers) — judge reasons logged. "
               "Data: LongMemEval-S, balanced across 6 question types. Auto-refresh: rerun the page.")
    st.markdown('<meta http-equiv="refresh" content="30">', unsafe_allow_html=True)  # auto-refresh 30s
    lme_f = REPO / "ctxpack" / "results" / "lme" / "traces.jsonl"
    if not lme_f.exists():
        st.warning("No LME run yet. Run: python -m ctxpack.lme_pack_eval")
    else:
        rows = [json.loads(l) for l in lme_f.read_text().splitlines()]
        good = [r for r in rows if "error" not in r]
        c = st.columns(4)
        c[0].metric("questions done", len(rows))
        if good:
            c[1].metric("pack accuracy", f"{sum(r['ok_pack'] for r in good)/len(good)*100:.0f}%")
            c[2].metric("RAG accuracy", f"{sum(r['ok_rag'] for r in good)/len(good)*100:.0f}%")
            c[3].metric("errors", len(rows) - len(good))
            by_t: dict = {}
            for r in good:
                by_t.setdefault(r["qtype"], []).append(r)
            st.dataframe([{ "type": t, "n": len(v),
                            "pack": f"{sum(r['ok_pack'] for r in v)/len(v)*100:.0f}%",
                            "rag": f"{sum(r['ok_rag'] for r in v)/len(v)*100:.0f}%"}
                          for t, v in sorted(by_t.items())],
                         use_container_width=True, hide_index=True)
            st.dataframe([{ "qid": r["qid"][:24], "type": r["qtype"],
                            "pack": "✅" if r["ok_pack"] else "❌",
                            "rag": "✅" if r["ok_rag"] else "❌",
                            "question": r["q"][:80]} for r in good],
                         use_container_width=True, hide_index=True)
            qpick = st.selectbox("inspect question", [r["qid"] for r in good])
            r = next(x for x in good if x["qid"] == qpick)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**Q:** {r['q']}\n\n**Gold:** {r['gold']}")
                st.markdown(f"**Pack answer** ({'✅' if r['ok_pack'] else '❌'}): {r['a_pack']}")
                st.caption(f"judge: {r.get('judge_pack','')}")
                st.markdown(f"**RAG answer** ({'✅' if r['ok_rag'] else '❌'}): {r['a_rag']}")
                st.caption(f"judge: {r.get('judge_rag','')}")
            with c2:
                st.code(r.get("pack", "")[:14_000], language="markdown")

# ================= mempol evolution tab =================
with tab_evo:
    st.caption("Experiment: corpus = mempol's own source code (43 files, 339k chars); 12 regex-"
               "scored questions about the code (6 train / 6 held-out); the PACK is evolved over "
               "rounds using blame-decomposed feedback from TRAIN questions only.")
    hist_f = EVO / "history.json"
    if not hist_f.exists():
        st.warning("No evolution run found. Run: python -m ctxpack.evolve")
        st.stop()
    hist = json.loads(hist_f.read_text())
    rounds = hist["rounds"]

    # ---- score trajectory ----
    st.subheader("Accuracy trajectory (deterministic scoring)")
    cols = st.columns([2, 1])
    with cols[0]:
        chart = {
            "train": [r["train_acc"] for r in rounds],
            "held-out": [r["heldout_acc"] for r in rounds],
        }
        st.line_chart(chart)
    with cols[1]:
        hw = hist.get("handwritten", {})
        st.metric("handwritten docs (held-out)", f"{hw.get('heldout_acc', 0)*100:.1f}%")
        best = max(rounds, key=lambda r: r["heldout_acc"])
        st.metric("best round (held-out)", f"r{best['round']} · {best['heldout_acc']*100:.1f}%")
        last = rounds[-1]
        gate = "SHIP r%d" % best["round"] if last["heldout_acc"] < best["heldout_acc"] else "SHIP latest"
        st.metric("regression gate says", gate)
        st.caption(f"budget: {hist['budget_tokens']} tokens · packs: " +
                   " → ".join(f"{r['pack_chars']//1000}k" for r in rounds) + " chars")

    # ---- blame matrix ----
    st.subheader("Per-question blame matrix")
    st.caption("OK = answered correctly from pack · MISSING = fact absent (writer fault) · "
               "BURIED = fact present but answer failed (organization fault)")
    qids = [row["id"] for row in rounds[0]["train"]] + [row["id"] for row in rounds[0]["heldout"]]
    split = {row["id"]: "train" for row in rounds[0]["train"]}
    split.update({row["id"]: "held-out" for row in rounds[0]["heldout"]})

    def cell(row: dict) -> str:
        if row["ok"]:
            return "✅ OK"
        return "🟡 BURIED" if row.get("fact_in_pack") else "🔴 MISSING"

    matrix = []
    for qid in qids:
        entry = {"question": qid, "split": split[qid]}
        for r in rounds:
            row = next(x for x in r["train"] + r["heldout"] if x["id"] == qid)
            entry[f"r{r['round']}"] = cell(row)
        matrix.append(entry)
    st.dataframe(matrix, use_container_width=True, hide_index=True)

    # ---- pack browser + diff ----
    st.subheader("Pack content & round-to-round diff")
    packs = sorted(EVO.glob("pack_r*.md"), key=lambda p: int(p.stem.split("_r")[1]))
    names = [p.stem for p in packs]
    c1, c2 = st.columns(2)
    sel = c1.selectbox("round", names, index=len(names) - 1)
    mode = c2.radio("view", ["pack", "diff vs previous"], horizontal=True)
    idx = names.index(sel)
    text = packs[idx].read_text()
    if mode == "pack" or idx == 0:
        st.code(text[:30_000], language="markdown")
    else:
        prev = packs[idx - 1].read_text()
        diff = "\n".join(difflib.unified_diff(
            prev.splitlines(), text.splitlines(),
            fromfile=names[idx - 1], tofile=sel, lineterm="", n=1))
        st.code(diff[:30_000] or "(no textual change)", language="diff")

    # ---- per-question answers drill-down ----
    st.subheader("Answers drill-down")
    qsel = st.selectbox("question", qids)
    drill = []
    for r in rounds:
        row = next(x for x in r["train"] + r["heldout"] if x["id"] == qsel)
        drill.append({"round": f"r{r['round']}", "status": cell(row), "answer": row["a"][:220]})
    st.dataframe(drill, use_container_width=True, hide_index=True)
    st.caption("Every number on this page is regenerable from ctxpack/results/evolution/ — no hidden state.")
