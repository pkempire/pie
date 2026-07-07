"""Hybrid condition on the existing n=30 LME traces — no recompilation needed.

Reuses the stored packs from traces.jsonl: answer from pack; if the pack answer signals
don't-know, escalate to half-pack + half raw-retrieval at the SAME total budget. Judges and
writes results/lme/hybrid.json. Cost ~$1.

Run: python -m ctxpack.hybrid_eval
"""
from __future__ import annotations
import json
from ctxpack.lme_pack_eval import DATA, OUT, hybrid_answer, judge


def main() -> None:
    data = {q["question_id"]: q for q in json.load(open(DATA))}
    rows = [json.loads(l) for l in (OUT / "traces.jsonl").read_text().splitlines()]
    rows = [r for r in rows if "error" not in r and r.get("pack")]
    out, esc_n = [], 0
    for r in rows:
        q = data[r["qid"]]
        budget_chars = r["budget_tokens"] * 4
        a, escalated = hybrid_answer(r["pack"], q, budget_chars)
        ok, reason = judge(q, a)
        esc_n += escalated
        out.append({"qid": r["qid"], "qtype": r["qtype"], "ok_hybrid": ok,
                    "escalated": escalated, "a": a, "reason": reason,
                    "ok_pack": r["ok_pack"], "ok_rag": r["ok_rag"]})
        print(f"{r['qid'][:22]:<22} {r['qtype'][:22]:<22} hybrid:{'OK' if ok else '..'}"
              f"{' (escalated)' if escalated else ''}")
    n = len(out)
    summ = {"n": n,
            "hybrid_acc": round(sum(r["ok_hybrid"] for r in out) / n, 4),
            "pack_acc": round(sum(r["ok_pack"] for r in out) / n, 4),
            "rag_acc": round(sum(r["ok_rag"] for r in out) / n, 4),
            "escalation_rate": round(esc_n / n, 3),
            "by_type": {}}
    for t in sorted({r["qtype"] for r in out}):
        tr = [r for r in out if r["qtype"] == t]
        summ["by_type"][t] = {"n": len(tr),
                              "hybrid": round(sum(r["ok_hybrid"] for r in tr) / len(tr), 3),
                              "pack": round(sum(r["ok_pack"] for r in tr) / len(tr), 3),
                              "rag": round(sum(r["ok_rag"] for r in tr) / len(tr), 3)}
    (OUT / "hybrid.json").write_text(json.dumps({"summary": summ, "rows": out}, indent=2))
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
