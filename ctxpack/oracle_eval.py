"""Full-context ceiling on the paired 30 LME questions — the missing calibration row.

Stuffs the raw haystack (up to a char cap well inside the model window) directly in context.
Every other number in the table is meaningless without this ceiling: it measures what the
answer model + judge can do with NO memory system at all, i.e. the judge-adjusted 100%.

Run: python -m ctxpack.oracle_eval   (post quota refill; ~30 questions x ~120k tok = ~$3)
"""
from __future__ import annotations
import json
from ctxpack.lme_pack_eval import DATA, OUT, answer, judge, sessions_text

CAP_CHARS = 700_000  # ~175k tokens; LME-S haystacks ~115k tokens fit whole


def main() -> None:
    data = {q["question_id"]: q for q in json.load(open(DATA))}
    qids = json.load(open(OUT / "qids30.json"))
    rows = []
    for qid in qids:
        q = data[qid]
        ctx = "\n\n".join(sessions_text(q))[:CAP_CHARS]
        a = answer(ctx, q)
        ok, reason = judge(q, a)
        rows.append({"qid": qid, "qtype": q["question_type"], "ok": ok, "a": a, "reason": reason})
        print(f"{qid[:22]:<22} {q['question_type'][:22]:<22} {'OK' if ok else '..'}")
    n = len(rows)
    summ = {"n": n, "oracle_acc": round(sum(r["ok"] for r in rows) / n, 4),
            "by_type": {}}
    for t in sorted({r["qtype"] for r in rows}):
        tr = [r for r in rows if r["qtype"] == t]
        summ["by_type"][t] = round(sum(r["ok"] for r in tr) / len(tr), 3)
    (OUT / "oracle.json").write_text(json.dumps({"summary": summ, "rows": rows}, indent=2))
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
