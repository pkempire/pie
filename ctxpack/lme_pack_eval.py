"""ctxpack on LongMemEval-S — the bigger experiment, with full trace logging.

Fair memory setting: the pack is compiled from the haystack BEFORE the question is seen
(generic task-distribution hint only). Conditions at matched token budget:

  turn-rag : lexical top-turns for the question, packed to budget (query-adaptive baseline)
  pack     : dated, timeline-structured compile of the full haystack (map/reduce), question-blind

Scoring: LLM judge (LongMemEval answers are free-form) — logged with reasons; this is NOT
deterministic scoring and is labeled as such. Every question writes a full trace row
(pack, contexts, answers, judge reason) to ctxpack/results/lme/traces.jsonl.

Run:  python -m ctxpack.lme_pack_eval --n-per-type 5 --budget-tokens 4000
Cost: ~$3-5 for n=30 (gpt-5-mini). Resumable: skips qids already in traces.jsonl.
"""
from __future__ import annotations
import argparse, json, random, re, time
from pathlib import Path

from ctxpack.run import REPO, chat

DATA = REPO / "benchmarks/longmemeval/data/longmemeval_s_cleaned.json"
OUT = REPO / "ctxpack/results/lme"

MAP_SYS = (
    "You are compiling memory notes from a user's chat history with an assistant. From the dated "
    "sessions below, extract dense bullet notes of everything durable about THE USER: facts, "
    "preferences, events, plans, and states — each with its session date. When something CHANGES "
    "across sessions, record it as a dated transition (was X -> became Y on <date>), not just the "
    "latest value. Tag each bullet [date]. Ignore assistant knowledge, puzzles, and generic chit-chat."
)
REDUCE_SYS = (
    "Merge these dated notes into ONE memory pack of AT MOST {budget} tokens (~{chars} chars). "
    "Organize per entity/topic as a dated timeline: keep transitions (was->became on date), current "
    "state last, exact names/numbers/dates. The pack will be the ONLY context to answer questions "
    "about the user's facts, preferences, changes over time, and multi-session history. Use the "
    "budget fully; do not stop early."
)
ANSWER_SYS = (
    "You answer questions about a user from memory context. Today's date is given. Use ONLY the "
    "context. For 'as of' or change questions, use the dated transitions. Be concise and specific. "
    "If the context lacks the answer, say you don't know."
)
JUDGE_SYS = (
    "Judge whether the model answer is correct given the gold answer. Minor phrasing differences "
    'are fine; the key fact(s) must match. JSON: {"correct": true|false, "reason": "<short>"}.'
)


def _tok(s: str) -> list[str]:
    return re.findall(r"[a-z0-9_']+", s.lower())


def sessions_text(q: dict) -> list[str]:
    out = []
    for date, sess in zip(q["haystack_dates"], q["haystack_sessions"]):
        turns = "\n".join(f"{m['role']}: {m['content']}" for m in sess if m.get("content"))
        out.append(f"=== session {date} ===\n{turns}")
    return out


def compile_pack(q: dict, budget_tokens: int, map_sys: str = MAP_SYS,
                 reduce_sys: str = REDUCE_SYS) -> str:
    # Deterministic per policy: cache the compiled pack by (qid, prompts, budget).
    # Kills within-policy compile stochasticity — paired comparisons become stable —
    # and makes re-evals free. (Variance finding: identical config scored 33/67/75%
    # held-out across three uncached runs.)
    import hashlib
    key = hashlib.sha1(f"{q['question_id']}|{map_sys}|{reduce_sys}|{budget_tokens}".encode()).hexdigest()[:20]
    cache = OUT / "pack_cache" / f"{key}.md"
    if cache.exists():
        return cache.read_text()
    groups, cur, used = [], [], 0
    for s in sessions_text(q):
        s = s[:24_000]
        if used + len(s) > 90_000 and cur:
            groups.append("\n\n".join(cur)); cur, used = [], 0
        cur.append(s); used += len(s)
    if cur:
        groups.append("\n\n".join(cur))
    notes = [chat(map_sys, g, max_tokens=6000, effort="minimal") for g in groups]
    budget_chars = budget_tokens * 4
    pack = chat(reduce_sys.format(budget=budget_tokens, chars=budget_chars),
                "\n\n---\n\n".join(notes), max_tokens=budget_tokens * 2 + 4000)[:budget_chars]
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(pack)
    return pack


def rag_context(q: dict, budget_chars: int) -> str:
    qt = set(_tok(q["question"]))
    turns = []
    for date, sess in zip(q["haystack_dates"], q["haystack_sessions"]):
        for m in sess:
            if m.get("content"):
                turns.append((date, f"{m['role']}: {m['content']}"))
    scored = sorted(turns, key=lambda t: -len(qt & set(_tok(t[1]))))
    out, used = [], 0
    for date, t in scored:
        t = t[:3000]
        if used + len(t) > budget_chars:
            break
        out.append(f"[{date}] {t}"); used += len(t)
    return "\n".join(out)


def answer(ctx: str, q: dict) -> str:
    return chat(ANSWER_SYS, f"Today: {q['question_date']}\n\nMemory context:\n{ctx}\n\n"
                            f"Question: {q['question']}", max_tokens=1200, effort="minimal")


def hybrid_answer(pack: str, q: dict, budget_chars: int) -> tuple[str, bool]:
    """Pack-first; escalate to targeted raw-haystack retrieval only if the pack can't answer.

    Returns (answer, escalated). Escalation budget = same total budget: half pack, half raw hits,
    so the hybrid never sees more tokens than either pure condition.
    """
    a = answer(pack, q)
    low = a.lower()
    if not any(s in low for s in ("don't know", "do not know", "not in", "no information",
                                  "cannot", "unsure", "not mentioned")):
        return a, False
    raw = rag_context(q, budget_chars // 2)
    ctx = pack[: budget_chars // 2] + "\n\n=== retrieved raw turns (escalation) ===\n" + raw
    return answer(ctx, q), True


def _judge_once(q: dict, pred: str) -> tuple[bool, str]:
    raw = chat(JUDGE_SYS, f"Question: {q['question']}\nGold: {q['answer']}\nModel answer: {pred}",
               max_tokens=900, effort="minimal")
    m = re.search(r'"correct"\s*:\s*(true|false)', raw.lower())
    reason = (re.search(r'"reason"\s*:\s*"([^"]*)"', raw) or [None, ""])[1]
    return (m.group(1) == "true" if m else False), reason


def judge(q: dict, pred: str) -> tuple[bool, str]:
    # Triple-judge majority: single LLM judges flip verdicts run-to-run (measured here and
    # documented field-wide, e.g. the Coin-Flip-Judge result). Majority of 3 stabilizes.
    votes = [_judge_once(q, pred) for _ in range(3)]
    ok = sum(v for v, _ in votes) >= 2
    reason = next((r for v, r in votes if v == ok and r), votes[0][1])
    return ok, f"{sum(v for v, _ in votes)}/3 · {reason}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-type", type=int, default=5)
    ap.add_argument("--budget-tokens", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    traces_f = OUT / "traces.jsonl"
    done = {json.loads(l)["qid"] for l in traces_f.read_text().splitlines()} if traces_f.exists() else set()

    data = json.load(open(DATA))
    data = [q for q in data if not str(q["question_id"]).endswith("_abs")]
    random.seed(args.seed)
    by_type: dict[str, list] = {}
    for q in data:
        by_type.setdefault(q["question_type"], []).append(q)
    sample = []
    for t, qs in sorted(by_type.items()):
        random.shuffle(qs)
        sample += qs[: args.n_per_type]
    print(f"sample: {len(sample)} questions across {len(by_type)} types; {len(done)} already done")

    budget_chars = args.budget_tokens * 4
    for q in sample:
        if q["question_id"] in done:
            continue
        t0 = time.time()
        row = {"qid": q["question_id"], "qtype": q["question_type"], "q": q["question"],
               "gold": q["answer"], "budget_tokens": args.budget_tokens}
        try:
            pack = compile_pack(q, args.budget_tokens)
            a_pack = answer(pack, q)
            ok_pack, r_pack = judge(q, a_pack)
            rag = rag_context(q, budget_chars)
            a_rag = answer(rag, q)
            ok_rag, r_rag = judge(q, a_rag)
            row.update({"pack": pack, "a_pack": a_pack, "ok_pack": ok_pack, "judge_pack": r_pack,
                        "rag_chars": len(rag), "a_rag": a_rag, "ok_rag": ok_rag, "judge_rag": r_rag,
                        "secs": round(time.time() - t0, 1)})
        except Exception as e:
            row["error"] = str(e)[:300]
        with open(traces_f, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"{q['question_id'][:20]:<20} {q['question_type'][:22]:<22} "
              f"pack:{'OK' if row.get('ok_pack') else '..'} rag:{'OK' if row.get('ok_rag') else '..'} "
              f"({row.get('secs','?')}s)")

    rows = [json.loads(l) for l in traces_f.read_text().splitlines()]
    rows = [r for r in rows if "error" not in r]
    n = len(rows)
    # cost accounting (tokens ~ chars/4). Answer-time context is budget-matched by design;
    # the asymmetry is amortization: the pack is compiled ONCE per corpus and is cache-stable
    # (same prefix every query -> prompt-cache rates); RAG re-retrieves and re-reads a fresh,
    # uncacheable context every query.
    pack_tok = [len(r.get("pack", "")) // 4 for r in rows]
    rag_tok = [r.get("rag_chars", 0) // 4 for r in rows]
    summ = {"n": n,
            "pack_acc": round(sum(r["ok_pack"] for r in rows) / max(1, n), 4),
            "rag_acc": round(sum(r["ok_rag"] for r in rows) / max(1, n), 4),
            "cost": {
                "answer_ctx_tokens_pack_mean": round(sum(pack_tok) / max(1, n)),
                "answer_ctx_tokens_rag_mean": round(sum(rag_tok) / max(1, n)),
                "pack_compile_secs_mean": round(sum(r.get("secs", 0) for r in rows) / max(1, n), 1),
                "pack_amortization": "compile once per corpus; cache-stable prefix at answer time",
                "rag_amortization": "no compile; per-query retrieval + uncacheable fresh context",
            },
            "by_type": {}}
    for t in sorted({r["qtype"] for r in rows}):
        tr = [r for r in rows if r["qtype"] == t]
        summ["by_type"][t] = {"n": len(tr),
                              "pack": round(sum(r["ok_pack"] for r in tr) / len(tr), 3),
                              "rag": round(sum(r["ok_rag"] for r in tr) / len(tr), 3)}
    (OUT / "summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
