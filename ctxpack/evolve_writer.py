"""Writer-policy evolution on LongMemEval — the outer loop (task #11).

Evolves the WRITER PROMPTS (map_sys + reduce_sys) — not any single pack — via reflective
prompt evolution (GEPA-style: reflect on failures in natural language, propose a revised
prompt pair, keep only if train improves). The policy claim: an evolved writer transfers
to UNSEEN haystacks.

Hygiene: reflection sees TRAIN failures only. Held-out is reported per iteration but never
optimized against. Every candidate prompt pair + scores saved to results/lme_writer/.

Run: python -m ctxpack.evolve_writer --iters 4 --n-train-per-type 2 --n-held-per-type 2
Cost: each iteration ≈ (train compile+answer+judge) ≈ 12 q × ~9 calls ≈ $2-3.
"""
from __future__ import annotations
import argparse, json, random, time
from pathlib import Path

from ctxpack.run import chat
from ctxpack.lme_pack_eval import (DATA, MAP_SYS, REDUCE_SYS, compile_pack, answer, judge)

OUT = Path(__file__).resolve().parent / "results" / "lme_writer"

REFLECT_SYS = (
    "You are improving the prompts of a memory-compilation system. It compiles a long, dated, "
    "multi-session chat history into ONE budgeted memory pack (map prompt extracts dated notes "
    "per session group; reduce prompt merges them). The pack is compiled WITHOUT seeing any "
    "question; questions arrive later. Below are failure cases: question, gold answer, the "
    "pack-only answer given, and the judge's reason. Diagnose the recurring failure patterns "
    "(e.g. which kinds of facts the writer drops, how dates/changes are recorded, what gets "
    "crowded out) and produce a REVISED map prompt and REVISED reduce prompt. Keep both concise "
    "and general — no question-specific hints, no dataset names. The reduce prompt must keep its "
    "{budget} and {chars} placeholders. "
    'Output JSON: {"map_sys": "...", "reduce_sys": "...", "diagnosis": "..."}.'
)


def _eval_one(q: dict, map_sys: str, reduce_sys: str, budget: int) -> dict:
    try:
        pack = compile_pack(q, budget, map_sys=map_sys, reduce_sys=reduce_sys)
        a = answer(pack, q)
        ok, reason = judge(q, a)
    except Exception as e:
        pack, a, ok, reason = "", f"[error {e}]", False, "error"
    return {"qid": q["question_id"], "qtype": q["question_type"], "q": q["question"],
            "gold": q["answer"], "a": a, "ok": ok, "reason": reason, "pack_head": pack[:800]}


def eval_prompts(qs: list[dict], map_sys: str, reduce_sys: str, budget: int,
                 workers: int = 5) -> list[dict]:
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(lambda q: _eval_one(q, map_sys, reduce_sys, budget), qs))


def acc(rows): return sum(r["ok"] for r in rows) / max(1, len(rows))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--n-train-per-type", type=int, default=2)
    ap.add_argument("--n-held-per-type", type=int, default=2)
    ap.add_argument("--budget-tokens", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    data = [q for q in json.load(open(DATA)) if not str(q["question_id"]).endswith("_abs")]
    random.seed(args.seed)
    by: dict[str, list] = {}
    for q in data:
        by.setdefault(q["question_type"], []).append(q)
    train, held = [], []
    for t, qs in sorted(by.items()):
        random.shuffle(qs)
        train += qs[: args.n_train_per_type]
        held += qs[args.n_train_per_type: args.n_train_per_type + args.n_held_per_type]
    print(f"train={len(train)} held={len(held)} across {len(by)} types")

    cur = {"map_sys": MAP_SYS, "reduce_sys": REDUCE_SYS}
    history = []
    best = None
    for it in range(args.iters + 1):
        t0 = time.time()
        tr = eval_prompts(train, cur["map_sys"], cur["reduce_sys"], args.budget_tokens)
        ho = eval_prompts(held, cur["map_sys"], cur["reduce_sys"], args.budget_tokens)
        rec = {"iter": it, "train_acc": acc(tr), "held_acc": acc(ho),
               "prompts": dict(cur), "train": tr, "held": ho, "secs": round(time.time() - t0)}
        history.append(rec)
        (OUT / f"iter_{it}.json").write_text(json.dumps(rec, indent=2))
        print(f"iter {it}: train {acc(tr)*100:5.1f}%  held {acc(ho)*100:5.1f}%  ({rec['secs']}s)")
        if best is None or acc(tr) > best["train_acc"] or (acc(tr) == best["train_acc"] and it == 0):
            best = rec
        if it == args.iters:
            break
        fails = [r for r in tr if not r["ok"]]
        if not fails:
            print("train perfect; stopping early"); break
        fb = "\n\n".join(f"Q: {r['q']}\nGold: {r['gold']}\nPack answer: {r['a'][:200]}\n"
                         f"Judge: {r['reason']}" for r in fails[:8])
        raw = chat(REFLECT_SYS, f"CURRENT MAP PROMPT:\n{cur['map_sys']}\n\nCURRENT REDUCE PROMPT:\n"
                                f"{cur['reduce_sys']}\n\nFAILURES (train only):\n{fb}",
                   max_tokens=4000, effort="low")
        try:
            prop = json.loads(raw[raw.index("{"): raw.rindex("}") + 1])
            assert "{budget}" in prop["reduce_sys"] and "{chars}" in prop["reduce_sys"]
            cur = {"map_sys": prop["map_sys"], "reduce_sys": prop["reduce_sys"]}
            (OUT / f"diagnosis_{it}.txt").write_text(prop.get("diagnosis", ""))
        except Exception as e:
            print(f"reflection parse failed ({e}); keeping current prompts")

    summary = {"iters": [{k: r[k] for k in ("iter", "train_acc", "held_acc")} for r in history],
               "best_iter_by_train": best["iter"], "budget_tokens": args.budget_tokens,
               "note": "held-out reported, never optimized; single seed"}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["iters"], indent=1))


if __name__ == "__main__":
    main()
