"""Structured rule accretion — the alternative to whole-prompt mutation (GEPA-style).

Hypothesis: for long-horizon self-improvement, LEARNING SHOULD BE STRUCTURED ACCRETION,
not prompt mutation. Instead of rewriting the writer prompt each iteration (lossy — a bad
rewrite silently destroys prior learning; nothing has provenance), we maintain a typed RULE
LEDGER: each learned behavior is a discrete rule with provenance (the failure cases that
spawned it), a measured train-delta (what admitting it bought), and status (active /
superseded / rejected). The working prompt is COMPILED from base + active rules.

Differences vs evolve_writer (whole-prompt mutation), same splits/seed for head-to-head:
  - additive & typed: base prompts never change; learning lands as auditable rules
  - gated admission: a proposed rule ships only if train accuracy improves (else rejected,
    but STILL RECORDED with its negative delta — failed lessons are data, not lost)
  - revertible: rules can be superseded/deactivated individually
  - ablatable: per-rule credit is measurable later (rule-granular counterfactual)

Run: python -m ctxpack.evolve_rules --iters 4
Artifacts: results/lme_rules/rules.json (the ledger), iter_*.json, compiled prompts.
"""
from __future__ import annotations
import argparse, json, random, time
from pathlib import Path

from ctxpack.run import chat
from ctxpack.lme_pack_eval import DATA, MAP_SYS, REDUCE_SYS
from ctxpack.evolve_writer import eval_prompts, acc

OUT = Path(__file__).resolve().parent / "results" / "lme_rules"

PROPOSE_SYS = (
    "You are improving a memory-compilation system by proposing DISCRETE RULES, not rewriting "
    "prompts. The system compiles a dated multi-session chat history into one budgeted memory "
    "pack (map stage: dated notes per session group; reduce stage: merge into the pack), "
    "question-blind. Below: the currently active learned rules, and failure cases from training "
    "(question, gold, pack-only answer, judge reason). Propose 1-3 NEW rules that would fix "
    "recurring failure patterns. Each rule: one imperative sentence, general (no question-"
    "specific hints), targeted at either the map or reduce stage. You may also supersede an "
    "existing rule by id if it is causing harm. "
    'Output JSON: {"rules": [{"scope": "map"|"reduce", "rule": "...", "rationale": "...", '
    '"supersedes": null|"<rule_id>"}]}.'
)


def compile_prompts(rules: list[dict]) -> tuple[str, str]:
    act_map = [r for r in rules if r["status"] == "active" and r["scope"] == "map"]
    act_red = [r for r in rules if r["status"] == "active" and r["scope"] == "reduce"]
    m = MAP_SYS + ("\n\nLearned rules (apply strictly):\n" +
                   "\n".join(f"- {r['rule']}" for r in act_map) if act_map else "")
    d = REDUCE_SYS + ("\n\nLearned rules (apply strictly):\n" +
                      "\n".join(f"- {r['rule']}" for r in act_red) if act_red else "")
    return m, d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--n-train-per-type", type=int, default=2)
    ap.add_argument("--n-held-per-type", type=int, default=2)
    ap.add_argument("--budget-tokens", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=11)  # SAME as evolve_writer -> same splits
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

    rules: list[dict] = []
    rid = 0
    m, d = compile_prompts(rules)
    tr = eval_prompts(train, m, d, args.budget_tokens)
    ho = eval_prompts(held, m, d, args.budget_tokens)
    base_train = acc(tr)
    history = [{"iter": 0, "train_acc": acc(tr), "held_acc": acc(ho), "n_active_rules": 0}]
    print(f"iter 0: train {acc(tr)*100:.1f}%  held {acc(ho)*100:.1f}%")
    (OUT / "iter_0.json").write_text(json.dumps({"train": tr, "held": ho}, indent=2))

    for it in range(1, args.iters + 1):
        fails = [r for r in tr if not r["ok"]]
        if not fails:
            print("train perfect; stopping"); break
        active_txt = "\n".join(f"[{r['id']}] ({r['scope']}) {r['rule']}"
                               for r in rules if r["status"] == "active") or "(none yet)"
        fb = "\n\n".join(f"Q: {r['q']}\nGold: {r['gold']}\nPack answer: {r['a'][:200]}\n"
                         f"Judge: {r['reason']}" for r in fails[:8])
        raw = chat(PROPOSE_SYS, f"ACTIVE RULES:\n{active_txt}\n\nTRAIN FAILURES:\n{fb}",
                   max_tokens=3000, effort="low")
        try:
            props = json.loads(raw[raw.index("{"): raw.rindex("}") + 1])["rules"][:3]
        except Exception as e:
            print(f"proposal parse failed ({e}); stopping"); break

        # tentatively admit proposals, apply supersessions
        new_ids = []
        for p in props:
            rid += 1
            r = {"id": f"R{rid}", "scope": p.get("scope", "map"), "rule": p["rule"],
                 "rationale": p.get("rationale", ""), "provenance": [f["qid"] for f in fails[:8]],
                 "added_iter": it, "status": "active", "train_delta": None}
            if p.get("supersedes"):
                for old in rules:
                    if old["id"] == p["supersedes"]:
                        old["status"] = "superseded"; old["superseded_by"] = r["id"]
            rules.append(r); new_ids.append(r["id"])

        m, d = compile_prompts(rules)
        t0 = time.time()
        tr_new = eval_prompts(train, m, d, args.budget_tokens)
        delta = acc(tr_new) - acc(tr)
        if delta >= 0:  # ADMIT (>=: keep neutral rules — they may help held-out; recorded either way)
            for r in rules:
                if r["id"] in new_ids:
                    r["train_delta"] = round(delta, 4)
            tr = tr_new
            verdict = "ADMITTED"
        else:  # REJECT but record — failed lessons are data
            for r in rules:
                if r["id"] in new_ids:
                    r["status"] = "rejected"; r["train_delta"] = round(delta, 4)
            m, d = compile_prompts(rules)
            verdict = "REJECTED"
        ho = eval_prompts(held, m, d, args.budget_tokens)
        n_active = sum(1 for r in rules if r["status"] == "active")
        history.append({"iter": it, "train_acc": acc(tr), "held_acc": acc(ho),
                        "n_active_rules": n_active, "proposed": new_ids, "verdict": verdict,
                        "delta": round(delta, 4), "secs": round(time.time() - t0)})
        (OUT / "rules.json").write_text(json.dumps(rules, indent=2))
        (OUT / f"iter_{it}.json").write_text(json.dumps({"train": tr, "held": ho}, indent=2))
        print(f"iter {it}: {verdict} {new_ids} (Δtrain {delta:+.3f}) -> "
              f"train {acc(tr)*100:.1f}%  held {acc(ho)*100:.1f}%  active_rules={n_active}")

    (OUT / "summary.json").write_text(json.dumps(
        {"history": history, "base_train": base_train,
         "note": "rule accretion with gated admission; same seed/splits as evolve_writer"}, indent=2))
    mp, dp = compile_prompts(rules)
    (OUT / "compiled_map_prompt.txt").write_text(mp)
    (OUT / "compiled_reduce_prompt.txt").write_text(dp)
    print(json.dumps(history, indent=1))


if __name__ == "__main__":
    main()
