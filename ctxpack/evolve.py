"""ctxpack evolve — artifact-level optimization: evolve the PACK against blame-decomposed feedback.

Inner loop of the bilevel design (outer loop = GEPA over writer/reader prompts):
  round k: eval(pack_k on TRAIN) -> per-question blame (fact MISSING from pack = writer fault;
  fact PRESENT but answer wrong = organization/reader fault) -> reviser rewrites pack_k -> pack_{k+1}

Hygiene: feedback sees ONLY train questions (never accept-patterns, never held-out items).
Held-out accuracy is the reported number. Each round's full pack is saved to
ctxpack/results/evolution/ for inspection.

Run: python -m ctxpack.evolve [--rounds 3] [--budget-tokens 4000]
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path

from ctxpack.run import (REPO, chat, load_corpus, handwritten_context, answer, score,
                         CHARS_PER_TOKEN)

TRAIN_IDS = {"reward-weights", "non-mutating-ops", "judge-bucket",
             "counterfactual-mech", "group-size", "embed-cache-key"}

MAP_SYS = (
    "You are compiling source notes for a codebase knowledge pack. From the files below, extract "
    "the facts most useful for answering precise questions: default constant values, class/function "
    "names and roles, environment variables, reward/weight/config values, tool/op names, and each "
    "component's mechanism in one line. Dense bullets. Prefix EVERY fact with its source file path "
    "in the form [src: path]."
)
REVISE_SYS = (
    "You maintain a context pack (max {chars} characters) that is the ONLY context available for "
    "answering precise questions about a codebase. You get: the current pack, evaluation feedback "
    "(failed questions, each tagged MISSING-FROM-PACK — the needed fact is absent, find it in the "
    "source notes and add it — or PRESENT-BUT-FAILED — the fact is in the pack but buried or badly "
    "stated; restate it plainly and prominently), and the source notes. Revise the pack: add what's "
    "missing with exact values/identifiers, clarify what's buried, cut the lowest-value content to "
    "stay in budget. Keep [src: path] anchors on every section. Output ONLY the revised pack."
)


def fact_in_pack(pack: str, accept: list[list[str]]) -> bool:
    p = pack.lower()
    return all(any(re.search(alt, p) for alt in alts) for alts in accept)


def evaluate(pack: str, tasks: list[dict]) -> list[dict]:
    rows = []
    for t in tasks:
        a = answer(pack, t["q"])
        rows.append({"id": t["id"], "q": t["q"], "a": a, "ok": score(a, t["accept"]),
                     "fact_in_pack": fact_in_pack(pack, t["accept"])})
    return rows


def acc(rows: list[dict]) -> float:
    return sum(r["ok"] for r in rows) / max(1, len(rows))


def feedback(rows: list[dict]) -> str:
    lines = []
    for r in rows:
        if r["ok"]:
            continue
        blame = "PRESENT-BUT-FAILED" if r["fact_in_pack"] else "MISSING-FROM-PACK"
        lines.append(f"- [{blame}] Q: {r['q']}\n  pack-only answer given: {r['a'][:160]}")
    return "\n".join(lines) if lines else "(all train questions passed)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--budget-tokens", type=int, default=4000)
    ap.add_argument("--tasks", default="ctxpack/tasks/mempol.jsonl")
    ap.add_argument("--init-pack", default="ctxpack/results/mempol_b4000_v2_pack.md")
    args = ap.parse_args()

    spec = [json.loads(l) for l in (REPO / args.tasks).read_text().splitlines() if l.strip()]
    meta, tasks = spec[0], spec[1:]
    train = [t for t in tasks if t["id"] in TRAIN_IDS]
    heldout = [t for t in tasks if t["id"] not in TRAIN_IDS]
    budget_chars = args.budget_tokens * CHARS_PER_TOKEN
    outdir = REPO / "ctxpack" / "results" / "evolution"
    outdir.mkdir(parents=True, exist_ok=True)

    # source notes (with [src] anchors), cached
    notes_f = outdir / "source_notes.md"
    if notes_f.exists():
        notes = notes_f.read_text()
    else:
        files = load_corpus(meta["corpus_globs"])
        groups, cur, used = [], [], 0
        for name, text in files:
            t = text[:30_000]
            if used + len(t) > 90_000 and cur:
                groups.append("\n\n".join(cur)); cur, used = [], 0
            cur.append(f"### FILE {name}\n{t}"); used += len(t)
        if cur:
            groups.append("\n\n".join(cur))
        notes = "\n\n---\n\n".join(chat(MAP_SYS, g, max_tokens=6000, effort="minimal") for g in groups)
        notes_f.write_text(notes)

    pack = (REPO / args.init_pack).read_text()[:budget_chars]
    hand = handwritten_context(meta["handwritten_docs"], budget_chars)

    history = []
    hand_tr, hand_ho = evaluate(hand, train), evaluate(hand, heldout)
    print(f"handwritten baseline   train {acc(hand_tr)*100:5.1f}%   heldout {acc(hand_ho)*100:5.1f}%")

    for k in range(args.rounds + 1):
        tr, ho = evaluate(pack, train), evaluate(pack, heldout)
        (outdir / f"pack_r{k}.md").write_text(pack)
        history.append({"round": k, "train_acc": acc(tr), "heldout_acc": acc(ho),
                        "pack_chars": len(pack),
                        "train": tr, "heldout": ho})
        print(f"round {k}: train {acc(tr)*100:5.1f}%   heldout {acc(ho)*100:5.1f}%   pack {len(pack):,} chars")
        if k == args.rounds:
            break
        fb = feedback(tr)
        pack = chat(
            REVISE_SYS.format(chars=budget_chars),
            f"CURRENT PACK:\n{pack}\n\nEVALUATION FEEDBACK (train questions only):\n{fb}\n\n"
            f"SOURCE NOTES:\n{notes[:180_000]}",
            max_tokens=args.budget_tokens * 2 + 4000,
        )[:budget_chars]

    (outdir / "history.json").write_text(json.dumps(
        {"budget_tokens": args.budget_tokens,
         "handwritten": {"train_acc": acc(hand_tr), "heldout_acc": acc(hand_ho)},
         "rounds": history}, indent=2))
    print(f"saved packs + history -> {outdir.relative_to(REPO)}/")


if __name__ == "__main__":
    main()
