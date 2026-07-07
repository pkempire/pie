"""Mastra OM exact prompts as the hand-tuned baseline condition (Q2).

Uses the REAL Observer/Reflector prompts from mempol/backends/mastra.py (the April repro of
Mastra's published 94.87% LongMemEval setup), expressed through the same map/reduce compile
path and evaluated on the SAME n=30 sample (seed 7) as the baseline run. This is the
"hand-tuned SOTA writer" row of the learned-vs-hand-tuned table.

Run: python -m ctxpack.mastra_eval
"""
from __future__ import annotations
import json, random
from pathlib import Path

from ctxpack.lme_pack_eval import DATA
from ctxpack.evolve_writer import eval_prompts, acc
from mempol.backends.mastra import _OBSERVER_SYS, _REFLECTOR_SYS

OUT = Path(__file__).resolve().parent / "results" / "lme_mastra"


def esc(s: str) -> str:
    return s.replace("{", "{{").replace("}", "}}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = [q for q in json.load(open(DATA)) if not str(q["question_id"]).endswith("_abs")]
    random.seed(7)  # SAME sample as the n=30 baseline
    by: dict[str, list] = {}
    for q in data:
        by.setdefault(q["question_type"], []).append(q)
    sample = []
    for t, qs in sorted(by.items()):
        random.shuffle(qs)
        sample += qs[:5]

    map_sys = esc(_OBSERVER_SYS)
    reduce_sys = (esc(_REFLECTOR_SYS) +
                  "\n\nHard limit: the condensed observational memory must fit in {budget} tokens "
                  "(~{chars} characters). Use the budget fully; keep dated, prioritized entries.")
    rows = eval_prompts(sample, map_sys, reduce_sys, budget=4000, workers=5)
    by_t: dict[str, list] = {}
    for r in rows:
        by_t.setdefault(r["qtype"], []).append(r)
    summ = {"n": len(rows), "mastra_acc": round(acc(rows), 4),
            "by_type": {t: {"n": len(v), "mastra": round(acc(v), 3)} for t, v in sorted(by_t.items())},
            "note": "Mastra OM Observer/Reflector prompts via same compile path, same sample/budget as baseline"}
    (OUT / "rows.json").write_text(json.dumps(rows, indent=2))
    (OUT / "summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
