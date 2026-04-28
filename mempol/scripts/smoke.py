"""End-to-end smoke test: load → ingest → policy → judge. Should run < 30s."""
from __future__ import annotations
import os
import sys

assert os.environ.get("OPENAI_API_KEY"), "set OPENAI_API_KEY"

from mempol.data.locomo import load
from mempol.backends.flat import FlatBackend
from mempol.policies.v0_naive import NaivePolicy
from mempol.policies.v1_heuristic import HeuristicPolicy
from mempol.eval.judge import judge
from mempol.eval.runner import conv_to_units


def main():
    convs = load(n_convs=1)
    conv, qas = convs[0]
    print(f"loaded {conv.sample_id}: {len(conv.turns)} turns, {len(qas)} qs")

    b = FlatBackend()
    b.ingest(conv_to_units(conv))
    print(f"ingested into {b.name}, units={len(b.units)}")

    qa = qas[0]
    for pol_cls in (NaivePolicy, HeuristicPolicy):
        p = pol_cls()
        t = p.run(qa.question, b)
        s, _ = judge(qa.question, qa.answer, t.answer)
        print(f"  {p.name:14s} steps={len(t.steps)}  ans={t.answer!r:50s} score={s}")
    print("OK")


if __name__ == "__main__":
    main()
