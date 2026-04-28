"""Random-retention efficiency frontier — the must-beat baseline.

Per the Discovery audit (Apr 2026): a learned write policy that does not
strictly outperform random subsampling at matched retention budget on the
LoCoMo and LongMemEval-S benchmarks has not earned its complexity. This
script materialises that frontier so the comparison can be made head-to-head
with whatever learned policy we ship.

Method
======
For each conversation in the eval set:
  for each F in {0.1, 0.2, ..., 1.0}:
    for each repeat r in 1..R:
      sample uniformly K = round(F * n_turns) turns from the conversation
      build a Backend (Flat or KG) populated only with those K turns
      for each q in conv.qa:
        answer = R(q, backend)
        score  = judge(answer, gold)
      record mean score
  aggregate (F, mean_score) → frontier

The result is a CSV per conversation and a JSON summary that the paper's
results table consumes directly.

Usage
=====
    # LoCoMo (2 held-out conversations × 10 retention fractions × 3 reps)
    python -m mempol.scripts.random_baseline \\
        --benchmark locomo --n_convs 2 --reps 3 \\
        --backend flat \\
        --out runs/random_baseline_locomo.json

    # LongMemEval-S (subset of n questions)
    python -m mempol.scripts.random_baseline \\
        --benchmark longmemeval --n_questions 50 --reps 2 \\
        --backend flat \\
        --out runs/random_baseline_lme.json
"""
from __future__ import annotations
import argparse
import json
import logging
import random
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

from mempol.backends.base import Unit
from mempol.backends.flat import FlatBackend
from mempol.backends.pie_kg import PIEBackend
from mempol.data.locomo import load as load_locomo
from mempol.eval.judge import judge as _judge
from mempol.policies.v1_heuristic import HeuristicPolicy

logger = logging.getLogger(__name__)


_RETENTION_FRACTIONS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]


@dataclass
class FrontierPoint:
    F: float
    mean_qa: float
    n_evaluated: int
    n_kept: int


@dataclass
class ConvFrontier:
    sample_id: str
    n_turns: int
    n_qas: int
    points: list[FrontierPoint] = field(default_factory=list)


# ─── Random ingestion ───────────────────────────────────────────────────────
def _conv_to_units(conv) -> list[Unit]:
    """Turn-level units with dia_id metadata. We keep one Unit per turn so
    the F=k/n quantisation is per-turn (matches the LoCoMo evidence
    granularity)."""
    return [
        Unit(
            uid=t.dia_id,
            text=f"{t.speaker}: {t.text}",
            metadata={"dia_ids": [t.dia_id], "speaker": t.speaker,
                       "session": t.session,
                       "session_date": t.session_date,
                       "timestamp": float(t.session)},
        )
        for t in conv.turns
    ]


def _random_sample_units(units: list[Unit], frac: float, rng: random.Random) -> list[Unit]:
    n_keep = max(1, round(frac * len(units)))
    if n_keep >= len(units):
        return list(units)
    return rng.sample(units, n_keep)


def _build_backend(units: list[Unit], backend_kind: str):
    if backend_kind == "flat":
        b = FlatBackend()
        b.ingest(units)
        return b
    if backend_kind == "kg":
        # The KG backend has its own ingest path that converts each Unit
        # into a low-importance EVENT entity. Use this for completeness;
        # for the random-baseline experiment Flat is the cleaner choice
        # since BM25 over the kept turns is the natural strong baseline.
        b = PIEBackend()
        b.ingest(units)
        return b
    raise ValueError(f"unknown backend kind: {backend_kind}")


# ─── Single-conv frontier ───────────────────────────────────────────────────
def conv_frontier(
    conv,
    qas,
    reader: HeuristicPolicy,
    fractions: list[float],
    reps: int,
    backend_kind: str,
    seed: int,
) -> ConvFrontier:
    units = _conv_to_units(conv)
    rng = random.Random(seed)
    out = ConvFrontier(sample_id=conv.sample_id,
                        n_turns=len(units), n_qas=len(qas))
    for F in fractions:
        rep_scores = []
        for r in range(reps):
            local_rng = random.Random(rng.randint(0, 1 << 30))
            kept = _random_sample_units(units, F, local_rng)
            backend = _build_backend(kept, backend_kind)

            qa_scores = []
            for qa in qas:
                trace = reader.run(qa.question, backend)
                ans = trace.answer or "not in context"
                s, _ = _judge(qa.question, qa.answer, ans)
                qa_scores.append(float(s))
            rep_scores.append(sum(qa_scores) / max(len(qa_scores), 1))
        mean_score = sum(rep_scores) / max(len(rep_scores), 1)
        # n_kept in this point: use the LAST repetition's n_kept (all reps same F)
        n_kept = max(1, round(F * len(units)))
        out.points.append(FrontierPoint(
            F=F, mean_qa=mean_score, n_evaluated=len(qas), n_kept=n_kept,
        ))
        logger.info("  conv=%s F=%.2f score=%.3f (kept %d/%d)",
                    conv.sample_id, F, mean_score, n_kept, len(units))
    return out


# ─── Aggregation ────────────────────────────────────────────────────────────
def _aggregate(frontiers: list[ConvFrontier]) -> dict:
    by_F: dict[float, list[float]] = defaultdict(list)
    for fr in frontiers:
        for p in fr.points:
            by_F[p.F].append(p.mean_qa)
    summary = {
        "frontier": {
            f"F={F:.2f}": {
                "mean": sum(scores)/len(scores),
                "min":  min(scores),
                "max":  max(scores),
                "n_convs": len(scores),
            }
            for F, scores in sorted(by_F.items())
        },
        "auec": _trapezoid_auec(by_F),
        "n_convs_total": len(frontiers),
    }
    return summary


def _trapezoid_auec(by_F: dict[float, list[float]]) -> float:
    """Area under efficiency curve via trapezoid rule on (F, mean_qa)."""
    pts = sorted([(F, sum(s)/len(s)) for F, s in by_F.items()])
    if len(pts) < 2:
        return 0.0
    auec = 0.0
    for (F1, y1), (F2, y2) in zip(pts, pts[1:]):
        auec += 0.5 * (y1 + y2) * (F2 - F1)
    return auec


# ─── CLI ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=["locomo", "longmemeval"],
                        default="locomo")
    parser.add_argument("--n_convs", type=int, default=2,
                        help="LoCoMo conversations to evaluate (held-out subset)")
    parser.add_argument("--n_questions", type=int, default=0,
                        help="Cap total questions per conv (0 = no cap)")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--backend", choices=["flat", "kg"], default="flat")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fractions",
                        default=",".join(str(f) for f in _RETENTION_FRACTIONS))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    fractions = [float(x) for x in args.fractions.split(",")]
    reader = HeuristicPolicy(first_k=8, final_k=4,
                              do_reformulate=True, do_expand=True)

    if args.benchmark == "locomo":
        convs = load_locomo(n_convs=args.n_convs)
    else:
        # LongMemEval subset support left as a TODO once the loader returns
        # the full-conv style; the LoCoMo path is the immediate need.
        raise SystemExit("longmemeval frontier path not yet wired; "
                          "run --benchmark locomo for now.")

    frontiers: list[ConvFrontier] = []
    for ci, (conv, qas) in enumerate(convs):
        if args.n_questions:
            qas = qas[: args.n_questions]
        logger.info("[%d/%d] conv=%s n_turns=%d n_qas=%d",
                    ci+1, len(convs), conv.sample_id, len(conv.turns), len(qas))
        fr = conv_frontier(
            conv=conv, qas=qas, reader=reader,
            fractions=fractions, reps=args.reps,
            backend_kind=args.backend,
            seed=args.seed + ci,
        )
        frontiers.append(fr)

    payload = {
        "config": vars(args) | {"out": str(args.out)},
        "summary": _aggregate(frontiers),
        "per_conv": [asdict(fr) for fr in frontiers],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, default=str))
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
