"""Run a (backend, policy) pair on LoCoMo. Logs traces JSONL + summary JSON."""
from __future__ import annotations
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from .. import config
from ..backends.base import Backend, Unit
from ..backends.flat import FlatBackend
from ..data.locomo import Conversation, QA, load
from ..policies.base import ReadPolicy
from ..policies.v0_naive import NaivePolicy
from ..policies.v1_heuristic import HeuristicPolicy
from .judge import judge
from .metrics import Result, summarise


def conv_to_units(conv: Conversation) -> list[Unit]:
    units = []
    for t in conv.turns:
        # cheap timestamp via session number — good enough for filter ordering
        ts = float(t.session)
        units.append(
            Unit(
                uid=f"{conv.sample_id}::{t.dia_id}",
                text=f"{t.speaker}: {t.text}",
                metadata={
                    "session": t.session,
                    "speaker": t.speaker,
                    "dia_id": t.dia_id,
                    "session_date": t.session_date,
                    "timestamp": ts,
                },
            )
        )
    return units


def evidence_recall(retrieved_uids: list[str], gold_dia_ids: list[str]) -> float:
    if not gold_dia_ids:
        return 0.0
    got = {u.split("::", 1)[-1] for u in retrieved_uids}
    return sum(1 for d in gold_dia_ids if d in got) / len(gold_dia_ids)


def run(
    backend_factory,
    policy: ReadPolicy,
    n_convs: int = 1,
    max_qs_per_conv: int | None = None,
    run_name: str = "smoke",
) -> dict:
    convs = load(n_convs=n_convs)
    out_dir = config.RESULTS_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    traces_path = config.TRACES_DIR / f"{run_name}.jsonl"
    results: list[Result] = []
    t0 = time.time()

    with traces_path.open("w") as ftrace:
        for ci, (conv, qas) in enumerate(convs):
            backend = backend_factory()
            backend.ingest(conv_to_units(conv))
            qas_to_run = qas if max_qs_per_conv is None else qas[:max_qs_per_conv]
            print(f"[conv {ci+1}/{len(convs)}] {conv.sample_id}: ingested {len(conv.turns)} turns, running {len(qas_to_run)} qs")
            for qi, qa in enumerate(qas_to_run):
                trace = policy.run(qa.question, backend)
                trace.qid = qa.qid
                score, reason = judge(qa.question, qa.answer, trace.answer)
                rec = evidence_recall([h.unit.uid for h in trace.final_hits], qa.evidence)
                r = Result(
                    qid=qa.qid,
                    category=qa.category,
                    category_name=qa.category_name,
                    score=score,
                    n_retrievals=trace.n_retrievals,
                    n_steps=len(trace.steps),
                    answer=trace.answer,
                    gold=qa.answer,
                    judge_reason=reason,
                    evidence_recall=rec,
                )
                results.append(r)
                ftrace.write(json.dumps({
                    "qid": qa.qid,
                    "question": qa.question,
                    "gold": qa.answer,
                    "answer": trace.answer,
                    "score": score,
                    "category": qa.category,
                    "policy": policy.name,
                    "backend": backend.name,
                    "steps": [asdict(s) for s in trace.steps],
                    "retrieved_uids": [h.unit.uid for h in trace.final_hits],
                    "evidence_recall": rec,
                }) + "\n")
                if (qi + 1) % 5 == 0:
                    print(f"  q {qi+1}/{len(qas_to_run)}  running_acc={sum(x.score for x in results)/len(results):.3f}")

    summary = summarise(results)
    summary["wall_time_s"] = time.time() - t0
    summary["backend"] = backend.name
    summary["policy"] = policy.name
    summary["n_convs"] = len(convs)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {traces_path}\nWrote {out_dir/'summary.json'}\n")
    print(json.dumps(summary, indent=2))
    return summary


_BACKENDS = {"flat": FlatBackend}
_POLICIES = {"v0_naive": NaivePolicy, "v1_heuristic": HeuristicPolicy}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=list(_BACKENDS), default="flat")
    ap.add_argument("--policy", choices=list(_POLICIES), default="v0_naive")
    ap.add_argument("--n-convs", type=int, default=1)
    ap.add_argument("--max-qs", type=int, default=20, help="0 = all")
    ap.add_argument("--run-name", default="smoke")
    args = ap.parse_args()
    max_qs = None if args.max_qs == 0 else args.max_qs
    run(
        backend_factory=_BACKENDS[args.backend],
        policy=_POLICIES[args.policy](),
        n_convs=args.n_convs,
        max_qs_per_conv=max_qs,
        run_name=args.run_name,
    )


if __name__ == "__main__":
    main()
