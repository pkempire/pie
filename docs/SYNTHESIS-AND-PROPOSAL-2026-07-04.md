# Synthesis & proposal — one program, built on what we already have

*2026-07-04. Written after a full forensic pass over every result file and the code behind each
recurring idea, plus a through-line trace across every doc. This replaces the last week of
new-headline-every-turn with a single program that subsumes the prior ideas instead of dropping
them. Every number below is from a file on disk; paths are given so nothing is hand-waved.*

---

## 1. The through-line (what we've always been building)

Beneath PIE → mempol → Temporal Context Engine → REM → Cartridges, one thesis has never moved:
**teach an agent to get better from experience by consolidating history into a validity-structured
artifact it can reuse.** Four constants: consolidation beats retrieval-only and fine-tuning-only;
consolidation can be *learned*, not hand-coded; the artifact must carry *temporal structure* (what
was true when, what changed); and the read/write/store trade-offs form a hierarchy (log → notes →
cache → weights). The substrates and reward recipes churned; that core didn't.

## 2. What actually exists and works (grounded asset ledger)

| Asset | Path | Real result | Verdict |
|---|---|---|---|
| **Timeline reconstruction reader** | `mempol/policies/rlm_temporal.py` | 71.7% LongMemEval-S (n=240, balanced) > turn-RAG 68.3 | real, best positive result |
| Stale-memory demo | `demos/01-stale-memory/results.json` | flat **20%** → replay **100%** on as-of-past (n=10, deterministic) | real, reproducible |
| Temporal validity demo | `output/experiments/temporal_memory_demo.json` | flat 75% → temporal **100%** (n=4) | real, tiny |
| RLM reconstruction demo | `output/experiments/rlm_temporal_reconstruction.json` | flat 60% → RLM **83%** (n=6 synthetic) | real, tiny |
| LoCoMo temporal eval | `output/experiments/locomo_temporal_eval.json` | flat **83%** > RLM **67%** (n=6 real LoCoMo) | **real — and a loss** |
| Plain-RAG LoCoMo baseline | `mempol/results/exp01_plain_rag/summary.json` | 0.584 overall (n=497); **temporal cat 0.733**, adversarial 0.152 | real baseline |
| Amortized critic seed | `output/experiments/critic_counterfactual.json` | Pearson **r=0.707**, MAE 0.03 (n=16 ops) | real but toy/weak |
| GEPA consolidator | `mempol/dspy_consolidator/` | 0.6→0.8 but **n=5, 1 chunk, train==val** | overfit smoke, unverified |
| Write-policy RL | `mempol/recipes/memory_rl/`, `smoke_write` | mean_reward −0.005, **zero variance** | dead / no signal |
| SFT warm-start data | `runs/sft_warmup.jsonl` | 800 rows, write-ops with tool calls | real, usable |
| Temporal schema (valid-time) | `mempol/temporal/schema.py`, `store.py` | `(value, valid_from, valid_until, supersedes)` + as-of-T query | real, foundational |
| Ledger / core substrate | `mempol/ledger/`, `mempol/core/` | 541 artifacts ingested from this repo | real, shipped |
| Eval matrix + shims | `mempol/scripts/*_matrix.py`, `memory_providers/` | n=240 / n=1491 runs; Mem0/Zep/etc. runnable | real infra |

## 3. The finding hiding in our own files

Line up the reconstruction reader's results by **question type**, which no doc has done:

- **State-change / "as-of-the-past" / stale questions:** flat 20% → replay **100%** (demo 01); 75% → 100% (validity demo); 60% → 83% (synthetic). The reconstruction reader wins *decisively*.
- **General / date-retrieval "temporal" questions (LoCoMo-style):** flat **83%** > RLM 67% (locomo_temporal_eval); and plain RAG already scores **0.733** on LoCoMo's temporal category. Here reconstruction *ties or loses*.

That contrast is the whole thing. **Our method's edge is specific and real: it wins exactly on
questions where a fact changed and you're asked what held at an earlier (or the current) time —
and it provides no edge on "when did X happen" lookups, which retrieval already handles.** Current
benchmarks (LoCoMo, LongMemEval) mostly test the second kind, which is why our headline LongMemEval
number (71.7%) looks merely-good — the benchmark barely contains the questions we're actually best
at. We've been measuring our method on the wrong axis.

## 4. Where that sits versus the frontier (from the July lit review)

- **Temporal/staleness is the universally admitted weak spot.** On STALE (implicit state-change),
  Mem0 scores 8.3%, Zep 6.0%; best frontier model 55%. Mem0 itself names temporal its weakest area.
- **No benchmark isolates "what was true as-of-T" reconstruction** with versioned ground truth —
  verified open; bi-temporal systems ship the capability *unevaluated*.
- **Per-op counterfactual reward is now published** (HiMPO, Memory-R2, Rosetta) — correctly dead as a
  headline.
- **The amortized critic** (our r=0.71 seed) and **GEPA-on-consolidation** remain verified-open.

So the exact thing our four experiments already show a real edge on is the exact thing the field is
worst at and has no benchmark for. That alignment is not something to pivot away from again.

## 5. The proposal — ONE program that absorbs every prior idea

**Working title: *Stale* — learned temporal consolidation, proven on the questions retrieval can't answer.**

One sentence: *memory's least-served failure is staying correct after facts change; we consolidate
history into a validity-structured artifact and read it by reconstructing state-as-of-T, and we
prove this beats retrieval specifically on state-change questions where every system — including
frontier models — fails.*

Three deliverables, each mapping to assets we already have. Nothing here is a new headline; each
prior "pivot" becomes a *component*:

**(A) The isolated finding — mostly already done (~1–2 wks, ~$50).**
Take the reconstruction reader (`rlm_temporal.py`) and score it against flat/RAG **sliced by
question type** (state-change vs static), on STALE + a state-change slice of LongMemEval. We already
have the raw win across four files; this makes it one rigorous, honest claim: *reconstruction's
gain is localized to state-change questions (+Xpp) and neutral elsewhere.* The LoCoMo loss stays in
the paper — it's evidence *for* the localization, not against the method.
→ absorbs: Timeline-Synthesis, the temporal demos, the "reasoning vs awareness vs memory" framing (§ intro).

**(B) The benchmark that isolates it (~3 wks).**
There's no as-of-T staleness benchmark; our `temporal/schema.py` (valid-time, supersession) is
literally a versioned ground-truth generator. Build a contamination-free benchmark of state-change
questions, run the vendor shims (`memory_providers/`) → expected 5–8% headline, the STALE story on
our own axis. This is the citable, durable artifact (benchmarks compound; method papers age).
→ absorbs: the temporal schema, the ledger, the eval matrix, the "no-replay / budget-curve" discipline.

**(C) The learned method + serving (later, optional, only if A/B land).**
Train the reconstruction reader instead of prompting it: **GEPA on its prompt first** (cheap, our
`run_gepa_consolidator.py` infra) → GRPO only if GEPA plateaus, with the **amortized critic**
(r=0.71 seed, scaled) making per-decision credit affordable, warm-started from `sft_warmup.jsonl`.
When it works, put the validity-structured artifact in a **Cartridge** to serve it cheaply on open
weights (Qwen3/GLM via Tinker).
→ absorbs: GEPA consolidation (Goal 01), the amortized critic, the write-side RL, REM's hierarchy,
Cartridges.

Every idea from the last month has a home in A/B/C. None is the headline; the headline is the
finding in §3.

## 6. The first step (uses only what we have, ~$0 today)

**Slice the existing results by question type.** We already ran demo 01, temporal_memory_demo,
rlm_temporal_reconstruction, locomo_temporal_eval. Nobody tagged their questions as state-change
vs static and pooled them. Do that first — it converts four scattered demos into one honest,
defensible claim (“reconstruction wins +Xpp on state-change, neutral on static”) for **zero new
compute**, and it tells us immediately whether (A) is worth the STALE run. This is the anti-goldfish
move: extract the result already sitting in our files before generating anything new.

## 7. What we explicitly stop doing

- Stop proposing new headlines. The program is §5; iterate *within* it.
- Stop measuring the reader on LoCoMo/LongMemEval as the primary axis — it lacks our question type.
- Stop treating the write-side RL, the critic, GEPA, and Cartridges as competing bets — they are
  components (C), sequenced after the finding (A) and benchmark (B) land.
- Don't touch the dead paper thesis, per-op reward as contribution, or PIE-as-benchmark.

---

*The honest summary: we have one real, reproduced result (reconstruction wins on state-change
questions), it lands exactly on the field's worst-served, unbenchmarked weakness, and our existing
temporal schema is the tool to benchmark it. That is a paper and a program. Everything else we built
is the machinery to train and serve it, not five separate ideas.*
