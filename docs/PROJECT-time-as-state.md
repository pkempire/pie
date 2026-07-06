# Project: Time-as-State — training temporal awareness into agents

*2026-07-04. Concrete plan. The first experiment already ran (demos/02); this is where it goes.*

## Claim
LLM agents are temporally blind — they treat time as passive text, not a state they reason over.
Giving them **time as an explicit, computed state variable** (age of each fact, deadline countdown,
process runtime) and **training the decision policy on it** beats both blind and raw-timestamp
baselines, and beats the 65% ceiling no model has passed on TicToc.

## Why this and not the other ideas (what happened to RESEARCH-FRONTIER-IDEAS)
Those 7 remain valid; #1 (label-free memory) and #2 (learning-to-study) are the bigger *memory*
bets, but each needs infrastructure we'd build from scratch. **This one is runnable now**: real
public data in hand, a **deterministic human-labeled metric** (no LLM judge), a train split nobody
exploited, and an **unsaturated ceiling (<65%)**. It's the temporal strand — the one constant across
every version of this project — made into a concrete, winnable target. Land it, then layer the
memory ideas on top (temporal awareness is the *sensor*; §continual-learning below).

## Data
- **TicToc** (primary — cloned, inspected): 3,630 train / 1,962 test, 50 scenarios across
  time-sensitivity levels, 4-point human preference (tool ↔ direct), `get_metric.py`. Deterministic.
- **Real-Time Deadlines** (arXiv:2601.13206) — negotiation under a deadline (4%→32% with time
  injected). A second, *generative* modality for the same capability.
- **Synthetic timed scenarios** — generate (event stream + timestamps + "as of now, act") for RL
  volume and controllability once the method is set.

## Method (phases; each ends in a number)
- **P0 — reproduce blindness. DONE.** demos/02: blind 41.7% < raw 47.2% < state 52.8% (n=36,
  gpt-5-mini). Raw timestamps barely help; computed freshness-as-state helps most. Direction only.
- **P1 — the time-state representation (~1 wk, ~$30).** Scale P0 to the full test set, multi-seed,
  add: per-fact age, domain volatility tag (derived, not hand-labeled), deadline/runtime where
  present. Report alignment vs blind/raw with CIs, sliced by time-sensitivity level. Deliverable: a
  clean "state > raw > blind" table on real n — the paper's motivating result.
- **P2 — GEPA on the decision prompt (~$50).** Optimize the temporal-reasoning prompt against
  train-split alignment (deterministic reward, our `run_gepa_consolidator.py` machinery repurposed).
  No weight training. Expect most of the gain here.
- **P3 — train the policy (Qwen3 via Tinker, ~$150).** The authors ship DPO scaffolding
  (`dpo_train_hf_margin.py`) and best is still <65%. Do it with the time-state features + GRPO on
  the human-preference reward. **Target: beat 65% test alignment = SOTA on TicToc.**
- **P4 — generalize.** Transfer the trained policy to Real-Time Deadlines (does temporal awareness
  learned on tool-calling transfer to negotiation?) — the generality claim.

## Models / stack
Qwen3-4B/8B via **Tinker** (train, already wired) + **vLLM** (serve). Frontier API for baselines.
Metric is human-labeled → **no LLM judge anywhere**, which kills the reproducibility problem that
sinks most memory papers.

## Target
TicToc normalized alignment. Nobody >65%. A trained, time-state policy that clears it is a clean,
deterministic-metric SOTA + paper — on the exact capability (feeling time) the field has only
measured, never fixed.

## Connection to continual learning (the thread this serves)
Temporal awareness is the **sensor**: an agent that can feel elapsed time is the precondition for
knowing when its own memory has gone stale — which is the trigger for *revising* consolidated
knowledge (the revision problem of continual learning). Ship the sensor first (this project, clean
and winnable); the memory/consolidation ideas (label-free training, revisable consolidation) plug
into it after. One program, sequenced — not another pivot.

## Next action
Run **P1**: full-test, multi-seed, sliced-by-time-sensitivity. That converts the demo's directional
smoke into the motivating result. ~1 day, ~$30.
