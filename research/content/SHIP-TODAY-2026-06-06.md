# Ship Today — 2026-06-06

## Ship first: temporal-awareness video

This is the cleanest public artifact today.

Claim:

```text
LLM agents can reason about dates, but they do not maintain elapsed time as a live state variable. Flat memory systems inherit that blindness by storing changing human context as timeless facts. A time-aware memory system should store and retrieve state transitions with validity, supersession, and confidence.
```

Use:

- `research/concepts/time-aware-memory.md` as the script.
- `research/content/temporal-awareness-video-runbook.md` as the production checklist.
- `scripts/temporal_memory_demo.py` as demo 1.
- `scripts/rlm_temporal_reconstruction.py` as demo 2.
- Paper screenshots from TicToc, Real-Time Deadlines, Robotouille, GEPA.

Numbers safe to show:

```text
Temporal demo: flat 75%, temporal 100% (+25pp)
RLM reconstruction: flat 67%, RLM 83% (+17pp)
Learned critic toy: Pearson r 0.71, MAE 0.030 with 8 exact deltas
Reflector matrix partial: flat 56.7%, cached PIE KG 70.0%, Mastra 40.0% on 30 conv-26 questions
GEPA tiny result: hand 60%, GEPA 80% on 5 questions
```

Say clearly:

```text
The synthetic demos prove the failure mode, not SOTA.
The GEPA result is promising but tiny.
The full backend/reflector matrix is the next real number.
```

## Do not claim yet

- Do not claim we solved temporal awareness.
- Do not claim GEPA beats all baselines on LoCoMo until `hand_flat` and `gepa_flat` matrix cells complete.
- Do not claim personal PIE data proves the system unless we run a clean export-safe eval.
- Do not lead with per-op counterfactual as the thesis; keep it as a critic-learning subproblem.

## Research artifact to ship second

Short blog / README:

```text
Memory Is Not a Fact. It Is a State Transition.
```

Structure:

1. Flat memory fails on time-sliced questions.
2. Temporal memory annotates validity and supersession.
3. RLM-style read-time reconstruction answers state-at-T.
4. Learned consolidation decides which transitions deserve storage.
5. GEPA/critic/RL are optimization strategies for that policy.

## Code artifact to clean next

Finish the reflector matrix:

```bash
python3 scripts/reflector_backend_matrix.py \
  --max-questions 30 \
  --max-chunks 8 \
  --cells flat_raw,kg_raw,mastra,hand_flat,gepa_flat \
  --model gpt-5-mini
```

Then full run:

```bash
python3 scripts/reflector_backend_matrix.py \
  --max-questions 0 \
  --max-chunks 0 \
  --cells flat_raw,kg_raw,mastra,hand_flat,gepa_flat \
  --model gpt-5-mini
```

If `--max-questions 0` is not implemented as "all", fix that before running.

## Product artifact after the video

Use `scripts/footnote/` on the finished recording:

```bash
python -m scripts.footnote.pipeline /path/to/video.mp4 --output-dir ./footnote_out
```

If the full Footnote pipeline is unstable, do the manual version:

- Premiere transcript.
- Paper screenshots.
- Terminal demos.
- Three title cards.
- One exported guide from the script.

## Repo cleanup after publishing

Commit in chunks:

1. Research wiki/content.
2. Temporal demos/scripts.
3. GEPA/consolidator eval code.
4. Universal memory core.
5. Provider/backend fixes.

Do not commit `paper/main.pdf` as authoritative until the dead counterfactual thesis is rewritten.
