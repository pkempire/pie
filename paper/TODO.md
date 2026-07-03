# mempol — full TODO and roadmap

This is the single source-of-truth document for what's left to ship the
paper, what's worth building beyond it, and the open conceptual
questions we want to answer along the way. It's organised in seven
parts: state of the project, critical path to publication,
conceptual deep-dives that change what we'd build, engineering plan to
make this useful to people who aren't us, the personalisation
question, paper-polish tasks, and a risk register.

---

## 1. State of the project, honestly

What is actually built and works:

The training pipeline runs end-to-end. We have `mempol/recipes/memory_rl/{write_env,write_reward,write_tools}.py`,
a per-op counterfactual reward in `mempol/eval/counterfactual.py`,
hybrid retrieval (NER + BM25 + dense + RRF) in `mempol/backends/pie_kg.py`,
and an `ops_log` in `WriteTool` that lets us replay leave-one-out
trajectories deterministically. The most recent Phase B v4 smoke
landed real positive signal: `full_state_score=0.4375`,
`per_op_delta_mean=+0.092`, `per_op_delta_max=+0.3125`,
`counterfactual_reward=+0.247`, `reward/total=+0.301`, `frac_mixed=1.0`.
That run died on JWT expiry at step 2; the checkpoint from step 1 is
saved at `/tmp/mempol/phaseB_v4_cf_20260502_1232/`.

What is built but broken / unused:

Phase A (training the read policy R) has never been run — R has been
hand-coded `HeuristicPolicy(first_k=8, final_k=4)` the entire time.
This is the largest known gap. The PIE baseline runs on disk all
errored out at 0% (`__init__() got an unexpected keyword argument
'content'`); the Mem0 baseline at 27% is the only working comparison
number we have on LoCoMo. There are 10 cached PIE world-models in
`benchmarks/locomo/cache/conv-*_wm.json` (full entities + transitions,
loadable via `WorldModel._load`), but no runner currently consumes
them — every baseline re-extracts.

What we tried and abandoned:

Reward design has burned through evidence-coverage (correlated 0.98
with turn count, non-content-sensitive), reader-overlap (granularity
mismatch), and answer-gain over random-K (one scalar per trajectory,
diffuses). The current reward is the weighted sum
`0.7·counterfactual + 0.3·qa + 0.05·coverage_floor − cost`. The
coverage floor exists because in early training the counterfactual
collapses to zero when R can't answer anything against any variant —
without the floor there's no gradient.

What's in production but undertested:

The hard retention budget `K_max=12`. The chunk windowing (W=6, S=3).
The hybrid retrieval recovers ~75% of the FlatBackend baseline
according to one paired audit. We've never ablated K_max, W, or the
0.7/0.3 reward weights.

---

## 2. Critical path to publish

Ordered by dependency. Each item has a budget, an expected outcome,
and a pass/fail criterion.

### 2.1 Phase A: train R on FlatBackend (~$30, 30 min)

The command from the previous response. The pass criterion is simple:
the trained R has to beat the `HeuristicPolicy` on the held-out 25%
of LoCoMo conversations by at least 5 absolute points of judge score
on multi-hop questions. If it doesn't, we have a more fundamental
problem (probably the prompt template) and Phase B numbers won't be
trustworthy. This is the unblock.

### 2.2 Cached-KG baseline runner (~2 hours of engineering)

Add a `--load_cached_kg` flag to `benchmarks/locomo/runner.py` that
reads `benchmarks/locomo/cache/conv-*_wm.json` directly via
`WorldModel._load` and skips the gpt-4o-mini extraction step. Run the
PIE baseline on those 10 cached KGs and report. This gives us the
"PIE-with-fixed-extraction" number the paper currently leaves blank,
and it's the strongest possible non-learned baseline because it has
already paid the extraction cost we'd otherwise be deferring to RL. If
this number is high (the user remembered ~75%, but disk-saved logs
disagree), the bar moves up substantially. If it's low (~30%), the
cost-of-extraction story we tell in the paper gains weight.

### 2.3 Phase B: train W with R from §2.1 frozen (~$300, ~6 hours)

Once R from §2.1 exists, train W with it as the reader inside
`WriteReward.__call__`. This is the actual headline experiment. Pass
criterion: the trained W beats the random-K efficiency frontier at
matched retention budget by at least 2 absolute points of judge score
across LoCoMo and LongMemEval-S. If it doesn't, we have a paper
without a result. Likely failure mode: R is weak enough that any
write policy looks roughly equally bad to it, so per-op deltas remain
in the ±0.05 range.

### 2.4 Random-K efficiency frontier sweep (~$50, 4 hours)

For each conversation in the held-out set and each `F ∈ {0.1, 0.2,
0.3, 0.5, 0.75, 1.0}`, retain `K = ⌈F·n_turns⌉` uniformly-sampled
turns and run the trained R against them. Report the AUEC. This is
the non-content-aware floor any contribution must clear. Already
scaffolded as `mempol/scripts/random_baseline.py`; just needs to be
run with the trained R.

### 2.5 Mode C: head-to-head vs. KGmem extraction on real data (~$80, 1 day)

Take the lead author's ChatGPT export, generate ~200 self-questions
from it with gpt-4o, then run two write paths on the same chunks:
KGmem's hand-coded extraction pipeline and the trained W. Read both
with the trained R; judge against the self-generated answers. This
is the experiment the paper most needs to publish a result that
generalises beyond the LoCoMo/LongMemEval setup. The paper currently
flags this as TBD; making it a real number is what turns the work
from a benchmark-shaped technique into a deployable claim.

### 2.6 Backend transfer table (~$60, half day)

Train W on FlatBackend, evaluate on FlatBackend / KGBackend /
MastraBackend. The hypothesis is within 5 absolute points across
backends; if the gap is >10 points the "backend-agnostic" claim
collapses and we have to soften that section of the paper.

### 2.7 Five paper-required ablations (~$200, ~1 day if parallelised)

The reader review demanded these and the paper currently has [TBD]
where each should be:

(a) `w_cf` ∈ {0.5, 0.7, 0.9, 1.0} — does the QA anchor matter, and
how much.

(b) `K_max` ∈ {6, 12, 18, 24} — when does the budget stop binding.
Memory-R1 doesn't ablate this; we should.

(c) Cost coefficient `λ_W` ∈ {0, 0.001, 0.01} — does the cost penalty
do real work given that K_max already prunes.

(d) Group size `G` ∈ {4, 8, 16} — variance/throughput tradeoff for
GRPO.

(e) Frozen-R variants: HeuristicPolicy vs. Phase-A-trained vs.
Phase-A-trained-larger-LoRA. Tells us how much W's ceiling depends on
R's ceiling.

### 2.8 TemporalBench v0 (~3 days of engineering, then $40 of judge calls)

Build the six-axis benchmark spec. Each axis gets ~10 hand-authored
scenarios scored by gpt-4o under a TicToc-style pairwise-preference
protocol. We don't need a leaderboard; we need a number the paper
can report so TemporalBench isn't TBD vapor.

---

## 3. Conceptual deep-dives that change the roadmap

### 3.1 The "store everything raw and traverse with tool calls" question

Yes — this is approximately what RLMs and Search-R1 do, and it's a
real design choice we should engage with rather than dismiss.

The argument for it is strong: extraction is lossy and brittle. Mem0's
own paper reports a 40% extraction-failure rate. KGmem's three-tier
resolver was tuned on inspection. Mastra ships a Reflector that runs
once at write time and produces a frozen artefact that downstream
queries read from — if the Reflector got the fact wrong, the system
never recovers. By contrast, "store the raw chunks, let the reader
retrieve aggressively at query time" pushes all the work to test
time, never loses information, and makes the storage layer a literal
flat list rather than a graph.

The argument against it is operational: test-time cost. A 1000-turn
conversation has ~30k tokens of raw text. At inference time, the
reader has to retrieve from that pile every time a question is asked.
If the reader's strategy is "expand neighbours on every miss," cost
scales with question hardness. RLM hits 89.8% on LongMemEval with
Gemini 3 Flash by burning a few thousand additional output tokens per
query in recursive decomposition; this is fine in a personal
assistant where queries are sparse, and ruinous in a high-QPS
deployment.

What this means for our roadmap: we should add a "naïve raw +
tool-traversal" baseline to §2 — store the raw windowed chunks
verbatim, give the reader a `lookup_neighbors(chunk_id, radius=k)`
tool, and run our trained R against it. If this baseline is competitive
with our trained-W story on LongMemEval-S, the paper's contribution
narrows to "we save inference cost at query time at the price of a
write-time training pass," which is honest and still publishable. If
the trained W beats raw-traversal cleanly, that's a stronger story.

### 3.2 Recursive Language Models, in depth

RLMs are simpler than the name suggests. The model isn't recursing in
the call-stack sense — it's emitting code that decomposes its input.
The data structure is a flat string sitting in a Python REPL
environment. The model gets task instructions plus a handle to the
string, and it generates a Python program that slices, processes,
and aggregates. A typical generated program looks like: "for chunk in
split(text, 4096): partial = LLM_call(chunk, query); results.append(partial);
return aggregate(results)." Each "recursion" is just another LLM call
on a smaller chunk. There's no fixed depth budget; the model decides
how many times to slice based on task structure.

What makes RLM work isn't a clever architecture — it's that DSPy's
typed-Pydantic output forces the per-chunk LLM calls to return
structured intermediate results, which the aggregation step can
combine deterministically. Without DSPy, Gemini 3 Flash gets 58% on
LongMemEval; with DSPy scaffolding, 87%. Cost is around $0.01-0.035
per query on LongMemEval, comparable to or cheaper than vanilla
GPT-5 (which often retries on long inputs).

The mempol-relevant insight: **memory is a compute/storage tradeoff,
and the field is bifurcating**. One side (Mem0, Mastra, mempol)
spends compute at write time to compress; the other (RLM, Search-R1)
spends compute at read time to decompress. These are not the same
problem and a write-time policy doesn't help the read-time recursion
strategy. mempol's contribution sits firmly on the write side and
should be presented as such — and an honest paper acknowledges the
RLM line as a complement, not a competitor. The natural next system
is mempol (write side, learned) + RLM (read side, learned), which
we don't yet have but should flag in §7.3 future work.

### 3.3 Classical retrieval ideas worth reconsidering

The standard memory pipeline has converged on dense embeddings + BM25
+ LLM rerank. There's a fairly large pile of classical IR theory
nobody is using because dense embeddings ate the world. Some of it is
worth re-examining specifically because we have token budgets and
want determinism:

**Suffix arrays / FM-index** for substring search. Build once,
O(m + log n) lookups for any literal substring across the entire
conversation. Free for "show me every mention of 'Caroline'" and
analogous keyword queries that BM25 currently approximates. This is
useful as a cheap read-side primitive that complements dense search
without compute cost.

**Locality-sensitive hashing (LSH)** for approximate nearest neighbour
without HNSW build cost. Useful if we want write-time entity
resolution to be more principled than the three-tier cascade.

**Min-hash sketches** for set similarity. If we represent each
conversation chunk as the set of its trigrams, min-hash gives fast
cheap "which chunks are textually similar to this one" without
embeddings. Useful for the merge-entity decision specifically.

**ColBERT-style late interaction.** Each token in a chunk gets its
own vector; retrieval is max-sim per query token against all chunk
tokens. Empirically much better recall than single-vector dense
retrieval on long documents. Worth trying as a read-side replacement
for our current dense-embedding step on the LongMemEval-S setup.

**Reservoir sampling** for budget-bounded retention without scoring.
Trivial to implement, O(1) per turn, and provably optimal under a
uniform-importance prior. A useful negative control: any learned
retention policy must beat reservoir sampling at matched K, which is
a strict efficiency-frontier baseline like random subsampling but
streaming.

**Bloom / cuckoo filters** for "have I seen this entity before?" at
write time without storing the full entity table. Useful when the K
budget bites.

The most actionable item from this list for us right now is **adding
an FM-index pass to read-side retrieval** — it adds a fourth fusion
input to RRF (BM25 + dense + NER + FM-index exact-substring) at near-
zero cost, and exact substring is exactly the failure mode dense
embeddings have on rare proper nouns.

### 3.4 KV cache, in depth

A KV cache is the inference-time memory of a transformer. When the
model generates token N, it needs the Key and Value tensors from
every previous token at every attention layer to compute the next
attention. Recomputing them for each generated token would be O(N²)
total; storing them after the first pass and indexing into them is
O(N). The cache is per-layer, per-head; for a 4B model with 32 layers,
32 heads, 128 head-dim, and a 4096-context the storage is roughly
`32 layers × 2 (K,V) × 4096 tokens × 4096 dim × 2 bytes ≈ 2GB`. That
size is why long-context inference is expensive and why prefix
caching matters.

**Prefix caching** is the practical optimisation. If two queries
share the first N tokens (e.g. the same system prompt, or the same
retrieved memory context), the inference engine can compute the KV
cache for those tokens once and reuse it. vLLM's prefix-caching
feature, OpenAI's prompt-caching API, and Anthropic's prompt-caching
header are all the same idea at different layers of the stack. The
savings are a function of how much of the prompt is shared: 80%
shared → ~80% latency saved on the prefill, plus you avoid hitting
the per-token rate limit.

### 3.5 Ramp Labs / multi-agent KV-cache sharing

Ramp's "Latent Briefing" (April 2026, per the research agent — the
exact post URL needs verification) frames the KV cache as a
shareable memory primitive across multi-agent systems. The setup:
an orchestrator agent breaks a task into sub-tasks and dispatches
worker agents, all of which need the same shared context (the
original task, the project background, the prior coordination
turns). Naively each worker re-prefills the shared prefix. Latent
Briefing computes the prefix's KV cache once at the orchestrator
level, compresses it (asymmetric quantisation: int8 for keys, 3-bit
Lloyd-Max for values), and injects it into each worker via
HuggingFace `DynamicCache` objects. Reported savings: 65% token
reduction on LongBench v2, +3pp accuracy, scales to 15+ workers at
~3× memory savings.

The mempol-relevant insight: **KV-cache sharing is not memory in our
sense, but it's the natural runtime substrate for serving a trained
W policy**. If we ship mempol-base as a LoRA adapter that anyone can
drop into their inference stack, the deployment path is "load the
base model + LoRA once, prefix-cache the system prompt, serve many
users from one weight set." This is what makes the
"facts-in-store, strategy-in-weights" framing economically viable —
the base model + LoRA is cacheable; the per-user store isn't, but it
doesn't need to be.

### 3.6 DeltaMem, in depth

The closest published comparison to mempol. DeltaMem trains a memory
manager with PPO on a per-op reward defined as a Memory-based
Levenshtein Distance — an edit-distance between the current memory
state and a reference memory state, weighted by semantic importance
(entities marked as "critical" by a teacher policy contribute more
to the distance). The op vocabulary is `INSERT(entity, attribute,
value)`, `DELETE`, `MODIFY`, `MERGE`. The architecture is a standard
transformer encoder over conversation + memory state, outputting
logits over (op, target) pairs. The reward at each op step is
proportional to how much that op reduced the distance; if op_i drops
the distance by 50%, the op gets reward 0.5. They report ~15%
absolute improvement in memory coherence over Memory-R1's
trajectory-level reward on LoCoMo/HaluMem/PersonaMem.

The honest comparison to ours: DeltaMem's reward asks "did the op
make the memory closer to a reference state?". Ours asks "did the op
make the reader answer better?". Theirs is cheaper (no replay
needed; you just compute the distance from a precomputed reference)
and has lower variance. Ours is more directly aligned with
downstream answer accuracy because the reader is actually run.
Theirs requires a reference memory state to exist (which DeltaMem
gets from a teacher policy — circular if the teacher is wrong); ours
requires a held-out QA battery to exist (which LoCoMo gives us, but
real deployments don't).

A clean experiment to add: implement DeltaMem-style state-distance
reward as a third reward variant in `WriteReward` and run it head-
to-head with our outcome-attribution reward on LoCoMo. If our reward
beats theirs by >2pp on LongMemEval-S, the paper has a clean
state-distance-vs-outcome-attribution contribution. If we lose, the
paper has to soften.

### 3.7 The personalisation question: generic adapter vs. per-user fine-tune

Two business models, both technically viable:

**(a) mempol-base.** One trained adapter applies to anyone's store.
Like an OS kernel. Anyone with a memory backend + their own
Qwen3-4B can drop in our LoRA. No per-user training. Same adapter
serves a million users. This is what the paper currently claims.

**(b) mempol-personal.** Fine-tune the adapter on each user's own
conversations. Better tail performance, expensive to train, expensive
to serve (one LoRA per user; KV-cache sharing for the base model
helps here, see §3.5).

The honest answer about what we have: we trained on LoCoMo's
two-speaker peer chats, and we have zero evidence the adapter
transfers to one-user-one-assistant data. Mode C in §2.5 is the
experiment that tells us whether (a) is real or not. If mempol-base
doesn't transfer to the lead author's ChatGPT export, the published
claim narrows to "trained on conversation data of style X, useful
for users in style X" — still a contribution, but a smaller one.

The right product story is probably mempol-base as the open-source
default, with mempol-personal as a paid hosted service for orgs that
care about the last few percentage points of accuracy. The cost
asymmetry (1 base model + 1 LoRA serves N users vs. N LoRAs in the
hosted case) makes this a defensible split. The research story is
just: ship mempol-base and the protocol for fine-tuning it; let
others run mempol-personal experiments.

---

## 4. Engineering plan to make this useful to others

The paper publishes a technique. The product needs to publish a
**piece of core engineering people can build with**. Concretely the
delivery surface should be:

**(i) A LoRA artefact** (~50-200MB) on Hugging Face that anyone can
load alongside Qwen3-4B-Instruct. The README is the operating manual:
"this LoRA expects tool-call observations in this format and emits
tool calls in this format. Your job is to wire the tool calls to your
storage backend." That's the contract.

**(ii) A reference backend interface.** A 200-line Python file with
the `Backend` ABC and three reference implementations: `FlatBackend`
(chunk store), `KGBackend` (typed graph), `MastraBackend` (bullet
log). Whatever store the user has, they implement the ABC; the LoRA
runs unchanged.

**(iii) A `mempol-serve` CLI.** One command. It loads the base model,
attaches the LoRA, attaches a backend, opens a chat loop. Drop-in
runtime so people can try the thing without standing up a server.

**(iv) A training recipe** — the existing `mempol/recipes/memory_rl/`
trees, cleaned up, with a `train_personal.py` that takes a directory
of someone's exported conversations and produces their personal LoRA.
This is the open-source path for mempol-personal.

**(v) A small benchmark suite.** Random-K, BM25-heuristic,
KGmem-extraction, mempol-base, mempol-personal (where applicable),
all on LoCoMo + LongMemEval-S + the user's own conversations. Lets
anyone reproduce our claim on their own data.

The deployment-readiness checklist before any of this ships:
**(a)** Phase A trained R passes its bar. **(b)** Phase B trained W
passes the random-K and BM25 bars. **(c)** Mode C shows non-trivial
transfer to ChatGPT-export data. **(d)** A 100-conversation latency
benchmark shows < 500ms median per-turn write decision. **(e)** The
LoRA is serializable with `peft.save_pretrained` and loads cleanly
on stock `transformers`.

---

## 5. Paper polish, with what I need from you

What's already in the paper after the rewrite: the new abstract,
intro with the worked example, related-work 2×2, per-op reward with
the Caroline/LGBTQ-group example before the math, deprecated rewards
in a footnote, chunking finding promoted to its own subsection, tool
errors as a general prescription, conclusion tightened.

What still needs to land before submission:

**Figures.** The current paper has Figure 1 (overview), Figure 2
(KGmem schema), Figure 3 (cotrain alternation), Figure 4 (chunking
windows). All are TikZ. We need three more for the rewrite to
feel right:

(a) **Per-op counterfactual figure** showing the leave-one-out
replay schematically — four boxes for the trajectory, four boxes
showing each leave-one-out variant, arrows from each to the judge,
deltas labelled. This is the figure the worked example in §3.5
points at.

(b) **2×2 placement diagram** — actually drawn, with the cells
populated by the systems we name. Right now this is just a text
table. Drawn with quadrant colours and named arrows showing the
direction of the field's movement (everyone is moving from
hand-coded ops toward learned ops; we're the leftmost outpost in the
learned-ops + KG cell).

(c) **The reward equation as a flowchart.** $r_W = w_{\text{cf}} \cdot
\Sigma\Delta_i + w_{\text{qa}} \cdot \text{score} + w_{\text{cov}}
\cdot \text{cov} - \lambda_W \cdot \text{cost}$, drawn with each
component fed by the right input (counterfactual replays, judge,
coverage scorer, op-counter).

**What I need you to provide or confirm:** I can generate the SVGs
inline from TikZ, but for two cases an actual screenshot would be
better:

- A real `streamlit run mempol/scripts/dashboard.py` screenshot at
the moment when a write trajectory is materialising — showing the KG
view side-by-side with the ops_log. This becomes Figure 5,
captioned "what the trained policy actually does." Take it; drop it
in `paper/figures/dashboard.png`.

- A real LongMemEval-S leaderboard screenshot (or the table from the
LongMemEval paper) — for the table in §2 that places mempol against
the published systems. Once Phase B lands and we have a real number,
we'd add ourselves to that figure.

Let me know if you want me to instead generate those as TikZ stubs;
either works.

---

## 6. Risk register

The four ways this paper could fail to publish, ordered by my
estimate of probability:

**(R1, ~40%)** Phase A's trained R doesn't decisively beat
HeuristicPolicy. Mitigation: try a larger LoRA rank (64 instead of
32), a different base model (Qwen3-7B), or an SFT warm-start from the
heuristic teacher's traces (we already have the SFT data generator).

**(R2, ~30%)** Phase B trains but doesn't beat the random-K
efficiency frontier. The cause would likely be K_max=12 being too
small to give the policy room to differentiate from random. Mitigation:
ablate K_max higher and lower; if 12 is binding, the contribution is
"we hit the random-K floor at small K" which is still publishable but
weaker. Also check whether the per-op deltas remain in the
±0.05-±0.30 range observed in v4; if they collapse back to ~0 in a
longer run, we've over-fit the smoke.

**(R3, ~20%)** Mode C (ChatGPT-export head-to-head) fails. The trained
adapter doesn't transfer to one-user-one-assistant data. Mitigation:
soften the deployment claim, narrow the contribution to LoCoMo-style
peer chats, and add LongMemEval-S as the primary out-of-distribution
target instead of personal data. We can still publish; the framing
just changes from "this generalises" to "this works on conversation
data of the form we trained on."

**(R4, ~10%)** Per-op counterfactual reward is empirically
indistinguishable from a much cheaper trajectory-level reward when
the same total compute is spent. Mitigation: this is the strongest
"the contribution didn't pan out" outcome. If a 5× cheaper
trajectory-level baseline matches our per-op approach at matched
training compute, the paper's core claim collapses. We'd retreat to
"per-op gives interpretability and per-op diagnostics even when
trajectory-level is sample-efficient enough at this scale" — still
true, but not what the title currently says.

---

## 7. Stretch goals (post-paper)

These are the experiments and ideas that don't fit the publication
window but would make the work generalise.

**Multi-turn write episodes.** Per-turn episodes give clean credit
assignment but rule out write strategies that span turns ("defer this,
revisit if it comes up"). Multi-turn episodes with discounted
across-turn reward are murkier but more realistic.

**Joint training of W and R end-to-end** rather than alternating.
Currently we freeze one to train the other. Concurrent reward
gradients would let the system co-evolve faster but might diverge.

**RLM read side + mempol write side.** The natural next system. Ship
mempol-base as the write LoRA; ship a separate read LoRA trained with
RLM-style recursion. Compose at inference time.

**Mode D (next-turn implicit reward).** The user's next message is
the reward signal. No annotation. The path to true label-free
training. We don't have the deployment surface to test this yet, but
the protocol is straightforward — train W to maximise the predictive
likelihood of the user's next turn given the post-W memory state.

**Memory-ops on code repos.** The same op vocabulary should work for
"remember what design decisions I made yesterday" on a coding agent.
The substrate is the file tree + git history + IDE buffer; the ops
are the same. This is the single largest applied direction the
research opens up.

**Architect product integration.** The architect/ subfolder is a
separate product, but its KG of AI components is exactly the kind of
substrate mempol's W policy could maintain — the architect's
knowledge graph grows from web-grounding ingestion, but it could
also grow from the user's own design conversations. Pairing
mempol-write with architect's planner would close the loop on
"learned memory for system design."
