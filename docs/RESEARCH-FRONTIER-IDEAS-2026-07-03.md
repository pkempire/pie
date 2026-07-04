# Frontier research directions — the hard problems, mid-2026

*Stripped of this repo's jargon. Seven directions aimed at the deepest open problems in memory /
continual learning / RL, each with: the gap and why it's genuinely hard, a concrete new method,
the main novel contribution, why it matters, the first experiment, an honest novelty check
(what's adjacent), and the outcome if it lands. Ranked by breakthrough potential × defensibility.*

The organizing belief behind all of them: **the field has spent two years making agents that
recall, and almost no effort making agents that get better. The next citations go to whoever
makes "learning from experience" a trainable, measurable, label-free thing.**

---

## 1. Memory you can train without labels — the predictive-sufficiency objective ★ top pick

**Paper title.** *What Should an Agent Remember? Learning Memory from the Next Interaction.*

**The gap (why it's hard).** Every memory system today is trained or tuned against a QA benchmark.
But (a) deployments have no QA labels, (b) the benchmarks are broken (LoCoMo's key is 6% wrong,
judges accept 63% of wrong answers), and (c) a policy tuned on one QA distribution doesn't transfer.
So the single thing blocking learned memory at scale is: **there is no label-free signal for
whether a memory write was good.** Everyone assumes you need downstream questions.

**The new method.** You don't. The user's *next interaction* is a free, abundant, un-gameable
label. Frame memory as an information bottleneck: the memory state `M` should be the **minimal
sufficient statistic** of history `H` for predicting future interactions `F` — maximize `I(M;F)`
(predictive sufficiency) while minimizing `I(M;H)` (compression). Operationally: compress history
into a bounded state, then reward the write/consolidation policy by how much `M` improves the
model's likelihood of (or reward on) the *actually-observed* future turns, versus a no-memory and
a full-history baseline. Train the policy — prompt-level first (cheap), weights-level second —
purely on raw interaction logs. No questions, no judge, no labels.

**Main novel contribution.** A **label-free training objective and recipe for memory** grounded in
predictive sufficiency, turning any interaction log into memory-training data. It converts the
eval crisis and the label bottleneck from blockers into non-issues.

**Why it matters / outcomes.** Every company sitting on chat/agent logs (i.e., all of them) could
train memory on their own data with zero annotation. It sidesteps the entire broken-benchmark
mess. If it works, it becomes *the* default way memory gets trained — the "next-token prediction"
moment for memory: a self-supervised objective that scales with data instead of labels.

**First experiment.** On a multi-session conversation corpus, train consolidation to maximize
held-out next-turn likelihood; then evaluate the resulting memory on QA it *never trained on*. The
headline plot: label-free memory beats importance-heuristic and even QA-tuned consolidation on
out-of-distribution questions.

**Honest novelty check.** A rate-distortion *framing* of memory exists ("Remember the Decision, Not
the Description," 2605.10870); next-turn prediction is an ancient LM objective; sleep-time compute
precomputes future-relevant inferences. What is not published: *training the memory write/
consolidation policy against future-interaction predictive likelihood as the label-free reward,
with an explicit minimal-sufficient-statistic objective.* Position it as the method that makes the
rate-distortion framing trainable.

**Risk.** Medium. The signal may be noisy; needs the right baselines to isolate memory's
contribution from the model just being good. But even a partial result is highly publishable.

---

## 2. Learning to study — self-generated curricula for autonomous domain mastery ★

**Paper title.** *Learning to Study: Self-Generated Curricula for Acquiring Expertise from a Corpus.*

**The gap.** The Machine Studying result (Jacob Li, 2026) is the sharpest recent finding: given a
corpus, retrieval ≠ expertise, fine-tuning fails, and the only thing that worked was a hand-built
"cheatsheet." But a cheatsheet is a fixed heuristic. The real skill — the one humans have — is
knowing *how to study*: what to focus on, what to quiz yourself on, what to work through. Nobody
has made *studying itself* a learned policy.

**The new method.** A studying policy that, given a corpus and a compute budget, **generates its own
practice tasks** (questions, exercises, worked problems), attempts them, critiques itself, and
writes an expertise artifact — with the entire loop optimized by a single reward: **held-out
expertise gain** (accuracy-per-compute on real downstream tasks it never saw during study). The
self-generated curriculum *is* the action space; the meta-objective is transfer. Optimize with RL /
meta-gradient over which questions to generate and what to retain.

**Main novel contribution.** Recasting continual learning as **learning a curriculum-generation
policy graded by downstream transfer** — an agent that gets better at *getting better*, with no
teacher and no labels on the target task.

**Why it matters / outcomes.** This is autonomous onboarding: point an agent at a new codebase,
a new API, a new field's papers, and it masters it overnight. That is the single most valuable
applied capability in agents right now (every "make Cursor/Devin learn our codebase" complaint).
If the studying policy transfers across domains, you've shown a general skill of self-directed
learning — a genuine step toward agents that improve on the job.

**First experiment.** On StudyBench (public, comparable), pit a fixed cheatsheet against a learned
studier that generates its own practice questions; measure the expertise curve. Key ablation: does
the *quality* of self-generated questions predict expertise gain (i.e., is the agent actually
learning to study, or just doing more compute)?

**Honest novelty check.** Pedagogical RL (Ziems) and self-instruct are adjacent — teaching signals
and self-generated data exist. The differentiator: **no privileged teacher; the curriculum is
self-generated and graded purely by held-out transfer**, and studying is optimized as a policy, not
a one-shot prompt. Cite pedagogical RL and self-play explicitly and draw the line.

**Risk.** Medium-high. Self-generated curricula can collapse (recursive drift, as SkillLearnBench
found). The anti-collapse mechanism is part of the contribution.

---

## 3. Where should knowledge live? Compute-optimal placement across context, cache, and weights ★

**Paper title.** *A Placement Law for Agent Knowledge: When to Retrieve, Cache, or Fine-Tune.*

**The gap.** A fact an agent knows can live in three places: in the prompt (read fresh every time —
flexible, forgetting-proof, expensive per query), in a precomputed KV cache (cheaper reads, stable),
or distilled into weights (near-zero read cost, but expensive to update and prone to forgetting).
The entire field argues *token-space vs weight-space* (Letta says context, Thinking Machines
distills, Google's Nested Learning blends) as if it's ideological. It's actually an **economics
question with a right answer per item**, and nobody has characterized it.

**The new method.** Model each knowledge item by three properties — access frequency, stability
(how often it changes), and value — and derive the **cost-optimal tier** as a function of these.
Learn a routing policy that places each item to minimize total serving cost at a target accuracy,
and characterize the **crossover conditions** (e.g., "distill to weights once an item is accessed
> f times and stable over > k updates"). Look for a scaling relationship: as interaction volume
grows, the optimal fraction of knowledge in weights grows in a predictable way.

**Main novel contribution.** The first **principled, quantitative account of where agent knowledge
should live** — a placement/crossover law that turns the token-vs-weights debate into an
optimization with measurable thresholds.

**Why it matters / outcomes.** This is a *foundational organizing result*, the "Chinchilla for
memory": it tells every agent builder when to retrieve vs cache vs fine-tune, instead of guessing.
It's the kind of result that gets cited by everyone downstream because it settles a debate the whole
field is having. Directly actionable and vendor-neutral.

**First experiment.** A controlled corpus with items of dialed frequency/stability; measure
accuracy-per-dollar for each fixed placement and for a learned router; show the router dominates any
fixed strategy and recover the crossover curve.

**Honest novelty check.** Memory hierarchies exist as *systems* (MemOS "activation memory," Letta
context repos, sleep-time compute); nobody has framed placement as **compute-optimal allocation with
a derived crossover/scaling law**. That analytical framing is the novelty. High citation ceiling,
harder to make rigorous — invest in clean measurement.

**Risk.** Medium. The clean law may be messier in practice; even an empirical crossover map is
valuable.

---

## 4. Turning any corpus into a verifiable continual-learning environment

**Paper title.** *Environments for Free: Auto-Generating Verifiable Continual-Learning Tasks from
Corpora.*

**The gap.** RL and continual learning are both bottlenecked on **environments** — Prime Intellect's
whole thesis, and labs are paying cash bounties for them. But environments are hand-built and don't
scale, and CL benchmarks (CL-Bench, SkillLearnBench, StudyBench) are small and semi-manual. There's
no way to *manufacture* verifiable "you must remember earlier things to do later things" tasks at
scale, contamination-free.

**The new method.** A generator that takes any corpus with dependency or temporal structure — a
codebase's commit history, versioned docs, a textbook, a repo's own decision log — and emits a
**stream of verifiable tasks where later success provably requires knowledge accumulated from
earlier items**. Verification is programmatic (code runs / tests pass / generated answer key
matches), and contamination is avoided by generating from post-cutoff or private corpora. The
generator also emits the "no-memory" and "full-context" reference curves automatically.

**Main novel contribution.** A **method to synthesize verifiable continual-learning environments
from arbitrary corpora**, decoupling CL/memory evaluation and RL training from hand-built benchmarks.

**Why it matters / outcomes.** It removes the environment bottleneck for an entire research
direction. Every memory/CL paper needs eval; every RL-for-agents effort needs environments; labs and
Prime Intellect literally pay for these. An adopted generator is infrastructure that gets cited by
default — the SQuAD-generator move for continual learning.

**First experiment.** Generate a CL stream from a real codebase's git history; show frontier agents
degrade catastrophically without cross-episode memory and improve with it; validate that the
verifier is sound (no false passes).

**Honest novelty check.** CL-Bench / SkillLearnBench / LongMemEval-V2 are hand-curated;
TDBench auto-generates temporal SQL QA. The novel bit is **automatic generation of dependency-linked,
verifiable, multi-episode tasks from arbitrary corpora** as reusable environments (not just a QA
set). Solid and defensible.

**Risk.** Low-medium. Main risk is verifier soundness; mitigated by programmatic checks.

---

## 5. Memory for RL, not RL for memory — retrospective credit via episodic recall

**Paper title.** *Hindsight from Memory: Retrospective Credit Assignment over Remembered Decisions.*

**The gap.** The literature is all "RL to train memory." The reverse — **memory to fix RL's hardest
problem** — is barely touched. Long-horizon agents fail because reward is sparse and terminal;
credit for the one pivotal decision 40 steps back is nearly impossible to assign. An agent that
*remembers its decisions across episodes* could, on success or failure, look back and attribute
credit to the choices that mattered.

**The new method.** Maintain an episodic store of decision points (state, choice, rationale) across
episodes. On a terminal outcome, a retrospective analyzer retrieves the pivotal remembered decisions
— including analogous ones from *past* episodes — and assigns dense counterfactual credit ("in three
prior runs this choice preceded failure"). Use that as shaped advantage for the next policy update.
Memory becomes the mechanism that makes long-horizon credit assignment tractable.

**Main novel contribution.** Using **cross-episode episodic memory as the substrate for retrospective,
counterfactual credit assignment** — connecting the memory and long-horizon-RL literatures that have
been running in parallel.

**Why it matters / outcomes.** If it improves long-horizon agent training (coding, web, research
agents), it matters far beyond memory — it's a contribution to core RL, which is a much larger,
higher-citation audience. It also gives episodic memory a *reason to exist* beyond QA.

**First experiment.** A sparse-reward long-horizon suite (ALFWorld/WebArena-style); compare GRPO,
hindsight baselines, and memory-indexed retrospective credit; show sample-efficiency gains.

**Honest novelty check.** HCAPO (LLM post-hoc critic with hindsight) and classic hindsight credit
assignment are adjacent, and within-episode. The differentiator is **cross-episode episodic
retrieval of analogous decisions** as the credit signal. Moderate novelty — cite HCAPO/hindsight
carefully and lead with the cross-episode framing.

**Risk.** Medium-high. Attribution quality is the crux.

---

## 6. Memory that checks itself — self-verifying writes and minimal belief revision

**Paper title.** *Self-Verifying Memory: Provenance-Grounded Writes and Minimal Revision under
Contradiction.*

**The gap.** Agents writing their own memory compound errors — Mem0 self-reports ~40% extraction
failure, and HaluMem shows hallucinations *originate at the write step* and propagate. Once a wrong
"fact" is stored, similarity retrieval surfaces it as truth forever. The write path is the least
verified, most consequential part of memory, and it's where trust dies for enterprises.

**The new method.** Make every write earn its place: (1) **verify against source** (does the raw
evidence entail this fact?), (2) **check against the store** (does it contradict existing memories?),
and (3) on conflict, perform **minimal belief revision** (AGM/JTMS-style: change the least, keep
provenance and justifications so revisions propagate). Learn the write/verify policy against a
hallucination-and-accuracy objective, not just recall.

**Main novel contribution.** A **learned, provenance-grounded write policy with minimal-revision
conflict handling** — bringing decades of belief-revision theory to bear on the LLM-memory write
path, where errors actually originate.

**Why it matters / outcomes.** Trust and auditability are *the* enterprise blocker (the "memory is a
review problem" discourse; multi-tenant contamination as the real buy-trigger). A memory that can
show why it believes something, and correctly un-believes it when contradicted, is what makes
learned memory deployable in regulated settings. Safety + robustness audience.

**First experiment.** On HaluMem's operation-level benchmark, show verify-at-write + minimal revision
cuts extraction/update hallucination and improves downstream accuracy vs write-everything and
overwrite-on-conflict.

**Honest novelty check.** TRUSTMEM (trustworthy consolidation), belief-revision-memory (2603.17244),
and HaluMem (measurement) are adjacent. Differentiate on **provenance-linked minimal revision as a
learned write policy** (not a fixed rule, not just measurement). Moderate novelty; the belief-revision
formalism is the anchor.

**Risk.** Medium.

---

## 7. Position / measurement — recall is not competence

**Paper title.** *Recall Is Not Competence: Measuring Continual Learning by Forward Transfer per Unit
Compute.*

**The gap.** The field's benchmarks score whether a system can *retrieve* a planted fact. That is not
the question. The question is whether the agent got *better* — more capable, more efficient — from
experience. Machine Studying showed retrieval and competence diverge; LoCoMo/LongMemEval are
corrupted/saturated; yet leaderboards keep reporting recall accuracy.

**The new method (measurement).** Propose **forward transfer per unit inference compute** as the
metric: after experience, how much does downstream task performance improve at a fixed compute
budget? Provide the protocol (fixed probe sets, compute-normalized scoring, adversarially-validated
judging or deterministic scoring) and re-measure a few named systems to show recall and competence
rank differently.

**Main novel contribution.** A **competence-based evaluation axis for memory/CL** that replaces
recall, with a concrete, gameable-resistant protocol.

**Why it matters / outcomes.** Reframes how the whole field reports results — the kind of measurement
paper that changes reviewer expectations and gets cited in every subsequent related-work section. Low
cost, high leverage, and it's the narrative spine (and the video) for everything else here.

**Honest novelty check.** Machine Studying introduced expertise-as-efficiency; various critiques of
LoCoMo/judges exist. The synthesis into a *standard competence protocol for continual learning* is
open. Low risk (position/measurement).

---

## How to read this list

- **If you want one breakthrough swing:** #1 (label-free memory). It attacks the deepest bottleneck,
  is principled, scales with data not labels, and is defensibly novel. It's the one that could become
  a default method used by thousands.
- **If you want the most *applied* pull:** #2 (learning to study) — autonomous onboarding to new
  domains is what every agent company is failing at right now.
- **If you want the most *citable-by-default* result:** #3 (placement law) — foundational, settles a
  live debate, vendor-neutral.
- **If you want the safest first ship that funds the rest:** #7 (recall≠competence) then #4
  (environment generator) — cheap, low-risk, and they build the measurement + eval ground the harder
  papers stand on.

They share a spine: **#7 defines the metric, #4 supplies the environments, #1 supplies the training
signal, #2/#3 are the methods, #5/#6 harden it.** Do #7 + #1 as the pair — a new metric and a new
label-free method — and you have a genuinely new research program, not another entry on a broken
leaderboard.
