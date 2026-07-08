# The Referee and the Compiler

**The program (2026-07-08).** One open-source system for engineers that does two things nobody
does rigorously: (a) **referee** — evaluate agent memory with paired design, cached compiles,
triple-judge majority, a full-context ceiling row, budget-matching, and amortization-aware cost
accounting; (b) **compiler** — compile and *maintain* budgeted context packs (the CLAUDE.md
replacement) behind that same regression gate. Launch paper: **the noise audit** — "your memory
benchmark is measuring your harness" — quantify how much of published memory/prompt-optimization
gains survive controlled reruns. Our receipts are first-person: identical config scored 33/67/75%
held-out before controls, and our own harness produced a fake 24% for Mastra via a wrong read
protocol. The field's receipts: the LoCoMo audit, the 93.4→73.8 reproductions, no neutral referee.

**Phased plan (mirrored on the task board):**
P1 *Calibrate* — ceiling row → corrected Mastra (full-context + their TS code) → controlled
evolution reruns. P2 *Audit* — 3-5 OSS systems + a prompt-opt loop under controls; effect-size
shrinkage table = paper 1. P3 *Ship* — harness + compiler + diff-invalidation/regression-gate
as installable OSS + GitHub Action. P4 *The swing* — **Corpus-RLVR** (below).

## P4 — Corpus-RLVR: the R1 recipe for corpus expertise (the breakthrough bet)

**The gap, at the intersection of the training literature and ours.** The frontier training
stack (2026 survey: 3D-parallel pre-training, RLHF/DPO, GRPO/RLVR, distillation, PEFT, TTC) has
a working recipe for *reasoning* (DeepSeek-R1: cold-start SFT → RLVR → rejection sampling →
distill; AIME 15.6→71.0) — and **no working recipe for turning a specific corpus into expert
weights.** "Continual pre-training for domain adaptation" is listed as unlocked by open weights,
but the empirical record says naive CPT on a fresh corpus *fails* and synthetic-QA SFT
*memorizes without competence* (Machine Studying, 2026 — the only thing that worked was a
studied context artifact). RLVR cracked reasoning because math/code have verifiers. Corpus
knowledge never had one. **That's the missing chapter of both literatures.**

**The claim.** The corpus IS the verifier. Rewards for knowledge-RLVR can be generated
deterministically from the corpus itself: string/AST-checkable questions from code, containment
and provenance checks, git-diff freshness oracles — exactly the machinery this repo spent six
months building, along with the referee discipline (paired, cached, triple-judge, ceiling rows)
that makes reward signals trustworthy. In RLVR the verifier is the moat; the RL loop is now
commodity (Tinker GRPO, QLoRA single-GPU).

**The recipe (R1 stages → corpus expertise):**
1. *Cold start*: self-study SFT — study data generated from the corpus (our compile/consolidation
   machinery is the data engine; Cartridges' self-study is the KV-space precedent, we target weights).
2. *RLVR*: GRPO on Qwen3-4B (Tinker, QLoRA) with corpus-verifiable rewards. Anti-memorization by
   design: evaluation uses **held-out question generators** (novel question styles, not held-out
   items from the training generator) and the competence metric is **accuracy at matched
   inference compute** (Machine Studying's expertise), closed-book and budget-matched.
3. *Rejection sampling*: keep high-reward study trajectories as new SFT data (R1 stage 3 analog).
4. *Distill/compose*: on-policy distillation from teacher-with-corpus into student-without;
   output = an **expert adapter** (LoRA) per corpus.
+ *Continual chapter*: on corpus diff, incremental RLVR on the delta + unlearning of superseded
  facts (the supersession machinery, now in weights) — measured with rot curves. The revision
  problem, finally with a verifier.

**Baselines to beat at matched inference compute:** long-context ICL · RAG · the studied
cheatsheet (current champion) · our pack+escalate. **Falsifier:** if expert adapters can't beat
the cheatsheet at matched cost on held-out question styles, publish the rigorous negative (*why*
knowledge-RLVR memorizes — credible because of the referee). **Product:** `expert adapters` —
compile a repo into a LoRA your local model loads; continual updates per commit.

Everything already built slots in: question harness = the verifier · compile machinery = the
data engine · pack/escalate numbers = the token-space baselines · referee = the credibility ·
temporal/supersession = the continual chapter · Tinker/Qwen3 = the substrate.

---

# ctxpack — a compiler for agent context

**Canonical research + product document (program details).** Supersedes the compiled-KV maintenance program (v1 of
this file) after competitive verification. Last updated 2026-07-06. Running results in
`ctxpack/results/`; code-substance notes in `MASTER-NOTES.md`; content pipeline in
`research/content/PIPELINE.md`.

## The program in one paragraph

One store, two compilers, one maintenance loop. **Compiler 1 (`compile`)**: corpus → budgeted,
source-anchored context pack, optimized against a task suite (replaces hand-written
CLAUDE.md/AGENTS.md). **Compiler 2 (`ledger`)**: long agent traces → a belief/experiment ledger —
what was believed, tried, resulted, and superseded. **The loop**: per-section `[src:]` anchors →
diff-scoped invalidation → incremental recompile → task-suite regression gate ("context under
CI"). Both compilers share the event-sourced store (`mempol/temporal` supersession chains +
`mempol/ledger` research objects — already built) and one observability dashboard.

## Why now — the evidence base (verified 2026-07-06)

1. **Unmeasured context files are a liability, published:** ETH SRI (arXiv:2602.11988) — across
   4 agents, AGENTS.md-style files do not improve task success and add >20% inference cost;
   LLM-generated files are net *negative*. Every first-party generator (/init, Copilot, Cursor,
   DeepWiki) is single-pass and unevaluated. → eval-gated compilation is a fix to a documented
   harm, not a nice-to-have.
2. **Our own result:** blame-guided pack evolution, 6-train/6-heldout, corrected keys:
   handwritten 0–17% < RAG 41.7% < evolved pack **83.3% held-out** at matched 4k budget
   (r2; r3 over-pruned and collapsed — demonstrating the regression gate). Pack provenance
   caught an answer-key error (group_size=4, not 8). Single seed, n=6: directional.
3. **Maintenance is the whitespace:** competitive scan verdict — generation is crowded;
   *nobody* combines anchors → diff-invalidation → incremental recompile → eval gate. The CI
   plumbing exists (promptfoo actions); the object under test does not.
4. **The ledger demand is documented by the failers themselves:** Google Co-Scientist reports
   "cycled through variations of the same failed approaches", "insights from failed attempts are
   lost across sessions", "confident rediscovery" of falsified hypotheses (arXiv:2602.03837).
   Kosmos's world model is additive-only (no supersession; 57.9% synthesis accuracy). Sakana's
   tree is per-run. Three June-2026 papers (StatefulDiscovery 2606.11851, Hypothesis-Tree
   Refinement 2606.11926, BeliefMem) call for the revisable belief ledger — **as papers, not
   tools.** The reusable component is unshipped.
5. **An industry spec is waiting for a compiler:** Agent Trace (Cognition + Cursor + Cloudflare +
   Vercel + Google Jules, Jan 2026) standardizes decision-provenance in git with no compilation
   step. ctxpack ledger can be *the* compiler for it.

## Positioning

> ctxpack is a compiler for agent context: it builds your CLAUDE.md from source with anchors,
> proves it helps against a task suite, and incrementally rebuilds it on every diff — because
> the published evidence says an unmaintained, unmeasured context file makes your agent worse
> while costing 20% more.

Enterprise wedges: (a) coding-agent context that is *proven and maintained* (anti-ETH pitch);
(b) research/agent fleets that *stop repeating failed experiments* (anti-Co-Scientist pitch).
OSS motion: MIT core compiler + emitters (CLAUDE.md/AGENTS.md/.cursorrules) + GitHub Action;
hosted CI + dashboard + fleet ledger as the commercial layer.

## The two papers

**P1 — "Compile, Don't Write: task-optimized, maintained context packs for coding agents."**
Direct response to ETH 2602.11988. Claims: (i) eval-gated compiled packs flip context files from
net-negative to strongly positive; (ii) budget–accuracy Pareto across corpora; (iii) a *writer
policy* (outer GEPA loop over writer/reviser/reader prompts) transfers to unseen corpora;
(iv) maintenance: under real git-history diffs, incremental recompile + regression gate holds
freshness at a fraction of full-recompile cost. Baselines: /init-style one-shot generation,
aider repo-map, RAG (lexical + dense), LLMLingua compression, human-written files. Metrics
deterministic where possible. Falsifier: if the transferred writer policy doesn't beat one-shot
generation on unseen corpora, the paper is the maintenance result only (workshop-grade).
Venue path: workshop → ACL/NeurIPS D&B; main-track if transfer is strong.

**P2 — "The Experiment Ledger: supersession-aware state prevents repeated failures in
long-horizon agents."** Compile agent traces into (belief, evidence, status) + (experiment,
result, artifact) objects with supersession; inject ledger at decision points. Metrics on
documented failure modes: repeated-failed-experiment rate, confident-rediscovery rate,
cross-session insight retention — measured on open discovery/coding agents over long runs.
Novelty: validated by three concurrent papers, contested by zero tools. Substrate:
`mempol/ledger` + `mempol/temporal`, already implemented. Falsifier: if ledger injection doesn't
reduce repeat-failure rate vs. transcript-replay and summary baselines, the primitive is
decorative — publish the negative and stop.

Shared machinery: anchors/provenance, supersession, regression gate, blame decomposition,
budget curves, one dashboard. P1 ships first (harness live today); P2 dogfoods on our own agent
traces (this project's sessions are the first corpus).

## Sequencing & scope gates

1. **Now (task #11):** outer GEPA loop — writer/reviser/reader prompts optimized on corpora A–C,
   evaluated on unseen D. Add understanding-slice questions; multi-seed; dense-RAG baseline.
2. **Then:** maintenance experiment on real git history (this repo + 2 public repos): diff →
   invalidate → recompile → gate; report freshness/cost vs full recompile. (The one surviving
   piece of the compiled-KV program, in token space where it's cheap and inspectable.)
3. **Then:** ledger MVP — one adapter from Agent Trace/Claude-Code session logs onto the
   existing ledger schema + injection experiment (P2).
4. **Scope gate:** no KV/weights work, no new benchmarks, no new stores until P1 experiments
   are done. Cartridges/KV remains a *future backend* of the same interface.

## Assets carried forward (nothing orphaned)

`temporal/store.py` supersession → ledger core · `ledger/schema.py` research objects → P2 ·
GEPA infra → outer loop · critic seed → refresh value-of-information (maintenance scheduling) ·
survival/hazard stats → refresh scheduling · eval matrix + provider shims → baselines ·
TicToc demo-02 + PIPELINE.md content → independent tracks (unchanged).
