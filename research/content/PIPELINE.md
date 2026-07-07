# Content & publication pipeline — what's real, ranked by readiness

*2026-07-06. Everything in this repo that can become a blog / video / paper, with what already
exists on disk and what's missing. Ordered: ship-now → near-ready → real-work-required.
Rule: nothing enters "ship-now" unless the asset exists and the numbers in it are reproducible.*

## Tier 1 — Ship this week (assets exist, ≤1 day of finishing each)

1. **"The Shape of Memory" essay + interactive field maps** — `blog-the-shape-of-memory.md`
   (~3,200 words, done) + three live artifacts (grid / scatter / systems views, sourced &
   verified). Missing: publish to own site, X thread (drafted earlier in session), HN post.
   The maps are 1-of-1 artifacts nobody else has.
2. **The eval-credibility story** — "we watched an LLM judge flip verdicts on identical answers"
   (happened live in demo-01) + the LoCoMo audit (6.4% wrong key, judge accepts 63% wrong) +
   claimed-vs-observed scandals. Format: X thread or short post. All receipts on disk
   (`demos/01-stale-memory/README.md` judge section). Highly shareable, zero new work.
3. **Operator's buyer's guide** — `memory-buyers-guide.md` (done, actionable pass complete).
   Missing: LinkedIn-format cut for the CTO/founder audience. Different audience than #1.
4. **"The Clock Agents Cannot Feel"** — `temporal-awareness-the-clock-you-cannot-feel.md` is
   near-publishable prose already; now strengthened by a real result (demo-02: blind 41.7 <
   timestamps 47.2 < computed-state 52.8, n=36) and the three-clocks frame (Peike Li) to cite.
   Missing: one editing pass + the demo-02 chart.

## Tier 2 — Near-ready (script/data exist; days of production, not research)

5. **Video: "How AI Learned to Remember"** — full script v1 exists
   (`02-how-ai-learned-to-remember.md`, the ladder narrative: goldfish → context → vectors → KG →
   reflection → learned policies → the fork), demo table mapped to real repo scripts, TOC v2
   locked, tone decided. Missing: record demos + produce. The single strongest content asset here.
6. **Video: "Your Agent Can't Feel Time"** — scripts + runbook exist
   (`temporal-awareness-video-2026-06.md`, `-runbook.md`, `-ASSETS.md`); demo-02 gives the live
   result; Real-Time-Deadlines (4%→32%) and TicToc (<65% ceiling) as the literature beats.
7. **"Substrate doesn't matter (we have receipts)"** — the honest-negative post: our own typed-KG
   lost to flat RAG (46.2 vs 58.6, n=1,491) + Hindsight's baseline-beats-own-KG + the field's
   89–95% convergence. Concept page exists (`substrate-design-space.md`). Honest negatives are
   rare and build more trust than wins.
8. **"Compress now or reconstruct later"** — the write-time/read-time bifurcation post; concept
   page exists (`write-time-vs-read-time.md`); RLM vs Mem0 as the poles; our timeline-reader
   numbers (71.7% LongMemEval-S) as the worked example.
9. **"Everyone is dreaming"** — sleep-consolidation concept page + the 2026 convergence receipts
   (OpenAI Dreaming, Letta sleep-time, Honcho, Supermemory) — the "industry converged on the
   mechanism, nobody optimizes it" angle. Concept page exists (`sleep-consolidation.md`).
10. **The research wiki, published** — 32 papers with verification tiers (abstract-verified /
    table-only / third-party), 9 concept pages, static-site generator already built
    (`research/scripts/build.py`). Goal 03 from months ago; it's a `vercel deploy` away.
11. **Recall ≠ competence** — the Machine Studying synthesis essay (position piece); half the
    argument is already written across the buyer's guide §2 and archived frontier docs.

## Tier 3 — Paper-grade (real experiments required; ranked by realism × payoff)

12. **TicToc time-as-state → trained policy** — the only line with a deterministic, human-labeled
    metric, an unbeaten ceiling (65%), an unexploited train split (3,630 ex), and a real smoke
    already showing the monotonic effect (P0 done). Path: P1 full-test slice (~$30) → P2 GEPA on
    the decision prompt (~$50) → P3 GRPO/DPO on Qwen3-4B via Tinker (authors ship DPO scaffolding).
    Beating 65% = clean SOTA on a metric nobody can dispute. **The most defensible paper bet in
    this repo.**
13. **Amortized write-utility critic** — verified-open (HiMPO explicitly avoids a critic; nobody
    amortizes per-op counterfactual credit); r=0.707 seed + exact-replay machinery exist. Risk:
    seed may not survive scale-up; niche moves fast. Medium paper, high prestige if it holds.
14. **Rot curves for compiled memory** — measurement-only slice of the maintenance idea: how fast
    does a Cartridge/consolidated artifact go stale on a fast-moving corpus? Cheap-ish, novel
    measurement, workshop-grade floor. Do NOT build the full maintenance program unless this
    number is dramatic.
15. **GEPA-consolidator held-out run** — ~$100 decides Goal 01; verified-open niche (MemPro took
    TextGrad, GEPA slot empty). Small paper or strong blog + the ep-1 live demo either way.
16. **SOTA pareto plot in axis-coordinates** — `paper_leaderboard` data + the field-map axes =
    the figure for ep-1 ch-8 and a standalone artifact/post.

## Anti-list (do not write about)
Per-op counterfactual as a contribution (scooped 3×) · LoCoMo numbers as headline (discredited) ·
PIE-as-SOTA (own negative) · GEPA +20pp (n=5, train==val) · any claim without a run artifact.

## Sequencing logic
Tier 1 items compound audience for everything else and cost ~a week total. Tier 2 videos are the
Working Memory channel launch (scripts exist — production is the only gap). Tier 3: run #12 and
#15 first (cheap, deterministic, decided quickly); #13/#14 only after those land. One experiment
in flight at a time.
