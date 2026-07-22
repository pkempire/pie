# Shape of Memory — filming & editing kit (2026-07-09, updated 07-22)

**FULL SCRIPT (use this, not the essay, as the spoken backbone): `research/content/video-script-shape-of-memory.md`** — zero-to-expert walk of every method with [SHOW] cues, timings, per-method implement-it steps, and the two framing corrections baked in (read/write = economics not ceiling; stores-knowledge vs learns-process). Map + buyer's guide revamped 07-22: map nodes now carry ◆/⚙ badges and "run it yourself" setups; guide has the play-along repo table (§6).

## Assets (paths + live URLs)
- **Field map, grid view**: `research/content/memory-map.html` · live: claude.ai/code/artifact/97163d29-90b1-4c34-a2f6-e1c50679642d — hoverable approaches w/ sources; the "everyone's in one row" reveal box
- **Field map, scatter view**: `research/content/memory-map-scatter.html` · live: claude.ai/code/artifact/1db6528e-ce6e-4fb7-8a5b-5f07b3d0c87e — better for camera pans
- **Systems view (companies)**: `research/content/memory-map-companies.html` · live: claude.ai/code/artifact/44b39d7f-b63a-42c9-896d-18f80b1a1fde
- **The essay** (script backbone): `research/content/blog-the-shape-of-memory.md`
- **Operator decisions segment**: `research/content/memory-buyers-guide.md` (§2 60-second test, §4 caching math)
- **Live demos**: `demos/01-stale-memory/` (flat 20% vs replay 100% as-of-past; run.py ~$0.01, terminal-friendly) · `demos/02-temporal-awareness/` (blind 41.7 < timestamps 47.2 < computed-state 52.8)
- **Dashboard b-roll**: `streamlit run ctxpack/dashboard.py --server.port 8601` — score trajectories, blame matrix, pack diffs (the r2→r3 gutting is a great visual)
- **Packs as artifacts**: `ctxpack/results/evolution/pack_r0..r3.md` (watch format evolve) · `ctxpack/results/lme/pack_cache/` (real LongMemEval packs)

## Numbers you can say on camera (with the honest caveat attached)
- ETH study: AGENTS.md-style files don't improve success, add >20% cost; LLM-generated = net negative [arXiv:2602.11988]
- LoCoMo audit: 6.4% wrong answer key; judge accepts 63% of wrong answers [github.com/dial481/locomo-audit]
- Claimed-vs-observed: Mem0 93.4% → 73.8% under frozen judge [maximem.ai]
- Ours (paired n=30, single seed — say "directional"): oracle ceiling 73.3%; pack+escalate 60.0% (=82% of ceiling at 3.4% of context); RAG 56.7; blind pack 40.0
- Our variance confession (great on camera): identical config scored 33/67/75% across reruns — "we caught our own harness lying before it caught us"
- STALE: production memory frameworks score 5–8% on implicit staleness [arXiv:2605.06527]
- RFT vs SFT forgetting: −2.3% vs −10.4% (RFT is the gentler consolidation operator) [see research/lit-review/2026-07-09-*]
- DO NOT cite: Mastra 24% or 30% as "their system" (our reimplementation); GEPA +20pp (n=5 overfit); any LoCoMo score without the audit

## Sources to review pre-edit (primary)
arXiv:2602.11988 (ETH AGENTS.md) · dial481/locomo-audit · arXiv:2506.06266 (Cartridges) ·
arXiv:2310.08560 (MemGPT) · arXiv:2308.10144 (ExpeL) · arXiv:2304.03442 (Generative Agents) ·
arXiv:2501.13956 (Zep) · arXiv:2507.19457 (GEPA) · jacobxli.com/blog/2026/machine-studying ·
thinkingmachines.ai/blog/on-policy-distillation · mastra.ai/research/observational-memory ·
arXiv:2510.23853 (TicToc) · arXiv:2601.13206 (Real-Time Deadlines) · Peike Li "Time is the
missing modality" · arXiv:2511.02824 (Kosmos) · Co-Scientist limitations arXiv:2602.03837 ·
docs/CONTRACT.md (the behavioral contract — closing beat)

## Recommended structure tweak (scope > "memory")
Cold open: the ETH fact, personal — "that CLAUDE.md you hand-wrote is probably making your agent
worse, and charging you 20% more for it." → the map ("everyone's in one cell, and it's the only
cell you can build on a closed API") → the six-worlds taxonomy (60s, the disambiguation nobody
does) → the four decisions the viewer makes this week (context file / RAG-vs-cache / when to buy
memory / what to measure + the variance confession) → the turn: recall ≠ competence (Machine
Studying) → the frontier: learning from experience, the behavioral contract, next-diff idea as
the closer. Title candidates: "The Shape of Memory" (keep) / "Your Agent's Memory Is Making It
Worse" (hook-forward) / "Memory Is the Wrong Word."
