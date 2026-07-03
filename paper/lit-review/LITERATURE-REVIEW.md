# Literature Review & Rewrite Plan — mempol paper

This is the master document. The four sub-reviews are in this folder:
- `memory-systems.md` — competitor memory systems (Mastra, Mem0, Zep, MemGPT, Cartridges, RLM, Honcho, OMEGA)
- `rl-memory.md` — RL-for-memory work (Memory-R1, Mem-α, DeltaMem, Search-R1, Graph-R1)
- `temporal.md` — temporal reasoning & time-aware agents (TicToc, Time-R1, Robotouille, Real-Time Deadlines, etc.)
- `reader-review.md` — brutal reader review of the current paper

## Where the field actually is, in three sentences

(1) Memory for LLM agents is converging on the same shape — *external store + write-time decisions + retrieval-time decisions* — but the design choices on the second and third axes are still being made by hand. (2) The newest published memory-RL work (Memory-R1 Aug 2025, Mem-α Sept 2025, DeltaMem Apr 2026) trains the write side but uses trajectory-level rewards that diffuse credit across many ops. (3) A separate cluster of work (TicToc, Real-Time Deadlines, Robotouille) is showing that agents fail at *behaving in time* — knowing when to act, when to refresh, when to surface — and these failures are orthogonal to whether the memory store is a graph, a chunk index, or a learned compression.

## The 2×2 we should actually use to position mempol

|  | **Hand-coded ops** | **Learned ops** |
|---|---|---|
| **Trajectory-level reward** | Mem0, Letta, Zep, Mastra, KGmem, generative agents | Memory-R1, Mem-α |
| **Per-op / state-distance reward** | (n/a — hand-coded never asks "did this op help?") | **mempol** + DeltaMem |

DeltaMem is the only other work in the bottom-right cell, and it computes an op-level reward via Levenshtein distance on the memory state — a *state-distance* signal, not an *outcome-attribution* signal. mempol is the only system that asks per-op: "if I leave this write out, does the downstream answer get worse?" That's the COMA-shaped attribution and it is genuinely uncrowded.

## Where competitors stand on LongMemEval

| System | LongMemEval | What they actually do |
|---|---|---|
| OMEGA | 95.4% | Closed; assume a hybrid pipeline |
| Mastra (Observational Memory) | 94.87% | Reflector LLM extracts (entity, property, value, timestamp) once; immutable |
| Hindsight | 91.4% | Episodic + reflection layers |
| Recursive Language Models (Gemini 3 Flash) | 89.8% | Test-time outer loop that recursively decompresses context |
| Zep / Graphiti | ~72% (varies; 94.8% DMR) | Bi-temporal KG with valid-time edges |
| Mem0 | ~69% | LLM-as-judge ADD/UPDATE/DELETE on flat KV store; reports 40% extraction-failure rate internally |
| Naïve RAG | ~60% | Chunk + embed + retrieve |

Critical observation: **the leaderboard is converging at 89-95% but on radically different storage substrates and inference budgets**. This is strong evidence that the substrate doesn't matter much above some quality floor, and the *amount of test-time compute spent on the read side* is doing most of the heavy lifting (Mastra runs the Reflector once at write time, RLMs spend it at query time, OMEGA presumably both). mempol's contribution is at a different layer: not "win the leaderboard" but "be the first system whose write-side decisions are *learned from outcome*, not hand-coded."

## What the reader review surfaced (5 most important)

1. **Equation 1 (per-op counterfactual) is buried after a justification that scatters the problem.** Lead with a worked example — four ops in a trajectory, replay without each, show the credit. Then the math.
2. **The chunking finding (0% multi-hop → workable) is in engineering notes.** Promote it. It tells the reader the substrate is doing real work and that the policy is operating on a substrate that already had to be made workable.
3. **"What we deprecated" eats three paragraphs of method.** Move to a footnote. Readers don't need the museum tour.
4. **Figure 1 is in the abstract but referenced six paragraphs later.** Move to end of intro.
5. **The "first to combine X+Y+Z" claim is unfalsifiable as written** — narrow it to "first to train memory-write ops with per-op outcome attribution, not first to combine discrete ops with learned policies."

## What we should add that we don't currently mention

- **Recursive Language Models** (raw.works blog, Weitekamp). Treats memory as an outer-loop test-time recursion: the model decompresses context lazily and decides recursion depth per query. Hits 89.8% on LongMemEval with Gemini 3 Flash. Worth a paragraph in related work as "a different way to spend test-time compute on memory" and worth flagging as a complement to mempol (RLM picks how much to retrieve, mempol picks what was worth storing).
- **Stanford Cartridges**. KV-cache compression artefact per knowledge bundle; very different problem than mempol's (model-internal not model-external) but worth one sentence as the "facts in the weights" extreme.
- **Mastra Observational Memory** is currently mentioned as a baseline but not engaged with as a *philosophy*. Their "no extraction" claim is sleight of hand — the Reflector is an extractor — but it's the strongest LongMemEval result among published systems and we should be honest about it. Better framing: Mastra is the upper bound of what aggressive write-time inference buys you on the existing benchmark; mempol asks whether a *learned* write policy can do as well at lower cost.
- **TicToc / Real-Time Deadlines / Robotouille** for the temporal section. Reframe TemporalBench as the natural-progression benchmark from these three: we test six axes instead of TicToc's one and we score *behaviour* not *math*.
- **DeltaMem** as the closest published comparison. They use op-level rewards via Memory-Levenshtein distance; we use op-level rewards via outcome-attribution. The difference is the right honest framing of our novelty.
- **Mem-α** as the long-horizon scaling story. They generalise from 30k → 400k tokens. We don't yet have a long-horizon claim; we should either run that experiment or carefully say we don't.

## What we should remove or soften

- Drop "backend-agnostic" claims until we have the transfer table. Replace with "designed to be backend-agnostic; transfer experiments in §X."
- Drop "no per-user fine-tuning needed" — we trained on LoCoMo's symmetric peer chats; we have no evidence this transfers to one-user-one-assistant deployments. Replace with "the same trained adapter compiles to multiple stores; we do not yet show user-style transfer."
- Drop "novel" framing on per-op counterfactual; replace with "an adaptation of COMA's counterfactual baseline (Foerster et al. 2018) to memory-write op attribution. The closest concurrent work, DeltaMem, uses a state-distance op-level reward; ours uses an outcome-attribution one."
- Drop or relegate the deprecated-rewards section. Footnote it.
- The "facts live in the store, strategy lives in the weights" line is good; keep one instance of it, kill the duplicate at the bottom.

## The new narrative arc (1-paragraph)

Long-running AI assistants need long-running memory; they handle this with an external store plus a control layer. Today every published system hand-codes the control layer — a prompt that decides what to write, a heuristic that decides what to evict. We ask whether the control layer can be learned end-to-end from outcome supervision. The technical obstacle is credit assignment: the policy emits a sequence of memory operations per turn, but the reward only arrives when a future question gets answered well or badly. Trajectory-level rewards, the standard solution in concurrent memory-RL work (Memory-R1, Mem-α), diffuse a single scalar over many ops; in our setting that gradient is too noisy. We propose per-op counterfactual marginal utility — for each mutating op, replay the trajectory without it and measure how the held-out QA accuracy changes. The op gets credit equal to the difference. This is the COMA baseline from cooperative multi-agent RL, applied at the granularity our typed op vocabulary makes natural; the closest published memory-RL work that gives op-level rewards (DeltaMem) uses a state-distance signal instead, which doesn't directly say what the downstream reader needs. We train a Qwen3-4B LoRA with GRPO under a hard retention budget and evaluate on LoCoMo and LongMemEval, with a controlled audit isolating the effect of the learned write side from the effect of the read-side retriever. The framing we leave the reader with: facts live in the store, strategy lives in the weights — and strategy can be learned, not hand-coded.

## Section-by-section rewrite plan

- **Abstract**: tighten to one paragraph. Lead with the budget+credit-assignment framing. Don't mention deprecated reward terms.
- **§1 Introduction**: rewrite around the 6-step structure from `reader-review.md` §10. Add a worked example of per-op credit before any math. Cite TicToc, Memory-R1, DeltaMem, RLM, Mastra. End with Figure 1.
- **§2 Related Work**: organise by the 2×2 above (storage substrate × control layer source) and end with the cell we sit in. Add RLM, Cartridges, DeltaMem, Mem-α, the temporal-benchmark cluster.
- **§3 Method**: keep the op vocabulary; tighten the KGmem background to a "why it matters" first sentence; promote the chunking finding from engineering-notes to its own subsection ("Substrate matters: a 0% baseline before any RL"); move the deprecated-rewards paragraph to a footnote; rewrite the per-op counterfactual subsection to lead with the worked example, then the math, then the COMA citation, then the cost analysis.
- **§4 Datasets and Evaluation**: keep mostly as is but reorganise tier-1/tier-2 baselines into prose. Move TemporalBench to its own short subsection that frames it relative to TicToc and Real-Time Deadlines.
- **§5 Experiments**: keep TBD honest, but add two paragraphs that *would* fit results when they land — one on the random-K efficiency frontier and one on the head-to-head against KGmem extraction.
- **§6 Analysis**: leave skeleton; add a placeholder for the failure-mode analysis the reader review asked for.
- **§7 Discussion**: rewrite limitations as honest rather than apologetic. Add a future-work paragraph on combining mempol-style write learning with RLM-style read recursion.
- **§8 Conclusion**: kill the duplicate "facts/strategy" tag line.
- **refs.bib**: add the new entries.

The rewritten paper should be ~30% shorter at the level of body text (the bullet dumps in §3 collapse into prose) and ~10% longer at the level of figures and worked examples.
