---
title: "Belief revision"
year: 2026
category: "capability-gap"
tags: ["belief", "revision", "AGM", "JTMS", "contradictions", "consistency"]
---

# Belief revision

When new information contradicts what's already in memory, you have a choice: keep the old, accept the new, or reframe both. Every memory system today either appends-only or overrides on conflict. Neither matches how real cognition handles contradictions.

## What current systems do

- **Append-only (Mem0, naive RAG, observation logs)**: contradictions sit side-by-side. The retriever returns both; the answer LLM picks one (often wrong).
- **Overwrite (most KG systems)**: new value replaces old. History destroyed. Can't answer "what did they believe last week."
- **Mark-as-contradiction (PIE, some KG variants)**: both states preserved with a flag. The reader must interpret. Usually does this poorly.
- **Bi-temporal edges (Zep / Graphiti)**: valid-time interval shrinks when contradicted. Closest to right. Doesn't propagate.

None of these do *belief revision* — re-evaluating dependent beliefs when an upstream claim changes status.

## The classical AI primitives that exist but aren't used

- **Truth Maintenance Systems (Doyle 1979)**: nodes are propositions; justification links record why each is believed. Retracting a premise propagates to dependents. Well-understood algorithm; never integrated with LLM-extracted facts.
- **AGM Belief Revision (Alchourrón, Gärdenfors, Makinson 1985)**: when new evidence contradicts the belief set, find the *minimal change* that restores consistency. Three operators: expansion, revision, contraction.
- **Dung's Argumentation Framework (1995)**: arguments + attack relations + formal semantics (grounded, preferred, stable extensions). Status of each argument is computed, not stored.

These are 30-50 year old techniques that LLM-memory systems haven't picked up.

## Why integration hasn't happened

Three reasons:
1. **Schema mismatch**: TMS nodes are propositions; LLM extractions are text. Identity of "the same fact across rephrasings" is fuzzy.
2. **Justification capture**: LLM systems rarely log *why* they wrote a fact. Without justification links, retraction propagation has no graph to walk.
3. **Performance**: AGM minimal-change computation is NP-hard in general; real implementations use heuristics. Adds overhead.

None of these are insurmountable. (1) is solved by normalizing extractions to canonical entity-attribute-value tuples. (2) is solved by writing source-trajectory references at extraction time ([[2605.20616-auto-dreamer|Auto-Dreamer]] does this). (3) is mitigated by bounding belief-set size.

## What an LLM-memory system with belief revision would look like

Every memory entry carries:
- Content (the claim)
- Derivation: which prior entries (and source trajectories) it depends on
- Status: Supported | Contested | Defeated (computed, not stored)
- Attacks: which entries it contradicts

When a new entry arrives that attacks an existing one, AGM revision picks the minimal-change update. Entries with broken justifications propagate Defeated downstream. The retriever at QA time only returns Supported entries by default; can return Contested with annotation when asked.

## Concrete experiment that would test this

Take LoCoMo's adversarial questions (the ones that test handling of contradicting facts like Caroline's-then-Melanie's-running-practice). Run two pipelines: (a) append-only memory, (b) memory with explicit derivation + AGM revision. Compare accuracy specifically on the adversarial split.

Hypothesis: belief-revision-augmented memory wins by 20+ points on adversarial (where the current PIE baseline collapses to 10.6%), with neutral effect on other categories.

## Why this is unclaimed

The combination of (i) LLM-extracted memory, (ii) JTMS-shaped derivation tracking, (iii) AGM-style revision operators isn't in any published paper as of May 2026. Recent work (Conversation as Belief Revision, AAAI 2026) applies AGM to single-conversation logical consistency, not multi-week memory state.

## See also

- [[sleep-consolidation]] — consolidator could be the place to run revision passes
- [[time-aware-memory]] — temporal validity + belief revision are different but compose
- [[substrate-design-space]] — derivation links require substrate support
