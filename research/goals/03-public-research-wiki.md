---
title: "Goal 03 — Public research wiki + educational content"
status: "active"
priority: 3
started: "2026-05-26"
owner: "us"
budget: "no $ — time only"
tags: ["wiki", "education", "public", "content", "active"]
---

# Goal 03 — Public research wiki + educational content

## What we're trying to prove

A Karpathy-spartan markdown wiki backed by `research/papers/`, `research/concepts/`, `research/systems/`, and `research/goals/` is sufficient to (a) navigate our research without restarting from scratch every session, and (b) anchor a series of educational videos / blog posts on AI memory.

## Current state (2026-05-26)

Built. 45 pages rendering. Backlinks bidirectional. Verified vs unverified leaderboard claims marked explicitly. Server runs locally with `python -m research.wiki.build --serve`.

Counts:
- 25 papers (top-10 LoCoMo SOTA + key foundational + GEPA + Auto-Dreamer + TML + etc.)
- 9 concept pages
- 7 system pages
- 3 active goals (this file is one of them)

## Next steps

1. **Push to GitHub Pages** — public URL, version-controlled, every commit redeploys. ~30 min.
2. **Fill remaining dead-link gaps** — `mastra-om`, `zep`, `search-r1` system pages exist; need a couple more concept pages.
3. **First educational video script** — "Why memory is the bottleneck for AI agents." Opens with Mem0's self-reported 40% extraction-failure rate. Walks through Mem0 → Mastra → TiMem → EverMemOS → Auto-Dreamer. ~15 min.
4. **Second script** — "GEPA: when natural-language gradient beats RL." Anchored on the [[gepa-vs-grpo]] concept page.
5. **Repeat** for the other 4 concept pages worth recording: sleep-consolidation, noreplay-vs-retrieval, time-aware-memory, multi-agent-delegation.

## Success criterion

- Wiki is publicly accessible at a URL anyone can link to
- ≥6 videos shipped over the next quarter
- Real engagement on at least one (≥1000 views, ≥10 substantive comments)
- Gets cited by at least one external paper in the memory-systems space

## Why this matters

The educational angle is its own product. The memory↔autoresearch bridge has no coverage today (Karpathy LLM wiki dominates memory discourse; Lex Fridman covers state-of-AI but not this specific space; Yannic/AI Coffee Break haven't touched 2026 memory work). Owning the explainer space for memory + GEPA is an unclaimed niche.

## Related

- Concept: [[sleep-consolidation]] (video 1)
- Concept: [[noreplay-vs-retrieval]] (video 2)
- Concept: [[gepa-vs-grpo]] (video 3)
- Concept: [[time-aware-memory]] (video 4)
- Concept: [[multi-agent-delegation]] (video 5)
- Concept: [[memory-budget-curves]] (video 6)
- Goal: [[goal-01-gepa-consolidator-on-locomo]] (provides the live-coding material for video 3)
