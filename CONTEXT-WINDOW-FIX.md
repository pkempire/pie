# Context Window Architecture — Revised

## Problem
Per-conversation mention detection fails when users work on subtasks without naming the parent project. "Write a curriculum outline" doesn't trigger entity resolution against "Lucid Academy" even though that's clearly what it's for.

## Solution: Sliding Window + Activity-Based Context

### Ingestion Flow (Revised)

```
Conversations sorted chronologically
         │
         ▼
┌─────────────────────────────┐
│  Group into daily batches    │  ← All conversations from same day
│  (or N-token windows)        │     processed as one unit
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│  Build context preamble:     │
│  1. Recently active entities │  ← Entities touched in last 3 days
│     (name, type, state)      │     regardless of mention in current batch
│  2. Recent state changes     │  ← Last 5-10 state changes across graph
│  3. Active project summaries │  ← Current state of all project entities
│                              │     with importance > 0.3
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│  Extract from daily batch    │
│  with context preamble       │  ← LLM sees: context + all convos from day
│                              │     Can connect subtasks to parent projects
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│  Entity resolution pass      │
│  Match extracted entities    │  ← Embedding similarity + LLM verify
│  against full graph          │     against recently active + top entities
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│  Update world model          │
│  Record transitions          │
└─────────────────────────────┘
```

### What "Recently Active" Means

Not just "recently mentioned" — recently *changed*. An entity is active if:
- It had a state transition in the last N conversations/days
- It was extracted from a conversation in the last N days
- It has high importance AND was active in the last 2 weeks

This is the key insight: the context window should be **activity-based, not mention-based**.

### Daily Batch vs Token Window

**Daily batches** (preferred):
- Natural boundary — a day's work is usually thematically coherent
- Average day in Parth's data: ~6 conversations, ~17K chars
- Fits comfortably in context with entity preamble
- Light days (1-2 convos) still get recent activity context
- Heavy days (15+ convos) might need sub-batching

**Token window** (fallback):
- Use if daily batches are too uneven
- Sliding window of N conversations (e.g., 10)
- Step by M conversations (e.g., 5) — 50% overlap ensures continuity
- Each window includes activity context from prior windows

### Context Preamble Template

```
=== CURRENT WORLD STATE (as of {date}) ===

ACTIVE PROJECTS:
- Lucid Academy: AI summer program for high schoolers. Currently in curriculum design phase.
  Last updated: 2 days ago. Changed 4 times this month.
- Real-Time API Doc Checker: POC for OpenAI API. Built with Streamlit + FAISS.
  Last updated: 1 week ago.

RECENTLY ACTIVE ENTITIES:
- Python curriculum (concept): being designed for 6-week syllabus
- TikTok Shop Affiliate (project): exploring affiliate marketing
- Neural networks (concept): topic for teaching materials

RECENT STATE CHANGES:
- AI Innovators → considering rename (3 days ago)
- TikTok Shop Affiliate → realized need Creator Center not Business Center (yesterday)

=== TODAY'S CONVERSATIONS ({N} conversations) ===
[conversation content follows]
```

### Why This Works

When the extractor sees "Write a lesson plan for week 3 — intro to neural networks" AND the preamble says "Lucid Academy: AI summer program, currently in curriculum design phase" + "Python curriculum: being designed for 6-week syllabus" — it can now:

1. Connect "lesson plan" to the Lucid Academy project
2. Connect "intro to neural networks" to the Python curriculum entity
3. Record a state change on the curriculum (now has week 3 content)

Without the preamble, all three would be orphaned generic entities.

### Token Budget

Rough estimates for context:
- Activity preamble: ~500-1000 tokens
- Daily batch (avg 6 convos): ~4000 tokens
- Extraction prompt: ~800 tokens
- Total input: ~6000 tokens per batch
- Output: ~500-1000 tokens

Much more efficient than per-conversation extraction (fewer LLM calls) and more accurate (richer context).

### Comparison: Per-Conversation vs Daily Batch

| Aspect | Per-Conversation | Daily Batch |
|--------|-----------------|-------------|
| LLM calls per 927 convos | ~1854 (2 per convo) | ~309 (2 per day × ~155 days) |
| Context quality | Poor (no surrounding context) | Good (day's work + recent activity) |
| Entity resolution | Relies on mention detection | Guided by active project context |
| Implicit connections | Misses most | Catches via activity preamble |
| Cost (gpt-4o) | ~$18.50 | ~$8-12 |
| Subtask → project linking | ❌ | ✅ |
```
