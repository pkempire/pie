# PIE Extraction Inspection Notes — 2025-02-06

## What's Working

### Event Date Extraction: 100%
All events have computed dates from the batch header. The prompt instructions for date computation are working:
- "today" / "I just [verb]" → batch date
- "yesterday" → batch date - 1
- Relative dates being correctly computed

### State Evolution Tracking
Example: **Framer** entity shows 6 transitions:
1. Created 2025-01-05 — "evaluating for popup implementation"
2. Updated — "building scroll-triggered slide-in popup"
3. Updated — "exploring cost-saving strategies for editor access"

This is the trajectory reconstruction that flat fact stores can't do.

### Entity Resolution Improving Over Time
- Batch 4: 9 new, 0 matches
- Batch 5: 3 new, 2 matches  
- Batch 6: 6 new, 2 matches

As context grows, more entities get matched to existing ones.

### Relationship Detection
Working relationships:
- `part_of` — Streamlit dashboard → sponsorFind
- `uses` — sponsorFind → Deepseek
- `caused_by` — question event → concept
- `related_to` — linking related topics

### Entity Type Diversity
After 7 batches: 5 projects, 5 tools, 5 decisions, 10 events, 2 orgs, 4 concepts, 1 belief

## Issues Found

### 1. Event/Concept Duplication
Same information extracted twice:
- "Asked about factors affecting blood cholesterol" (event)
- "Factors affecting blood cholesterol" (concept)

**Fix idea:** Don't extract concepts from low-significance batches, or dedupe event+concept pairs in post-processing.

### 2. Verbose Entity Names
"Decision: Parallelize LLM calls without multiple API keys" — could be shorter.

**Fix idea:** Add instruction to use concise canonical names.

### 3. Some Batches Have 0 Matches
Batch 4 and 7 had 0 matched entities — could be genuinely new topics, or resolution is missing matches.

**Investigate:** Check if embedding similarity is working correctly for these batches.

### 4. Duplicate Projects
"Streamlit dashboard" and "sponsorFind" are the same project but extracted separately. The `part_of` relationship exists but they're not merged.

**Fix idea:** Post-processing pass to merge entities with `part_of` relationships where the child is small.

## Performance Notes

- Small batches (< 20K chars): 40-60s
- Medium batches (20-60K chars): 60-80s
- Large batches (100K+ chars): 80-120s (truncated to 60K)

Bottleneck is 100% OpenAI API. Parallel batching would break sliding window.

## Key Learning

The sliding window context preamble is crucial. It shows:
1. Active projects with state
2. Recently active entities
3. Recent state changes with relative time ("2 days ago")

This gives the LLM context to match new extractions to existing entities.

## Recommendations

1. **Run full extraction** — Let it complete all 203 batches
2. **Analyze resolution stats** — Check string vs embedding vs LLM match rates
3. **Review entity quality** — Sample 50 entities for precision/recall
4. **Test temporal queries** — Run actual trajectory reconstruction queries
5. **Benchmark comparison** — Run ToT and LongMemEval with PIE data
