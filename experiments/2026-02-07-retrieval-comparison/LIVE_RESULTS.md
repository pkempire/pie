# PIE Retrieval Experiment — Live Results

**Started:** 2026-02-07 18:16 EST  
**Status:** Running

## Streams

### Stream A: naive_rag on LongMemEval (100 questions)
- **Progress:** 22/100
- **Running accuracy:** **91%** 🔥
- **Session:** good-seaslug

### Stream C: Extraction Quality Audit
- **Status:** ✅ Complete
- **Finding:** **CRITICAL — 0 event entities in world model**
- See: stream_c_extraction_audit.md

### Stream D: Temporal-reasoning category (naive_rag)
- **Progress:** 34/133
- **Running accuracy:** **72%**
- **Session:** oceanic-reef

## Key Finding: No Temporal Entities

The main PIE pipeline (amber-seaslug run) extracted:
- 1,057 entities
- 0 events with dates
- 0 valid_from/valid_to timestamps
- Only 6 "period" entities

**This means pie_temporal CANNOT work on temporal queries** — there's nothing to retrieve.

## Implication for Paper/Blog

1. **naive_rag is the real baseline** — ~88% on simple facts, ~76% on temporal
2. **pie_temporal failures are extraction bugs, not retrieval bugs**
3. **Need to re-run extraction with event-focused prompt before comparing**

## Next Steps

1. Wait for Stream A/D to complete
2. Fix extraction prompt to produce events with dates
3. Re-run pipeline on ChatGPT history with fixed prompt
4. Then compare graph_aware vs naive_rag fairly
