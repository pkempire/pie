# PIE Temporal Reasoning Fix — Detailed Proposal

## The Problem (Precisely Diagnosed)

### What's Failing

Question: "At what time did E74 stop being the R20 of E63?"
- **Baseline sees**: `E74 was the R20 of E63 from 1964 to 1973.`
- **Baseline answers**: 1973 ✅
- **PIE sees**: 2000+ line reformulation with entity timelines, overlaps, durations
- **PIE answers**: 1974 ❌ (off by 1 year — noise drowning signal)

Question: "In 1988, E41 was the R53 of which entity?"
- **Baseline sees**: Direct lookup in raw facts
- **PIE sees**: Facts scattered across entity timelines with duration descriptions
- **PIE answers**: Wrong entity (can't find the right fact in the noise)

### Root Cause

**The reformulation REPLACES raw facts instead of AUGMENTING them.**

Current approach:
```
Raw facts → LLM reformulation → Semantic narrative (loses precision)
```

What we need:
```
Raw facts → BOTH preserved AND augmented with semantic context
```

### Evidence from SOTA

**Graphiti (Zep)** achieves 18.5% improvement over baselines on temporal tasks using:
- **Bi-temporal model**: Every fact has event_time AND ingestion_time
- **Raw facts preserved**: Never delete, just timestamp as invalid
- **Semantic layer ON TOP**: Graph structure + narrative as additional signal

**Honcho** achieves 94.9% on knowledge-update questions using:
- **Explicit extraction first**: Store exactly what was said
- **Deductive reasoning second**: Derive implications
- **Multiple representation layers**: Facts → Conclusions → Peer representations

---

## The Fix: Hybrid Temporal Context

### Principle: Don't Replace, Augment

For any temporal query, the model should see:

```
=== RAW TEMPORAL FACTS ===
(Filtered to relevant entities, preserving exact dates)

E74 was the R20 of E63 from 1964 to 1973.
E74 was the R20 of E91 from 1969 to 1978.
E74 was the R20 of E10 from 1962 to 1970.

=== DERIVED TEMPORAL FACTS ===
(Explicit facts derived from raw data — things the LLM shouldn't have to compute)

E74 stopped being R20 of E63 in 1973.
E74 started being R20 of E63 in 1964.
E74's R20 tenure with E63 lasted 9 years.
E74 held R20 of E63 and E10 concurrently from 1964-1970.

=== SEMANTIC TEMPORAL CONTEXT ===
(For understanding evolution, ordering, patterns)

Entity E74 — R20 Role Summary:
- First R20 role: E76 (1957)
- Last R20 role: E91 (ended 1978)
- Most concurrent R20 roles: 4 entities during 1962-1964
- Longest R20 tenure: E91 (9 years)
- Total span: 21 years (1957-1978)
```

### Implementation

#### 1. Fact-Level Derivation (NEW)

When storing `E74 was the R20 of E63 from 1964 to 1973`, automatically derive:

```python
derived_facts = [
    f"E74 started being R20 of E63 in 1964",
    f"E74 stopped being R20 of E63 in 1973",
    f"E74's R20 tenure with E63 lasted 9 years",
    f"E74 was R20 of E63 during the period 1964-1973",
]
```

**Why**: The LLM doesn't have to parse "from X to Y" and extract the end date. The derived fact explicitly answers "when did it stop?"

#### 2. Query-Adaptive Context Selection (NEW)

Detect query type and adjust context:

| Query Type | Raw Facts | Derived Facts | Semantic Context |
|------------|-----------|---------------|------------------|
| Point-in-time ("in 1988") | ✅ Full | ✅ Full | ❌ Skip |
| End/start time ("when did X stop") | ✅ Relevant | ✅ Full | ❌ Skip |
| Ordering ("what happened first") | ⚠️ Minimal | ✅ Full | ✅ Full |
| Duration ("how long") | ⚠️ Minimal | ✅ Full | ✅ Full |
| Evolution ("how did X change") | ❌ Skip | ✅ Full | ✅ Full |

**Implementation**:
```python
def classify_temporal_query(question: str) -> str:
    """Classify query to determine optimal context strategy."""
    
    patterns = {
        "point_in_time": ["in 19", "in 20", "during 19", "at the time"],
        "end_time": ["stop", "end", "finish", "cease", "no longer"],
        "start_time": ["start", "begin", "first", "initial"],
        "ordering": ["first", "last", "before", "after", "earlier", "later"],
        "duration": ["how long", "duration", "years", "span"],
        "evolution": ["change", "evolve", "transition", "shift"],
    }
    
    q_lower = question.lower()
    for qtype, keywords in patterns.items():
        if any(kw in q_lower for kw in keywords):
            return qtype
    return "general"
```

#### 3. Targeted Retrieval (NEW)

Instead of dumping the entire reformulated graph, retrieve ONLY:
- Entities mentioned in the question
- Direct relationships of those entities
- 1-hop neighbors if needed for context

For "At what time did E74 stop being R20 of E63?":
- Extract: E74, R20, E63
- Retrieve: All facts involving E74 as R20, all facts involving E63
- NOT: Entire 1111-fact corpus reformulated

#### 4. Keep Original Format Available

For arithmetic queries, the original format is optimal:
```
E74 was the R20 of E63 from 1964 to 1973.
```

Don't reformulate. The LLM can parse this directly.

---

## Concrete Code Changes

### 1. New Module: `pie/temporal/derived_facts.py`

```python
def derive_temporal_facts(fact: TemporalFact) -> list[str]:
    """Generate explicit derived facts from a temporal fact."""
    derived = []
    
    # Start/end times (critical for temporal lookup)
    derived.append(f"{fact.subject} started being {fact.role} of {fact.obj} in {fact.start}.")
    derived.append(f"{fact.subject} stopped being {fact.role} of {fact.obj} in {fact.end}.")
    
    # Duration
    duration = fact.end - fact.start
    derived.append(f"{fact.subject}'s {fact.role} tenure with {fact.obj} lasted {duration} years.")
    
    # Period membership
    derived.append(f"{fact.subject} was {fact.role} of {fact.obj} during {fact.start}-{fact.end}.")
    
    return derived
```

### 2. Modified Benchmark Runner

```python
def build_hybrid_context(facts: list[TemporalFact], question: str) -> str:
    """Build hybrid context with raw + derived + semantic layers."""
    
    qtype = classify_temporal_query(question)
    relevant_facts = filter_relevant_facts(facts, question)
    
    sections = []
    
    # Section 1: Raw facts (always include for precision)
    sections.append("=== RAW TEMPORAL FACTS ===")
    for f in relevant_facts:
        sections.append(f.raw)
    
    # Section 2: Derived facts (explicit answers to common queries)
    sections.append("\n=== DERIVED TEMPORAL FACTS ===")
    for f in relevant_facts:
        for derived in derive_temporal_facts(f):
            sections.append(derived)
    
    # Section 3: Semantic context (only for ordering/evolution queries)
    if qtype in ["ordering", "duration", "evolution", "general"]:
        sections.append("\n=== SEMANTIC TEMPORAL CONTEXT ===")
        sections.append(build_semantic_summary(relevant_facts))
    
    return "\n".join(sections)
```

### 3. Entity Filtering

```python
def filter_relevant_facts(facts: list[TemporalFact], question: str) -> list[TemporalFact]:
    """Extract only facts relevant to the question."""
    
    # Extract entity IDs from question
    entities = re.findall(r'E\d+', question)
    roles = re.findall(r'R\d+', question)
    
    relevant = []
    for f in facts:
        # Include if any entity/role matches
        if f.subject in entities or f.obj in entities:
            relevant.append(f)
        elif f.role in roles and (f.subject in entities or f.obj in entities):
            relevant.append(f)
    
    # If no matches, fall back to all facts (shouldn't happen for well-formed queries)
    return relevant if relevant else facts
```

---

## Expected Impact

### Test of Time (ToT)

| Query Type | Current | Expected | Improvement |
|------------|---------|----------|-------------|
| event_at_what_time | 54% | 85%+ | +31% |
| event_at_time_t | 0% | 70%+ | +70% |
| before_after | 0% | 60%+ | +60% |
| first_last | 58% | 75%+ | +17% |
| relation_duration | 58% | 80%+ | +22% |
| **Overall** | 34% | **70%+** | +36% |

### LongMemEval Temporal

Current naive_rag: 60% on temporal
Expected with hybrid: 80%+

### Why This Will Work

1. **Preserves precision**: Raw facts are ALWAYS available
2. **Reduces noise**: Only relevant facts included
3. **Explicit answers**: Derived facts directly answer common query patterns
4. **Best of both**: Semantic context available when it helps, skipped when it hurts

---

## Implementation Priority

1. **P0**: Implement derived fact generation
2. **P0**: Implement query classification
3. **P0**: Implement entity-based filtering
4. **P0**: Build hybrid context function
5. **P1**: Re-run ToT benchmark with hybrid approach
6. **P1**: Re-run LongMemEval temporal with hybrid approach
7. **P2**: Integrate into main PIE pipeline
8. **P2**: Add to all benchmark runners

---

## Open Questions

1. **How much filtering is optimal?** Too aggressive filtering might miss relevant context. Need to tune.

2. **Should derived facts be stored or computed?** Storing is faster but uses more space. Computing is slower but always fresh.

3. **Query classification accuracy**: What if we misclassify? Need fallback to hybrid mode.

4. **How does this interact with the main PIE world model?** Need to ensure the temporal fix integrates cleanly.
