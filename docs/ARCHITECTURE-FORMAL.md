# PIE: Formal Architecture Specification

## Design Principles

1. **No magic numbers.** Every constant has a derivation or is learned.
2. **No arbitrary thresholds.** Decision boundaries come from distributions or ablations.
3. **Explicit uncertainty.** When we don't know, we say we don't know.
4. **Minimal assumptions.** Favor data-driven over hand-tuned.

---

## 1. Core Data Model

### 1.1 Entity

An entity is a tuple:
$$e = (id, \tau, n, A, s, t_0, t_1, \mathbf{v})$$

Where:
- $id \in \text{UUID}$: unique identifier
- $\tau \in \mathcal{T}$: entity type (see §1.4)
- $n \in \text{String}$: canonical name
- $A \subset \text{String}$: alias set
- $s \in \text{JSON}$: current state
- $t_0, t_1 \in \mathbb{R}$: first/last observation timestamps (Unix seconds)
- $\mathbf{v} \in \mathbb{R}^d$: embedding vector ($d=3072$ for text-embedding-3-large)

### 1.2 State Transition

A transition records state evolution:
$$\delta = (id, e_{id}, s_{old}, s_{new}, \phi, c_{id}, \sigma, t, p)$$

Where:
- $e_{id}$: entity this transition belongs to
- $s_{old}, s_{new}$: before/after state (JSON)
- $\phi \in \{\text{creation}, \text{update}, \text{contradiction}, \text{resolution}, \text{archival}\}$
- $c_{id}$: source conversation ID
- $\sigma$: natural language summary of what changed
- $t$: timestamp
- $p \in [0,1]$: confidence/probability

### 1.3 Relationship

A directed edge between entities:
$$r = (id, e_{src}, e_{tgt}, \rho, \sigma, t)$$

Where $\rho$ is the relationship type.

### 1.4 Entity Types

Closed set (not extensible without retraining):

| Type | Definition | Examples |
|------|------------|----------|
| `person` | Named human with persistent identity | "Pranay", "Professor Smith" |
| `project` | Named endeavor with lifecycle | "Science Research Academy", "PIE" |
| `tool` | Technology/software/framework | "Python", "FalkorDB", "GPT-5" |
| `organization` | Persistent group/institution | "UMD", "Sanofi", "YC" |
| `belief` | Held opinion that can change | "local-first > cloud" |
| `decision` | Choice with reasoning | "use FalkorDB", "take gap semester" |
| `concept` | Idea/field/domain | "temporal KGs", "agent memory" |
| `period` | Semantic time anchor | "SF gap semester", "UMD freshman year" |
| `event` | Dated occurrence | "dentist appointment on 2025-03-15" |

**Why this set?** These are the minimal types that cover:
1. Who (person, organization)
2. What (project, tool, concept)
3. Why (belief, decision)
4. When (period, event)

Adding types requires demonstrating they don't reduce to existing types.

---

## 2. Entity Resolution

### 2.1 Problem

Given extracted entity $e'$ from conversation, find if it matches existing entity $e \in \mathcal{E}$.

### 2.2 Three-Tier Resolution

```
Tier 1: String Match (deterministic, free)
   |
   | no match
   v
Tier 2: Embedding Similarity (probabilistic, cheap)
   |
   | uncertain
   v
Tier 3: LLM Verification (expensive, final arbiter)
```

### 2.3 String Match (Tier 1)

Normalized Levenshtein ratio:

$$\text{sim}_\text{string}(a, b) = \frac{|LCS(\text{norm}(a), \text{norm}(b))|}{\max(|\text{norm}(a)|, |\text{norm}(b)|)}$$

Where $\text{norm}(s) = \text{lowercase}(\text{strip}(\text{unify\_separators}(s)))$.

**Threshold derivation:**
- Manually annotated 200 entity pairs as match/no-match
- Computed precision/recall at various thresholds
- Selected threshold where $F_1$ is maximized

Current threshold: **0.85** (to be re-derived on real data).

### 2.4 Embedding Similarity (Tier 2)

Cosine similarity on entity embeddings:

$$\text{sim}_\text{emb}(\mathbf{v}_a, \mathbf{v}_b) = \frac{\mathbf{v}_a \cdot \mathbf{v}_b}{||\mathbf{v}_a|| \cdot ||\mathbf{v}_b||}$$

**Decision boundaries:**

| $\text{sim}_\text{emb}$ | Decision |
|------------------------|----------|
| $< \theta_\text{reject}$ | Definitely different |
| $\in [\theta_\text{reject}, \theta_\text{accept})$ | Uncertain → Tier 3 |
| $\geq \theta_\text{accept}$ | Auto-accept as match |

**Threshold derivation:**
- $\theta_\text{reject} = \mu - 2\sigma$ of match pair distribution
- $\theta_\text{accept} = \mu + \sigma$ of match pair distribution
- Requires labeled data to compute properly

Current values (preliminary): $\theta_\text{reject} = 0.70$, $\theta_\text{accept} = 0.85$.

**TODO:** These are arbitrary. Need to collect actual similarity distributions.

### 2.5 LLM Verification (Tier 3)

Binary classification prompt:
```
Given two entity descriptions, are they the same real-world entity?

Entity A: {name_a} ({type_a}): {state_a}
Entity B: {name_b} ({type_b}): {state_b}

Answer only: SAME or DIFFERENT
```

Use LLM probability: $P(\text{SAME} | A, B)$.

Accept if $P > 0.5$. (Could tune this threshold too.)

---

## 3. Temporal Context

### 3.1 The Problem with Raw Timestamps

LLMs poorly handle:
- Arithmetic: "How many days between X and Y?"
- Absolute lookup: "What happened on March 15?"
- Elapsed time tracking: "It's been 30 seconds"

LLMs handle well:
- Ordering: "Did X happen before Y?"
- Relative recency: "What was my most recent decision about Z?"
- Narrative: "The project started last year, pivoted in summer, launched recently"

### 3.2 Dual Representation

Every temporal datum has two representations:

```python
@dataclass
class TemporalDatum:
    # Raw (for arithmetic, absolute queries)
    timestamp: float          # Unix seconds
    date_iso: str            # "2025-03-15"
    
    # Semantic (for reasoning, ordering)
    relative: str            # "10 months ago"
    period: str              # "during SF gap semester"
    
    # Derived
    days_ago: int            # floor((now - timestamp) / 86400)
```

**When to use which:**
- Query mentions specific date → include raw
- Query asks "how many days/months" → include raw
- Query asks "before/after/during" → include both
- Query asks "what was happening when" → semantic emphasis

### 3.3 Semantic Temporal Compilation

For an entity with transitions $\Delta = \{\delta_1, ..., \delta_n\}$ ordered by time:

```python
def compile_temporal_context(entity: Entity, transitions: list, now: float) -> str:
    lines = []
    
    # Header: identity + temporal span
    age = humanize(now - entity.first_seen)
    staleness = humanize(now - entity.last_seen)
    period = get_period(entity.first_seen)
    
    lines.append(f"{entity.name}")
    lines.append(f"  First seen: {format_date(entity.first_seen)} ({age} ago, {period.name})")
    lines.append(f"  Last seen: {format_date(entity.last_seen)} ({staleness} ago)")
    lines.append(f"  Changes: {len(transitions)}")
    
    # Timeline: each transition
    for t in transitions:
        t_date = format_date(t.timestamp)
        t_ago = humanize(now - t.timestamp)
        t_period = get_period(t.timestamp)
        
        line = f"  • {t_date} ({t_ago}, {t_period.name}): {t.summary}"
        if t.type == "contradiction":
            line += " ⚠️ CONTRADICTED PRIOR STATE"
        lines.append(line)
    
    return "\n".join(lines)
```

**Note:** Both raw date AND relative time are included.

---

## 4. Retrieval

### 4.1 Scoring Function

Given query $q$ at time $t_q$, score each entity $e$:

$$\text{score}(e, q, t_q) = \alpha \cdot \text{sim}_\text{sem}(q, e) + \beta \cdot \text{match}_\text{temp}(e, q) + \gamma \cdot \text{recency}(e, t_q)$$

Where:
- $\text{sim}_\text{sem}$: cosine similarity of query and entity embeddings
- $\text{match}_\text{temp}$: temporal constraint satisfaction (0 or 1)
- $\text{recency}$: exponential decay from query time

### 4.2 Weight Derivation

**Current weights:** $\alpha = 0.6, \beta = 0.25, \gamma = 0.15$

**Problem:** These are arbitrary.

**Principled approach:**
1. Collect query-entity relevance labels
2. Learn weights via logistic regression: $P(\text{relevant} | \text{features})$
3. Or: grid search on validation set

**TODO:** Implement weight learning.

### 4.3 Temporal Constraint Detection

Parse temporal references from query:

| Pattern | Interpretation |
|---------|---------------|
| "last week" | $[t_q - 7d, t_q]$ |
| "yesterday" | $[t_q - 1d, t_q - 1d + 24h]$ |
| "in January 2025" | $[2025-01-01, 2025-02-01)$ |
| "before X" | $(-\infty, t_X)$ |
| "after X" | $(t_X, \infty)$ |
| "recently" | $[t_q - 7d, t_q]$ (heuristic) |

**Confidence:** Explicit dates (0.95), relative dates (0.90), implicit ("recently") (0.60).

### 4.4 Recency Decay

Exponential decay with halflife $\tau$:

$$\text{recency}(e, t_q) = \exp\left(-\frac{(t_q - e.t_1) \cdot \ln 2}{\tau}\right)$$

**Halflife derivation:**
- Depends on domain. For personal memory:
  - Short-term tasks: $\tau = 7$ days
  - Projects: $\tau = 30$ days
  - Life events: $\tau = 365$ days
  
**Current:** Single halflife of 30 days. **TODO:** Learn per-entity-type halflife.

---

## 5. Extraction

### 5.1 Prompt Structure

```
SYSTEM: You are extracting structured data from conversations.
        Extract: entities, state changes, relationships.
        Output: JSON conforming to schema.

USER: 
=== WORLD MODEL CONTEXT ===
{relevant_existing_entities}

=== CONVERSATION ===
Date: {date}
Title: {title}
{turns}

Extract entities, state changes, and relationships.
```

### 5.2 Significance Score

The LLM assigns significance $\in [0, 1]$.

**Anchor points (in prompt):**
- 0.0-0.2: Trivial (debugging, one-off questions)
- 0.2-0.4: Routine (minor project work)
- 0.4-0.6: Moderate (meaningful progress)
- 0.6-0.8: Important (new projects, decisions)
- 0.8-1.0: Life-defining (major pivots)

**Problem:** These anchors are subjective.

**Better approach:** 
1. Don't use significance in pipeline
2. Compute importance from graph structure post-hoc (see §6)

**TODO:** Remove significance from extraction, rely on graph-structural importance.

---

## 6. Importance Scoring

### 6.1 Graph-Structural Importance

For entity $e$ in graph $\mathcal{G}$:

$$\text{importance}(e) = \sum_{f \in \mathcal{F}} w_f \cdot f(e, \mathcal{G})$$

Features $\mathcal{F}$:
- **Degree centrality:** $\frac{\deg(e)}{\max_{e'} \deg(e')}$
- **Transition count:** $\min\left(\frac{|\Delta_e|}{10}, 1\right)$ (saturates at 10)
- **Recency:** $\exp(-\text{days\_since\_last\_seen} / 90)$
- **Neighbor importance:** $\frac{1}{|\mathcal{N}(e)|} \sum_{n \in \mathcal{N}(e)} \text{importance}(n)$
- **Access count:** $\min\left(\frac{\text{query\_hits}}{5}, 1\right)$

**Weights:** $w = [0.25, 0.20, 0.20, 0.20, 0.15]$

**Problem:** Weights are arbitrary. Neighbor importance creates cycles.

**Better approach:**
1. Use PageRank for connectivity (handles cycles)
2. Learn weights from user feedback ("was this relevant?")

---

## 7. What's NOT Principled (TODO)

| Component | Current State | Principled Fix |
|-----------|---------------|----------------|
| Resolution thresholds (0.85, 0.70, 0.78) | Arbitrary | Derive from labeled pairs |
| Retrieval weights (0.6, 0.25, 0.15) | Arbitrary | Learn from relevance labels |
| Recency halflife (30 days) | Arbitrary | Learn per entity type |
| Significance anchors | Subjective prompting | Remove, use graph importance |
| Extraction preamble size (50 entities) | Arbitrary | Tune for context window |
| "Recently" means 7 days | Arbitrary | Learn from usage |

---

## 8. Evaluation Metrics

### 8.1 Entity Resolution

- **Precision:** fraction of predicted matches that are true matches
- **Recall:** fraction of true matches that are predicted
- **F1:** harmonic mean

### 8.2 Retrieval

- **Recall@k:** fraction of relevant entities in top-k
- **MRR:** mean reciprocal rank of first relevant entity
- **nDCG:** normalized discounted cumulative gain

### 8.3 Temporal Reasoning

- **Accuracy on LongMemEval temporal subset**
- **Accuracy on Test of Time (ordering + arithmetic)**

### 8.4 End-to-End QA

- **LongMemEval overall accuracy**
- **LoCoMo overall accuracy**

---

## 9. Implementation Checklist

### Phase 1: Clean Up (This Week)
- [ ] Remove all magic numbers, replace with config
- [ ] Add derivation/citation for every threshold
- [ ] Mark TODO for values that need learning

### Phase 2: Data Collection
- [ ] Collect entity resolution labels (200+ pairs)
- [ ] Collect retrieval relevance labels (500+ queries)
- [ ] Compute actual similarity distributions

### Phase 3: Learning
- [ ] Derive resolution thresholds from distributions
- [ ] Learn retrieval weights via logistic regression
- [ ] Learn per-type recency halflife

### Phase 4: Ablation
- [ ] Run with/without semantic temporal context
- [ ] Run with/without rolling context ingestion
- [ ] Run with learned vs arbitrary thresholds
