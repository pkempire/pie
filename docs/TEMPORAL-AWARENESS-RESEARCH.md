# Temporal Awareness in LLMs: Research & Implementation Deep Dive

**Goal:** Make LLMs internally track and reason about time without explicit injection at every turn.

**Core problem:** LLMs are stateless next-token predictors. They have no internal clock, no sense of elapsed time, no temporal state. Time only exists if it's in the tokens.

---

## 1. Temporal Position Embeddings

### Concept
Like positional encoding gives tokens spatial position, temporal embeddings give them "when" position. Each token gets a temporal vector that encodes its absolute or relative time.

### Implementation Approaches

**a) Absolute temporal encoding:**
```python
class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, max_time=365*24*3600):  # 1 year in seconds
        super().__init__()
        self.time_encoder = nn.Linear(1, d_model)
        self.max_time = max_time
    
    def forward(self, timestamps):
        # timestamps: [batch, seq_len] unix timestamps
        normalized = timestamps / self.max_time
        return self.time_encoder(normalized.unsqueeze(-1))

# Inject into transformer:
# hidden = token_emb + pos_emb + temporal_emb(timestamps)
```

**b) Relative temporal encoding (like RoPE but for time):**
```python
def temporal_rope(x, timestamps):
    """Apply rotary temporal embedding based on time deltas."""
    time_deltas = timestamps[:, :, None] - timestamps[:, None, :]  # pairwise
    # Apply sinusoidal encoding to time deltas
    freqs = 1.0 / (10000 ** (torch.arange(0, d_model, 2) / d_model))
    angles = time_deltas.unsqueeze(-1) * freqs
    return x * torch.cos(angles) + rotate_half(x) * torch.sin(angles)
```

**c) Continuous-time transformers (from temporal point processes):**
The TPP-TAL paper (arxiv 2601.00845) aligns temporal dynamics with contextual semantics *before* feeding to the LLM:
```python
# Instead of concat(event_emb, time_emb), they do:
temporal_context = cross_attention(event_emb, time_emb)
llm_input = project(temporal_context)
```

### Pros/Cons
- **Pro:** Native temporal reasoning, no external scaffolding needed
- **Con:** Requires retraining or significant fine-tuning
- **Con:** Unclear how to represent "elapsed time during conversation" vs "timestamp of events"

### Research Questions
- Does temporal encoding help more for retrieval (finding relevant past events) or reasoning (ordering, duration calculation)?
- What granularity matters? Seconds? Hours? Days?
- How to encode uncertain/fuzzy times ("last week", "recently")?

---

## 2. Clock Token Injection

### Concept
Insert explicit time markers at regular intervals during generation. Stateless but effective — what the Real-Time Deadlines paper's "time-aware" condition does.

### Implementation

**a) Simple injection:**
```python
def inject_clock_tokens(conversation, interval_seconds=30):
    """Insert [TIME: Xs remaining] tokens at fixed intervals."""
    result = []
    elapsed = 0
    for turn in conversation:
        elapsed += estimate_turn_duration(turn)
        if elapsed >= interval_seconds:
            result.append(f"[TIME: {deadline - elapsed:.0f}s remaining]")
            elapsed = 0
        result.append(turn)
    return result
```

**b) Adaptive injection (more urgent = more frequent):**
```python
def adaptive_clock(remaining_time, total_time):
    """More frequent updates as deadline approaches."""
    urgency = 1 - (remaining_time / total_time)
    interval = max(5, 60 * (1 - urgency))  # 60s -> 5s as deadline nears
    return interval
```

**c) Semantic clock (not just numbers):**
```python
def semantic_time_marker(remaining, total, context):
    """Natural language time awareness."""
    pct = remaining / total
    if pct > 0.75:
        return ""  # No urgency yet
    elif pct > 0.5:
        return "(Roughly half your time has passed.)"
    elif pct > 0.25:
        return "(Time is running short—consider moving toward agreement.)"
    elif pct > 0.1:
        return "(URGENT: Very little time remains. Finalize now or lose the deal.)"
    else:
        return "(CRITICAL: Seconds left. Accept or walk away NOW.)"
```

### Key Finding from Real-Time Deadlines Paper
Urgency cues ("Deadline approaching!") **outperformed** numeric countdowns. LLMs respond better to semantic pressure than raw numbers.

### Research Questions
- Optimal injection frequency?
- Does the model habituate to repeated clock tokens?
- How to balance informativeness vs context pollution?

---

## 3. Semantic Temporal Compilation (PIE's Approach)

### Concept
Don't give raw timestamps — convert temporal graph state into natural language narratives that LLMs can reason about natively.

### Implementation

```python
def compile_temporal_context(entity, world_model, current_time):
    """Convert temporal state to narrative."""
    
    history = world_model.get_transitions(entity)
    parts = []
    
    # Relative time framing
    for transition in history[-5:]:  # Last 5 changes
        delta = current_time - transition.timestamp
        relative = humanize_delta(delta)  # "3 days ago", "last week"
        parts.append(f"- {relative}: {transition.description}")
    
    # State velocity
    if len(history) > 1:
        recent_rate = len([t for t in history if current_time - t.timestamp < 7*24*3600])
        if recent_rate > 3:
            parts.append(f"(This entity has been changing rapidly—{recent_rate} updates this week.)")
    
    # Contradictions
    contradictions = world_model.get_contradictions(entity)
    if contradictions:
        parts.append(f"⚠️ Note: There are unresolved contradictions about this entity.")
    
    # Period anchoring
    period = world_model.get_period(current_time)
    parts.append(f"(Current period: {period.name})")
    
    return "\n".join(parts)
```

**Example output:**
```
Timeline for "Project Alpha":
- 3 days ago: Moved to testing phase
- 1 week ago: Core features completed
- 2 weeks ago: Initial prototype deployed
(This entity has been changing rapidly—4 updates this week.)
(Current period: Q1 2026 sprint)
```

### Advanced: Temporal Narrative Templates

```python
TEMPLATES = {
    "negotiation": """
You are in a negotiation that started {duration} ago.
Key timeline:
{timeline}
Current state: {state}
Deadline: {deadline_relative}
Competitor activity: {competitor_timeline}
""",
    "project": """
Project "{name}" history:
Started: {start_relative}
Current phase: {phase} (entered {phase_duration} ago)
Recent changes: {recent_changes}
Upcoming deadline: {next_deadline}
""",
}
```

### Research Questions
- Which narrative elements matter most? (relative time? velocity? contradictions?)
- Does more temporal detail always help, or is there a ceiling?
- How to handle conflicting temporal information?

---

## 4. Temporal Fine-Tuning / Curriculum

### Concept
Train the model on temporal reasoning tasks in a curriculum, from simple to complex.

### Implementation: Time-R1 Style Curriculum

**Stage 1: Basic temporal comprehension**
```
Input: "Event A happened on March 1. Event B happened on March 5. Which came first?"
Output: "Event A"
```

**Stage 2: Relative reasoning**
```
Input: "The meeting was 3 days before the deadline. The deadline is Friday. What day was the meeting?"
Output: "Tuesday"
```

**Stage 3: Duration + ordering**
```
Input: "Project X took 2 weeks. Project Y started when X ended. Project Y took 10 days. If X started Jan 1, when did Y end?"
Output: "January 24"
```

**Stage 4: Implicit temporal reasoning**
```
Input: "Sarah mentioned the contract 'last summer'. It's now February 2026. What year was the contract discussed?"
Output: "2025"
```

**Stage 5: Multi-entity temporal synchronization**
```
Input: "Company A acquired B in 2019. B had launched product X in 2017. A discontinued X in 2020. How long did A own X before discontinuing?"
Output: "1 year (2019-2020)"
```

### Synthetic Data Generation

```python
def generate_temporal_qa(difficulty):
    """Generate temporal reasoning training data."""
    if difficulty == 1:
        # Simple ordering
        events = generate_random_events(2)
        events.sort(key=lambda e: e.date)
        return {
            "input": f"{events[0].desc} happened on {events[0].date}. {events[1].desc} happened on {events[1].date}. Which came first?",
            "output": events[0].desc
        }
    elif difficulty == 2:
        # Relative time
        anchor = random_date()
        offset = random.choice(["3 days before", "1 week after", "2 months before"])
        target = compute_offset(anchor, offset)
        event = random_event()
        return {
            "input": f"{event} was {offset} {anchor.strftime('%B %d')}. What date was {event}?",
            "output": target.strftime('%B %d')
        }
    # ... higher difficulties
```

### Research Questions
- Does temporal fine-tuning transfer across domains?
- Minimum data needed for meaningful improvement?
- Does it hurt other capabilities (alignment tax)?

---

## 5. External Temporal Agent

### Concept
Separate lightweight model that tracks time and injects alerts/context into the main model.

### Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Main LLM                          │
│  (Handles conversation, reasoning, generation)     │
└──────────────────────────┬──────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────┐
│              Temporal Supervisor Agent              │
│  - Tracks elapsed time                              │
│  - Monitors deadline proximity                      │
│  - Detects temporal references in conversation     │
│  - Injects alerts: "ALERT: 30s left"               │
│  - Suggests temporal context when relevant         │
└─────────────────────────────────────────────────────┘
```

### Implementation

```python
class TemporalSupervisor:
    def __init__(self, deadline: float):
        self.start_time = time.time()
        self.deadline = deadline
        self.last_alert = 0
        self.alert_thresholds = [0.75, 0.5, 0.25, 0.1, 0.05]  # % remaining
    
    def check(self) -> str | None:
        elapsed = time.time() - self.start_time
        remaining = self.deadline - elapsed
        pct = remaining / self.deadline
        
        for threshold in self.alert_thresholds:
            if pct <= threshold and self.last_alert > threshold:
                self.last_alert = threshold
                return self.generate_alert(remaining, pct)
        return None
    
    def generate_alert(self, remaining, pct):
        if pct > 0.5:
            return f"[TEMPORAL NOTE: {remaining:.0f}s remaining—plenty of time.]"
        elif pct > 0.2:
            return f"[TEMPORAL ALERT: {remaining:.0f}s remaining—consider wrapping up.]"
        else:
            return f"[URGENT: Only {remaining:.0f}s left! Finalize immediately.]"
    
    def analyze_turn(self, text: str) -> str | None:
        """Detect temporal references and add context."""
        patterns = [
            (r"last (week|month|year)", self.contextualize_relative),
            (r"before the deadline", self.add_deadline_context),
            (r"how long", self.add_duration_context),
        ]
        for pattern, handler in patterns:
            if re.search(pattern, text, re.I):
                return handler(text)
        return None
```

### Pros/Cons
- **Pro:** No retraining, plug-and-play
- **Pro:** Lightweight, fast
- **Pro:** Can use specialized small model (or rules)
- **Con:** Adds latency
- **Con:** Coordination between agents can be brittle

---

## 6. Decay Attention / Temporal Attention Bias

### Concept
Modify attention weights to decay based on temporal distance. Recent events weight more than distant ones.

### Implementation

```python
class TemporalAttention(nn.Module):
    def __init__(self, d_model, decay_rate=0.1):
        super().__init__()
        self.decay_rate = decay_rate
        self.base_attention = nn.MultiheadAttention(d_model, 8)
    
    def forward(self, q, k, v, timestamps):
        # Compute base attention scores
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        
        # Compute temporal decay mask
        time_diffs = timestamps.unsqueeze(-1) - timestamps.unsqueeze(-2)
        decay_mask = torch.exp(-self.decay_rate * torch.abs(time_diffs))
        
        # Apply temporal bias
        attn_weights = attn_weights * decay_mask
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        return torch.matmul(attn_weights, v)
```

### Variations

**a) Learned decay rate:**
```python
self.decay_rate = nn.Parameter(torch.tensor(0.1))
```

**b) Asymmetric decay (future vs past):**
```python
past_decay = 0.1   # Slow decay for past
future_decay = 0.5  # Fast decay for future (less relevant)
decay = torch.where(time_diffs > 0, future_decay, past_decay)
```

**c) Query-dependent decay (some queries need recent, some need historical):**
```python
decay_gate = self.decay_predictor(q)  # [batch, heads, 1]
decay_mask = torch.exp(-decay_gate * torch.abs(time_diffs))
```

### Research Questions
- Fixed vs learned decay rate?
- Per-head decay (different heads specialize in different temporal ranges)?
- How does this interact with existing attention patterns?

---

## 7. Memory-Augmented Temporal Reasoning (MemoTime Style)

### Concept
From MemoTime paper (arxiv 2510.13614): hierarchical "Tree of Time" decomposition + experience memory.

### Key Ideas

**a) Tree of Time decomposition:**
Complex temporal questions are decomposed into a hierarchy where each branch inherits temporal constraints from parent.

```
Q: "Who did X hire after Y resigned during Q3 2017?"
            │
    ┌───────┴───────┐
    ▼               ▼
"When did Y     "Who did X hire
resign?"        in Q3 2017?"
    │               │
    ▼               ▼
  2017-08-15    [filter: after 2017-08-15]
                    │
                    ▼
                  Answer
```

**b) Monotonic timestamp enforcement:**
Each hop in reasoning must preserve or narrow temporal bounds — prevents "time travel" errors.

**c) Experience memory:**
Store successful reasoning traces with embeddings. On new questions, retrieve similar traces to guide reasoning.

```python
class TemporalExperienceMemory:
    def __init__(self):
        self.traces = []  # (question_emb, operator_type, trace)
    
    def add(self, question, operator, trace):
        emb = embed(question)
        self.traces.append((emb, operator, trace))
    
    def retrieve(self, question, operator, k=3):
        q_emb = embed(question)
        # Filter by operator type first
        candidates = [t for t in self.traces if t[1] == operator]
        # Rank by embedding similarity
        scored = [(cosine_sim(q_emb, t[0]), t[2]) for t in candidates]
        return sorted(scored, reverse=True)[:k]
```

### Temporal Operators to Handle
- `before(X)` / `after(X)` — point-relative
- `during(X, Y)` — interval containment
- `first(set)` / `last(set)` — ordinal
- `between(X, Y)` — range
- `overlaps(X, Y)` — interval intersection

---

## 8. Continuous-Time State Machines

### Concept
Maintain explicit temporal state that updates with each turn. The LLM reads/writes to this state.

### Implementation

```python
@dataclass
class TemporalState:
    start_time: float
    current_time: float
    deadline: float | None
    events: list[tuple[float, str]]  # (timestamp, description)
    active_intervals: dict[str, tuple[float, float | None]]  # name -> (start, end)
    
    def elapsed(self) -> float:
        return self.current_time - self.start_time
    
    def remaining(self) -> float | None:
        return self.deadline - self.current_time if self.deadline else None
    
    def add_event(self, description: str):
        self.events.append((self.current_time, description))
    
    def start_interval(self, name: str):
        self.active_intervals[name] = (self.current_time, None)
    
    def end_interval(self, name: str):
        start, _ = self.active_intervals[name]
        self.active_intervals[name] = (start, self.current_time)
    
    def to_context(self) -> str:
        """Render state as context for LLM."""
        parts = [
            f"Elapsed: {self.elapsed():.1f}s",
            f"Remaining: {self.remaining():.1f}s" if self.deadline else "",
            f"Recent events: {self.events[-3:]}",
            f"Active: {[k for k, v in self.active_intervals.items() if v[1] is None]}",
        ]
        return " | ".join(filter(bool, parts))
```

### LLM Interface

```python
SYSTEM_PROMPT = """
You have access to a temporal state machine. Use these commands:
- [TIME:EVENT description] — log an event at current time
- [TIME:START interval_name] — start tracking an interval
- [TIME:END interval_name] — end an interval
- [TIME:QUERY] — request current temporal context

The system will inject temporal state into your context automatically.
"""

def process_llm_output(output: str, state: TemporalState):
    """Parse temporal commands from LLM output."""
    for match in re.finditer(r'\[TIME:(\w+)\s*(.*?)\]', output):
        cmd, arg = match.groups()
        if cmd == "EVENT":
            state.add_event(arg)
        elif cmd == "START":
            state.start_interval(arg)
        elif cmd == "END":
            state.end_interval(arg)
```

---

## 9. Experimental: Temporal Dreaming / Consolidation

### Concept
Like Honcho's "dialectical dreaming" but for temporal knowledge. Periodically run a background process that:
1. Reviews temporal events
2. Consolidates patterns (repeated behaviors, cycles)
3. Infers implicit temporal relationships
4. Compresses old events into summaries

### Implementation

```python
async def temporal_dreaming(world_model, interval_hours=24):
    """Background consolidation of temporal knowledge."""
    while True:
        await asyncio.sleep(interval_hours * 3600)
        
        # 1. Identify temporal patterns
        patterns = detect_patterns(world_model.events)
        # e.g., "User checks email every morning", "Project updates happen weekly"
        
        # 2. Consolidate old events
        old_events = [e for e in world_model.events if age(e) > 30]  # 30+ days
        summaries = summarize_events(old_events)
        world_model.archive(old_events, summaries)
        
        # 3. Infer implicit relationships
        # "X happened 2 days before Y" -> explicit before relation
        inferred = infer_temporal_relations(world_model.events)
        world_model.add_relations(inferred)
        
        # 4. Predict upcoming events based on patterns
        predictions = predict_from_patterns(patterns)
        world_model.add_predictions(predictions)
```

---

## 10. Experimental: Temporal Contrastive Learning

### Concept
Train embeddings where temporally close events are close in embedding space.

### Implementation

```python
class TemporalContrastiveLoss(nn.Module):
    def __init__(self, time_scale=24*3600):  # 1 day
        super().__init__()
        self.time_scale = time_scale
    
    def forward(self, embeddings, timestamps):
        # embeddings: [batch, dim]
        # timestamps: [batch]
        
        # Compute pairwise similarities
        sims = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=-1)
        
        # Compute temporal distance (normalized)
        time_dists = torch.abs(timestamps.unsqueeze(1) - timestamps.unsqueeze(0)) / self.time_scale
        
        # Target: temporally close events should have high similarity
        targets = torch.exp(-time_dists)
        
        # MSE loss between similarity and temporal target
        return F.mse_loss(sims, targets)
```

### Use Cases
- Retrieval: find temporally relevant context
- Clustering: group events by temporal proximity
- Anomaly detection: events that are temporally close but embedding-distant

---

## Summary: Implementation Complexity vs Impact

| Approach | Complexity | Retraining? | Expected Impact | Best For |
|----------|------------|-------------|-----------------|----------|
| Clock Token Injection | Low | No | Medium | Quick wins, negotiations |
| Semantic Compilation (PIE) | Medium | No | High | Knowledge work, memory |
| External Temporal Agent | Medium | No | Medium | Real-time systems |
| Temporal State Machine | Medium | No | Medium | Structured dialogues |
| Temporal Embeddings | High | Yes | High | Native temporal reasoning |
| Decay Attention | High | Yes | Medium | Long-context retrieval |
| Temporal Fine-tuning | High | Yes | High | General improvement |
| MemoTime-style Memory | High | No | High | Complex multi-hop queries |
| Temporal Dreaming | Medium | No | Unknown | Long-term agents |
| Contrastive Learning | High | Yes | Unknown | Temporal retrieval |

---

## Recommended Research Path

1. **Baseline:** Clock token injection + urgency cues (replicate Real-Time Deadlines)
2. **PIE enhancement:** Semantic temporal compilation with ablations
3. **Eval creation:** Multi-domain temporal awareness benchmark (negotiation, scheduling, memory)
4. **Architecture exploration:** Temporal attention bias (no full retraining, just attention mod)
5. **Advanced:** MemoTime-style hierarchical decomposition + experience memory

The key insight from the research: **LLMs fail at mapping time to policy, not at understanding time exists.** Focus on approaches that translate temporal state into actionable context.
