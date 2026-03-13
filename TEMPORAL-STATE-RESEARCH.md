# Elapsed Time as First-Class State: Research Directions & Experiments

## The Build Agenda

Five buildable directions, ordered from "can start this week" to "needs a research lab." Each has a concrete experiment to validate whether it works.

---

## Direction 1: Temporal Runtime ("The Clock Layer")

### What to build

A process that runs *between* LLM calls. Not the LLM itself — a runtime that maintains temporal state the way an OS maintains a system clock. The LLM is stateless between calls. The runtime is not.

**Robotics analogy: the ROS clock.** In Robot Operating System, every node has access to `rospy.Time.now()`. Planning, perception, and control all condition on this shared clock. No node "reasons about" time — time is an ambient signal available to every computation. LLM agents have no equivalent. We build one.

**Concrete architecture:**

```
┌─────────────────────────────────────────────┐
│              TEMPORAL RUNTIME                │
│                                             │
│  Wall Clock ─── Current Time                │
│       │                                     │
│  Thread Registry                            │
│    ├── thread_A: {last_touch: T1,           │
│    │     survival: 0.73, deadline: T5}      │
│    ├── thread_B: {last_touch: T2,           │
│    │     survival: 0.91, deadline: null}     │
│    └── thread_C: {last_touch: T0,           │
│          survival: 0.12, deadline: T3}       │
│       │                                     │
│  Temporal Event Queue                       │
│    ├── [T3 - 1day]: ALERT deadline_C        │
│    ├── [T_now + 7d]: CHECK stale_threads    │
│    └── [monday 9am]: PREDICT rhythm_A       │
│       │                                     │
│  State Vector (injected into every LLM call)│
│    temporal_context = {                     │
│      now: "2026-03-04T14:32:00",            │
│      since_last: "5d 3h",                   │
│      active_threads: [...],                 │
│      approaching_deadlines: [...],          │
│      stale_threads: [...],                  │
│      predicted_topic: "project_A"           │
│    }                                        │
└─────────────────────────────────────────────┘
          │
          ▼  (injected as structured context)
┌─────────────────────────────────────────────┐
│              LLM (stateless)                │
│  System prompt includes temporal_context    │
│  Model conditions behavior on this signal   │
└─────────────────────────────────────────────┘
```

**PIE reuse:** This is literally what `temporal.py` already computes — survival functions, hazard rates, Hawkes process intensities. The missing piece is wrapping it in a *persistent runtime* with timers, event queues, and injection into the LLM call path.

### Experiment 1: Proactive Deadline Surfacing

**Setup:**
- 20 synthetic users, each with 5-10 active threads
- Each thread has metadata: last discussed, deadline (if any), importance
- Simulate multi-session conversations over 30 "days"
- At each session, the temporal runtime computes state and injects it

**Conditions:**
1. **Baseline:** No temporal context injected. Standard system prompt.
2. **Timestamp injection:** Raw timestamps in system prompt ("Thread A last discussed 5 days ago, deadline in 2 days")
3. **Runtime injection:** Full temporal state vector including survival probabilities, urgency scores, predicted topic

**Metric:** Proactive Surfacing Rate (PSR) — percentage of approaching deadlines that the agent mentions unprompted. Measure at 7 days, 3 days, 1 day, and 0 days before deadline.

**Hypothesis:** Condition 3 (runtime) > Condition 2 (timestamps) > Condition 1 (baseline). The runtime's survival scores and urgency computation provide signal that raw timestamps don't — the model doesn't have to do temporal math itself, it receives pre-computed urgency.

**Why this matters:** If even Condition 2 (raw timestamps) significantly beats Condition 1, it confirms the thesis: the model CAN do temporal behavior, it just lacks the state. If Condition 3 beats Condition 2, it shows that pre-computed temporal features (survival, urgency) are better than making the LLM do temporal arithmetic.

### Experiment 2: Staleness Detection

**Setup:**
- Feed the agent facts at various timestamps
- At query time, some facts are current, some are stale (the world has changed)
- Agent must decide: answer from memory or flag as potentially stale

**Conditions:**
1. No temporal metadata
2. Timestamp-only (fact stored at time T)
3. Runtime with decay score (fact stored at T, current confidence = survival(T, now))

**Metric:** Staleness F1 — precision and recall on correctly identifying stale information. Compare to the <65% human alignment from Tic-Toc benchmark.

---

## Direction 2: Temporal Attention Bias ("Time-ALiBi")

### What to build

Modify the attention mechanism so that temporal distance between messages biases attention weights — not token distance, not semantic similarity, but *how much wall-clock time separates two pieces of context*.

**Video generation analogy: temporal attention.** Video transformers (VideoGPT, Sora-class models, CogVideo) use temporal attention where frame i's attention to frame j is biased by temporal distance. Nearby frames get higher attention. This is what gives video temporal coherence — the model "knows" that frame 100 is temporally close to frame 99 and far from frame 1.

Apply the same principle to conversations. A message from 5 minutes ago should get different attention than a message from 5 months ago, regardless of where they sit in the token sequence.

**Formal definition:**

Standard attention: `attn(Q, K) = softmax(QK^T / √d)`

ALiBi: `attn(Q, K) = softmax(QK^T / √d - m * |i - j|)` where |i-j| is token distance

**Time-ALiBi:** `attn(Q, K) = softmax(QK^T / √d - f(Δt_ij))` where Δt_ij is wall-clock time between message i and message j, and f is a learned function.

**Design choices for f:**
- **Linear:** `f(Δt) = α * log(1 + Δt)` — log-scale because the difference between 1 hour and 2 hours matters more than 1001 and 1002 hours
- **Multi-scale:** `f(Δt) = Σ_k α_k * exp(-Δt / τ_k)` — different attention heads use different timescales (τ_k), like how different brain regions process time at different granularities. Some heads specialize in "minutes ago" (τ=3600), others in "weeks ago" (τ=604800)
- **Learned:** Small MLP that takes Δt and outputs a scalar bias. Most flexible, needs training data.

**Robotics analogy: multi-rate sensor fusion.** Robots fuse data from sensors operating at different rates — camera at 30Hz, LIDAR at 10Hz, GPS at 1Hz. The Kalman filter handles this via time-indexed covariance updates. Multi-scale temporal attention is the same idea: different "sensors" (attention heads) operating at different temporal resolutions, fused into a unified representation.

### Experiment 3: Time-ALiBi on Multi-Session Conversations

**Setup:**
- Take LongMemEval dataset, add realistic timestamps to each session (sessions separated by hours to weeks)
- Fine-tune Llama-3-8B with LoRA, adding Time-ALiBi bias to attention

**Conditions:**
1. Standard positional encoding (no temporal info)
2. Timestamps as text tokens in the conversation
3. Time-ALiBi with linear f
4. Time-ALiBi with multi-scale f (different timescales per head group)

**Metrics:**
- LongMemEval accuracy (does temporal bias hurt standard memory tasks?)
- Temporal ordering accuracy (new task: "did I mention X before or after Y?")
- Recency preference (when facts conflict, does the model prefer the more recent one?)

**Hypothesis:** Condition 4 (multi-scale) > Condition 3 (linear) > Condition 2 (text) > Condition 1 (none). Multi-scale captures both "5 minutes ago" and "5 months ago" relationships, which no single timescale can.

**What this borrows from video:** CogVideo and similar models use *hierarchical temporal attention* — local temporal attention within scene segments + global temporal attention across scenes. Multi-scale Time-ALiBi is the conversational analog: local attention for within-session coherence + global attention for cross-session memory.

---

## Direction 3: FiLM Temporal Conditioning ("Temporal Modulation")

### What to build

Inject continuous temporal features into every transformer layer via adaptive modulation — the model's internal representations shift based on temporal context without consuming context window tokens.

**Diffusion model analogy (extended):** In Stable Diffusion, the denoising U-Net conditions on:
- Noise level σ (continuous scalar → modulates all layers)
- Text embedding (high-dimensional → cross-attention)
- Timestep t (discrete → embedded and added to residual blocks)

The key insight: σ doesn't appear as a "token" in the input. It's a *conditioning signal* that modulates the computation globally. The model learns completely different behavior at σ=0.99 vs σ=0.01 — same architecture, same weights, different functional mode.

**Our temporal analog:**

```
Temporal Features:
  - Δt (log seconds since last interaction)
  - day_of_week (cyclical: sin/cos)
  - hour_of_day (cyclical: sin/cos)
  - thread_survivals[] (per-thread survival probabilities from PIE)
  - thread_urgencies[] (deadline proximity scores)
  - session_count (how many sessions with this user)
  - gap_histogram (distribution of recent inter-session gaps)
         │
         ▼
  ┌──────────────┐
  │  Temporal MLP │  (2-3 layers, ~1M params)
  │  [64 → 128 → │
  │   2 * d_model]│
  └──────┬───────┘
         │
    ┌────┴────┐
    │  γ   β  │  (scale and shift vectors, dimension = d_model)
    └────┬────┘
         │
         ▼  (applied at every transformer layer after LayerNorm)

  For each layer l:
    h = LayerNorm(x)
    h = γ_l * h + β_l    ← FiLM modulation
    h = Attention(h)      ← standard attention
    h = FFN(h)            ← standard FFN
```

**What the model learns (hypothesized):**

When Δt is large (days):
- γ amplifies "summary" and "context-setting" features
- β shifts toward "resumption" behaviors ("Last time we discussed...")

When Δt is small (minutes):
- γ amplifies "continuation" features
- β shifts toward "flow state" behaviors (no re-introduction needed)

When deadline urgency is high:
- γ amplifies "action-oriented" and "deadline-aware" features
- β shifts toward proactive surfacing

When day_of_week matches a learned rhythm:
- γ amplifies features associated with that day's typical topics
- β shifts toward predicted topic context

### Experiment 4: FiLM Conditioning with PIE Survival Features

**Setup:**
- Use PIE's existing world model (3,998 entities, 6,706 transitions) as a source of temporal features
- Generate synthetic conversations where temporal behavior matters:
  - Deadline approaching → agent should mention it
  - Thread dormant for 2 weeks → agent should ask about it
  - Same day of week as a prior topic → agent should anticipate it
- Fine-tune Llama-3-8B with LoRA + FiLM temporal layers

**Training data construction:**
1. Take real multi-session conversation logs (or generate synthetic ones)
2. For each session, compute PIE temporal features (survival, Hawkes intensity, rhythm scores)
3. Create (input, temporal_features, desired_output) triples where desired_output includes appropriate temporal behaviors
4. Train the FiLM layers + LoRA adapters jointly

**Conditions:**
1. Baseline: no temporal info
2. Timestamps as text in prompt
3. Temporal features as structured JSON in prompt (uses context tokens)
4. FiLM conditioning (temporal features as modulation signal, zero context tokens)

**Metrics:**
- Temporal behavior score (composite of: deadline surfacing, gap acknowledgment, rhythm prediction, staleness flagging)
- Context efficiency (how many tokens used for temporal info: Condition 3 = 200+ tokens, Condition 4 = 0 tokens)
- Standard task quality (temporal conditioning shouldn't degrade normal conversation quality)

**Hypothesis:** Condition 4 matches or beats Condition 3 on temporal behavior while using zero context tokens. This would prove that temporal awareness can be "free" — no context window cost.

**Robotics analogy: proprioception.** Robots have proprioceptive sensors (joint angles, forces) that don't go through the vision pipeline — they condition motor control directly. FiLM temporal conditioning is proprioceptive temporal sensing: the model "feels" time passing without having to "see" it as text tokens. This is more efficient and potentially more robust than text-based temporal injection, just as proprioception is more reliable than visual self-monitoring for motor control.

---

## Direction 4: Continuous-Time State via SSM Adapter ("The Heartbeat")

### What to build

A small state space model that maintains a hidden state evolving in *continuous real time* between LLM interactions. This is the most architecturally ambitious direction — it gives the agent something no transformer has: dynamics that don't stop when inference stops.

**Robotics analogy: the Kalman filter.** A robot's Kalman filter maintains a state estimate that does two things:
1. **Predict:** Between observations, the state evolves according to a process model: `x(t+dt) = Ax(t) + noise`. The robot's belief about its position drifts even when it's not looking.
2. **Update:** When a new observation arrives, the state gets corrected: `x_new = x_predicted + K(observation - predicted_observation)`.

This predict-update cycle is exactly what an agent needs for temporal awareness:
- **Predict (between sessions):** Thread urgencies evolve. Deadlines approach. Information decays. Rhythms cycle.
- **Update (during session):** New conversation data corrects the state. Thread X was mentioned → its survival resets. Deadline passed → remove it.

**SSM formulation:**

State vector h(t) ∈ R^d represents the agent's temporal context. Between observations:

```
dh/dt = Ah(t)
```

where A is a learned matrix with:
- Negative real eigenvalues → exponential decay of thread urgency/relevance
- Complex eigenvalues → oscillating features for rhythms (weekly, daily cycles)
- Near-zero eigenvalues → long-term persistent features (user identity, stable preferences)

At observation time (new conversation):

```
h_new = h_predicted + B * extract(conversation)
```

B is a learned input matrix. `extract(conversation)` maps the conversation to a state update vector — which threads were discussed, what deadlines were set, what commitments were made.

**Discrete-time implementation (practical version):**

Since we don't actually run continuous dynamics, we compute the state update at each session:

```python
def evolve_state(h_prev, delta_t, A):
    """Evolve state by delta_t seconds."""
    # Matrix exponential: exact solution to dh/dt = Ah
    return scipy.linalg.expm(A * delta_t) @ h_prev

def update_state(h_predicted, conversation_embedding, B):
    """Update state with new observation."""
    return h_predicted + B @ conversation_embedding
```

The matrix exponential `expm(A * Δt)` is the key operation. For Δt = 5 minutes, the state barely changes. For Δt = 5 days, fast-decaying components have gone to zero (forgotten), slow-decaying components are reduced (faded), and oscillating components have cycled (rhythm features).

**Video generation analogy: temporal latent diffusion.** Video diffusion models (ModelScope, AnimateDiff) maintain a latent representation that evolves across frames. The temporal dynamics are learned, and the model generates each frame conditioned on the evolved latent. Our SSM state is analogous: a latent that evolves in real time, conditioning each conversation on the evolved temporal context.

**Neural ODE analogy:** Neural ODEs (Chen et al. 2018) parameterize continuous dynamics with neural networks: `dh/dt = f_θ(h, t)`. This allows non-linear state evolution — thread urgencies could interact (deadline A approaching makes thread B less relevant because attention shifts), rhythms could modulate decay rates (things decay slower during active work periods). More expressive than linear dynamics, but harder to train.

### Experiment 5: SSM Temporal Adapter

**Setup:**
- Train a small SSM (d=256) on multi-session conversation data with timestamps
- The SSM state evolves between sessions via matrix exponential
- At each session, the state is injected into a frozen Llama-3-8B via cross-attention (the SSM state becomes additional "virtual tokens")

**Training:**
1. Process conversation datasets with timestamps
2. At each session t_i, the SSM state h(t_i) is computed by evolving from h(t_{i-1}) + update
3. Train end-to-end: SSM parameters (A, B, C, D) + cross-attention adapter on frozen LLM
4. Loss: standard next-token prediction + temporal behavior rewards (deadline surfacing, gap acknowledgment)

**Diagnostic experiments:**
- **Eigenvalue analysis:** After training, examine A's eigenvalues. Do we see the expected structure? (Fast decay for short-term, slow decay for long-term, complex eigenvalues for rhythms?)
- **State ablation:** Zero out different components of h and measure which temporal behaviors break. This reveals what the SSM learned to represent.
- **Δt sensitivity:** Plot model behavior as a function of inter-session gap. Does it smoothly interpolate? (Minutes → continuation, hours → session shift, days → resumption, weeks → re-engagement)

**Robotics analogy: the proprioceptive state estimator.** Modern robots (Boston Dynamics, ANYmal) run a learned state estimator that fuses proprioceptive signals (joint angles, IMU data) into a latent state used by the locomotion policy. The state estimator runs at 500Hz even when observations arrive at 30Hz — it interpolates. Our SSM does the same: it interpolates the temporal state between conversation "observations."

---

## Direction 5: Temporal RL Policy ("Learning When to Act")

### What to build

Train an agent via RL to make *temporally optimal decisions* — not just "what to say" but "when to say it" and "what to proactively surface."

**The insight from Time-R1:** RL curriculum can train temporal reasoning into small models. Time-R1 used a 3-stage curriculum (understanding → prediction → generation) with rule-based temporal rewards. We adapt this for *temporal behavior*:

**Stage 1: Temporal Grounding**
- Reward: correctly answering "how long ago did I mention X?" and "what's the most recent information about Y?"
- Trains basic temporal localization — the model learns to use temporal signals to locate information in time

**Stage 2: Temporal Prediction**
- Reward: correctly predicting "what topic will the user bring up next?" given the temporal state (thread survivals, day of week, recent patterns)
- Trains temporal anticipation — the model learns to use rhythms and patterns to predict

**Stage 3: Proactive Temporal Action**
- Reward: surfacing information at the *right time* — mentioning deadlines before they pass, noting stale threads when relevant, following up on commitments
- Penalty: surfacing information at the *wrong time* — irrelevant interruptions, incorrect deadline urgency, false staleness alarms
- Trains temporal initiative — the model learns when to act vs. when to wait

**Reward design (critical):**

```
R_temporal = w1 * R_deadline + w2 * R_staleness + w3 * R_rhythm + w4 * R_precision - w5 * R_noise

where:
  R_deadline  = 1 if agent surfaces deadline within window before due date, 0 otherwise
  R_staleness = 1 if agent flags stale info that IS stale, -1 if flags current info as stale
  R_rhythm    = 1 if agent anticipates correct topic for temporal pattern, 0 otherwise
  R_precision = -|t_surfaced - t_optimal| / T_window  (closer to optimal surfacing time = higher reward)
  R_noise     = 1 for each unprompted surfacing that is irrelevant (penalizes spam)
```

### Experiment 6: RL Temporal Policy Training

**Setup:**
- Environment: simulated multi-session conversations with a synthetic user who has defined:
  - Weekly rhythms (works on project A Mon-Wed, project B Thu-Fri)
  - Active deadlines (project A due March 15, personal goal C review March 20)
  - Information that becomes stale (job title changed on March 1)
  - Commitments ("I'll send that doc by Friday")
- Agent: Llama-3-8B + temporal runtime (Direction 1) + optionally FiLM conditioning (Direction 3)
- Training: GRPO (Group Relative Policy Optimization), same as used in AgeMem and DeepSeek-R1

**Evaluation:**
- Compare RL-trained temporal policy vs:
  1. No temporal awareness (baseline)
  2. Hand-crafted rules (if deadline < 24h, mention it)
  3. Prompted temporal awareness ("You should proactively mention approaching deadlines")
  4. RL-trained policy

**Metrics (our novel eval from Part 8 of landscape doc):**
- Proactive Surfacing Rate
- Surfacing Precision (right time, right topic)
- False Alarm Rate (irrelevant proactive mentions)
- Gap-Aware Resumption Quality
- Commitment Follow-Through Rate

---

## Direction 6: Multimodal Temporal Fusion ("ChronusOmni Extended")

### What to build

ChronusOmni (Zheng et al.) interleaves timestamp tokens with multimodal features for temporal video grounding. Extend this to agent memory: interleave temporal tokens with conversation embeddings to create a temporally-structured memory representation.

**Video understanding analogy:** A video understanding model processes a sequence of frames, each with a timestamp. The model learns temporal relationships between frames — "this happened THEN that happened BECAUSE of the first thing." For conversations:

```
Standard memory:  [msg1_embed, msg2_embed, msg3_embed, ...]
                   (temporal order implicit in sequence)

Temporal memory:  [T1_embed, msg1_embed, T2_embed, msg2_embed, ΔT_embed, msg3_embed, ...]
                   (temporal structure explicit and learnable)
```

Where T_embed encodes absolute time and ΔT_embed encodes the gap between messages. The model can learn that ΔT = 5 days means "topic likely shifted" while ΔT = 5 seconds means "continuation."

**Event camera analogy from robotics:** Standard cameras capture frames at fixed intervals. Event cameras capture *changes* — they fire when something happens, producing a continuous stream of timestamped events. This is more efficient (no redundant frames) and naturally temporal (every event has a precise timestamp).

Agent memory could work the same way: instead of storing every conversation turn at a fixed "frame rate," store *events* (topic changes, deadlines set, commitments made, facts learned) with precise timestamps. Retrieval becomes temporal event querying rather than semantic similarity search.

### Experiment 7: Temporal Token Interleaving

**Setup:**
- Take multi-session conversation dataset
- Create three memory representations:
  1. Standard: concatenated conversation embeddings
  2. Temporal tokens: interleave learned temporal embeddings between conversations
  3. Event-based: extract events from conversations, store with timestamps, query temporally

**Task:** Temporal QA — "When did I mention X?", "What was I working on before Y?", "Has the information about Z changed since I last mentioned it?"

**Metric:** Temporal localization accuracy (how precisely can the model place events in time).

---

## What to Build First: The Minimum Viable Temporal Agent

Directions 1-6 are a research agenda spanning months to years. But the user can build something demonstrably novel in weeks by combining Directions 1, 5 (the rule-based version), and the eval:

### The MVP: Temporal Runtime + Eval

**Week 1-2: Build the temporal runtime**
- Wrap PIE's `temporal.py` in a persistent service
- Maintain thread registry with survival functions
- Implement event queue with timer-based triggers
- Output: structured temporal_context JSON

**Week 3-4: Build the eval**
- Generate 50 synthetic multi-session user profiles (10-50 sessions each, spanning 1-6 months)
- Each profile has: threads, deadlines, rhythms, stale facts, commitments
- Define ground truth: at each session, what should a temporally-aware agent do?
- Implement scoring: PSR, temporal precision, gap-aware resumption, commitment follow-through

**Week 5-6: Run baselines**
- Condition 1: GPT-5 / Claude with no temporal context
- Condition 2: GPT-5 / Claude with timestamps in prompt
- Condition 3: GPT-5 / Claude with full temporal runtime injection
- Publish results regardless of outcome — the eval itself is a contribution

**Week 7-8: FiLM conditioning experiment**
- LoRA + FiLM on Llama-3-8B using temporal features from the runtime
- Compare against text-injection baselines
- This is the paper's core experiment

### What Victory Looks Like

**If the runtime injection (Direction 1) significantly beats text timestamps:**
→ Paper: "Temporal State as Infrastructure: Pre-Computed Temporal Features Outperform Text-Based Temporal Injection for Agent Memory"

**If FiLM conditioning (Direction 3) matches text injection at zero context cost:**
→ Paper: "Free Temporal Awareness: Conditioning LLMs on Continuous Time Without Consuming Context"

**If the SSM adapter (Direction 4) shows smooth interpolation across time gaps:**
→ Paper: "Continuous-Time Temporal State for LLM Agents via State Space Model Adapters"

**The eval alone, with baselines showing failure:**
→ Paper: "Benchmarking Lived Temporal Awareness: Why No LLM Agent Knows What Day It Is"

Any ONE of these is publishable. The combination is a research program.

---

## Cross-Cutting Insights from Robotics & Multimodal

### From robotics:

1. **Separation of estimation and control.** Robots separate state estimation (Kalman filter) from control policy (RL/MPC). The estimator runs at high frequency, the controller at lower frequency. Our analog: the temporal runtime (estimator) runs continuously, the LLM (controller) runs on-demand. Don't make the LLM do temporal estimation — give it pre-computed temporal state.

2. **Multi-rate fusion.** Robots fuse sensors at different rates. Our analog: different temporal features operate at different timescales. Thread survival decays on the scale of days, deadline urgency on the scale of hours, rhythm features on the scale of weeks. Multi-scale attention (Direction 2) handles this naturally.

3. **Predictive control (MPC).** Model Predictive Control looks ahead — "given current state, what's the best action sequence for the next N steps?" Our analog: the temporal runtime could predict not just "what should I do now" but "what will I need to do in the next 3 sessions?" This enables pre-computation of context.

4. **Event-driven vs. polling.** Robots use interrupt-driven architectures for time-critical signals. Our analog: don't check all deadlines every session — set timers that fire when deadlines approach. The temporal runtime's event queue is this interrupt mechanism.

### From video generation:

1. **Temporal coherence as a learned property.** Video models don't enforce temporal coherence via rules — they learn it from data (temporal attention, temporal convolutions). Our analog: don't hand-code "if deadline < 24h, mention it." Let the model learn temporal behavior from examples of good temporal behavior. Direction 5 (RL) does this.

2. **Hierarchical temporal modeling.** Video models use multi-scale temporal representations — frame-level, clip-level, video-level. Our analog: message-level, session-level, relationship-level temporal representations. Different granularities of temporal awareness for different tasks.

3. **Autoregressive temporal generation.** Video models generate the next frame conditioned on previous frames with temporal attention. Our analog: generate the next response conditioned on the *temporal evolution* of the conversation history, not just its content. The temporal attention bias (Direction 2) enables this.

### From audio/speech:

1. **Streaming vs. batch.** Speech models can operate in streaming mode — processing audio as it arrives, maintaining hidden state between chunks. Our analog: the SSM adapter (Direction 4) maintains hidden state between conversation "chunks" (sessions), enabling streaming temporal awareness.

2. **Voice Activity Detection.** Speech systems detect when someone is speaking vs. silent. Our analog: the temporal runtime detects when a thread is "active" vs. "dormant" — a kind of "topic activity detection."

---

## Open Questions

1. **How much of temporal awareness is inferrable from text alone?** If you just write "5 days have passed since our last conversation" in the system prompt, how much temporal behavior does a frontier model (GPT-5, Claude) already exhibit? We need the eval (Direction 1) to answer this — it's possible that the problem is 80% solved by good prompting and 20% requires architectural innovation.

2. **Does the FiLM conditioning generalize?** If you train FiLM temporal conditioning on conversations with Δt up to 30 days, does it generalize to Δt = 90 days? Or do the learned γ/β distributions break down at unseen timescales? This is an empirical question.

3. **Can RL discover temporal behaviors that humans haven't specified?** The RL reward in Direction 5 is hand-designed (deadline surfacing, staleness flagging). But maybe there are temporal behaviors we haven't thought of that would emerge from a more general reward signal (user satisfaction, task completion rate). Open-ended RL exploration could discover novel temporal strategies.

4. **What's the right dimensionality for temporal state?** The SSM adapter (Direction 4) has a state dimension d. Too small → can't represent all active threads. Too large → overfitting, slow evolution. Is d=64 enough? d=256? d=1024? How does it scale with number of active threads?

5. **Is temporal awareness transferable across users?** If you train FiLM conditioning on user A's temporal patterns, does it help with user B? Or are temporal patterns so user-specific that each user needs their own temporal adapter? If the latter, this connects to Stanford Cartridges — per-user temporal cartridges.
