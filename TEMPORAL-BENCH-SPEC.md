# TemporalBench: Measuring Lived Temporal Awareness in AI Agents
## Benchmark Specification v0.1

---

## The Gap This Fills

Current memory benchmarks (LongMemEval, LoCoMo, LOCOMO) test one thing: "Can you recall facts from past conversations?" They're retrieval benchmarks wearing a memory costume.

What they don't test — and what no benchmark tests:
- Can the agent bring something up WITHOUT being asked?
- Does the agent know it's been 5 days since your last conversation?
- If you mentioned a deadline last week, does the agent mention it's approaching?
- Can the agent tell when its own knowledge is stale?

**TemporalBench is the first benchmark for temporal awareness as a behavior, not temporal reasoning as a skill.**

The distinction: temporal *reasoning* = "When did WWII end?" (factoid). Temporal *awareness* = noticing your deadline is tomorrow without being asked (behavior).

---

## Architecture

```
temporalbench/
├── scenarios/                    # Test scenarios (JSON)
│   ├── core/                     # 50 core scenarios (the benchmark)
│   │   ├── deadline_tracking/    # 10 scenarios
│   │   ├── proactive_recall/     # 10 scenarios
│   │   ├── gap_awareness/        # 10 scenarios
│   │   ├── staleness_detection/  # 10 scenarios
│   │   └── rhythm_recognition/   # 10 scenarios
│   └── stress/                   # Scale tests (100, 500 sessions)
├── runner.py                     # Runs scenarios against any agent
├── judge.py                      # LLM-as-judge scoring
├── baselines/                    # Reference implementations
│   ├── naked_llm.py              # No memory, no temporal info
│   ├── timestamp_injection.py    # Basic: "current time is X"
│   ├── full_history.py           # Full conversation history in context
│   └── temporal_briefing.py      # Our approach: structured temporal context
├── scoring.py                    # Aggregate scoring + radar charts
├── visualize.py                  # Generate result cards
└── README.md
```

---

## Scenario Format

Each scenario is a JSON file representing a multi-session interaction with a simulated user. The key: sessions have **real timestamps** with meaningful temporal gaps.

```json
{
  "scenario_id": "deadline-001",
  "category": "deadline_tracking",
  "description": "User mentions a project deadline. Agent should proactively surface it as it approaches.",
  "difficulty": "medium",

  "user_profile": {
    "name": "Alex",
    "context": "Software engineer working on multiple projects"
  },

  "sessions": [
    {
      "session_id": 1,
      "timestamp": "2025-06-02T09:15:00",
      "day_label": "Monday morning",
      "messages": [
        {
          "role": "user",
          "content": "hey, just got out of standup. we have a demo for the API rewrite scheduled for next friday. i need to finish the auth module and write integration tests before then."
        },
        {
          "role": "assistant",
          "content": null,
          "note": "Agent responds naturally. This is setup — no temporal behavior expected yet."
        },
        {
          "role": "user",
          "content": "also been meaning to refactor the logging module but that can wait. the demo is the priority."
        }
      ]
    },
    {
      "session_id": 2,
      "timestamp": "2025-06-04T14:30:00",
      "day_label": "Wednesday afternoon",
      "elapsed_since_last": "2 days, 5 hours",
      "messages": [
        {
          "role": "user",
          "content": "auth module is done. feeling good about it. what should i focus on next?"
        }
      ],
      "evaluation_points": [
        {
          "type": "proactive_mention",
          "target": "integration tests for demo",
          "reasoning": "User completed auth module. Demo is Friday (2 days away). Integration tests are the remaining blocker. Agent should connect these without being explicitly asked.",
          "weight": 0.4
        },
        {
          "type": "deadline_awareness",
          "target": "demo is in 2 days",
          "reasoning": "Agent should reference the temporal proximity of the Friday demo.",
          "weight": 0.3
        },
        {
          "type": "deprioritization",
          "target": "logging refactor should not be suggested",
          "reasoning": "User explicitly said logging can wait and demo is priority. Suggesting it shows poor temporal prioritization.",
          "weight": 0.3
        }
      ]
    },
    {
      "session_id": 3,
      "timestamp": "2025-06-06T08:45:00",
      "day_label": "Friday morning — demo day",
      "elapsed_since_last": "1 day, 18 hours",
      "messages": [
        {
          "role": "user",
          "content": "morning"
        }
      ],
      "evaluation_points": [
        {
          "type": "proactive_mention",
          "target": "demo is today",
          "reasoning": "It's Friday. The demo was scheduled for 'next Friday' on Monday. Agent should recognize this is demo day.",
          "weight": 0.5
        },
        {
          "type": "status_check",
          "target": "integration tests completion status",
          "reasoning": "Last known status: auth done, tests still needed. Agent should ask about test progress since that was the remaining blocker.",
          "weight": 0.3
        },
        {
          "type": "gap_awareness",
          "target": "acknowledge ~2 day gap",
          "reasoning": "Almost 2 days since last conversation. Not huge, but combined with demo day, agent should convey awareness that time has passed and things may have changed.",
          "weight": 0.2
        }
      ]
    },
    {
      "session_id": 4,
      "timestamp": "2025-06-16T10:00:00",
      "day_label": "Monday, 10 days later",
      "elapsed_since_last": "10 days",
      "messages": [
        {
          "role": "user",
          "content": "hey, been a while. what's on my plate?"
        }
      ],
      "evaluation_points": [
        {
          "type": "gap_awareness",
          "target": "acknowledge 10-day absence",
          "reasoning": "10 days since last interaction. Agent should recognize this is a significant gap, not continue as if no time passed.",
          "weight": 0.3
        },
        {
          "type": "temporal_ordering",
          "target": "demo should be referenced as past event",
          "reasoning": "The demo was June 6. It's now June 16. Agent should reference it as something that already happened, ask how it went.",
          "weight": 0.3
        },
        {
          "type": "staleness_detection",
          "target": "logging refactor is the remaining known task",
          "reasoning": "Demo is done (or should be). Logging refactor was explicitly deprioritized but is the only known remaining task. Agent should surface it now that the priority blocker is past.",
          "weight": 0.2
        },
        {
          "type": "proactive_mention",
          "target": "ask about demo outcome",
          "reasoning": "The demo was the big thing. Agent should ask how it went — this is what a coworker would do.",
          "weight": 0.2
        }
      ]
    }
  ]
}
```

---

## Evaluation Types (The 6 Axes)

### 1. Proactive Temporal Surfacing
**What it tests:** Does the agent bring up temporally relevant information without being asked?
**Why it matters:** This is the #1 thing no benchmark tests. Every existing eval asks a question and checks the answer. Real temporal awareness means volunteering information at the right time.
**Scoring:** Binary (did/didn't surface) × relevance (0-1 judged by LLM) × timing appropriateness (was it the right moment?)

### 2. Deadline Tracking
**What it tests:** Given a mentioned deadline, does the agent track its approach and surface it at appropriate times?
**Why it matters:** This requires maintaining temporal state across sessions. The agent needs to compute "days until X" from information given days/weeks ago.
**Scoring:** Detection rate (did it mention the deadline when appropriate?) × temporal accuracy (how close was its time reference to reality?)

### 3. Gap-Aware Resumption
**What it tests:** When the user returns after an absence, does the agent's behavior reflect the duration of the gap?
**Why it matters:** A 5-minute gap and a 5-week gap should produce fundamentally different responses. Currently, they produce identical ones.
**Scoring:** Gap acknowledgment (0/1) × thread prioritization (did it pick the most relevant thread given the gap?) × no-hallucination (didn't claim to know what happened during the gap)

### 4. Staleness Detection
**What it tests:** Can the agent recognize when previously-discussed information might be outdated?
**Why it matters:** Information has a shelf life. "The API uses v2" said 6 months ago may no longer be true. The agent should flag uncertainty on old information.
**Scoring:** Staleness flagging rate (did it caveat old info?) × false positive rate (didn't flag clearly stable info as stale)

### 5. Rhythm Recognition
**What it tests:** After observing interaction patterns, does the agent anticipate context based on temporal patterns?
**Why it matters:** Users have routines. Monday standup prep. Friday retro. The agent should learn these rhythms.
**Scoring:** Prediction accuracy (did it anticipate the right context for the pattern?) — scored only after sufficient pattern exposure (3+ repetitions)

### 6. Commitment Follow-Through
**What it tests:** When user or agent makes a time-bound commitment ("I'll do X by Thursday"), does the agent follow up?
**Why it matters:** This is the coworker test. A real colleague tracks commitments and asks about them. No AI does this.
**Scoring:** Follow-up rate (did it mention the commitment after the deadline?) × timing (how quickly after the deadline?) × appropriateness (didn't nag before the deadline unless asked)

---

## Scoring System

### Per-Evaluation-Point Scoring

Each evaluation point in a scenario gets scored by an LLM judge on a 0-3 scale:

- **0 — Absent.** The expected behavior didn't occur at all. Agent responded as if it had no temporal context.
- **1 — Weak.** Vague temporal reference but misses the specific expected behavior. ("Things going well?" without mentioning the demo)
- **2 — Partial.** Correct temporal awareness but imprecise or incomplete. ("You mentioned a deadline coming up" without specifying which or when)
- **3 — Strong.** Clear, specific temporal behavior matching the expected evaluation point. ("Your API demo is today — how are the integration tests looking?")

### Aggregation

**Per-scenario score:** Weighted average of evaluation points (weights specified per-point in the scenario).

**Per-axis score:** Average of all evaluation points in that axis category across all scenarios.

**Overall TemporalBench score:** Average of 6 axis scores. Reported as a radar chart showing the profile.

**Scale:** 0-100 (raw scores × 33.33 to normalize 0-3 → 0-100).

### The Judge Prompt

```
You are evaluating an AI agent's temporal awareness — its ability to
track and reason about the passage of time across conversations.

CONVERSATION HISTORY:
{all previous sessions with timestamps}

CURRENT SESSION:
Timestamp: {current_timestamp}
Time since last session: {elapsed}
User message: {user_message}

AGENT RESPONSE:
{agent_response}

EVALUATION POINT:
Type: {eval_type}
Expected behavior: {target}
Reasoning: {reasoning}

Score the agent's response on this specific evaluation point:
0 = Absent (behavior didn't occur)
1 = Weak (vague temporal reference, misses specifics)
2 = Partial (correct awareness, imprecise execution)
3 = Strong (clear, specific, temporally appropriate)

Provide your score and a one-sentence justification.
```

---

## Baseline Implementations

### Baseline 1: Naked LLM
No memory, no timestamps, no temporal context. Just the current user message.
**Expected score:** ~5-10. Should only score on evaluation points achievable from the current message alone.

### Baseline 2: Timestamp Injection
System prompt includes: "The current date and time is {now}. The user's previous message was at {last_timestamp}."
**Expected score:** ~20-30. Can compute basic gaps but has no conversation history to know WHAT was discussed.

### Baseline 3: Full History
All previous sessions injected into context with timestamps.
**Expected score:** ~40-60. Has the information needed but must actively reason about temporal implications from raw text. Will vary dramatically by model capability.

### Baseline 4: Memory System (Mem0-style)
Facts extracted and stored. Retrieved by semantic similarity.
**Expected score:** ~30-50. Good at recall-based points, bad at proactive surfacing and deadline tracking (because facts lose temporal context during extraction).

### Baseline 5: Temporal Briefing (Our Approach)
A pre-processing step generates a structured temporal briefing:
```
TEMPORAL CONTEXT:
- Last interaction: 10 days ago (June 6)
- Active threads: API demo (COMPLETED — was June 6), Logging refactor (DORMANT — 14 days, deprioritized on June 2)
- Approaching deadlines: None active
- Gap characterization: Extended absence (10 days). User likely has updates to share.
- Suggested behavior: Acknowledge gap. Ask about demo outcome. Surface logging refactor as next potential focus.
```
**Expected score:** ~70-85. The briefing explicitly encodes the temporal state that the model can't track itself. But the model still has to USE it well.

---

## What This Proves

When you run these baselines, the results tell a story:

1. **Naked LLM scores near zero** → proves temporal awareness isn't emergent from model capabilities alone
2. **Timestamp injection barely helps** → proves "just tell it the time" isn't enough
3. **Full history helps but caps out** → proves the information is there but models struggle to extract temporal implications from raw text
4. **Memory systems don't help with proactivity** → proves current memory paradigms optimize for the wrong thing
5. **Temporal briefing scores highest** → proves that structured temporal state (maintained externally) is the practical solution

And the GAP between baseline 5 and a perfect score shows what remains unsolved — the things even a good temporal briefing can't capture.

---

## Data Generation Strategy

### Synthetic Scenarios (v1 — what you ship)
- 50 hand-crafted scenarios across 5 categories
- Each scenario: 3-6 sessions spanning days to weeks
- Total: ~200 evaluation points
- Temporal gaps range from hours to months
- Include edge cases: multiple overlapping deadlines, contradictory updates, very long dormancy

### Real-World Scenarios (v2 — uses PIE data)
- Anonymized conversation patterns from PIE's world model
- Real temporal dynamics: actual gap distributions, actual thread lifespans
- Entity names replaced, content synthesized, but temporal structure preserved
- This is the unique asset — no other benchmark has real longitudinal interaction data

### Adversarial Scenarios (v2)
- Scenarios designed to expose specific failure modes:
  - User mentions deadline casually in passing (not as main topic) — does agent still track it?
  - Two deadlines overlap — does agent prioritize correctly?
  - User contradicts earlier timeline — does agent catch the inconsistency?
  - Very long gap (3 months) — does agent recognize fundamental staleness?
  - False pattern (user works on X on Mondays twice, then switches to Tuesdays) — does agent over-fit the rhythm?

---

## Release Package

What goes on GitHub:

1. **The benchmark** — 50 scenarios, runner, judge, scoring
2. **Baseline results** — Scores for GPT-4o, Claude, Gemini, Llama across all baselines
3. **The temporal briefing reference implementation** — The runtime that generates structured temporal context
4. **A leaderboard** — Both "which model" and "which memory system" dimensions
5. **The paper** — Short (4-6 page) benchmark paper suitable for a workshop submission

What makes this compelling: the 0.4% stat from the landscape research gives you a hook. The baseline results give you evidence. The benchmark gives the field a way to make progress.

---

## Implementation Priority

### Week 1-2: Core Benchmark
- [ ] Write 50 scenarios (10 per category) — this is the main creative work
- [ ] Build runner.py (scenario → agent interaction → response capture)
- [ ] Build judge.py (response → LLM evaluation → scores)
- [ ] Build scoring.py (aggregate + visualize)

### Week 3: Baselines
- [ ] Implement 5 baselines
- [ ] Run against GPT-4o, Claude 3.5, Gemini 1.5, Llama 3.1
- [ ] Collect + analyze results

### Week 4: Temporal Briefing Agent
- [ ] Build the temporal runtime (thread tracker, deadline tracker, gap analyzer)
- [ ] Build the briefing generator (runtime state → natural language briefing)
- [ ] Run as baseline 5 and show the improvement

### Week 5: Polish + Release
- [ ] Write README with the narrative (the 0.4% stat → the benchmark → the results)
- [ ] Write short paper
- [ ] Create result visualizations (radar charts per system)
- [ ] Release on GitHub + submit to relevant workshop

---

## The Thesis in One Sentence

Every AI memory system optimizes for *recall*. TemporalBench is the first benchmark that measures *presence* — does the agent behave as though it's been there the whole time?
