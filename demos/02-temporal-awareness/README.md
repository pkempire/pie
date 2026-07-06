# Demo 02 — Temporal awareness: does the model *feel* elapsed time?

**Claim.** Agents are "temporally blind": they treat timestamps as passive text and don't reason
about how much wall-clock time has passed. Making time an **explicit computed state** — the *age*
of each piece of data as of now — changes their decisions in the right direction, where raw
timestamps barely do.

**Task.** [TicToc](https://github.com/chengez/TicToc) ("Your LLM Agents are Temporally Blind",
arXiv:[2510.23853](https://arxiv.org/abs/2510.23853)): a timestamped tool-use conversation where
the agent must decide **call a tool to refresh** vs **answer directly from what it has**, scored
against **human preference labels** (deterministic — no LLM judge). 50 scenarios spanning high
time-sensitivity (stock, tide, ambulance dispatch) to low (baggage policy, degree requirements),
3,630 train / 1,962 test. No published model exceeds 65% even *with* timestamps.

**Result** (n=36 balanced, gpt-5-mini, low reasoning, single seed — a smoke, not significance):

| Condition | What the model sees | Alignment |
|---|---|---|
| **blind** | timestamps stripped | 41.7% |
| **raw** | ISO timestamps present (what TicToc tests) | 47.2% |
| **state** | timestamps + a computed line: *"the data you retrieved is 8 minutes old"* | **52.8%** |

Monotonic: **blind < raw < state.** Raw timestamps buy +5.5pp — the model can barely use them
(blindness). Computing the elapsed time into a state buys **+11pp over blind, +5.6pp over raw
timestamps.** The information was already in the transcript; making it a *state* is what the model
could actually act on.

## Run it

```bash
git clone https://github.com/chengez/TicToc demos/02-temporal-awareness/TicToc
python demos/02-temporal-awareness/run.py     # needs OPENAI_API_KEY or .env; ~$1
```

## Why this is the wedge, not a one-off

"Temporal awareness" here means the specific capability agents lack: **maintaining time as a state
variable and acting on it** — sensing that data is stale, that a deadline is near, that a process
has been running too long. TicToc isolates it with a deterministic, human-labeled, *unsaturated*
(<65%) metric and a **train split nobody has exploited**. That makes it a real target: reproduce
the blindness (done), engineer the time-state representation (this demo, first cut), then **train**
the decision policy (GEPA on the prompt, then GRPO on an open-weight model via Tinker) to beat 65%.

Full plan: [docs/PROJECT-time-as-state.md](../../docs/PROJECT-time-as-state.md).
