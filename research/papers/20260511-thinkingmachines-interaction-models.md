---
arxiv_id: ""
title: "Interaction Models: A Scalable Approach to Human-AI Collaboration"
authors: ["Thinking Machines Lab"]
year: 2026
date_published: "2026-05-11"
date_ingested: "2026-05-12"
source_url: "https://thinkingmachines.ai/blog/interaction-models/"

approach_class: "infrastructure"
problem: "Turn-based LLMs can't perceive continuous time, can't react mid-turn, can't speak and listen simultaneously."
approach: "TML-Interaction-Small (276B MoE, 12B active) with time-aligned 200ms micro-turns: input AND output streams interleaved at 5Hz."
benchmarks: ["TimeSpeak", "CueSpeak", "RepCount-A", "ProactiveVideoQA"]
results:
  - "TimeSpeak: 64.7 (vs GPT-Realtime-2 minimal 4.3)"
  - "Direct sense of elapsed time built into architecture"
  - "Split design: interaction model + background model"
reward_shape: "none"
base_model: "TML-Interaction-Small (276B MoE, 12B active)"

relevance: "high"
relevance_reason: "Defines the continuous-time-perception problem at the model level. Validates background/interaction split."
steal:
  - "Time-aligned micro-turn architecture (200ms chunks)"
  - "Encoder-free early fusion across modalities"
  - "Background-model + interaction-model split as architectural pattern"
limitations:
  - "Requires training a new model from scratch (can't apply to closed APIs)"
  - "Long-session context management still unsolved"
  - "Background reasoner not yet sophisticated"
tags: ["thinking-machines", "continuous-time", "streaming", "multimodal", "background-agent"]
---

# Interaction Models: Thinking Machines Lab, May 11, 2026

## Quick read

TML's flagship architecture: time-aligned 200ms micro-turns that interleave input and output streams. 276B MoE (12B active). The model has a direct sense of elapsed time built into the architecture — answers "how long did this take" without seeing a timestamp. Scores 64.7 on TimeSpeak vs GPT-Realtime-2 minimal's 4.3.

## Why it matters to us

This solves continuous-time perception at the model level — for seconds-to-minutes. Crucial implications:

1. Closed APIs (GPT-5, Claude) will remain turn-based for some time; we cannot replicate this with prompt engineering.
2. They explicitly state long-session context management is unsolved.
3. They explicitly use a "split: interaction model + background model" architecture, validating our intuition that long-horizon state needs a separate reasoner.

The long-horizon week-to-month state layer is the unclaimed opening.

## Method in one paragraph

Multi-stream architecture: at each 200ms tick, the model receives a chunk of audio/video/text input AND emits a chunk of output, in parallel. Streaming bidirectional. Built-in timing structure means duration is structural, not metadata. Encoder-free early fusion. Mixture-of-experts (276B total / 12B active).

## Results in numbers

- TimeSpeak: 64.7 (best baseline GPT-Realtime-2 minimal: 4.3)
- Other benchmarks: CueSpeak, RepCount-A, ProactiveVideoQA — all custom-built by TML because no existing benchmarks measured the right thing

## What they don't do

- Long-session memory across days/weeks (explicitly flagged as open)
- Their background model is not yet a sophisticated long-horizon reasoner
- Doesn't address project-state tracking or research workflows

## Open questions / followups

- How long can a session actually run before context management breaks?
- What's the background model's actual architecture and capability?
- Can we build a persistent state layer that interfaces with their interaction model?
