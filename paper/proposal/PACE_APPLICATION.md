# Pace Fellowship — Application (Summer 2026)
**For: Parth Kocheta**

> Copy each block into the matching form field at pacefellows.com.

---

## Full name
Parth Kocheta

## Email address
pranayko021@gmail.com

---

## Research proposal *(≤500 words)*

**Memory as a learned policy: the operating system for compounding-knowledge AI agents.**

The constraint on AI-for-science is not model intelligence — it is memory. A scientist running a six-month investigation accumulates context across hundreds of papers and dozens of failed experiments. A frontier language model on the same investigation forgets what it read on Monday by Wednesday. The compounding asset of a scientific career — knowing more about your subfield each week than the week before — has no operating system in current AI agents.

I think this layer can be learned. Every shipping memory system today (Mem0, Letta, Mastra, Zep, my own prior KG system) pairs an external store with a hand-tuned control layer that decides what to write and how to read. The store grows from use; the control layer does not. I am training two LoRA adapters on a 4B open-source model sharing an op vocabulary across read and write sides of memory. The write policy ingests one conversation turn and emits operations into a typed knowledge graph (creates, updates, contradictions, merges, archivals). The read policy retrieves and answers. Both train with GRPO on the LoCoMo and LongMemEval benchmarks.

The hard part is the write reward — a write op only matters if the read policy can later use what was stored, a delayed and noisy signal. My contribution is a dense companion reward I call evidence coverage: the fraction of dialogue turns required to answer a held-out question that the policy actually preserved in the graph, computed deterministically from existing benchmark annotations and the graph's provenance fields. Coverage carries 60% of the reward; an LLM-judge correctness term carries 40%. The mix trains ~6× faster per step and is robust to judge noise.

The hypothesis I most want to falsify: a single trained adapter runs unchanged across three different storage backends (a flat chunk store, a typed-transition knowledge graph, a Mastra-style observation log) with under a 5-point accuracy drop. If true, memory becomes a substrate-agnostic layer rather than per-vendor lock-in.

Why this sits at Pace's intersection: long-horizon agents are bottlenecked on context-window economics. A learned memory operating system pushes the expensive recall work into a small adapter (~200 MB) and a substrate-agnostic vocabulary, decoupling agent intelligence from the linear-in-history cost of every retrieval. The unit economics of customer support, sales engineering, scientific research, and personal AI all change once an agent compounds instead of re-derives. Whoever owns the adapter gets a network effect the foundation-model layer does not.

The fellowship period would convert this into two publishable artifacts: (1) the empirical paper with the full backend-transfer table on LoCoMo and LongMemEval, and (2) a short essay arguing the memory OS is the next vertical of LLM-agent infrastructure worth investing in — with a taxonomy of what is durable (substrate-agnostic adapters), what is interchangeable (the storage), and what gets disrupted (per-customer fine-tuning).

Code, paper, and logs are public at github.com/pkempire/pie.

---

## How are you sharing your work sample?
Both — pasted below and attached as PDF.

## Work sample — paste *(use the abstract + opening of the proposal)*

The most important constraint on AI-for-science is not model intelligence. It is memory. A research scientist running a six-month investigation accumulates context across hundreds of papers, dozens of failed experiments, and a steadily evolving model of which lab techniques work in their hands. A frontier language model running the same investigation forgets what it read on Monday by Wednesday. The compounding asset of a scientific career — knowing more about your specific subfield each week than you did the week before — has no operating system in current AI agents.

Every shipping memory system today pairs an external store with a hand-tuned control layer that decides what to write and how to read. The store grows from use; the control layer does not. I think this is the layer that needs to be learned.

I am training two LoRA adapters on a 4B-parameter open-source language model that share an op vocabulary across read and write sides of memory. The write policy ingests one conversation turn at a time and emits operations into a typed knowledge graph — creates, updates, marks of contradiction, merges, archivals. The read policy retrieves from the same graph and answers questions. Both policies are trained with GRPO (DeepSeek's critic-free RL algorithm) on the LoCoMo and LongMemEval benchmarks of long-horizon conversational memory.

The hard problem is the write reward. Outcome supervision is sparse and delayed: a write op only matters if the read policy can later use what was stored. My contribution is a dense companion signal I call evidence coverage — the fraction of dialogue turns required to answer a held-out question that the policy actually preserved in the graph, computed deterministically from existing benchmark annotations and the graph's provenance fields. Coverage carries 60% of the reward; an LLM-judge-anchored answer-correctness term carries 40%. The mix trains ~6× faster per step and is robust to judge noise.

The same trained adapter compiles to three storage backends: a flat chunk store, a typed-transition knowledge graph, and a Mastra-style observation log. Backend transfer (one policy, three stores) is the empirical claim that lets this work be a layer rather than a system.

[Continued in attached PDF — full proposal including infrastructure / economics / physical-world framing and the deployment-reward gap.]

## Upload work sample
Attach: **paper/proposal/proposal.pdf** (3-page research proposal)

---

## Short bio *(≤200 words)*

Parth Kocheta. CS at the University of Maryland, currently on a gap semester in San Francisco.

I have been building ML projects since 8th grade. The two threads I keep coming back to are AI for science and the infrastructure that lets small teams do disproportionately ambitious work. I work on AI optimization for high-density solar plant design with a company backed by Breakthrough Energy and Khosla Ventures — applied research turning generative layout models into real megawatts. Before that I improved Dice score by 34% across 3,000+ mice MRI scans by implementing a Residual Attention U-Net, and built a pipeline with senior scientists for automating the processing of optical imaging data from animal studies.

Alongside research, I run an AI education program that has mentored 150+ high school students into portfolio-ready AI work, hosted 1:1s with PhDs and ISEF winners, and built custom tools to automate feedback and curriculum.

The thing I want to spend the next year on is the memory operating system for long-horizon AI agents. It is the unsexy infrastructure layer between today's RAG stacks and AI collaborators that actually compound knowledge over months. The proposal attached is where I am.
