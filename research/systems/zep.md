---
title: "Zep / Graphiti"
year: 2024
category: "memory-system"
tags: ["KG", "bi-temporal", "valid-time", "Zep", "Graphiti", "open-source"]
---

# Zep / Graphiti

Bi-temporal knowledge graph memory. Every edge has valid-time (when the fact was true) AND ingestion-time (when we learned it). Edges can expire. Open-source implementation as Graphiti.

## Paper

[[2501.13956-zep|Zep: A Temporal Knowledge Graph Architecture for Agent Memory]]

## Numbers

- LoCoMo: ~85% (varies by setup; 85.22% in EverMemOS paper's reproduction)
- DMR: 94.8% (their own claim, contested in independent runs)
- Their own newer setups: 79.09% LoCoMo in some configs

The variance across reproductions is informative — bi-temporal edges help, but the gap to current SOTA is real.

## What's interesting

Closest existing thing to time-aware memory primitives. Bi-temporal edges are a real architectural choice; most systems just timestamp entries.

## What it doesn't do

- No derivation propagation (retracting an edge doesn't propagate)
- No [[belief-revision|formal belief revision]]
- Edge-validity rules are hand-coded

## See also

- [[time-aware-memory]] — Zep is the closest existing primitive
- [[belief-revision]] — what Zep's bi-temporal layer doesn't do
