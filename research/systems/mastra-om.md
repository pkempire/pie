---
title: "Mastra Observational Memory"
year: 2025
category: "memory-system"
tags: ["LongMemEval", "observer", "reflector", "observation-log", "Mastra"]
---

# Mastra OM

Observer + Reflector pipeline that produces a dated bullet-point observation log. Reported **94.87% on LongMemEval** with gpt-5-mini — the highest published LongMemEval number among write-time-compression systems.

## Architecture

- **Observer**: scans each chunk of conversation, extracts candidate observations as (entity, property, value, timestamp) tuples.
- **Reflector**: runs over collected observations, compresses into a coherent log.
- Both run as background passes on a schedule. Per-turn LLM cost amortized.

## What's interesting

The Reflector runs *once at write time, never again*. So the artifact is frozen once produced. This is closer to [[sleep-consolidation]] than to per-turn extraction. Mastra is the prototype that "consolidation matters" was right; Auto-Dreamer is what happens when you learn the consolidator.

## Numbers

- LongMemEval (with gpt-5-mini): **94.87%**

This is the strongest published LongMemEval result among open systems. OMEGA (95.4%, closed) is higher.

## What we steal

- Observer + Reflector split (close to the [[sleep-consolidation|fast + slow]] CLS split)
- Tuple-shaped extraction format
- Test-time-friendly: queries hit the consolidated log, not the raw transcript

## See also

- [[2605.20616-auto-dreamer|Auto-Dreamer]] — the learned variant
- [[sleep-consolidation]] — the architectural family
