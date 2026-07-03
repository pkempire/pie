# Memory Strategy Cards Video

This folder contains a Manim video explaining nine AI memory strategies:

1. RAG
2. Temporal knowledge graphs
3. Observation logs
4. Timeline reconstruction
5. Offline consolidation
6. Recursive read-time memory
7. Learned memory policies
8. Learned chunking
9. Latent and KV cache memory

## Render

```bash
cd /Users/parthkocheta/personal-intelligence-system/research/videos/memory_strategy_cards
chmod +x render.sh
./render.sh m
```

Use `./render.sh h` for a higher quality 1080p render.

## Voiceover

`make_voiceover.py` uses OpenAI TTS if `OPENAI_API_KEY` is available. It falls back to macOS `say`.

Voiceover script: `script.md`

Sources: `sources.json`

## Design Notes

The opening map uses two axes:

- `text / symbols` to `learned / latent`: what form carries the memory. RAG stores text chunks. A graph stores symbolic state. Learned chunkers, RL controllers, neural memory modules, and KV caches move more of the memory into learned representations.
- `compute at write time` to `compute at read time`: when the system spends intelligence. Observation logs, graphs, chunking, and consolidation do more work before the question is asked. RAG, recursive reading, timeline reconstruction, and cache loading spend more work when a specific question arrives.

Those axes are useful because most memory systems differ less by product name and more by these two engineering choices: representation and compute timing.

`rag_similarity.json` contains a small embedding check used in the RAG scene. It was computed with `text-embedding-3-small`.
