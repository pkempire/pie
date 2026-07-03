# Memory Strategy Cards

Working title: **Nine Ways To Give An AI Agent Memory**

Target length: 5 to 7 minutes.

## 00. Hook

Most debates about AI memory are confused because people compare different layers of the system as if they were the same thing.

A vector database, a knowledge graph, a summary log, a trained KV cache, and an RL memory policy are not competing answers to one question.

They answer different questions.

What gets stored? When is compute spent? What is retrieved? What is learned? And how does the system know if the memory helped?

This video is a map.

## 01. RAG

The simplest strategy is retrieval augmented generation.

Store documents or chunks. Embed them. At question time, retrieve the top matching passages and put them into the prompt.

RAG is great when the problem is external facts or document lookup.

But RAG does not know what is currently true. It mostly knows what text is similar.

So “I will move to New York” and “I moved to New York” can both look relevant, even though they imply different states of the world.

As a concrete check, with `text-embedding-3-small`, “I will buy milk after work” and “I bought milk after work” score about 0.89 cosine similarity. An unrelated sentence about a broken laptop scores about 0.16.

That is the point. Similarity is useful, but it is not the same thing as current truth.

## 02. Temporal Knowledge Graphs

The next strategy is structured memory.

Instead of storing only text chunks, extract entities, relations, and facts.

If the user changes jobs, moves cities, or updates a preference, a temporal graph can keep the old fact and the new fact with time attached.

This helps with questions like, “what was true then?” or “what changed?”

The risk is extraction quality.

If the system merges the wrong person, misses a fact, or writes the wrong edge, the graph can become clean looking but wrong.

## 03. Observation Logs

Mastra’s Observational Memory is a useful version of a different strategy.

It uses background agents, an Observer and a Reflector, to turn long conversations into a dense observation log.

Your agent does not retrieve raw messages forever. It sees a compact running record.

This can work very well on chat memory benchmarks because the hard work is prepaid.

But it also means the observer and reflector matter a lot. If they miss something, the compact log may not contain what the future question needs.

## 04. Timeline Reconstruction

Another strategy is to keep the event log and reconstruct state at read time.

Suppose the user lived in Boston, moved to New York in August, and later asks, “where did I live in May?”

A flat store may retrieve New York because it is the latest or most similar fact.

Timeline reconstruction sorts the relevant events, replays them up to May, and answers Boston.

This is not magic. It is the same basic idea as event sourcing and temporal databases, applied to agent memory.

## 05. Offline Consolidation

Online memory writes are usually too early.

Turn 12 may look unimportant until turn 80 makes it important.

Offline consolidation separates fast acquisition from slow compression.

During the session, keep the raw trace. Later, with hindsight, rewrite a working region into a smaller, better memory.

Mastra’s Reflector is in this family as a prompt based consolidator.

Auto-Dreamer is the learned version: it trains the consolidator with downstream task reward and a compactness objective.

## 06. Recursive Read-Time Memory

Instead of deciding everything at write time, you can spend compute at read time.

Recursive Language Models treat the long context like something the model can inspect in parts.

The model decomposes the question, reads relevant slices, aggregates partial answers, and decides whether to inspect more.

This is powerful when the raw context is huge and each question needs a different slice.

The tradeoff is latency and cost per query.

## 07. Learned Memory Policies

The next strategy is not a storage format. It is a controller.

Create an environment where the model can search raw traces, retrieve memory, write memory, update memory, delete memory, or answer.

Then reward it for future task success, evidence support, temporal correctness, and low cost.

This is what work like Memory-R1 points toward.

The hard part is credit assignment.

If the answer is wrong 200 turns later, which earlier memory operation deserved the blame?

## 08. Learned Chunking

Before memory, there is a smaller question: what is the unit?

Most RAG systems cut text into fixed windows.

But the useful boundary might be a topic shift, a code function, a claim, a plan, a contradiction, or a whole task episode.

Learned chunking is the idea that boundaries should be learned from downstream use, not hard coded as 800 tokens.

For RAG it improves retrieval units.

For memory it improves the regions that get consolidated.

## 09. Latent And KV Cache Memory

The final card is not text memory at all.

RAG stores text. A knowledge graph stores symbols. KV and latent memory store model state.

Cartridges train a small reusable KV cache for a large corpus, so future queries can load the cache instead of reprocessing all the text.

Titans adds a neural long term memory module that learns to memorize at test time.

These methods are fast and compact, but harder to inspect and cite.

That makes them powerful for stable corpora, but less ideal when the system must show exact evidence.

## 10. Synthesis

The winning system is probably not one card.

It keeps raw traces forever.

It indexes exact evidence spans.

It builds compressed memory views with provenance.

It tracks time, validity, and state changes.

It consolidates offline with hindsight.

It spends read time compute only when the question is hard.

And eventually, the controller over all of this is learned from downstream outcomes.

So the real question is not “vector database or knowledge graph?”

The real question is:

Given the task, current time, raw history, and budget, what should the agent see next?
