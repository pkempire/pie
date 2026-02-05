# Raw Notes & Ideas — Parth's Brain Dump (2026-02-02)

## Core Vision
- Local context manager + MCP server → all LLMs read/write to one ground truth
- Don't lose information in the LLM — persist everything locally
- "PIE: Personal Intelligence Engine" — one DB with key memories, interchangeable via MCP
- Own your context, edit, understand, give any model access

## Architecture Ideas
- Temporal understanding → proactivity → agency
- Daily state change logs for temporal understanding
- Recursive folder creation / project labeling with max-depth
- Parametric vs non-continuous learning approaches
- ThinkyMachines LORA infra
- World models and simulation
- Context-based / dynamic chunking (NOT static sized)
- Combine semantic search + keyword search
- Combine different embedding models for different aspects of conversations
- Temporal GNNs for novel link prediction
- Citations for where info comes from
- Multi-hop RAG + knowledge graphs + proactive agents

## Data Sources to Connect
- ChatGPT logs
- Google Drive
- Apple Notes
- Twitter bookmarks
- All browser history
- Google Photos
- Email
- Instagram + LinkedIn + iMessage
- Local hard drive
- Slack
- Everything timestamped

## Key Technical Questions
- How to encode temporal data?
- How to extract world view and underlying reasoning from texts?
- Can we continually fine-tune?
- How to handle context switching in stream-of-consciousness notes?
- How to decide when to delete from graph?
- Context window management for graphs
- How to come up with objects and relations for graph DB?
- How niche to get with graph schema?
- Is graph best, or combine with vector DB?
- How to eval RAG quantitatively?
- How to find novel connections between ideas using graph DB?
- Cosine/semantic similarity — does it make sense for this?
- LLM-generated regex for finding relevant info?
- How to keep adding new info in a way that makes sense?
- Can MCP server send new chat logs back to update KG in real-time?
- How to chunk conversational data (traditional RAG fails here)?
- GPT answers are super long — how to extract key knowledge?

## Product Ideas
- "What did I say about…?" lightning recall
- Global ⌘-K search bar (browser extension, Raycast-like)
- Automagic Daily Brief (7am: yesterday's decisions, unfinished items)
- Insight Cards (topic appears 3rd time → suggest primer, Tinder-swipe UX)
- See how thinking evolved — supporting, contradictory, evolving perspectives
- Break GPT sycophancy with unbiased reasoning LLM / fact checker
- Cluster ideas into projects automatically → todolist / PM software
- Proactive agent for connecting dots, brainstorming
- AI Architect / Toolkit vertical — understand many products, how they connect
- Enterprise: company knowledge graph connecting people working on similar things
- Semantic distance to embed excitement/curiosity about topics

## Experiments to Run
- Pre-process conversations → chat with in NotebookLM / Google AI Studio
- Semantic distance tracking as you go through a project
- Chunk by conversation vs by topic
- Have LLM bucket convos into categories (health, RAG, etc.)
- Recursive categorization — model finds new subcategories
- AllenAI re-query style search with multi-step retrieval

## Projects/Links to Research
- Plastic Labs Honcho: https://github.com/plastic-labs/honcho
- Basic Memory: https://github.com/basicmachines-co/basic-memory
- Your Memory (Politzki): https://github.com/jonathan-politzki/your-memory
- Graphlit: https://www.graphlit.com/
- SuperMemory: supermemory.ai
- Mem0: mem0.ai
- Graphiti (Zep)
- H-net personal AI memory layer
- Google blog: multi-agent framework
- OpenAI cookbook: temporal agents with knowledge graphs
- Qdrant vector DB on n8n
- ThinkyMachines
- AllenAI re-query
