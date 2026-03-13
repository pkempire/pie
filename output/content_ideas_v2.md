# Content Ideas v2 — From Real Knowledge, Not Personal Vlogs

**The vibe:** AI/tech content creator who actually builds things. Educational, entertaining, genuinely insightful. Think: Fireship's density + real builder credibility. Not confessional, not "day in my life," not flexing age. Pure value — things the viewer learns that they can't get from the 500 other AI YouTubers parroting the same surface-level takes.

**Your edge (from PIE):** You've actually built across the full stack — from ESP32 hardware sending WiFi CSI data, to LSTM training pipelines, to LLM orchestration libraries, to cold email automation workflows, to processing 29M rows of YouTube sponsorship data. You've evaluated the tools (LangChain vs LlamaIndex, Apollo vs Clay vs Instantly, Browserbase, n8n, Pinecone, DeepSeek API). You've hit the real walls (last-batch hangs in concurrent LLM processing, PyTorch safe-unpickling breaking sklearn objects, ESP32 Bluetooth connection instability, dependency hell in HuggingFace + TRL + bitsandbytes). That's the stuff nobody else talks about.

---

## YOUTUBE (10-15 min, tech-first, each video has a clear thesis)

### Agents & Automation

**1. "MCP vs Browser Agents — Which One Actually Wins?"**
You literally have a shortform script already drafted AND deep research comparing the two architectures. Your thesis from PIE: MCPs are fast/deterministic but limited without APIs. Browser agents are GUI-native (work behind logins, complex UIs) but slow and brittle. The real answer is hybrid. You can demo both live — show an MCP tool call completing in 200ms vs a browser agent taking 15 seconds to do the same thing, then show a task only the browser agent can handle. Nobody has made this video with actual side-by-side demos.

**2. "I Processed 29 Million YouTube Videos to Find Every Sponsorship Deal"**
The sponsorFind pipeline. Two-pass LLM detection (fast heuristic filter → deep GPT parse), domain normalization, URL expansion caching. Show the actual data — 150K clean records, what patterns emerged (which brands sponsor the most, which niches are growing). The technical breakdown of how you built the pipeline is genuinely useful for anyone doing large-scale LLM data processing. The business insights from the data are interesting to anyone in the creator economy.

**3. "Why Your LLM Pipeline is 1000x Slower Than It Should Be"**
HyperLLM. Dynamic token batching, concurrent API calls, the difference between naive sequential processing and properly orchestrated parallel LLM calls. Show the actual benchmarks. Talk about the real bugs you hit — last-batch hang behavior, KeyError from dataset shape mismatches between generation and consumption phases, how batched responses weren't being parsed into per-item outputs. This is a video every AI engineer needs.

**4. "Building a Cold Email System That Actually Works (Full Technical Breakdown)"**
Not "how to write cold emails" — the actual technical infrastructure. n8n orchestration, Apollo/Clay enrichment, email warmup and deliverability (SPF/DKIM/DMARC), secondary domain strategy, the difference between Instantly vs building your own. You've evaluated all of these tools. Show the n8n workflow, explain why you chose each component, what breaks at scale.

**5. "I Built an AI Agent That Finds and Qualifies Leads Automatically"**
The Hermes architecture — but framed as a technical deep-dive, not a product pitch. LangGraph orchestration, how you chain research → strategy → email generation → sending. The real challenge: making the agent's output actually good enough that a human doesn't need to review every email. Show the prompting strategy, the evaluation approach, what "good" looks like.

### AI/ML Deep Dives

**6. "WiFi Signals Can Detect Your Heart Rate — Here's How"**
The WiFi CSI pipeline. Not a personal project video — a genuine technical explainer of how RF sensing works, why it's possible (micro-Doppler from chest wall movement), the signal processing chain (Hampel outlier removal → bandpass filtering → Savitzky-Golay smoothing → LSTM), what accuracy looks like (sub-1 bpm MAE). Show real CSI data. Explain why this matters for eldercare, sleep monitoring, privacy-preserving health tech. This is genuinely novel content that almost nobody is making.

**7. "Fine-Tuning an LLM on WiFi Signals (WiFiGPT)"**
Treating RF telemetry as language. How you tokenize CSI data, what a 1B-parameter model learns from WiFi signals, ablation results across different environments. The non-obvious insight: the model was relying on prompt templates / memorized medians rather than actually learning CSI patterns (>85% attribution mass on first prompt token). That's a real finding that changes how you think about applying LLMs to non-text domains.

**8. "The Real Bottleneck in LLM Applications (It's Not What You Think)"**
Your belief + real experience: the main bottleneck is the LLM API (network + model latency + token processing), not local compute. How this changes architecture decisions — bigger batches, smarter caching, when to use local models vs API. Dynamic token batching explained. The chunked scaling + memory-mapped loading approach for handling datasets that don't fit in RAM. Practical, specific, useful.

**9. "RAG Is Broken — Here's What Actually Works"**
You've built multiple RAG systems (PIE, research search engine, LLM Doc Integration). The real problems: naive chunking destroys context, vector similarity isn't semantic understanding, LLM staleness means your retrieval needs real-time augmentation. Your concept of selective semantic chunking + chunk scoring + post-processing auto-fix layer. Show the difference between a basic RAG pipeline and one that actually works.

**10. "How to Build a Personal Knowledge Graph from Your Chat History"**
PIE — but framed as a technical tutorial anyone can follow. Entity extraction from conversations, temporal state management, TF-IDF embeddings for semantic search (no API needed), UMAP clustering for visualization. Show the 3,998-entity graph, the HDBSCAN clustering finding 76 natural domains, the semantic search returning relevant results. This is a project video that teaches real techniques.

### Business/Industry Analysis (with technical depth)

**11. "The Computational Arbitrage Window is Closing"**
Your thesis: VC-subsidized compute (free Cursor, cheap DeepSeek, generous Claude tiers) is a temporary phenomenon. Smart builders ship products NOW while the cost of intelligence is artificially low. When subsidies compress, the unit economics change. Show specific examples of what's subsidized, estimate the real cost, explain what this means for what you should build today vs in 2 years.

**12. "AI SEO: Getting Cited by LLMs (New Category)"**
You've researched offering guaranteed LLM citations as a service. The insight: as people shift from Google to AI for answers, the SEO game completely changes. What makes an LLM cite your content? How do you test it? Is this a real business? Technical breakdown of how LLM retrieval actually works under the hood (from your RAG experience), what that means for content strategy.

**13. "Open Source Wins Infra, Closed Labs Win Frontiers — What to Build"**
Your belief, backed by real experience using both. Open source dominates tooling and infra (LangChain, n8n, llama.cpp). Closed labs lead frontier breakthroughs. The implication: don't compete on infra (you'll lose to open source). Build verticalized products, distribution moats, batteries-included SaaS on top. Show specific examples from your own stack choices.

**14. "The AI Engineer Role Is Changing — What Actually Matters Now"**
Your belief that AI engineering is evolving toward orchestration and product-focus. Show what the real work looks like: it's not training models, it's designing systems that chain multiple LLMs, handle failures gracefully, manage context windows, route between tools. Walk through a real system architecture (Hermes or PIE) and explain what each decision teaches about what the role actually is.

### Tool Evaluations / Tutorials (high search volume)

**15. "I Evaluated Every Cold Email Tool — Here's What I'd Actually Use"**
Apollo vs Clay vs Instantly vs PhantomBuster vs building your own with n8n. You've actually used or deeply evaluated all of these. Honest comparison: what each does well, where each breaks, pricing reality, and when to just build it yourself.

**16. "LangChain vs LlamaIndex vs Just Writing Code — Honest Take"**
You've used LangChain for ReAct agents and RAG, evaluated LlamaIndex for document agent pipelines. The real tradeoff: abstraction saves time but hides what's happening. When to use a framework vs when it's faster to write 50 lines of Python yourself. Show specific examples where the framework helped and where it got in the way.

**17. "Building an LLM App? Here's the Stack I'd Actually Choose in 2026"**
Based on everything in your world model: Fastify or Next.js API, Supabase + pgvector (not Pinecone unless you need scale), OpenAI primary with Anthropic fallback, LangGraph for complex orchestration, n8n for workflow automation, Browserbase for anything that needs a real browser. Explain each choice from experience, not from reading docs.

**18. "Deploying ML on a Raspberry Pi (What They Don't Tell You)"**
ESP32 CSI sensing, llama.cpp quantization (Q4_K_M/Q5_0 on Pi 4B), the real constraints: CPU/host overhead is the bottleneck not VRAM, how quantization tradeoffs actually play out, why inference at >30 FPS on a Jetson Nano matters. Practical edge AI content that's grounded in your actual deployment experience.

---

## SHORT-FORM (Reels / Shorts / TikTok)

These should be dense, fast, one-insight-per-video. Not vlogs. Tech content.

**19.** "Browser agents take 15 seconds to click a button. MCP does it in 200ms. Here's why you need both." (MCP vs browser agents — you have the script drafted)

**20.** "Your LLM pipeline has a last-batch bug and you don't know it." (HyperLLM debugging insight)

**21.** "WiFi routers can detect if someone falls. 99% accuracy. No cameras." (Pulse-Fi stat, elder care angle)

**22.** "The real bottleneck in your AI app isn't GPU — it's API latency." (LLM API bottleneck belief)

**23.** "I processed 29 million YouTube videos with a two-pass LLM pipeline. Here's the architecture." (sponsorFind — whiteboard-style 60s breakdown)

**24.** "Open source wins infra. Closed labs win frontiers. Here's what that means for what you should build." (Quick thesis)

**25.** "RAG doesn't work because you're chunking wrong." (Selective semantic chunking insight)

**26.** "LLMs are getting stale. Your docs from 6 months ago are already wrong." (LLM staleness concept, real-time retrieval solution)

**27.** "Dynamic token batching: why your LLM calls are wasting 60% of your context window." (HyperLLM technique)

**28.** "ESP32 streams WiFi channel state information over Bluetooth to an Android app. No Arduino needed." (Hardware insight — skip Arduino, go direct)

**29.** "I tried fine-tuning an LLM on WiFi signals. 85% of the attention mass was on the prompt template, not the data." (WiFiGPT finding — genuinely surprising)

**30.** "The cold email stack nobody talks about: secondary domains, SPF/DKIM/DMARC, and why your emails are landing in spam." (Deliverability infra)

---

## WRITTEN (LinkedIn / X threads)

**31.** Thread: "I built a system that processes 1000x more LLM calls than naive sequential code. Here's the architecture." (HyperLLM — technical thread with diagrams)

**32.** Thread: "MCP is fast. Browser agents are flexible. Here's a decision framework for which to use when." (Your researched tradeoff framework)

**33.** Post: "The AI engineer role in 2026 isn't about training models. It's about orchestrating systems." (Your belief — sharp, specific)

**34.** Thread: "I scraped 29M YouTube videos and found the real patterns in brand sponsorships." (Data insights from sponsorFind)

**35.** Post: "VC-subsidized compute is a temporary gift. Build now or pay 10x later." (Computational arbitrage — provocative, time-sensitive)

---

## CONTENT ENGINE FORMATS (from your world model)

These are recurring series concepts, not one-offs:

**Workflow Autopsy** — Pick a real AI workflow (email automation, lead enrichment, content generation). Time yourself rebuilding it from scratch. Show what works, what breaks, what the actual time/cost is. 5-min videos.

**Agent in 60 Minutes** — Live-build a working AI agent in one hour. Start with a prompt, end with something deployed. Stream it or record it. Ship the code.

**Prompt Cage Match** — Take the same task, run it through GPT-4, Claude, DeepSeek, Llama. Show the actual outputs side by side. No opinion-first — let the outputs speak.

**ROI Diary** — Monthly transparent post: here's what I built, here's what it cost, here's what it made. Real numbers. This is the rarest and most valuable content format in AI — everyone talks about what's possible, almost nobody shows real economics.

---

The through-line across all of this: **you've actually built these things and hit the real walls.** That's the content moat. Anyone can explain what RAG is. Almost nobody can explain why their RAG pipeline's chunking strategy was destroying context and how they fixed it with a scoring + post-processing layer. That specificity IS the content.
