# Content Ideas v4 — February 2026

Reference accounts: Rishab Jain, Boris Meinardus, Liam Ottley, Nick Saraev, HealthyGamerGG, Ras Mic, Nicky Case, Dan Koe, Naval, Hormozi. Write like you're texting a group chat. Smart friend energy, not professor energy. Casual but dense.

Your POV: "You need to figure out how to use AI to enhance yourself. To fill in your weaknesses, to collect data on your workflows, habits, etc. This is the best preparation for AGI."

---

## 1. ONE PAPER A DAY

iPad + paper + you learning in real time. Dr. K style handwritten notes. 3-5 min reels + 15-20 min YouTube for full breakdowns.

**Timely papers to cover right now:**

- **WebMCP just dropped (Feb 10).** Chrome 146 shipped `navigator.modelContext` — websites can now expose structured tools directly to AI agents. No more screenshot-scraping. 89% token efficiency improvement over visual methods. Google + Microsoft W3C standard. This is the biggest shift in how AI agents interact with the web since... ever? Read the spec with me. Sketch the architecture on iPad — declarative API (HTML forms) vs imperative API (JS handlers). Connect to your own MCP vs browser agents research. You literally predicted this convergence.

- **Anthropic's context engineering guide.** They formalized it. Context engineering = optimizing the information you feed an LLM to maximize output quality, subject to context window constraints. This is the real skill now, not prompt engineering. Connect to your PIE system — you've been doing context engineering for personal memory for months.

- **"Agents need runbooks, not bigger context windows."** Memory engineering paper/framework. The distinction between conversational memory (preferences, history) and operational memory (procedures, runbooks). Sketch both on iPad. Why cramming more tokens in isn't the answer.

- **Context rot (Chroma research).** 18 models tested, all degrade as context grows. Even on simple tasks. Performance drops are predictable and measurable. Most people building RAG don't know this. Draw the degradation curves, talk about what it means for everyone building on long-context models.

- **Whatever model dropped this week.** GPT-5.3-Codex merged the Codex + GPT-5 training stacks (Feb 5). Claude Opus 4.6 shipped 1M token context + agent teams (Feb 5). Gemini 3.1 Pro just announced (Feb 19). Gemini 3 Deep Think solved grad-level physics. Don't just screenshot benchmarks — read the methods section, explain what actually changed architecturally.

- **MCP hit 97 million SDK downloads.** OpenAI, Microsoft, Google, Linux Foundation all backed it. It won the "tool protocol war." Read the adoption trajectory paper, sketch why this protocol won and others didn't. The network effects argument.

- **Agent-to-agent security crisis.** Emerging problem: when agents call other agents, trust chains break. Prompt injection through tool results. There's new research on sandboxing, capability-based security for multi-agent systems. Read it, sketch the attack vectors on iPad.

- **X-Humanoid TienKung 3.0.** First full-size humanoid with touch-interactive whole-body control. Just launched. Read their paper — what's the control architecture? How does tactile feedback change locomotion? Connect to your robotics interest.

- **WiFi sensing papers (your domain).** CSI-based heart rate monitoring, the "Wireless as Language" framing for transformers. You've actually built this stuff. "Read a paper with me, but I've actually implemented this one."

- **Subject bias in WiFi HAR.** Evaluation leakage where same subjects in train/test makes models learn person-specific patterns, not activity patterns. Underrated ML methodological issue that applies to basically every domain.

**Format:** iPad screen recording + handwritten annotations. Post paper link in comments. Film a batch of 3-4 on the weekend.

---

## 2. PARTH KOCHETA (main channel)

The everything channel. Educational, entertaining, dense. One person per video you're trying to help.

### Longer videos (10-20 min)

**Timely right now:**

- **WebMCP killed browser agents. Or did it?** Chrome 146 just shipped `navigator.modelContext`. Websites expose structured tools to AI agents natively. No more DOM scraping, no more screenshot analysis. But here's the thing — WebMCP only works on sites that implement it. Legacy sites, behind-login stuff, anything not updated? You still need browser agents. The real architecture is hybrid and you've been building with both sides. Do a live demo: show a WebMCP tool call completing instantly vs a browser agent taking 15 seconds on the same task, then show a task only the browser agent can handle. Sketch the decision framework on iPad. This is THE video to make right now because WebMCP is 9 days old and nobody's done a real technical breakdown yet.

- **Context engineering is the new prompt engineering. I built a system that does it.** Anthropic formalized context engineering. Karpathy coined it, Tobi Lutke evangelized it. The optimization problem: finding the right context-generating functions (retrieval, summarization, selection) that maximize LLM output quality, subject to context length constraints. You've literally built PIE — a personal context engineering system. Show the math on iPad: it's a constrained optimization problem. Show your system working. This is timely because Anthropic just published their guide and the concept is entering mainstream discourse.

- **The tool protocol war is over. MCP won. Here's what that means.** 97M SDK monthly downloads. OpenAI, Microsoft, Google, Linux Foundation backing. WebMCP extending it to browsers. Smithery's thesis: "tool calls are the new clicks." Walk through the timeline of how MCP went from Anthropic side project to industry standard. What does it mean when every app exposes structured tools? The death of the GUI? The rise of agent-native software? Your take from actually building MCP integrations.

- **I've used every AI coding tool. Here's the thing nobody tells you.** Cursor, Claude Code, Windsurf, Replit Agent, v0, Bolt.new, Lovable — they're converging. The differentiator isn't the IDE, it's the tools you give your LLM to work with. Your 3 biggest unlocks when vibe coding. Why everyone asking "which AI coding tool" is asking the wrong question. This is evergreen but also timely because vibe coding went fully mainstream in 2026 — 25-30 million solopreneurs in the US are now building software.

- **Soloware — the future is software that only you use.** Cost of software → 0. What happens when anyone can build custom tools for their exact workflow? You've built PIE, sponsorFind, HyperLLM, Cold-Email Autopilot — all for yourself first. The solopreneur boom is real (25-30M in US). Who wins, who loses when every person can be their own software company? Connect to the vibe coding explosion.

- **GPT sycophancy is making you dumber and you don't notice.** The commitment jail / ChatGPT psychosis stuff. Why AI agreeing with everything you say is actually the most dangerous feature of current LLMs. Not a rant — a genuine technical + psychological exploration. Your three-layer strategy for breaking it (balanced challenge, adversarial red-team, meta-review). Connect to your interest in applied psychology. Timely because Anthropic just published their agent autonomy study.

- **How to prevent AI brainrot.** Your notes: "Prompting an LLM is like taking a stimulant." The implementation gap — everyone knows everything now, fat people know about nutrition and peptides but haven't run in weeks, business owners know the AI tools but don't use them. Does knowledge even matter anymore? What was Leonardo da Vinci's information diet? What's the right AI diet? You actually care about this question. iPad sketches, genuine exploration.

- **What is AGI? 3 Turing tests for 3 domains.** The OG Turing test is conversational. Physical AI: give a robot a dirty kitchen, can it clean it? (Timely — humanoid robots are actually shipping now. TienKung 3.0, Figure, 1X.) Economic: can you hire it as a contractor and it delivers? Your take: fully automated self-improving scientific research = AGI. Sketch the different tests on iPad.

- **The evolution of RAG — from broken to barely working to... actually useful?** Where it started, why naive chunking sucks, what vector similarity actually captures vs misses. Context rot research showing the problem is worse than people think. Where it's going: context engineering, outcome-based memory scoring, memories that learn. Draw the evolution on iPad. You've built multiple RAG systems and hit all the real walls.

- **Here's every AI tool I pay for.** Go through your actual subscriptions. What's worth it, what's not, what you'd cut. The video you'd want to watch before buying a bunch of AI tools. Honest, specific, no affiliate-link energy.

### Shorter stuff (5-10 min or shorts, misc interests)

- **I tried using AI as a pre-production tool for this video.** Use AI video generators (Sora, Veo, Midjourney Video) to storyboard and pre-visualize a tech explainer. Show the process — what prompts you used, what worked, what looked like garbage. This is both content AND practice for your videography skills. Meta.

- **Training BJJ but tracking it with ML.** What if you logged rolling sessions and ran pattern analysis? Movement classification from accelerometer data. The intersection of your martial arts interest and your ML skills. Actually try it, show the results (even if they suck).

- **My biohacking stack — the n-of-1 approach.** HRV tracking with Polar H10, what the data actually shows, creatine + retatrutide, sleep optimization experiments. Not advice — data. Show the dashboards, be honest about what's working.

- **Making a beat with AI.** Cloud rap production in FL Studio but using AI for sound design / generation. Show the full process from blank project to finished beat. Where AI helps, where it makes everything sound generic.

- **Buddhism says craving causes suffering, but what about ambition?** The taṇhā vs chanda distinction. Your actual belief about this — you've thought about it. Connect to tech culture's relationship with ambition. 5-minute iPad notes style.

---

## 3. ARTISTIC / FILM EXPERIMENTS

This is the "learn videography, editing, and storytelling" bucket. The point isn't to go viral — it's to develop craft.

- **Video essay: "The Latent Space of Cinema."** AI video generators (Sora, Veo, Kling) are becoming pre-production tools in real filmmaking. Directors use them to pre-visualize shots and communicate pacing. But they also create a new kind of image — not captured light, but steered patterns in latent space. Make a video essay exploring this. Use AI-generated footage intercut with real footage. Practice: voiceover narration, editing rhythm, visual argument construction. Research is fresh — "The Principia of Cinema" essay just dropped in Jan 2026.

- **"One Day in [City]" — cinematic mini-doc, no dialogue.** Pick a location. Film it entirely on your phone. Tell a story purely through visuals, cuts, and music. No narration, no text overlay. Practice: shot composition, color grading, pacing, sound design. This is the filmmaking equivalent of doing scales. Post it, don't overthink it.

- **Nicky Case-style interactive explainer.** Pick one concept (attention mechanisms, WiFi sensing, context windows) and build an interactive web explanation. Animated, playful, you can click things. This forces you to think about storytelling differently — the viewer controls the pace. Put it on your site. Make a short video showing it off.

- **Timelapse build videos.** Film yourself building something (a script, a circuit, a pipeline) compressed into 3-5 minutes with music and text overlays explaining decisions. Practice: time management in editing, text animation, pacing. Low production effort, high learning value.

- **"How I'd shoot this" reaction series.** Watch a well-made tech video (Vsauce, Veritasium, Fireship) and break down WHY it works visually. What shots they used, how they structured the argument, where the music shifts. Screen recording + your commentary. You learn filmmaking by studying it.

- **Shot-on-iPhone short film about your workspace.** 60 seconds. Macro shots of keyboards, screens with code, soldering irons, the router streaming CSI data. Practice: close-up cinematography, shallow depth of field (portrait mode), ambient sound design. The content is "what does building AI stuff actually look like?"

---

## 4. LUCID ACADEMY (brand account)

3 short-form videos a day. Target: high school students and parents. Batch film on weekends.

**Pillar 1: Advice that unlocks opportunities**
- How I interned at CMU as a high schooler (the actual cold email template that worked)
- Blueprint of a science fair project that wins at ISEF
- 3 CS opportunities most students miss: research via cold email, startups via cold email, open source contributions
- The timeline from learning to code → first research project → publication
- Best competitive summer research programs (you've researched this extensively)
- What working in a research lab with PhDs as a high schooler is actually like
- The best mentors are 1-2 steps ahead, not 10 — why peer mentorship works

**Pillar 2: Actionable AI/CS learning**
- How to understand ML research papers without the crazy math
- 3 AI research project ideas you can actually start this weekend
- What language should high school students learn? (Python, but here's what to learn first)
- Use Kaggle to start ML — how to actually pick a competition and learn
- How to come up with unique research ideas: end of papers, GitHub trending, intersection of 2 interests, ask GPT
- How AI research sets you up to build startups later
- **NEW (timely):** What is MCP and why every CS student should learn it now. MCP just became the industry standard for AI tools. Explain it simply.
- **NEW (timely):** Build your first AI agent this weekend using Claude Code or Cursor. Step-by-step for beginners.

**Pillar 3: Quick interesting AI/tech things**
- AI news from a student perspective — what actually matters for someone learning
- **NEW:** WebMCP explained in 60 seconds. Websites just became function calls.
- **NEW:** Gemini 3 Deep Think can solve graduate physics. What does that mean for education?
- Quick concept breakdowns: transformers in 60 sec, what is RLHF, what is context engineering
- Cool project ideas: "here's something you could build this weekend"

**Testimonials / lead magnets:**
- Student project showcases (Tanvi winning 5 science fair awards)
- Free mini-course per video (comment for full guide)
- Weekly newsletter

---

## 5. AI THEME PAGE (automated content experiment)

Semi-automated with n8n + AI summarization. Source from arXiv, X, TechCrunch, a16z. Your editorial curation is the human touch.

- AI news, funding announcements, breakthroughs
- Edits of AI founders talking — podcast clips
- Quick takes on new releases
- **Currently hot topics to cover:** WebMCP launch, MCP standardization, Claude Opus 4.6 agent teams, humanoid robot launches (TienKung 3.0), the vibe coding boom, GPT-5.3-Codex, Gemini 3.1 Pro, agent security concerns
- Funnel everything to newsletter signup

---

## 6. PODCAST: Young Founders

Interview Xav, Ritesh, Alex, other builders you know.

Riff topics:
- How they got started, what they're building now
- The real challenges nobody talks about
- AI tools in their workflow
- **Timely frontier questions:** WebMCP and the "agent-ready web" — does every startup need to implement it? The soloware thesis — is the age of big software companies ending? Vibe coding — are non-technical founders now technical founders? When does AGI actually arrive and what happens to their businesses?

---

## CONTENT FROM PIE — NON-OBVIOUS ANGLES

**Your beliefs that are genuinely interesting takes:**
- "Everything is a memory problem." Most AI/search problems reduce to memory. Now validated by Anthropic formalizing context engineering.
- "The AI engineer role is evolving toward orchestration & product-focus." Less training, more system design. Increasingly true as MCP makes tool orchestration the core skill.
- "Open source wins infra, closed labs win frontiers." What to build and where.
- "ChatGPT memory is a valuable but siloed data layer." You built PIE partly because of this. Still true — nobody's solved cross-model memory portability.
- "Natural-language browser agents are the future interface." Now getting tested in real time as WebMCP ships.

**Your research that makes unique content:**
- WiFiGPT: model was relying on prompt template (>85% attention mass on first token), not learning from CSI data. Genuine finding about limits of applying LLMs to non-text domains. Still novel, nobody else has talked about this.
- Cross-environment transfer learning for localization (Locaris). The reviewer feedback process. "What it's like getting paper reviews" is content in itself.
- Two-pass LLM pipeline for sponsor detection. Fast heuristic filter → deep LLM parse. Applicable pattern for any large-scale classification problem.
- 29M YouTube videos processed to find every sponsorship deal. The data patterns are genuinely interesting to the creator economy.

---

## SCHEDULE (realistic)

**Week 1-2:** One Paper a Day (daily, 30 min to read + record) + Lucid Academy (3 shorts/day, batch film weekends)

**Week 3-4:** Add main channel (1 longer video/week, start with WebMCP video — it's the most timely) + AI theme page (mostly automated) + 1 artistic experiment

**Month 2:** Add podcast (1 interview every 2 weeks) + continue artistic experiments

**The WebMCP video is the #1 priority for the main channel.** It's 9 days old, directly in your wheelhouse (you've built with both MCP and browser agents), and nobody's done a real technical breakdown with live demos yet. First-mover advantage on a topic you actually know deeply.
