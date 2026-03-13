# Content Ideas v3

Your reference accounts tell the story: Rishab Jain (young researcher making ISEF/research accessible), Boris Meinardus (documenting ML learning journey), Liam Ottley (AI agency how-to), Nick Saraev (n8n automations you can sell), HealthyGamerGG (iPad notes + deep dives on psychology), Ras Mic (web dev + AI tools), Nicky Case (interactive explanations), Dan Koe, Naval, Hormozi. Plus your own notes about writing like you're emailing a friend.

The vibe isn't "professional AI educator." It's: smart friend who's deep in the weeds on AI/tech/science/life and shares what they're actually learning, building, and thinking about. Casual but dense. More like a group chat than a lecture.

Your unique POV (from your own notes): "You need to figure out how to use AI to enhance yourself. To fill in your weaknesses, to collect data on your workflows, habits, etc. This is the best preparation for AGI."

---

## ACCOUNTS

### 1. One Paper a Day
Read 1 interesting AI/ML paper daily with the audience. Understand it, break it down, implement something from it. Short reels + full YouTube breakdowns.

**The vibe:** You + iPad + paper. Dr. K style handwritten notes. Not a lecture — you're genuinely learning in real time and taking the viewer with you.

Ideas (pull from what's actually interesting to you right now + PIE research):

- Context Rot (Chroma research) — LLMs don't actually use their context window uniformly. Performance degrades as input grows. Even on simple tasks. 18 models tested, all of them break. This is wild and most people building RAG don't know it. Break down the paper, sketch the degradation curves, talk about what this means for everyone building on long-context models.

- DeepSeek V3's 1M tokens/minute inference architecture. How did they actually do it? What's the mixture of experts setup, what's the training trick. Read the paper with me.

- Whatever new model just dropped that beats OpenAI/Claude benchmarks. Fresh take, not just benchmark table screenshots — actually read the methods section, explain what they changed.

- TikTok's recommendation algorithm paper. Break down how the "For You" page actually works at a systems level. The real ML behind it, not the influencer speculation.

- WiFi sensing papers (your own domain). CSI-based heart rate monitoring. You can actually explain this better than anyone because you've implemented it. "Read a paper with me, but I've actually built this one."

- The "Wireless as Language" framing — treating RF telemetry like text for transformers. This is a genuinely novel idea most ML people haven't seen.

- Attention-driven pruning for transformers. Your Locaris work. What does attention actually tell you about what the model learned?

- Context engineering as a formal discipline. Andrej Karpathy's framing — prompt engineering is dead, context engineering is the real skill. The optimization problem of filling the context window with the right info for the next step.

- Subject bias in WiFi HAR — the paper about evaluation leakage where same subjects in train/test makes models learn person-specific patterns, not activity patterns. Underrated ML methodological issue that applies way beyond WiFi.

- Any paper that connects to your interests: applied neuroscience, biohacking, nutrition science, robotics, RL, anything.

**Format:** iPad screen recording with handwritten annotations. 3-5 min reels for the "one insight" version, 15-20 min YouTube for the full breakdown. Post the paper link in comments.

---

### 2. Parth Kocheta (main channel)

This is the everything channel. Not just AI — the full range of stuff you're interested in. Write as if you were emailing a friend. Hyperfixate on one person this video would help.

**Longer videos (10-20 min):**

- **How to prevent AI brainrot.** Your own notes: "Prompting an LLM is like taking a stimulant — you get work done short term but your brain has downregulated and the ceiling for..." Read a book. Write every day. The implementation gap: everyone knows everything now, fat people know about nutrition and peptides but haven't run in weeks, business owners know the AI tools but don't use them. Does knowledge even matter anymore with AI? How should education change? What was Leonardo da Vinci's information diet? What's the right AI diet? This is genuinely an interesting question you care about and have real opinions on.

- **What is context engineering? Towards infinite memory.** Break down the Chroma research, the Karpathy/Tobi Lutke quotes, and connect it to your own PIE system. You've literally built a system that does context engineering for personal memory. Show the optimization problem: finding the right context-generating functions that maximize LLM output quality, subject to context length constraints. Sketch it on iPad.

- **Computer use vs tool calls — how AI agents actually interact with software.** The Smithery thesis: tool calls are the new clicks. Computer use is like building a robot to turn pages instead of giving it the text. But MCP can't do everything — you need browser agents for legacy systems, behind-login stuff. You've built with both. Show the actual tradeoffs, not the theory.

- **What is AGI? 3 different Turing tests for 3 domains.** The OG Turing test is conversational. Physical AI: give a robot a dirty kitchen, can it clean it? Economic: can you hire it as a contractor and it delivers? Your take: once we hit fully automated self-improving scientific research, that's AGI. Sketch the different tests on iPad, explain why most people are talking about AGI wrong.

- **Soloware — the future is hyperpersonalized software.** On the cost of software going to 0. What happens when anyone can build custom software for their exact workflow? Who wins, who loses? Connect to your own experience building tools (PIE, sponsorFind, HyperLLM).

- **The evolution of RAG.** From basic semantic search to current SOTA. Where it started, why naive chunking sucks, what vector similarity actually captures vs what it misses, and where it's going (context engineering, outcome-based memory scoring, the Roampal approach of memories that learn). Draw the evolution on iPad.

- **GPT sycophancy — is reasoning the way out?** The commitment jail / ChatGPT psychosis stuff. Why AI agreeing with everything you say is actually dangerous for your thinking. Connect to your own experience: breaking GPT sycophancy is a real problem you've thought about (three-layer strategy: balanced challenge, adversarial red-team, meta-review).

- **I've used every AI coding tool. They're all basically the same.** Cursor, Claude Code, Windsurf, etc. What actually makes a difference is the tools you give your LLM to use. Your 3 biggest unlocks when vibe coding. Must-have: Perplexity MCP for deep research. This is a genuinely useful video because everyone's asking "which AI coding tool" and the real answer is "they're interchangeable, focus on the MCP setup."

- **Here's every AI tool I pay for — and my current stack.** Go through your actual subscriptions. What's worth it, what's not, what you'd cut. Honest, useful, the kind of video you'd want to watch before buying a bunch of AI tools.

- **Solving the implementation gap.** The deeper version of the AI brainrot video. AI and social media make it easy to "learn" without acting. How to create your own AI diet. How to use AI to enhance yourself rather than replace your thinking. Your POV: use AI to fill weaknesses, collect data on workflows/habits. This is the best preparation for AGI.

- **The psychology of why people don't use AI tools they already pay for.** Tie into behavioral psych (your interest), dopamine/reward systems, the knowing-doing gap. Not preachy — genuinely curious exploration.

**Shorter stuff (misc interests, 5-10 min or shorts):**

- Applied neuroscience / biohacking experiments you're running. What's working, what's not. Peptides, supplements, sleep optimization. Not advice — just sharing what you're trying. The n-of-1 approach.

- BJJ/MMA content tied to AI. "What if you could track your rolling sessions with ML?" Training logs, movement analysis, the intersection.

- Music production + AI. Cloud rap production in FL Studio. Using AI for sound design, beat generation. Making something and showing the process.

- Philosophy takes that connect to tech. "Buddhism says craving causes suffering, but what about ambition?" Your actual belief about the taṇhā vs chanda distinction.

- Fitness content with a data angle. HRV tracking, what your Polar H10 data actually shows, training with creatine + retatrutide.

---

### 3. Lucid Academy (brand account)

3 short-form videos a day (from your notes). Target: high school students and parents.

**Pillar 1: Advice that unlocks opportunities**
- How I interned at CMU as a high schooler (cold emailing professors)
- Blueprint of an ISEF-winning science fair project
- 3 unique CS opportunities most students and parents miss (research via cold email, startups via cold email, open source contributions)
- The snowball effect in CS: do your first independent project fast
- The timeline from learning to code → first research project → publication
- Best competitive summer research programs (you've actually researched this extensively)
- The best mentors are only 1-2 steps ahead of you, not 10. That's why peer mentorship works.
- What not to do as a high school student who wants to pursue CS
- How to cold-email professors for research (your actual template that worked)
- What working in a research lab with PhDs in high school is actually like

**Pillar 2: Actionable AI learning tips**
- How to understand ML research papers without crazy math
- 3 unique AI research project ideas to learn LLM/agents
- Use Kaggle to start ML — here's how to actually pick a competition and learn from it
- What language should high school students learn? (Python, but here's why and what to learn first)
- How to come up with unique research ideas: end of papers, GitHub trending, intersection of 2 interests, talk to GPT
- How AI research sets students up to build their own startups

**Pillar 3: Quick interesting AI/tech things**
- AI news summaries but from a student perspective — what matters for someone learning
- Cool project ideas: "here's something you could build this weekend"
- Quick breakdowns of interesting concepts (transformers in 60 seconds, what is RLHF, etc.)

**Testimonials / case studies:**
- How we helped Tanvi win 5 science fair awards
- Student project showcases
- Record zoom testimonials with Aryan and other students

**Lead magnets:**
- Free course that helps achieve 1 mini-goal (per video, comment for full guide)
- Put all old newsletters on the site so people can read them
- Weekly newsletter (consistently)

---

### 4. AI Theme Page (content automation experiment)

This one you literally automate as much as possible. It's a funnel to a newsletter.

- AI news, funding announcements, breakthroughs
- Edits of AI founders talking — podcast clips from all the AI podcasts
- Quick takes on new model releases, tool drops, funding rounds
- Funnel everything to newsletter signup

You can semi-automate this with n8n + AI summarization. Source from arXiv, Twitter, TechCrunch, a16z, Sequoia podcasts. The human touch is your editorial curation — what's actually important vs what's noise.

---

### 5. Podcast: Interviews with young founders

Interview Xav, Ritesh, Alex, other young builders you know. Good tool to meet people too.

Topics to riff on:
- How they got started, what they're building now
- The real challenges nobody talks about
- AI tools in their workflow
- The overarching frontier questions: what happens when everyone can build software? How does education change? What's the actual timeline to AGI?

---

## CONTENT FROM PIE THAT ISN'T OBVIOUS

Stuff buried in your world model that maps to content you'd actually want to make:

**Your beliefs that are genuinely interesting takes:**
- "Everything is a memory problem" — most AI/search problems reduce to memory. Content angle: where is the field going on memory/context for AI?
- "The AI engineer role is evolving toward orchestration & product-focus" — less training, more system design. Useful perspective for anyone trying to break into AI.
- "Open source wins infra, closed labs win frontiers" — what to build and where
- "ChatGPT memory is a valuable but siloed data layer" — commercial opportunity in enabling export/versioning/cross-model syncing. You built PIE partly because of this.
- "Natural-language browser agents are the future interface" — a real thesis about where computing is going

**Your research that makes unique content:**
- WiFiGPT finding: model was relying on prompt template (>85% attention mass on first token), not actually learning from CSI data. This is a real finding about the limits of applying LLMs to non-text domains.
- Cross-environment transfer learning for localization. Your Locaris work. The reviewer feedback and how you're addressing it. "What it's like getting paper reviews" is content in itself.
- The two-pass LLM pipeline for sponsor detection. A real architecture pattern (fast heuristic filter → deep LLM parse) that's applicable to any large-scale classification problem.

**Topics from your notes that connect to PIE entities:**
- The Chroma "context rot" paper → connects to your PIE work, your RAG experience, your belief about memory
- "Tool calls are the new clicks" → connects to your MCP vs browser agents research, your Hermes architecture
- "Soloware" / cost of software going to 0 → connects to your HyperLLM, PIE, the tools you've built for yourself
- GPT sycophancy → connects to your "breaking GPT syncophancy" entity (three-layer strategy)

---

## THE ACTUAL SCHEDULE (realistic)

Don't try to do all 5 accounts at once. Start with 2:

**Week 1-2:** One Paper a Day (daily, 30 min to read + record) + Lucid Academy (3 shorts/day, batch film on weekends)

**Week 3-4:** Add main Parth Kocheta channel (1 longer video per week) + AI theme page (mostly automated)

**Month 2:** Add podcast (1 interview every 2 weeks)

The One Paper a Day is the easiest to start because the content comes to you — you just need to read and talk. The Lucid Academy content is closest to revenue because it directly markets the program. The main channel is where the interesting stuff goes but needs the most production effort.
