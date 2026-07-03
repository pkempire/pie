---
title: "Lucid Academy → Research Residency: 12-week curriculum + business pivot"
audience: "internal — Parth + co-founders"
status: "v1 proposal"
last_updated: 2026-05-30
---

# Lucid Academy → Research Residency

A 12-week, 1-on-1 AI research apprenticeship for high school students, plus a full business pivot from "summer AI bootcamp" to "youngest publishable researcher in your subfield."

This doc has three parts. Read them in order.

- **Part A** — Honest diagnosis of where Lucid is now, and what's structurally wrong with the current offer.
- **Part B** — The 12-week curriculum, week by week, three assigned papers, all deliverables.
- **Part C** — The business pivot. Pricing, positioning copy you can paste into the site, mentor model, marketing, 90-day execution plan.

---

## Part A — Diagnosis: what's actually wrong with Lucid today

### A.1 What you have

Pulled from your own PIE entry and `system_prompt_lucid_academy.md`:

- Founded 2023-08 (originally "Science Fair Academy," rebranded after C&D)
- 115 students, 95%+ completion, $40k net, 35% MoM
- 600 newsletter subs, ~2k parent signups
- Current product: 8-week cohort, 10 hrs/wk, 50% live / 50% project, $2,400, cap 25
- Current pitch: *"end-to-end launchpad … bridge CS theory and real-world impact … LLMs, agents, hands-on labs"*
- 1-yr target: $1M ARR. Stated short-term: close cohort-1 (20×$2.5k = $50k)
- Status flag in your own system: **"fading,"** last meaningful update ~45 days ago, next-step "Decide single offer" still pending

### A.2 The honest read

Here's what the copy actually says, translated.

> *"While schools teach syntax and sorting algorithms, we teach students how to actually build with AI—LLMs, agents, and all."*

This positioning **was true in 2023** and is no longer differentiated. In 2026 it describes Inspirit AI, Veritas AI, Algoverse, AI4ALL, BWSI, COSMOS, every YC-backed "AI for kids" company, and roughly forty Discord servers. "Hands-on AI projects for high schoolers" is the most crowded category in pre-college edtech right now. The pricing is in the middle of the pack. The format is identical to the pack. The credentialing language ("ISEF, CMU, regenerons") is the same language every competitor uses.

The single most predictive signal is the one in your own system: **"fading, 45 days of silence, next-step 'decide single offer' still pending."** That's not a marketing problem. That's the founder smelling that the offer doesn't differentiate hard enough to be worth pouring energy into.

### A.3 What's structurally broken

Five problems, in order of severity.

**1. The deliverable is a feeling, not an artifact.** A student finishes Lucid with "they built AI projects." A student finishes a research residency with *a paper at a workshop, a PR merged into a maintained repo, an arXiv submission, a Kaggle gold*. The first is unverifiable. The second is the highest-leverage college-app signal in 2026, because the bar for "they built something with AI" has collapsed. Every junior built a chatbot last summer. The bar is now "they published" or "they shipped to users."

**2. The cohort model commoditizes you.** Twenty-five students in a Zoom is a community college class. The competitor on $1500 is doing the same shape with worse mentors. The competitor on $15000 (Polygence, Inspirit's premium tier) is doing the same shape with one-on-one shellac. You're in the middle. Pick a side. The high side is where the margin and the differentiation are.

**3. The mentors are pitched as credentials, not as research collaborators.** Your own copy says *"mentors who've done it at ISEF, CMU, and beyond."* Every competitor says this. The actual differentiation is: *who is the mentor doing research with right now, and will they bring the student into that work?* That's a 1-of-1 thing nobody at scale can copy.

**4. The curriculum is a survey.** "Basic Python → ML theory → transformers" is the same arc as every Coursera. The student finishes knowing the names of things. A research residency teaches by *reproducing one paper deeply* — the student finishes having actually inhabited one corner of the field.

**5. The output never reaches a real venue.** Lucid programs (and 90% of competitors) end in a "showcase" — internal demo day, no external eyes. A research residency ends with a submission to a real workshop, a blog post on a maintained site, a PR on a maintained repo, or a Kaggle/Atlas leaderboard entry. The thing that survives the program is the only thing the student can put on a college app three years later.

### A.4 The pivot, stated once

**Old:** Lucid Academy is an end-to-end launchpad bridging CS theory and real-world AI impact for high school students.

**New:** Lucid Academy is a research residency. We pair one ambitious high school student with one active researcher for twelve weeks. The student picks an open problem from our curated frontier wiki, reproduces a recent SOTA paper, runs an original experiment, and ships a public artifact — a workshop paper, a merged PR to a maintained repo, an arXiv submission, or a benchmark leaderboard entry. By week 12 they are the youngest person doing real work in their corner of AI.

That sentence is the whole pivot. Everything below executes against it.

---

## Part B — The 12-week curriculum

### B.1 Philosophy: four principles

**Principle 1: Build before survey.** No lecture about a concept the student hasn't already tried to implement and watched fail. Karpathy's nanoGPT video works because by the time he names "attention" the viewer has already written a working line-by-line version. The curriculum follows that order in every week.

**Principle 2: One subfield, deeply.** The student does not learn "AI." They learn one corner — *LLM memory*, or *interpretability*, or *RL for agents*, or *medical imaging*, or *protein design*. The mentor's corner. You will know the SOTA, the open problems, the maintained repos, and the workshops where their paper has a real shot. Surface area kills depth.

**Principle 3: Three papers, three roles.** The reading list isn't "10 papers." It's three papers, each playing a distinct role.
- Paper 1 (Foundation): the canonical paper that defines the subfield. Read it to *understand the language.*
- Paper 2 (Reproduction target): a recent SOTA paper, ideally with public code. The student reproduces a key result. Read it to *understand what shipping looks like.*
- Paper 3 (Extension target): a frontier paper from the last 6 months that opens a question. The student designs an experiment that tests one specific claim or extension. Read it to *understand what is unsolved.*

That's it. Three papers, deeply. Not a survey of forty.

**Principle 4: The artifact is the grade.** No quizzes. No projects-for-projects-sake. The student's final week produces one of four canonical artifacts: workshop paper draft, merged repo PR, arXiv submission, or named leaderboard entry. The artifact is what they show colleges, employers, and future collaborators. The artifact is the only thing the program promises.

### B.2 The arc

Twelve weeks, structured as four three-week sprints. Each sprint produces a tangible artifact. Each sprint's artifact is a precondition for the next sprint.

- **Sprint 1 (W1–W3) — Tooling and foundations.** Get the student up the build curve and into the language of the subfield. Artifact: a from-scratch reimplementation of the field's foundational primitive (e.g., a 200-line transformer; a vanilla policy gradient; a from-scratch SAM segmenter).
- **Sprint 2 (W4–W6) — Pick the problem, reproduce the SOTA.** Curated frontier-problems wiki → student picks → reproduce one key result of Paper 2 on a tractable scale. Artifact: a runnable repo that reproduces a paper number within 3 percentage points.
- **Sprint 3 (W7–W9) — Run their first original experiment.** Design one experiment that extends the SOTA or tests one claim from Paper 3. Run it. The result is allowed to be negative. Artifact: a writeup with plots, comparing the student's experiment to the baseline.
- **Sprint 4 (W10–W12) — Polish, write, submit.** Convert the experiment into a final artifact appropriate for the venue. Submit. Artifact: the public submission, plus a code release with a README anyone can run.

### B.3 The 3 papers (default: LLM memory subfield)

The exact picks depend on the student and the mentor's subfield. Below is the **default reading list for an LLM-memory student**, which is the subfield you (Parth) know cold and have a curated wiki for. Substitute by subfield in Part C.

| # | Paper | Role | Why |
|---|---|---|---|
| 1 | **LongMemEval — Wu et al., 2024** ([[2410.10813-longmemeval]]) | Foundation | Defines the field's eval discipline. Reading it teaches the student what "memory" even means operationally — 5 capabilities, 30-40 sessions, abstention. Cannot do any other paper without this vocabulary. |
| 2 | **Mem0 — Mehta et al., 2024** ([[2504.19413-mem0]]) | Reproduction target | Production-style memory system with clean public code and a small dependency footprint. The student can reproduce Mem0's LongMemEval numbers in 2 weeks on a single laptop. Their own paper reports a 40% extraction-failure rate, which is a natural launchpad for a student experiment. |
| 3 | **Auto-Dreamer — Ye et al., 2026** ([[2605.20616-auto-dreamer]]) | Extension target | Six months old. RL-trained offline consolidator. ScienceWorld 41.1% vs UMEM 34.1% at 12× less memory. Opens a clean extension question: *does prompt-evolution (GEPA-style) get a meaningful fraction of Auto-Dreamer's gains at a fraction of the compute?* The student can run this on LoCoMo conv-26 in a week. (This is also Goal-01 of your own research program — the student is literally working on a problem you're working on, in parallel, with guidance.) |

The student reads them in this order: 1 → 2 → 3. Each is paired with a week of code work that exercises the paper's claims. By week 9 the student has read three papers carefully enough to *argue* with them, which is the actual cognitive move research demands.

### B.4 Week-by-week

Hours below assume 10 student-hours/week + 2 mentor-hours/week (one 90-min session + 30 min Slack/Discord async).

#### Sprint 1: Tooling and foundations

**Week 1 — Setup, fluency, the first build.**
- Tooling: Python venv, PyTorch, git, GitHub, VS Code remote, OpenAI API, a personal Notion or Obsidian for the lab notebook.
- Fluency check: implement a 3-layer MLP on MNIST, no copy-paste, no LLM help. Sub-50 lines. The student either has the fluency or builds it this week.
- Build of the week: Karpathy's nanoGPT walk, paused at every block. Student must explain `q @ k.T / sqrt(d_k)` in their own words on camera.
- Mentor session (90 min): set the subfield. Set the wiki. Walk through the 3 papers' abstracts together. Choose Paper 1.
- Read: Paper 1, sections 1–3 only.
- Deliverable: a 200-line `nanogpt.py` that trains on a 1 MB text corpus and generates the correct next token >50% of the time on a held-out small string set.

**Week 2 — Paper 1, deeply.**
- Read: Paper 1 in full. Two passes. First pass for narrative, second pass annotating claims, methods, ablations, limitations.
- Build: implement the paper's evaluation protocol from scratch. For LongMemEval this means a runnable scorer over the 500 questions. The student does NOT use the official repo. Reimplementing the eval is how you learn what the paper actually says.
- Mentor session: spend the full 90 min on three questions: *what is the paper measuring, what is it not measuring, and what would break the eval?* This is the only meta-skill that matters in research. The student finishes the session able to read the limitations section without taking it at face value.
- Deliverable: a markdown writeup answering those three questions plus a working scorer.

**Week 3 — The vocabulary check.**
- Build: a "naive baseline" of the subfield's task. For memory: stuff-it-all-in-context on LongMemEval-S. Get a real number. The number is bad. That's the point.
- Read: skim the related-work section of the SOTA paper (Paper 2). Identify the 4–5 systems that compete on Paper 1's benchmark.
- Mentor session: review the naive baseline result. Ask the student to predict what each of the 4–5 systems would change. Force articulation of the design space *before* showing them the answer.
- Deliverable: a "design space" markdown — one paragraph per system describing what changes from the naive baseline and what they'd predict it buys.
- **Sprint 1 artifact:** the from-scratch reimplementation + the design space writeup.

#### Sprint 2: Pick the problem, reproduce the SOTA

**Week 4 — Paper 2, reproduce a number.**
- Read: Paper 2 in full. The student must locate the headline number, identify the exact dataset + split + metric, and find the corresponding code.
- Build: stand up the official Paper 2 repo. Reproduce the headline LongMemEval number, even if it takes the full week. The lesson is not "succeed" — it's "actually grind through a research codebase, with its broken deps and its undocumented configs." Mentor's job is to keep the student from quitting on day 3 when the conda env breaks.
- Mentor session: pair-debug the reproduction. Show the student how a senior researcher reads a broken `requirements.txt`. (This is the unsexy core skill that nobody teaches and that determines whether someone can ever publish anything.)
- Deliverable: a number on Paper 2's benchmark within 3 percentage points of the published number, plus a `REPRODUCTION.md` documenting every divergence.

**Week 5 — Pick the open problem.**
- Pick a frontier problem from your curated wiki (the same one you've built at `personal-intelligence-system/research/`). The student picks from a list of 5–10 mentor-curated problems sized for 4-week experiments.
- Read: the relevant section of Paper 3.
- Mentor session: this is the critical session of the program. Walk through 5 candidate problems. The student picks. Justify in writing why this problem, why now, why it's tractable in 4 weeks. The justification becomes the introduction of the eventual paper.
- Deliverable: a one-page problem statement following the structure: *current SOTA = X; gap = Y; minimal experiment that would resolve part of Y = Z*.

**Week 6 — Build the minimum experimental harness.**
- Build: take the reproduction from W4, strip it to the minimum loop you need to run a single ablation. Set up a clean experiment-tracking spreadsheet or wandb.
- Read: Paper 3 in full.
- Mentor session: walk through Paper 3 together. Pause every claim. Ask: "do you believe this?" Train the student to read papers as adversarial documents.
- Deliverable: a runnable `experiment.py` that the student can sweep over one knob in one command.
- **Sprint 2 artifact:** the reproduction repo + the chosen problem statement + the experimental harness.

#### Sprint 3: Run the first original experiment

**Week 7 — First experiment.**
- Build: run the first sweep. Three conditions: baseline, baseline + the student's idea, baseline + an ablation of the student's idea.
- The student WILL hit some bug or stupid blocker that costs 2 days. This is the curriculum, not a flaw in the curriculum.
- Mentor session: live debug. Then look at the numbers together. Demand a prediction *before* the numbers are revealed.
- Deliverable: three runs, three numbers, a writeup of one paragraph each interpreting the result.

**Week 8 — Surprise, iterate, replicate.**
- Build: based on Week 7's surprise (there always is one), design the follow-up. Run it on 3 seeds (not 1). The 3-seeds rule is non-negotiable; this is where "I had a result" becomes "I have a result."
- Read: revisit Paper 3. Are your numbers consistent with their reported scaling? Why or why not?
- Mentor session: figure design. The student drafts the headline figure on paper. Mentor critiques. Iterate until the figure tells one clear story.
- Deliverable: the headline plot (PDF), 3-seed numbers, a 400-word interpretation.

**Week 9 — Make it real.**
- Build: scale to the larger split of the benchmark if compute allows, OR run an additional ablation if you're compute-bound. The goal is to make the result robust enough to defend.
- Mentor session: the "is this publishable?" conversation. The four canonical paths:
  1. **Workshop paper** at a NeurIPS/ICLR/ACL workshop (yes, high schoolers can submit; many workshops accept anonymous submissions).
  2. **PR to a maintained repo** (Mem0, Letta, LangGraph, etc.). The student's contribution: a new memory backend, a benchmark, a bug fix, a documentation rewrite.
  3. **arXiv submission** under a sponsor (the mentor co-authors and endorses).
  4. **Public blog post** on the mentor's research site with the code release.

   Pick the path that fits the result honestly. Negative results go to a workshop or blog. Strong positive results can go to arXiv.
- Deliverable: chosen artifact path, plus a draft outline.
- **Sprint 3 artifact:** experiment writeup + headline figure + 3-seed numbers + chosen submission path.

#### Sprint 4: Polish, write, submit

**Week 10 — Draft the artifact.**
- Build: write the artifact draft. For a workshop paper: 4 pages, Introduction / Background / Method / Results / Discussion. For a PR: the actual code + tests + a writeup in the PR description. For a blog: 2000 words and the code release.
- Mentor session: line edit the introduction together. The intro is 80% of what reviewers read.
- Deliverable: full first draft.

**Week 11 — Tighten.**
- Build: figures to print-quality (matplotlib config, fonts, sane sizing). Citations correct. Reproducibility appendix (commands, seeds, hardware).
- Mentor session: tear-down review of the full draft. The student must accept ~20 specific edits and push back on ~5.
- Deliverable: revision 2.

**Week 12 — Ship.**
- Submit. Hit the button. Post the blog. Open the PR. Push the arXiv.
- Final mentor session: walk through the student's lab notebook from W1 to W12. Identify the moment the student stopped being a student and started being a researcher. There always is one. Name it for them.
- Deliverable: **the public artifact**, link-shareable, the thing that goes on every college app, every internship application, every email to a future PI for the next four years.

### B.5 The student's three "deliverable cards" — what to publicly publish

Every Lucid Residency graduate leaves with three artifacts, each presented as a portfolio card on a personal site Lucid hosts for them:

1. **The artifact** — workshop paper PDF / merged-PR link / arXiv link / blog URL.
2. **The reproduction** — the working repo that reproduces Paper 2's headline number, public.
3. **The lab notebook** — the 12-week chronological log of decisions, dead ends, and pivots. This is the most unique credential of the three. No college applicant has ever submitted one. Every PhD has one.

These three together are the thing that no $2,400 cohort can ever produce.

### B.6 How this scales (and how it doesn't)

A 1-on-1 residency is **deliberately** non-scalable. That's the moat. Three structural points:

- **One mentor handles 2–3 students per cohort.** Beyond 3, the daily Slack quality drops and the program becomes Inspirit-with-extra-steps.
- **Mentors are real working researchers**, not undergrads. Recruit from: your CMU AirLab network, the PhD students whose papers you cite, your sponsorFind contacts, your peers in `paper_leaderboard`. The mentor needs to actually *be doing research* in the student's subfield while the student is in the program. That's what makes the assignment of frontier-problem options non-fake.
- **Subfield concentration matters more than headcount.** If you have 5 mentors, they should cover at most 3 subfields, deeply. The wiki of frontier problems is per-subfield. Pretending to mentor across all of AI is the failure mode.

Starting target capacity: **4 students × $12,000 = $48,000 per cohort, two cohorts a year (fall + spring), one parallel summer cohort of 6 students = $120k/year per active mentor.** Three mentors and you're at $360k/year before any other product. That's the unit economics.

---

## Part C — Business pivot: pricing, copy, operations, 90-day plan

### C.1 The new offer (single, focused)

Drop everything else. One offer, one outcome:

> **Lucid Academy Research Residency**
> 12 weeks, 1-on-1 with an active AI researcher.
> One subfield, deeply. Three papers, one of which you'll reproduce and one of which you'll extend.
> Output: a public artifact — workshop paper, merged PR to a maintained AI repo, arXiv preprint, or a research blog post on Lucid's site.
> $12,000.
> Five seats per cohort. Three cohorts per year (fall, spring, summer).

Kill the 8-week generic cohort. It is not the cash cow you think it is. Inspirit will outspend you on it forever.

### C.2 Pricing strategy

| Variant | Price | Capacity | Notes |
|---|---|---|---|
| Residency (default) | $12,000 | 5/cohort | The product. |
| Residency + co-authored arXiv (mentor explicitly co-authors) | $18,000 | 2/cohort | Premium tier. Mentor's name lends the credibility; the student does the work and is first author. |
| **Scholar seat** | $0 | 1/cohort | One per cohort, full-ride, by application. Income-or merit-gated. This is your social proof and your conscience. |

Total cohort revenue: 5 × $12k + 2 × $18k - 1 × $0 = **~$96k/cohort**. Three cohorts/yr × this mix = **~$288k/yr per active mentor**. You currently have you. With one peer mentor recruited in the next 6 months, you're at ~$580k/yr. With two, you cross $1M, which is the stated one-year target.

This is the only pricing path I can see that hits $1M ARR without a sales team. The cohort model gets you there only if you can grow seat count past 100/cohort, which means a sales team, which means a real series A, which is a different company.

### C.3 Positioning copy (paste this onto the site)

Headline:

> **Be the youngest person doing real research in your corner of AI.**
>
> 12 weeks. One mentor. One paper.

Subhead:

> Lucid Academy pairs ambitious high school students with active AI researchers for a 12-week, one-on-one research residency. You'll reproduce a recent paper, run an original experiment, and ship something the field will actually see — a workshop paper, a merged PR to a maintained AI repo, or an arXiv submission. By week twelve, you are the youngest contributor in your subfield.

The three-card "what's included":

1. **A mentor who's actively publishing.** Not a TA. Not an alum. Someone whose paper you just read. You'll work on a problem they're working on. Daily Slack, weekly 90-minute pair sessions.
2. **A curated frontier problem.** We maintain a private wiki of open research problems sized for 12-week experiments, updated weekly from arXiv. You pick one in week 5. You don't waste a month searching.
3. **A public artifact.** Every residency ends with a submission to a real venue. Not a showcase. Not a demo day. A submission with your name on it, public, link-shareable, defensible.

Below that, three named alumni cards (do these for real, with publicly shippable artifacts and consenting students — start with one in this first cohort, then accumulate):

> **Anika M., class of 2027.** Workshop paper at NeurIPS Memorization in Foundation Models, 2026: *"Counterfactual rewards for offline memory consolidation."* Co-authored with Lucid mentor Parth Kocheta.
>
> [photo] [paper PDF] [code]

That single card is worth more than all the "trusted by parents" stock copy. It is also the only thing that lets you charge $12,000 with a straight face. Build the first one in cohort one, even if it means the mentor (you) co-authors.

### C.4 What to kill from the current site

Delete from your current copy:

- *"end-to-end launchpad"* — vague, every competitor uses it.
- *"data science, ML, LLMs, and agents"* laundry-listed — survey language. Bad.
- *"mentors who've done it at ISEF, CMU, and beyond"* — drop "and beyond." Name the actual papers your mentors are on. Names and links, not categories.
- *"hands-on labs, industry-style mentorship"* — every competitor word for word.
- Cohort sizes ≥ 10 — the moment you list "25 students per cohort" you're priced like one.

What to keep:

- The "gap between theory and impact" framing — it's still true, just sharpen the *impact* word into *artifact*.
- The 35% MoM growth number — keep it on the page for social proof.
- The 95% completion stat — keep it. But add a row: "Public artifacts shipped per cohort: TBD → 5."

### C.5 The mentor model (recruit slowly, pay properly)

Three structural rules:

1. **Mentors are paid as researchers, not as instructors.** Floor: $300/hr for sessions, $5,000/cohort retainer. This is fundable from the $12k tuition and is the only number that gets you actual PhDs, not undergrads who'll ghost in week 6.
2. **Mentors bring their own subfield wiki.** The wiki of frontier problems is the asset that justifies their cost and the program's premium. You (Parth) have already built one for LLM memory. Other mentors need one for their subfield before they take students. The wiki-build is an unpaid 20-hour onboarding cost. It is also why mentor recruitment is the long pole, not student recruitment.
3. **Mentors co-author when the result merits it.** This is the actual North Star recruitment pitch: *"I'll pay you to mentor a smart kid for 12 weeks. You get a co-author on a workshop or blog. Your student does the grindy reproduction work. Your name attaches when you would have stamped the result anyway."*

The mentor recruitment funnel:
- Your CMU AirLab network → 2 candidates
- PhD students at the labs of papers you cite → cold email 30, expect 3 interested
- Sponsorfind / Lucid Labs network → 2 candidates
- LinkedIn outreach to AI4ALL alumni who are now late-stage PhDs → 5 candidates

Target: 3 mentors signed by end of 2026.

### C.6 Marketing — channels and message

You will not buy your way to the residency customer with ads. The audience is too narrow. The five channels that work:

1. **Working Memory (YouTube).** Your channel is the marketing engine. Every episode ends with: *"if you're a high schooler who wants to actually do this work, the Lucid Residency is open for applications."* The audience self-selects. This is the highest-leverage channel and the one nobody else has.
2. **Existing Lucid alumni.** 115 students, 600 newsletter subscribers, 2k parent signups. Email the parent list with "we've pivoted; here's the new offer; first cohort is half-priced for legacy families." This is your first 5 seats.
3. **High-school research Discords + Reddit.** r/ApplyingToCollege, r/AI_Agents, several invite-only Discord servers (RSI alumni, ISEF community). Show up with the artifact card, not the marketing copy.
4. **Mentor's own network.** Each mentor brings 5–10 student inquiries on their first cohort because their grad-school friends with HS-aged cousins exist.
5. **One conference appearance per quarter.** NeurIPS, ICLR student volunteers. Wear the t-shirt.

Stop running paid ads to the cohort program. The CAC is fighting Inspirit's funded ad spend on the same Google keywords. You can't out-bid them, and you don't need to.

### C.7 Operations — the SOP doc you need next

Once the offer is locked, you need three SOPs. Each takes a day to write, once.

- **Admissions SOP.** Application form (one essay, one code sample, one 90-min mentor-fit interview). Notion-based pipeline. Time-to-decision: 10 days max.
- **Onboarding SOP.** Day-0 checklist for the student: GitHub, Slack, calendar, reading list, lab notebook template, the W1 syllabus PDF. Sets the tone that this is professional, not summer camp.
- **Delivery SOP.** Mentor's weekly checklist, the 12-week curriculum doc (Part B above, distilled into a printable PDF), the artifact-submission templates for each of the four output paths.

These three SOPs become your only "scaling layer." Everything else is the mentor doing the work.

### C.8 90-day execution plan

Concrete, dated, dependency-ordered.

**Days 1–7 (this week):**
- Lock the offer. One offer, $12k, 12-week residency. Don't ship variants until the base sells.
- Pick the first subfield: LLM memory (the one you know).
- Email the parent list with a 200-word "we've pivoted, here's why, applications open" note. Stagger to 100/day to manage replies.

**Days 8–21:**
- Rewrite the homepage with the Part C.3 copy. Keep the brand colors. Replace every CTA with "apply."
- Build the application form (Typeform or Tally).
- Draft the three SOPs.
- Open applications.

**Days 22–49:**
- First 30 applications come in. Run interviews. Accept 4 paying + 1 scholar.
- Recruit mentor 2 (target: an LLM-memory or interpretability PhD student). Pay the $5k retainer up front. This is the highest-leverage check you write this quarter.
- Build the Lucid mentor-facing wiki of frontier problems for cohort 1 subfield. (You already have most of this in `personal-intelligence-system/research/` — repackage.)

**Days 50–89:**
- Cohort 1 starts (target: late August). 5 students × 12 weeks. You are the mentor for 3, mentor 2 takes 2.
- During the cohort: film the first Working Memory episodes with the cohort's permission. Every episode is also a marketing asset for cohort 2.

**Days 90 (end of November):**
- Cohort 1 mid-program. First public artifacts shipping in 6 weeks.
- Open cohort 2 applications. Price $12k → $14k for cohort 2 (you'll have cohort-1 artifacts as proof).
- Decide if you have signal for a third mentor.

### C.9 What success looks like at +12 months

By May 2027:

- 3 cohorts shipped (Aug 2026, Jan 2027, May 2027 starts).
- 15 paying students × $13k average = $195k delivered.
- 3 public artifacts: 1 workshop paper, 1 merged PR, 1 research blog. (Even one workshop paper is enough to anchor the marketing for year 2.)
- 3 active mentors.
- Working Memory YouTube: 5–10k subs, every video closing with the residency CTA.
- Your stated $1M ARR target is in reach for year 2 if mentor 4 lands and you push capacity to 8 per cohort.

This is the path. It is slower than the $1M-this-year version. It also has a real moat that the cohort version does not.

---

## Appendix: substituting the curriculum for non-memory subfields

The Part B curriculum is structurally subfield-agnostic. Swap the three papers and the example problems. Templates below; pick the column that matches the mentor and the student.

| Sprint | LLM memory | Interpretability | RL for agents | Medical imaging | Protein design |
|---|---|---|---|---|---|
| W1 build | nanoGPT | Anthropic toy-model SAE | nanoVPG (vanilla PG) | SAM minimal seg | AlphaFold-mini residue contact |
| Paper 1 (foundation) | LongMemEval (2410.10813) | Anthropic "Toy Models of Superposition" | Sutton & Barto PG chapter | nnU-Net | AlphaFold2 |
| Paper 2 (reproduce) | Mem0 (2504.19413) | "Scaling Monosemanticity" SAE | Search-R1 (2503.09516) | TotalSegmentator | ESM-2 fine-tune for binding |
| Paper 3 (extend) | Auto-Dreamer (2605.20616) | latest SAE-circuits paper | latest agent-RL paper | latest foundation-medseg | latest binder-design RL paper |
| Default extension Q | GEPA-vs-GRPO for the consolidator | feature-stability across model sizes | reward-shaping for tool-use | OOD generalization across modalities | inverse folding + binder design |

The substitution lets you scale across mentors without rebuilding the curriculum each time. Each mentor brings the three papers and the wiki problems; the structural template is fixed.

---

*End of doc. This replaces the existing Lucid copy as the canonical offer. If you want, the next thing to write is the actual homepage HTML mockup with the Part C.3 copy filled in — say the word and I'll do it.*
