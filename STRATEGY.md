# PIE Strategy: Academic Paper, Monetization, and Release Plan

## 1. Academic Paper Strategy

### Title Options
- "Typed State Transitions as Memory Primitives for Long-Term Agent Memory"
- "Beyond Embedding Retrieval: Temporal Knowledge Graphs for Conversational Memory"
- "PIE: A Personal Intelligence Engine with State-Transition Memory"

### Novel Contributions (what passes peer review)
1. **Typed state transitions as memory primitives** — 5 types (creation, update, contradiction, resolution, archival) that capture HOW knowledge evolves, not just WHAT was said
2. **Temporal validity profiling** — Different fact types (emotions, preferences, biographical) have different natural decay rates
3. **Procedural memory from lifecycle analysis** — Detecting behavioral patterns across entity transition chains (nobody has published this for conversational memory)

### Paper Structure
1. Introduction — LLMs are time-blind, current memory is flat
2. Related Work — Zep/Graphiti (bi-temporal), Mem0 (flat), Letta (blocks), Mastra (three-date), MemWalker, COMEDY
3. Method — State transition model, extraction pipeline, resolution, temporal retrieval
4. Experiments
   - LongMemEval (500 questions, 6 categories) — compare to published baselines
   - LoCoMo (1986 QA pairs from 10 long conversations)
   - MSC (multi-session chat consistency)
   - Ablation studies: with/without contradiction detection, with/without temporal validity
5. Analysis — Where PIE wins (knowledge-update, temporal-reasoning) and where it loses (single-session)
6. Discussion — Procedural memory, enterprise applications, multi-agent orchestration

### Target Venues
- **NeurIPS 2026 Workshop** (Deadline ~Aug 2026) — Workshop on Memory in LLMs
- **EMNLP 2026** (Deadline ~Jun 2026) — Main conference, competitive but high impact
- **ACL 2026** (Deadline ~Feb 2026) — Might be too soon
- **arXiv preprint ASAP** — Establish priority, get citations going
- **COLM 2026** (Conference on Language Models) — Good fit

### What's Needed for Submission
1. Full benchmark results on LongMemEval, LoCoMo, MSC (not 5-sample runs — need 100+ per benchmark)
2. Ablation studies (remove each component and measure impact)
3. Reproducible code + data release
4. Clear comparison table against published baselines
5. Error analysis and failure mode discussion

### Estimated Timeline
- Week 1-2: Run full benchmarks, fix PIE baseline issues
- Week 3-4: Ablation studies, optimize extraction prompts
- Week 5-6: Write paper (sections 1-4 done in parallel)
- Week 7: Internal review, revisions
- Week 8: Submit to arXiv + conference

---

## 2. Monetization Strategy

### Option A: Open Source + Hosted Service (RECOMMENDED)
- **Open source**: Core PIE library (extraction, world model, retrieval)
- **Hosted service**: Managed temporal memory API
- **Revenue**: Usage-based pricing for API, enterprise support contracts

**Pricing model:**
- Free tier: 10K extractions/month, single world model
- Pro: $49/month — 100K extractions, 10 world models, priority support
- Enterprise: Custom — unlimited, SLA, on-prem deployment, custom extraction

### Option B: SDK + Enterprise License
- Open source the research code and benchmarks
- Sell commercial SDK license for production use
- Enterprise license includes: support, custom integration, fine-tuned extraction models

### Option C: Vertical SaaS (Highest Revenue, Most Work)
Pick one enterprise vertical and build a product:
- **Sales Intelligence**: Temporal deal tracking, contradiction detection for pipeline
- **Customer Success**: Track customer sentiment evolution, detect churn signals
- **Legal/Compliance**: Track evolving regulatory positions, detect contradictions in testimony
- **Healthcare**: Patient history tracking with temporal validity (symptoms vs conditions)

### Quick Revenue Ideas (< 30 days)
1. **MCP Server**: Package PIE as an MCP server for Claude Desktop / Claude Code. Charge $19/month.
2. **ChatGPT Memory Analyzer**: Tool that ingests ChatGPT exports and builds a temporal knowledge graph. One-time $9.99 purchase.
3. **API Wrapper**: Host PIE as an API, charge per extraction. $0.001/extraction.
4. **Consulting**: Offer temporal memory consulting for AI teams building agents. $200/hr.

---

## 3. What to Release NOW for Momentum

### Phase 1: This Week — Establish Presence
1. **Blog post / Twitter thread**: "LLMs Can't Tell Time: 8 Experiments Showing Why Agent Memory is Broken"
   - Use results from `experiments/llm_temporal_gaps.py` and `experiments/temporal_reasoning_tests.py`
   - Include charts, diagrams, concrete examples
   - End with: "We built PIE to fix this. Paper + code coming soon."

2. **GitHub repo (public)**:
   - README with problem statement, approach, experiments
   - Experiments directory (self-contained, no API needed)
   - Benchmark infrastructure (shows rigor)
   - World model core (the novel part)
   - DON'T release: extraction prompts (competitive advantage), full pipeline (save for paper)

3. **arXiv preprint**: Short 4-page workshop paper with:
   - Problem definition + literature review
   - State transition model description
   - Preliminary benchmark results
   - "Full results forthcoming"

### Phase 2: Week 2-3 — Build Community
1. Run full LongMemEval benchmark (500 questions, all baselines)
2. Post results with comparison table
3. Release benchmark runner so others can reproduce
4. Create a Discord / GitHub Discussions

### Phase 3: Month 2 — Productize
1. Package as pip-installable library: `pip install pie-memory`
2. MCP server for Claude Desktop
3. REST API for integration
4. Documentation with quickstart guide

---

## 4. Terminal Commands — Benchmarks

### Quick Test (2-3 minutes, ~$0.50)
```bash
cd /path/to/personal-intelligence-system

# Run naive_rag on 5 LongMemEval questions
python -m benchmarks.eval_harness -b naive_rag -n 5 --benchmarks longmemeval

# Run full_context on 5 questions
python -m benchmarks.eval_harness -b full_context -n 5 --benchmarks longmemeval
```

### Medium Run (15-30 minutes, ~$5)
```bash
# Compare all baselines on 10 questions per benchmark
python -m benchmarks.eval_harness -b all -n 10

# Run PIE with caching on 10 LongMemEval questions
python -m benchmarks.longmemeval.runner -b pie_temporal_cached -n 10 --cache-dir benchmarks/longmemeval/cache

# Run on specific category (temporal-reasoning)
python -m benchmarks.longmemeval.runner -b all -n 20 -c temporal-reasoning
```

### Full Benchmark (2-6 hours, ~$50-100)
```bash
# Full LongMemEval (500 questions) — ALL baselines
python -m benchmarks.longmemeval.runner -b all --cache-dir benchmarks/longmemeval/cache -o results/full

# Full LoCoMo (1986 QA pairs)
python -m benchmarks.eval_harness -b all --benchmarks locomo

# Full MSC
python -m benchmarks.eval_harness -b all --benchmarks msc

# Everything at once
python -m benchmarks.eval_harness -b all
```

### Debug Single Question
```bash
# Run single question with full debug output
python -m benchmarks.longmemeval.runner --question-id e47becba --debug

# See dataset statistics
python -m benchmarks.longmemeval.runner --stats
```

---

## 5. Speed Analysis

### Current Architecture
- **LongMemEval**: Each question has a UNIQUE haystack of ~53 sessions
  - PIE must build a separate world model per question
  - First run: ~2-5 min/question (10-11 LLM extraction calls)
  - Cached re-run: ~5s/question (just retrieval + answer)
  - Full 500 questions first run: ~16-42 hours
  - Full 500 questions cached: ~42 minutes

- **LoCoMo**: 10 conversations, 1986 questions
  - Each conversation shared across ~200 questions
  - Build 10 world models total, reuse across questions
  - Much more efficient: ~10 min build + ~3 hours for all QA

### Parallelization Opportunities
1. **Per-question parallelism**: Different questions can be processed simultaneously since they have independent haystacks (LongMemEval) or share a pre-built world model (LoCoMo)
2. **Batch extraction**: Multiple sessions can be batched into single LLM calls
3. **Embedding batch**: Entity embeddings computed in batches (already implemented in CachedWorldModel)
4. **Async API calls**: OpenAI API supports async — could 5x throughput

### What To Build
A parallel runner that:
1. Groups questions by shared haystack (LoCoMo) or treats independently (LongMemEval)
2. Builds world models in parallel (up to 5 concurrent)
3. Caches everything to disk
4. Does retrieval + answer + judge in parallel batches

---

## 6. Core Code Still Needed

### Critical (blocks benchmark runs)
- [x] Fix eval_harness BASELINES to include pie_temporal_cached
- [ ] Fix LoCoMo adapter to share world models across questions from same conversation
- [ ] Add `--parallel N` flag to eval_harness for concurrent question processing
- [ ] Improve PIE extraction prompt for benchmark format (currently tuned for ChatGPT exports)

### Important (improves scores)
- [ ] Tune retrieval to blend embedding similarity + temporal proximity to question date
- [ ] Add "hybrid" baseline: RAG + PIE (retrieve raw chunks AND compiled entities)
- [ ] Increase max_context_chars for PIE baseline (currently 12K, could be 30K)
- [ ] Better entity embedding text (include transitions summary, not just current state)

### Nice to Have (for paper)
- [ ] Ablation framework (run with/without each component)
- [ ] Error analysis script (categorize failures by type)
- [ ] Visualization of world model graph
- [ ] Fine-tune extraction model on benchmark data
