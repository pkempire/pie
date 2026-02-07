# PIE Launch Plan — Weekend of Feb 7-8, 2026

**Goal:** Ship PIE publicly with blog post and video by Sunday night.

---

## Current Status

### ✅ Done
- [x] Core extraction pipeline (1,057 entities from 203 batches)
- [x] World model with transitions + relationships
- [x] Benchmark runs (LongMemEval 66%, LoCoMo 58%, ToT 56%, MSC 46%)
- [x] Graph-aware retrieval (new, needs testing)
- [x] GitHub repo exists (pkempire/pie)
- [x] Blog draft (BLOG-DRAFT.md — 40KB, needs number updates)
- [x] Video outline (VIDEO-OUTLINE.md — 25KB)
- [x] Architecture docs

### ⚠️ Needs Work
- [ ] Update blog with actual benchmark numbers (not placeholders)
- [ ] Clean README for OSS release
- [ ] Add LICENSE file
- [ ] Record video
- [ ] Demo/visualization that works reliably
- [ ] Push clean code to GitHub

### ❌ Cut for Launch (Nice to Have)
- RL training (documented in RESEARCH-IDEAS.md for future)
- Sales product demo
- Full benchmark suite reruns

---

## Saturday Feb 7 (Today)

### Morning/Afternoon ✓
- [x] Built graph retriever with intent parsing
- [x] Created research ideas doc with RL directions
- [x] Ran retriever comparison

### Evening (Now → 11pm)

**1. Fix Demo Query Interface** [1hr]
- The comparison showed 0% overlap and missed "PIE" — retrieval isn't working right
- Need a reliable demo that actually answers questions about the extracted world model
- Test queries that work:
  - "What projects have I worked on?"
  - "How has my tech stack evolved?"
  - "What tools do I use most?"

**2. Update Blog Numbers** [1hr]
- Replace [TODO] placeholders with actual benchmark results
- Key numbers to add:
  - LongMemEval: 66.3% (naive_rag baseline)
  - LoCoMo: 58%
  - ToT: 56.2% (naive_rag), 31.2% (pie_temporal)
  - Task-adaptive finding: helps ordering +8-25%, hurts date lookup -23-38%

**3. Clean README.md** [30min]
- Installation instructions
- Quick start example
- Link to blog/video

**4. Git Cleanup** [30min]
- Add .gitignore for output/, __pycache__, .env
- Add MIT LICENSE
- Commit all changes

---

## Sunday Feb 8

### Morning (9am-12pm)

**5. Record Video** [3hr]
- Follow VIDEO-OUTLINE.md structure
- Screen recordings needed:
  - ChatGPT export → PIE extraction
  - Query interface demo
  - Viz.html graph visualization
  - Benchmark results comparison
- Keep to 15-20min

### Afternoon (12pm-6pm)

**6. Edit Video** [3hr]
- Basic cuts, add music
- Screen overlays for code/architecture
- Export 1080p

**7. Finalize Blog** [1hr]
- Final read-through
- Add video embed once uploaded
- Check all links work

### Evening (6pm-10pm)

**8. Publish** [2hr]
- Upload video to YouTube (unlisted first, then public)
- Post blog (where? personal site? Medium? Substack?)
- Push final code to GitHub
- Tweet/share announcement

---

## Key Deliverables

| Deliverable | Target | Status |
|-------------|--------|--------|
| GitHub repo clean + public | Sun 6pm | 🔄 In progress |
| Blog post live | Sun 8pm | 📝 Draft done |
| Video uploaded | Sun 9pm | 📋 Outline done |
| Social announcement | Sun 10pm | ⏳ Waiting |

---

## Demo Script

For video recording, show this flow:

```bash
# 1. Export ChatGPT data (show download)
# Goes to ~/Downloads/conversations.json

# 2. Run extraction
cd personal-intelligence-system
python3 run.py ingest --input ~/Downloads/conversations.json

# 3. Show results
python3 run.py stats
# → 1,057 entities, 2,278 transitions, 1,187 relationships

# 4. Query interface
python3 -m pie.eval.query_interface
> What projects have I worked on?
> How has my tech stack evolved?
> What patterns do you see in my decision-making?

# 5. Visualization
open output/viz.html
# → Force-directed graph, click to explore
```

---

## Blog Distribution

1. **Primary:** Personal site or Substack (own the content)
2. **Cross-post:** 
   - HN: "Show HN: PIE – Temporal Memory for AI Agents"
   - r/MachineLearning
   - Twitter/X thread
   - LinkedIn (if applicable)

---

## Video Distribution

1. **YouTube:** Main upload
2. **Twitter:** 2-3min teaser clip
3. **LinkedIn:** Same teaser

---

## Risks

| Risk | Mitigation |
|------|------------|
| Demo fails during recording | Pre-record demo segments, use cached results |
| Video editing takes too long | Keep it simple, minimal effects |
| Blog too long | Cut to core narrative, move details to appendix |
| Benchmark numbers questioned | Be honest about what's baseline vs PIE |

---

## Tonight's Priority Order

1. **Fix retrieval demo** — can't show broken queries
2. **Update blog numbers** — credibility
3. **Git cleanup** — ship-ready code
4. Rest, record fresh tomorrow

---

## Commands to Run Now

```bash
# Test query interface with good queries
cd /Users/parthkocheta/.openclaw/workspace/personal-intelligence-system
python3 -m pie.eval.query_interface --query "What projects have I worked on?"

# Check viz works
open output/viz.html

# Git status
git status

# Add LICENSE
echo "MIT License..." > LICENSE  # Actually write the full text
```
