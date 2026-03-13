#!/usr/bin/env python3
"""
Run PIE benchmarks locally.

Usage:
    # Quick test (5 questions, see if extraction works):
    OPENAI_API_KEY="sk-proj-..." python run_benchmark.py --quick-test

    # Full LongMemEval run (500 questions):
    OPENAI_API_KEY="sk-proj-..." python run_benchmark.py --benchmark longmemeval --baseline pie_temporal

    # Compare PIE vs naive RAG:
    OPENAI_API_KEY="sk-proj-..." python run_benchmark.py --benchmark longmemeval --baseline naive_rag
    OPENAI_API_KEY="sk-proj-..." python run_benchmark.py --benchmark longmemeval --baseline pie_temporal

    # Run with caching (builds world model once, reuses for all questions):
    OPENAI_API_KEY="sk-proj-..." python run_benchmark.py --benchmark longmemeval --baseline pie_temporal_cached --cache-dir benchmarks/longmemeval/cache

    # Debug mode (shows extraction output):
    OPENAI_API_KEY="sk-proj-..." python run_benchmark.py --quick-test --debug
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import re as _re

def _normalize_for_match(text: str) -> str:
    """Normalize text for rough matching — strip punctuation, collapse whitespace."""
    text = text.lower().strip()
    text = _re.sub(r'["\'\.\,\!\?\;\:\(\)\[\]\{\}]', ' ', text)
    text = _re.sub(r'\s+', ' ', text).strip()
    return text


def run_quick_test(debug: bool = False, num_questions: int = 5):
    """Run a quick test on a few LongMemEval questions to verify extraction works."""
    from benchmarks.longmemeval.adapter import load_dataset as load_longmemeval_dataset
    from benchmarks.longmemeval.baselines import (
        pie_temporal, naive_rag, full_context,
        PIE_EXTRACTION_PROMPT, _build_world_model_for_question,
        _retrieve_entities_for_question, _compile_temporal_context,
    )
    from benchmarks.longmemeval.adapter import parse_question_date
    from pie.core.llm import LLMClient

    print("=" * 70)
    print("PIE BENCHMARK QUICK TEST")
    print("=" * 70)

    # Load dataset
    print("\n[1/4] Loading LongMemEval dataset...")
    data = load_longmemeval_dataset()
    print(f"  Loaded {len(data)} questions")

    # Pick a diverse sample
    # Try to get one of each type
    by_type = {}
    for item in data:
        qt = item.get("question_type", "unknown")
        if qt not in by_type:
            by_type[qt] = item
    sample = list(by_type.values())[:num_questions]
    if len(sample) < num_questions:
        # Fill with more questions
        for item in data:
            if item not in sample:
                sample.append(item)
            if len(sample) >= num_questions:
                break

    print(f"  Selected {len(sample)} questions: {[s['question_type'] for s in sample]}")

    # Initialize LLM
    print("\n[2/4] Initializing LLM client...")
    llm = LLMClient()

    # Test API connectivity
    print("  Testing API connection...")
    try:
        result = llm.chat(
            messages=[{"role": "user", "content": "Say 'API works' in 2 words"}],
            model="gpt-4o-mini",
            max_tokens=10,
        )
        print(f"  API test: {result['content'].strip()}")
    except Exception as e:
        print(f"  ERROR: API connection failed: {e}")
        print("  Make sure OPENAI_API_KEY is set correctly")
        sys.exit(1)

    # Run each question through PIE pipeline
    print(f"\n[3/4] Running PIE temporal on {len(sample)} questions...")
    results = []

    for i, item in enumerate(sample):
        qid = item["question_id"]
        qtype = item["question_type"]
        question = item["question"]
        gold = item["answer"]

        print(f"\n{'─' * 60}")
        print(f"  Q{i+1}/{len(sample)} [{qtype}] {qid}")
        print(f"  Question: {question}")
        print(f"  Gold answer: {gold}")

        t0 = time.time()

        if debug:
            # Step-by-step with debug output
            print(f"\n  [Building world model...]")
            wm = _build_world_model_for_question(item, llm, "gpt-4o-mini")
            print(f"  World model: {len(wm.entities)} entities")

            if wm.entities:
                print(f"\n  Entities extracted:")
                for eid, entity in list(wm.entities.items())[:20]:
                    state_desc = entity.current_state.get("description", str(entity.current_state)[:100]) if isinstance(entity.current_state, dict) else str(entity.current_state)[:100]
                    print(f"    - {entity.name} ({entity.type.value}): {state_desc}")

            print(f"\n  [Retrieving relevant entities...]")
            retrieved = _retrieve_entities_for_question(question, wm, llm, top_k=15)
            print(f"  Retrieved {len(retrieved)} entities")

            if retrieved:
                print(f"  Top 5 by relevance:")
                for eid, entity, sim in retrieved[:5]:
                    print(f"    - {entity.name} (sim={sim:.3f})")

                question_ts = parse_question_date(item["question_date"])
                context = _compile_temporal_context(retrieved, wm, question_ts, max_chars=30_000)
                print(f"\n  Compiled context ({len(context)} chars):")
                print(f"  {'─' * 40}")
                # Show first 2000 chars of context
                print(f"  {context[:2000]}")
                if len(context) > 2000:
                    print(f"  ... ({len(context) - 2000} more chars)")

            # Now run the full baseline to get the answer
            result = pie_temporal(item, world_model=wm, llm=llm)
        else:
            result = pie_temporal(item, llm=llm)

        elapsed = time.time() - t0

        pred = result.hypothesis
        print(f"\n  Predicted: {pred}")
        print(f"  Time: {elapsed:.1f}s | Context: {result.context_chars} chars | Entities: {result.retrieval_count}")

        # Simple match check
        gold_lower = str(gold).lower().strip()
        pred_lower = str(pred).lower().strip()
        match = gold_lower in pred_lower or any(w in pred_lower for w in gold_lower.split() if len(w) > 3)
        print(f"  Match: {'✓' if match else '✗'}")

        results.append({
            "question_id": qid,
            "question_type": qtype,
            "question": question,
            "gold": gold,
            "predicted": pred,
            "match": match,
            "time_s": round(elapsed, 1),
        })

    # Summary
    print(f"\n{'=' * 70}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 70}")
    matches = sum(1 for r in results if r["match"])
    print(f"  Approximate accuracy: {matches}/{len(results)} ({100*matches/len(results):.0f}%)")
    print(f"  (Note: this is a rough string match, not the official LLM judge)")

    for r in results:
        icon = "✓" if r["match"] else "✗"
        print(f"  {icon} [{r['question_type']}] Q: {r['question'][:60]}...")
        print(f"     Gold: {r['gold'][:60]}")
        print(f"     Pred: {r['predicted'][:60]}")

    # Save results
    out_path = Path("benchmarks/quick_test_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {out_path}")

    return results


def run_full_benchmark(
    benchmark: str,
    baseline: str,
    cache_dir: str | None = None,
    num_questions: int | None = None,
    model: str = "gpt-4o",
    debug: bool = False,
):
    """Run a full benchmark evaluation."""
    from pie.core.llm import LLMClient

    print("=" * 70)
    print(f"PIE BENCHMARK: {benchmark} / {baseline}")
    print("=" * 70)

    llm = LLMClient()

    if benchmark == "longmemeval":
        from benchmarks.longmemeval.adapter import load_dataset as load_longmemeval_dataset
        from benchmarks.longmemeval.baselines import BASELINES, PIETemporalCachedBaseline

        data = load_longmemeval_dataset()
        print(f"Loaded {len(data)} questions")

        if num_questions:
            data = data[:num_questions]
            print(f"Using first {num_questions} questions")

        if baseline not in BASELINES:
            print(f"Unknown baseline: {baseline}. Available: {list(BASELINES.keys())}")
            sys.exit(1)

        # Special handling for cached baseline
        if baseline == "pie_temporal_cached":
            cache_path = Path(cache_dir) if cache_dir else Path("benchmarks/longmemeval/cache")
            cached_baseline = PIETemporalCachedBaseline(
                cache_dir=cache_path,
                llm=llm,
                model=model,
                extraction_model="gpt-4o-mini",
            )
            baseline_fn = lambda item: cached_baseline.run(item)
        else:
            baseline_fn = lambda item: BASELINES[baseline](item, llm=llm, model=model)

        results = []
        errors = 0
        t_total = time.time()

        for i, item in enumerate(data):
            t0 = time.time()
            result = baseline_fn(item)
            elapsed = time.time() - t0

            results.append(result.to_dict())

            if result.error:
                errors += 1

            if debug or (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(data)}] {item['question_type']:15s} | "
                      f"{elapsed:.1f}s | "
                      f"ctx={result.context_chars:,} | "
                      f"Q: {item['question'][:50]}...")
                if debug:
                    print(f"    Gold: {item['answer'][:80]}")
                    print(f"    Pred: {result.hypothesis[:80]}")

        total_time = time.time() - t_total

        # Save results
        out_dir = Path(f"benchmarks/results/{benchmark}")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{baseline}_{model.replace('/', '_')}_{int(time.time())}.json"

        with open(out_path, "w") as f:
            json.dump({
                "benchmark": benchmark,
                "baseline": baseline,
                "model": model,
                "num_questions": len(data),
                "total_time_s": round(total_time, 1),
                "errors": errors,
                "results": results,
            }, f, indent=2)

        print(f"\n{'=' * 70}")
        print(f"COMPLETE: {len(data)} questions in {total_time:.0f}s ({total_time/len(data):.1f}s avg)")
        print(f"Errors: {errors}")
        print(f"Results saved to: {out_path}")
        print(f"\nTo score results, run:")
        print(f"  python -m benchmarks.eval_harness score {out_path}")

    elif benchmark == "locomo":
        from benchmarks.locomo.adapter import load_dataset as load_locomo_dataset, flatten_qa
        from benchmarks.locomo.baselines import (
            BASELINES as LOCOMO_BASELINES,
            _build_world_model_for_conversation, pie_temporal,
        )
        from collections import defaultdict

        data = load_locomo_dataset()
        qa_items = flatten_qa(data)
        print(f"Loaded {len(data)} conversations, {len(qa_items)} QA items")

        if num_questions:
            qa_items = qa_items[:num_questions]

        if baseline not in LOCOMO_BASELINES:
            print(f"Unknown baseline: {baseline}. Available: {list(LOCOMO_BASELINES.keys())}")
            sys.exit(1)

        # Collect debug data for the viewer
        viewer_data = {"conversations": {}, "results": []}

        # Group questions by conversation
        by_convo = defaultdict(list)
        for item in qa_items:
            by_convo[item["sample_id"]].append(item)

        # Limit number of conversations if requested
        num_convos = int(os.environ.get("PIE_NUM_CONVOS", "0"))
        if num_convos > 0:
            limited = dict(list(by_convo.items())[:num_convos])
            by_convo = limited
            qa_items = [item for items in by_convo.values() for item in items]
            print(f"Limited to {num_convos} conversations ({len(qa_items)} questions)")

        results = []
        t_total = time.time()
        q_idx = 0  # global question counter

        # Set up viewer path (written after each conversation for live viewing)
        out_dir = Path(f"benchmarks/results/{benchmark}")
        out_dir.mkdir(parents=True, exist_ok=True)
        run_id = int(time.time())
        viewer_path = out_dir / f"viewer_{baseline}_{run_id}.html"
        print(f"\nLive viewer: {viewer_path}  (open in browser, refresh to see progress)")

        # INTERLEAVED: Build world model for each conversation, then immediately answer its Qs
        print(f"{len(by_convo)} conversations to process (build + answer interleaved)...")

        for convo_id, items in by_convo.items():
            # ── Build world model ──
            if baseline == "pie_temporal":
                print(f"\n{'─' * 60}")
                print(f"  {convo_id}: building world model ({len(items)} questions)...")
                t0 = time.time()
                extraction_log = []
                wm = _build_world_model_for_conversation(
                    items[0], llm, "gpt-4o-mini",
                    debug=debug,
                    debug_log=extraction_log if debug else None,
                )
                elapsed = time.time() - t0
                print(f"  → {len(wm.entities)} entities in {elapsed:.1f}s")

                # Compute dynamics for this conversation's world model
                dynamics_summary = ""
                try:
                    from pie.core.dynamics import TransitionDynamics
                    dyn = TransitionDynamics(wm)
                    dynamics_summary = dyn.summarize()
                except Exception as dyn_err:
                    dynamics_summary = f"Dynamics analysis failed: {dyn_err}"

                # Build transitions data for the viewer
                transitions_data = {}
                for eid in wm.entities:
                    trans = wm.get_transitions(eid, ordered=True)
                    if trans:
                        transitions_data[eid] = [
                            {
                                "type": t.transition_type.value,
                                "timestamp": t.timestamp,
                                "trigger": t.trigger_summary,
                                "to_state": (
                                    t.to_state.get("description", str(t.to_state)[:150])
                                    if isinstance(t.to_state, dict)
                                    else str(t.to_state)[:150]
                                ),
                            }
                            for t in trans
                        ]

                viewer_data["conversations"][convo_id] = {
                    "num_questions": len(items),
                    "num_entities": len(wm.entities),
                    "num_transitions": len(wm.transitions),
                    "build_time_s": round(elapsed, 1),
                    "extraction_log": extraction_log,
                    "dynamics_summary": dynamics_summary,
                    "entities": {
                        eid: {
                            "name": e.name,
                            "type": e.type.value,
                            "state": e.current_state if isinstance(e.current_state, dict) else str(e.current_state),
                        }
                        for eid, e in wm.entities.items()
                    },
                    "transitions": transitions_data,
                }
            else:
                wm = None

            # ── Answer questions for this conversation ──
            convo_correct = 0
            print(f"  Answering {len(items)} questions...")
            for item in items:
                q_idx += 1
                t0 = time.time()
                if baseline == "pie_temporal" and wm is not None:
                    result = pie_temporal(
                        item, world_model=wm, llm=llm, model=model,
                    )
                else:
                    result = LOCOMO_BASELINES[baseline](item, llm=llm, model=model)
                elapsed = time.time() - t0
                results.append(result.to_dict())

                gold_lower = _normalize_for_match(str(item['answer']))
                pred_lower = _normalize_for_match(str(result.hypothesis))
                match = gold_lower in pred_lower or any(
                    w in pred_lower for w in gold_lower.split() if len(w) > 3
                )
                if match:
                    convo_correct += 1

                if debug or q_idx % 25 == 0:
                    match_icon = "✓" if match else "✗"
                    print(f"    [{q_idx}/{len(qa_items)}] {match_icon} {item['question_type']:15s} | "
                          f"{elapsed:.1f}s | Q: {item['question'][:50]}...")
                    if debug:
                        print(f"      Gold: {str(item['answer'])[:80]}")
                        print(f"      Pred: {result.hypothesis[:80]}")

                viewer_data["results"].append({
                    "question_id": item["question_id"],
                    "question_type": item["question_type"],
                    "question": item["question"],
                    "gold": str(item["answer"]),
                    "predicted": result.hypothesis,
                    "match": match,
                    "sample_id": item.get("sample_id", ""),
                    "context_chars": result.context_chars,
                    "retrieval_count": result.retrieval_count,
                    "time_s": round(elapsed, 1),
                    "context_preview": getattr(result, '_context_preview', ''),
                })

            print(f"  → {convo_id}: {convo_correct}/{len(items)} correct ({100*convo_correct/len(items):.0f}%)")

            # Live-update viewer after each conversation (so you can open it mid-run)
            _generate_viewer_html(viewer_data, viewer_path, benchmark, baseline, model)

        total_time = time.time() - t_total

        out_path = out_dir / f"{baseline}_{model.replace('/', '_')}_{run_id}.json"

        with open(out_path, "w") as f:
            json.dump({
                "benchmark": benchmark,
                "baseline": baseline,
                "model": model,
                "num_questions": len(qa_items),
                "total_time_s": round(total_time, 1),
                "results": results,
            }, f, indent=2)

        print(f"\nResults saved to: {out_path}")

        # Final viewer
        _generate_viewer_html(viewer_data, viewer_path, benchmark, baseline, model)
        print(f"Pipeline viewer: {viewer_path}")


def _generate_viewer_html(
    viewer_data: dict,
    out_path: Path,
    benchmark: str,
    baseline: str,
    model: str,
):
    """Generate an interactive HTML viewer for inspecting the benchmark pipeline."""
    import html as html_lib

    convos = viewer_data.get("conversations", {})
    results = viewer_data.get("results", [])

    total_match = sum(1 for r in results if r.get("match"))
    total_q = len(results)
    pct = (100 * total_match / total_q) if total_q else 0

    # Group results by question type
    from collections import Counter, defaultdict
    type_counts = Counter(r["question_type"] for r in results)
    type_correct = Counter(r["question_type"] for r in results if r.get("match"))

    # Group results by conversation
    by_convo = defaultdict(list)
    for r in results:
        by_convo[r.get("sample_id", "unknown")].append(r)

    # Build HTML
    lines = []
    lines.append(f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><title>PIE Benchmark Viewer</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #0a0a0a; color: #e0e0e0; padding: 20px; }}
  h1 {{ color: #fff; margin-bottom: 8px; }}
  h2 {{ color: #a78bfa; margin: 24px 0 12px; border-bottom: 1px solid #333; padding-bottom: 4px; }}
  h3 {{ color: #7dd3fc; margin: 16px 0 8px; }}
  .summary {{ background: #1a1a2e; padding: 16px; border-radius: 8px; margin: 16px 0;
              display: flex; gap: 24px; flex-wrap: wrap; }}
  .stat {{ text-align: center; }}
  .stat .num {{ font-size: 32px; font-weight: bold; color: #a78bfa; }}
  .stat .label {{ font-size: 12px; color: #888; }}
  .type-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
                gap: 8px; margin: 12px 0; }}
  .type-card {{ background: #1e1e2e; padding: 10px; border-radius: 6px; text-align: center; }}
  .type-card .name {{ font-size: 12px; color: #888; }}
  .type-card .score {{ font-size: 20px; font-weight: bold; }}
  .convo-section {{ background: #111; border: 1px solid #222; border-radius: 8px;
                    margin: 12px 0; overflow: hidden; }}
  .convo-header {{ background: #1a1a2e; padding: 12px 16px; cursor: pointer; display: flex;
                   justify-content: space-between; align-items: center; }}
  .convo-header:hover {{ background: #222244; }}
  .convo-body {{ padding: 16px; display: none; }}
  .convo-body.open {{ display: block; }}
  .entity-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
                  gap: 8px; margin: 8px 0; }}
  .entity-card {{ background: #1a1a2e; padding: 8px 12px; border-radius: 6px; border-left: 3px solid #a78bfa; }}
  .entity-card.has-transitions {{ border-left-color: #facc15; }}
  .entity-name {{ font-weight: bold; color: #fff; }}
  .entity-type {{ font-size: 11px; color: #888; }}
  .entity-state {{ font-size: 12px; color: #aaa; margin-top: 4px; }}
  .entity-transitions {{ font-size: 11px; color: #7dd3fc; margin-top: 4px; }}
  .entity-transitions .trans {{ margin: 2px 0; }}
  .trans-creation {{ color: #4ade80; }}
  .trans-update {{ color: #facc15; }}
  .trans-contradiction {{ color: #f87171; }}
  .trans-archival {{ color: #888; }}
  .dynamics-box {{ background: #0d0d1a; padding: 12px; border-radius: 6px; margin: 8px 0;
                   font-family: monospace; font-size: 12px; white-space: pre-wrap;
                   max-height: 300px; overflow-y: auto; border: 1px solid #333; }}
  .extraction-log {{ font-family: monospace; font-size: 12px; background: #0d0d1a;
                     padding: 8px; border-radius: 4px; margin: 8px 0; max-height: 200px;
                     overflow-y: auto; }}
  .extraction-log .ok {{ color: #4ade80; }}
  .extraction-log .err {{ color: #f87171; }}
  .qa-table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  .qa-table th {{ text-align: left; padding: 8px; background: #1a1a2e; color: #a78bfa;
                  border-bottom: 1px solid #333; position: sticky; top: 0; }}
  .qa-table td {{ padding: 8px; border-bottom: 1px solid #1a1a1a; vertical-align: top; }}
  .qa-table tr:hover {{ background: #111122; }}
  .pass {{ color: #4ade80; }}
  .fail {{ color: #f87171; }}
  .q-text {{ max-width: 300px; }}
  .ans-text {{ max-width: 200px; word-break: break-word; }}
  details {{ margin: 8px 0; }}
  details summary {{ cursor: pointer; color: #7dd3fc; font-size: 13px; }}
  .filter-bar {{ margin: 12px 0; display: flex; gap: 8px; flex-wrap: wrap; }}
  .filter-btn {{ background: #1e1e2e; border: 1px solid #333; color: #e0e0e0; padding: 4px 12px;
                 border-radius: 4px; cursor: pointer; font-size: 12px; }}
  .filter-btn.active {{ background: #a78bfa; color: #000; border-color: #a78bfa; }}
</style></head><body>
<h1>PIE Benchmark Viewer</h1>
<p style="color:#888">{benchmark} / {baseline} / {model}</p>
""")

    # Summary stats
    total_ents = sum(c.get('num_entities', 0) for c in convos.values())
    total_trans = sum(c.get('num_transitions', 0) for c in convos.values())
    lines.append(f"""<div class="summary">
  <div class="stat"><div class="num">{total_match}/{total_q}</div><div class="label">Correct (rough match)</div></div>
  <div class="stat"><div class="num">{pct:.0f}%</div><div class="label">Accuracy</div></div>
  <div class="stat"><div class="num">{len(convos)}</div><div class="label">Conversations</div></div>
  <div class="stat"><div class="num">{total_ents}</div><div class="label">Total Entities</div></div>
  <div class="stat"><div class="num">{total_trans}</div><div class="label">Total Transitions</div></div>
</div>""")

    # Type breakdown
    lines.append('<h2>By Question Type</h2><div class="type-grid">')
    for qtype in sorted(type_counts.keys()):
        c = type_correct.get(qtype, 0)
        t = type_counts[qtype]
        p = 100 * c / t if t else 0
        color = "#4ade80" if p >= 60 else "#facc15" if p >= 30 else "#f87171"
        lines.append(f'<div class="type-card"><div class="name">{qtype}</div>'
                      f'<div class="score" style="color:{color}">{c}/{t} ({p:.0f}%)</div></div>')
    lines.append('</div>')

    # Per-conversation sections
    lines.append('<h2>World Models &amp; Extraction</h2>')
    for cid in sorted(convos.keys()):
        cdata = convos[cid]
        n_ent = cdata.get("num_entities", 0)
        n_q = cdata.get("num_questions", 0)
        build_t = cdata.get("build_time_s", 0)
        elog = cdata.get("extraction_log", [])
        entities = cdata.get("entities", {})

        n_trans_c = cdata.get("num_transitions", 0)
        lines.append(f"""<div class="convo-section">
  <div class="convo-header" onclick="this.nextElementSibling.classList.toggle('open')">
    <span><strong>{cid}</strong> — {n_ent} entities, {n_trans_c} transitions, {n_q} questions, {build_t}s</span>
    <span>▼</span>
  </div>
  <div class="convo-body">""")

        # Extraction log
        if elog:
            lines.append('<h3>Extraction Log</h3><div class="extraction-log">')
            for entry in elog:
                if "error" in entry:
                    lines.append(f'<div class="err">Session {entry["session_index"]+1}: '
                                  f'FAILED — {html_lib.escape(str(entry["error"]))}</div>')
                else:
                    lines.append(f'<div class="ok">Session {entry["session_index"]+1}: '
                                  f'{entry.get("input_chars",0):,} chars → '
                                  f'{entry.get("entities_found",0)} extracted, '
                                  f'{entry.get("entities_new",0)} new '
                                  f'(total: {entry.get("total_entities",0)})</div>')
            lines.append('</div>')

        # Dynamics summary
        dynamics_summary = cdata.get("dynamics_summary", "")
        if dynamics_summary:
            lines.append('<h3>Transition Dynamics</h3>')
            lines.append(f'<div class="dynamics-box">{html_lib.escape(dynamics_summary)}</div>')

        # Entities with transitions
        transitions = cdata.get("transitions", {})
        n_trans = cdata.get("num_transitions", 0)
        if entities:
            lines.append(f'<h3>Entities ({len(entities)}) — {n_trans} transitions</h3>')
            lines.append('<div class="entity-grid">')
            for eid, edata in sorted(entities.items(), key=lambda x: x[1].get("name", "")):
                name = html_lib.escape(str(edata.get("name", "")))
                etype = html_lib.escape(str(edata.get("type", "")))
                state = edata.get("state", {})
                if isinstance(state, dict):
                    state_str = "; ".join(f"{k}: {v}" for k, v in state.items() if v and k != "embedding")
                else:
                    state_str = str(state)
                state_str = html_lib.escape(state_str[:200])

                # Transitions for this entity
                etrans = transitions.get(eid, [])
                has_trans_cls = " has-transitions" if len(etrans) > 1 else ""
                lines.append(f'<div class="entity-card{has_trans_cls}">'
                              f'<div class="entity-name">{name}</div>'
                              f'<div class="entity-type">{etype}</div>'
                              f'<div class="entity-state">{state_str}</div>')

                if etrans and len(etrans) > 1:
                    lines.append('<div class="entity-transitions">')
                    for t in etrans:
                        ttype = t.get("type", "update")
                        trigger = html_lib.escape(str(t.get("trigger", ""))[:80])
                        ts = t.get("timestamp", 0)
                        from datetime import datetime, timezone as tz_
                        try:
                            dt = datetime.fromtimestamp(ts, tz=tz_.utc)
                            date_s = dt.strftime("%Y-%m-%d")
                        except Exception:
                            date_s = "?"
                        lines.append(f'<div class="trans trans-{ttype}">'
                                      f'{date_s} [{ttype}] {trigger}</div>')
                    lines.append('</div>')

                lines.append('</div>')
            lines.append('</div>')

        # Questions for this conversation
        cq = by_convo.get(cid, [])
        if cq:
            lines.append(f'<h3>Questions ({len(cq)})</h3>')
            lines.append('<table class="qa-table"><tr><th>✓</th><th>Type</th>'
                          '<th>Question</th><th>Gold</th><th>Predicted</th><th>Time</th></tr>')
            for r in cq:
                icon = '<span class="pass">✓</span>' if r.get("match") else '<span class="fail">✗</span>'
                q = html_lib.escape(str(r["question"])[:120])
                g = html_lib.escape(str(r["gold"])[:100])
                p = html_lib.escape(str(r["predicted"])[:100])
                lines.append(f'<tr><td>{icon}</td><td>{r["question_type"]}</td>'
                              f'<td class="q-text">{q}</td>'
                              f'<td class="ans-text">{g}</td>'
                              f'<td class="ans-text">{p}</td>'
                              f'<td>{r.get("time_s","")}s</td></tr>')
            lines.append('</table>')

        lines.append('</div></div>')  # close convo-body and convo-section

    lines.append('</body></html>')

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(
        description="Run PIE benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXPERIMENTS (copy-paste these):

  # Exp 0: Baseline — full LoCoMo run (1 conversation first to validate)
  python run_benchmark.py --benchmark locomo --baseline pie_temporal --num-convos 1 --debug

  # Exp 0b: Full LoCoMo (all 10 conversations, ~2000 Qs)
  python run_benchmark.py --benchmark locomo --baseline pie_temporal --debug

  # Exp 1: Embedding-only retrieval (disable BM25) — compare with Exp 0
  python run_benchmark.py --benchmark locomo --baseline pie_temporal --ablation no-bm25 --debug

  # Exp 2: No timelines (flat facts only, no transition history)
  python run_benchmark.py --benchmark locomo --baseline pie_temporal --ablation no-timeline --debug

  # Exp 3: Vary top_k retrieval (default=20)
  python run_benchmark.py --benchmark locomo --baseline pie_temporal --top-k 5 --debug
  python run_benchmark.py --benchmark locomo --baseline pie_temporal --top-k 50 --debug

  # Exp 4: Different answer models
  python run_benchmark.py --benchmark locomo --baseline pie_temporal --model gpt-4o-mini --debug

  # LongMemEval quick test (5 diverse questions)
  python run_benchmark.py --quick-test --debug
""",
    )
    parser.add_argument("--quick-test", action="store_true", help="Quick test on 5 questions")
    parser.add_argument("--benchmark", choices=["longmemeval", "locomo"], default="longmemeval")
    parser.add_argument("--baseline", default="pie_temporal",
                        help="Baseline to run (full_context, naive_rag, pie_temporal, pie_temporal_cached)")
    parser.add_argument("--cache-dir", default=None, help="Cache directory for cached baseline")
    parser.add_argument("--num-questions", type=int, default=None, help="Limit number of questions")
    parser.add_argument("--num-convos", type=int, default=None,
                        help="Limit number of conversations (LoCoMo only, for quick testing)")
    parser.add_argument("--model", default="gpt-4o", help="LLM model for answering")
    parser.add_argument("--debug", action="store_true", help="Show detailed extraction output")
    parser.add_argument("--ablation", default=None,
                        choices=["no-bm25", "no-timeline", "no-dates"],
                        help="Ablation: no-bm25 (embedding-only), no-timeline (flat facts), no-dates (strip temporal)")
    parser.add_argument("--top-k", type=int, default=None, help="Override top_k for retrieval (default: 20)")
    args = parser.parse_args()

    # Check API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: Set OPENAI_API_KEY environment variable")
        print('  export OPENAI_API_KEY="sk-proj-..."')
        sys.exit(1)

    # Set ablation flags as environment variables so baselines can read them
    if args.ablation:
        os.environ["PIE_ABLATION"] = args.ablation
        print(f"ABLATION: {args.ablation}")
    if args.top_k:
        os.environ["PIE_TOP_K"] = str(args.top_k)
        print(f"TOP_K: {args.top_k}")
    if args.num_convos:
        os.environ["PIE_NUM_CONVOS"] = str(args.num_convos)

    if args.quick_test:
        run_quick_test(debug=args.debug, num_questions=args.num_questions or 5)
    else:
        run_full_benchmark(
            benchmark=args.benchmark,
            baseline=args.baseline,
            cache_dir=args.cache_dir,
            num_questions=args.num_questions,
            model=args.model,
            debug=args.debug,
        )


if __name__ == "__main__":
    main()
