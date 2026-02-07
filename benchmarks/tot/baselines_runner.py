#!/usr/bin/env python3
"""
ToT Benchmark: Naive vs Hybrid Baselines
Compares direct LLM prompting against temporal-structured approaches.
"""

import os
import re
import json
import time
import ast
from collections import defaultdict
from datetime import datetime
from openai import OpenAI

# --------------- Config ---------------
MODEL = "gpt-4o-mini"
MAX_CHAR_ESTIMATE = 16000
SAMPLES_PER_TYPE = 10  # 10 per question type = ~50 total semantic questions
MAX_RESPONSE_TOKENS = 1024

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY environment variable not set.")

client = OpenAI(api_key=api_key)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# --------------- Common Helpers ---------------

def call_llm(prompt, system=None):
    """Send a prompt to the LLM and return the text response."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    
    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_completion_tokens=MAX_RESPONSE_TOKENS,
    )
    return resp.choices[0].message.content.strip()


def extract_json_from_response(text):
    """Try to extract a JSON object from an LLM response."""
    m = re.search(r'JSON\s*=\s*(\{.*?\})', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except:
            try:
                return ast.literal_eval(m.group(1))
            except:
                pass
    
    matches = re.findall(r'\{[^{}]*\}', text, re.DOTALL)
    for match in matches:
        try:
            return json.loads(match)
        except:
            try:
                return ast.literal_eval(match)
            except:
                pass
    try:
        return json.loads(text)
    except:
        pass
    return None


def normalize(val):
    """Normalize a value for comparison."""
    if isinstance(val, str):
        return val.strip().lower()
    if isinstance(val, (int, float)):
        return str(int(val) if val == int(val) else val).strip().lower()
    if isinstance(val, list):
        return sorted([normalize(v) for v in val])
    return str(val).strip().lower()


def parse_label_str(label_str):
    """Parse the arithmetic label string to dict."""
    try:
        return json.loads(label_str)
    except:
        try:
            return ast.literal_eval(label_str)
        except:
            return {"raw": label_str}


# --------------- Semantic Evaluation ---------------

def evaluate_semantic_answer(predicted_text, label, question_type):
    """Check if the LLM's response matches the semantic label."""
    pred = predicted_text.strip()
    gt = label.strip()
    
    if pred.lower() == gt.lower():
        return True
    
    if question_type == "timeline":
        gt_list = [x.strip().lower() for x in gt.split(",")]
        m = re.findall(r'E\d+', pred, re.IGNORECASE)
        if m:
            pred_list = [x.lower() for x in m]
            if pred_list == gt_list:
                return True
            if len(pred_list) >= len(gt_list):
                for i in range(len(pred_list) - len(gt_list) + 1):
                    if pred_list[i:i + len(gt_list)] == gt_list:
                        return True
        return False
    
    if gt.isdigit():
        nums = re.findall(r'\b' + re.escape(gt) + r'\b', pred)
        if nums:
            return True
    
    if re.match(r'^E\d+$', gt, re.IGNORECASE):
        if re.search(r'\b' + re.escape(gt) + r'\b', pred, re.IGNORECASE):
            return True
    
    if re.match(r'^\d{4}$', gt):
        if gt in pred:
            return True
    
    return False


def evaluate_arithmetic_answer(response_text, label_str, question_type):
    """Check if the LLM's response matches the arithmetic label."""
    label = parse_label_str(label_str)
    extracted = extract_json_from_response(response_text)
    
    if extracted is None:
        if 'answer' in label:
            ans = str(label['answer']).strip().lower()
            if ans in response_text.lower():
                return True
        return False
    
    if 'answer' in label:
        gt_ans = normalize(label['answer'])
        pred_ans = normalize(extracted.get('answer', ''))
        if gt_ans == pred_ans:
            return True
        try:
            if float(gt_ans) == float(pred_ans):
                return True
        except:
            pass
    
    if 'unordered_list' in label:
        gt_list = sorted([normalize(x) for x in label['unordered_list']])
        pred_list = extracted.get('unordered_list', extracted.get('answer', []))
        if isinstance(pred_list, str):
            pred_list = [x.strip() for x in pred_list.split(',')]
        if isinstance(pred_list, list):
            pred_sorted = sorted([normalize(x) for x in pred_list])
            if gt_list == pred_sorted:
                return True
    
    multi_keys = [k for k in label.keys() if k not in ('answer', 'unordered_list', 'explanation', 'raw')]
    if multi_keys:
        all_match = True
        for k in multi_keys:
            gt_val = normalize(label[k])
            pred_val = normalize(extracted.get(k, ''))
            if gt_val != pred_val:
                try:
                    if float(gt_val) != float(pred_val):
                        all_match = False
                except:
                    all_match = False
        if all_match and multi_keys:
            return True
    
    return False


# --------------- Temporal Fact Parsing ---------------

def parse_temporal_facts(prompt_text):
    """Parse temporal facts from prompt into structured format."""
    facts = []
    pattern = r'(E\d+) was the (R\d+) of (E\d+) from (\d{4}) to (\d{4})'
    
    for match in re.finditer(pattern, prompt_text):
        entity, relation, target, start, end = match.groups()
        facts.append({
            'entity': entity,
            'relation': relation,
            'target': target,
            'start': int(start),
            'end': int(end),
            'duration': int(end) - int(start)
        })
    
    return facts


def build_temporal_index(facts):
    """Build indexes for temporal querying."""
    by_entity = defaultdict(list)
    by_target = defaultdict(list)
    by_relation = defaultdict(list)
    by_year = defaultdict(list)
    
    for f in facts:
        by_entity[f['entity']].append(f)
        by_target[f['target']].append(f)
        by_relation[f['relation']].append(f)
        for year in range(f['start'], f['end'] + 1):
            by_year[year].append(f)
    
    return {
        'by_entity': dict(by_entity),
        'by_target': dict(by_target),
        'by_relation': dict(by_relation),
        'by_year': dict(by_year),
        'all_facts': facts
    }


def get_temporal_summary(index):
    """Generate a temporal summary of the data."""
    facts = index['all_facts']
    if not facts:
        return "No temporal facts found."
    
    min_year = min(f['start'] for f in facts)
    max_year = max(f['end'] for f in facts)
    entities = set(f['entity'] for f in facts)
    targets = set(f['target'] for f in facts)
    relations = set(f['relation'] for f in facts)
    
    summary = f"""TEMPORAL SUMMARY:
- Time span: {min_year} to {max_year}
- Entities involved: {len(entities)} ({', '.join(sorted(entities)[:5])}...)
- Target entities: {len(targets)} ({', '.join(sorted(targets)[:5])}...)
- Relation types: {len(relations)} ({', '.join(sorted(relations))})
- Total facts: {len(facts)}

KEY TEMPORAL PATTERNS:
"""
    
    # Find first/last events for each target
    for target in sorted(targets)[:3]:
        target_facts = index['by_target'].get(target, [])
        if target_facts:
            first = min(target_facts, key=lambda x: x['start'])
            last = max(target_facts, key=lambda x: x['end'])
            summary += f"- {target}: First event {first['start']} ({first['entity']}/{first['relation']}), Last event {last['end']} ({last['entity']}/{last['relation']})\n"
    
    return summary


# --------------- BASELINE 1: Naive RAG ---------------

def naive_baseline_semantic(prompt, question):
    """Direct prompting - send raw facts + question to LLM."""
    full_prompt = prompt + "\n\n" + question
    return call_llm(full_prompt)


def naive_baseline_arithmetic(question):
    """Direct prompting for arithmetic questions."""
    return call_llm(question)


# --------------- BASELINE 2: Hybrid Temporal ---------------

TEMPORAL_SYSTEM = """You are a temporal reasoning expert. You analyze temporal facts systematically.

For questions about temporal data:
1. First identify the relevant entities, relations, and time periods
2. Apply temporal logic (before, after, during, overlapping)
3. Use the structured summary to verify your reasoning
4. Give a precise answer

Be methodical. Check your work."""


def hybrid_baseline_semantic(prompt, question, question_type):
    """Structured temporal reasoning with fact parsing and guided prompting."""
    
    # Parse and index facts
    facts = parse_temporal_facts(prompt)
    index = build_temporal_index(facts)
    
    # Generate temporal summary
    summary = get_temporal_summary(index)
    
    # Determine query type and add type-specific guidance
    guidance = ""
    if question_type == "first_event":
        guidance = "\nFOCUS: Find the EARLIEST start year for the specified condition."
    elif question_type == "last_event":
        guidance = "\nFOCUS: Find the LATEST end year for the specified condition."
    elif question_type == "event_at_time_t":
        guidance = "\nFOCUS: Find what was active/true at the specific year mentioned."
    elif question_type == "timeline":
        guidance = "\nFOCUS: Order events chronologically by their start times."
    elif question_type == "time_compare":
        guidance = "\nFOCUS: Compare two events/entities and determine temporal relationship."
    
    # Build structured prompt
    structured_prompt = f"""{summary}

TEMPORAL FACTS:
{prompt}

{guidance}

QUESTION: {question}

Think step by step:
1. What entities/relations does the question ask about?
2. What temporal constraints apply?
3. What facts are relevant?
4. What is the answer?

Answer concisely with just the answer (e.g., E42, 1985, etc.)."""

    return call_llm(structured_prompt, system=TEMPORAL_SYSTEM)


def hybrid_baseline_arithmetic(question, question_type):
    """Structured approach for arithmetic temporal questions."""
    
    guidance = ""
    if "duration" in question_type.lower():
        guidance = "Calculate time periods carefully. Duration = End - Start."
    elif "count" in question_type.lower():
        guidance = "Count systematically. List items before counting."
    elif "clock" in question_type.lower():
        guidance = "Handle clock arithmetic with 12/24 hour awareness."
    
    structured_prompt = f"""TEMPORAL ARITHMETIC PROBLEM:
{question}

{guidance}

Solve step by step, then provide your answer in JSON format:
{{"answer": <your_answer>}}

If multiple values, use: {{"H": <hours>, "M": <minutes>}} or {{"unordered_list": [...]}}"""

    return call_llm(structured_prompt, system=TEMPORAL_SYSTEM)


# --------------- Main Runner ---------------

def run_comparison():
    print("=" * 70)
    print("ToT BENCHMARK: NAIVE vs HYBRID BASELINES")
    print("=" * 70)
    print(f"Model: {MODEL}")
    print(f"Samples per question type: {SAMPLES_PER_TYPE}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Load data
    with open(os.path.join(SCRIPT_DIR, "tot_semantic.json")) as f:
        ds_sem = json.load(f)
    with open(os.path.join(SCRIPT_DIR, "tot_arithmetic.json")) as f:
        ds_arith = json.load(f)
    
    print(f"\nLoaded: {len(ds_sem)} semantic, {len(ds_arith)} arithmetic questions")
    
    results = {
        'naive': defaultdict(lambda: {"correct": 0, "total": 0}),
        'hybrid': defaultdict(lambda: {"correct": 0, "total": 0})
    }
    
    # --------------- Semantic Questions ---------------
    print("\n" + "=" * 70)
    print("SEMANTIC QUESTIONS")
    print("=" * 70)
    
    sem_types = sorted(set(r['question_type'] for r in ds_sem))
    print(f"Question types: {sem_types}")
    
    for qt in sem_types:
        print(f"\n--- {qt.upper()} ---")
        candidates = [r for r in ds_sem if r['question_type'] == qt]
        candidates = [r for r in candidates if len(r['prompt'] + '\n' + r['question']) < MAX_CHAR_ESTIMATE]
        
        if not candidates:
            print("  SKIPPED (prompts too long)")
            continue
        
        candidates.sort(key=lambda r: len(r['prompt']))
        selected = candidates[:SAMPLES_PER_TYPE]
        
        for i, row in enumerate(selected):
            prompt = row['prompt']
            question = row['question']
            label = row['label']
            
            try:
                # Run naive baseline
                naive_resp = naive_baseline_semantic(prompt, question)
                naive_correct = evaluate_semantic_answer(naive_resp, label, qt)
                results['naive'][f"sem_{qt}"]["total"] += 1
                if naive_correct:
                    results['naive'][f"sem_{qt}"]["correct"] += 1
                
                time.sleep(0.2)
                
                # Run hybrid baseline
                hybrid_resp = hybrid_baseline_semantic(prompt, question, qt)
                hybrid_correct = evaluate_semantic_answer(hybrid_resp, label, qt)
                results['hybrid'][f"sem_{qt}"]["total"] += 1
                if hybrid_correct:
                    results['hybrid'][f"sem_{qt}"]["correct"] += 1
                
                n_status = "✓" if naive_correct else "✗"
                h_status = "✓" if hybrid_correct else "✗"
                print(f"  {i+1:2d}. naive={n_status} hybrid={h_status}  expected={label[:30]}")
                
                time.sleep(0.2)
                
            except Exception as e:
                print(f"  {i+1:2d}. ERROR: {e}")
                results['naive'][f"sem_{qt}"]["total"] += 1
                results['hybrid'][f"sem_{qt}"]["total"] += 1
    
    # --------------- Arithmetic Questions ---------------
    print("\n" + "=" * 70)
    print("ARITHMETIC QUESTIONS")
    print("=" * 70)
    
    arith_types = sorted(set(r['question_type'] for r in ds_arith))
    print(f"Question types: {arith_types}")
    
    for qt in arith_types:
        print(f"\n--- {qt.upper()} ---")
        candidates = [r for r in ds_arith if r['question_type'] == qt]
        selected = candidates[:SAMPLES_PER_TYPE]
        
        for i, row in enumerate(selected):
            question = row['question']
            label_str = row['label']
            
            try:
                # Run naive baseline
                naive_resp = naive_baseline_arithmetic(question)
                naive_correct = evaluate_arithmetic_answer(naive_resp, label_str, qt)
                results['naive'][f"arith_{qt}"]["total"] += 1
                if naive_correct:
                    results['naive'][f"arith_{qt}"]["correct"] += 1
                
                time.sleep(0.2)
                
                # Run hybrid baseline
                hybrid_resp = hybrid_baseline_arithmetic(question, qt)
                hybrid_correct = evaluate_arithmetic_answer(hybrid_resp, label_str, qt)
                results['hybrid'][f"arith_{qt}"]["total"] += 1
                if hybrid_correct:
                    results['hybrid'][f"arith_{qt}"]["correct"] += 1
                
                n_status = "✓" if naive_correct else "✗"
                h_status = "✓" if hybrid_correct else "✗"
                label_short = label_str[:40]
                print(f"  {i+1:2d}. naive={n_status} hybrid={h_status}  expected={label_short}")
                
                time.sleep(0.2)
                
            except Exception as e:
                print(f"  {i+1:2d}. ERROR: {e}")
                results['naive'][f"arith_{qt}"]["total"] += 1
                results['hybrid'][f"arith_{qt}"]["total"] += 1
    
    # --------------- Results Summary ---------------
    print("\n" + "=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)
    
    print(f"\n{'Category':<40} {'Naive':>12} {'Hybrid':>12} {'Δ':>8}")
    print("-" * 72)
    
    naive_sem_correct, naive_sem_total = 0, 0
    hybrid_sem_correct, hybrid_sem_total = 0, 0
    naive_arith_correct, naive_arith_total = 0, 0
    hybrid_arith_correct, hybrid_arith_total = 0, 0
    
    all_keys = sorted(set(list(results['naive'].keys()) + list(results['hybrid'].keys())))
    
    for key in all_keys:
        nr = results['naive'][key]
        hr = results['hybrid'][key]
        
        naive_acc = nr['correct'] / nr['total'] * 100 if nr['total'] > 0 else 0
        hybrid_acc = hr['correct'] / hr['total'] * 100 if hr['total'] > 0 else 0
        delta = hybrid_acc - naive_acc
        delta_str = f"+{delta:.1f}%" if delta > 0 else f"{delta:.1f}%"
        
        print(f"  {key:<38} {naive_acc:>10.1f}% {hybrid_acc:>10.1f}% {delta_str:>8}")
        
        if key.startswith("sem_"):
            naive_sem_correct += nr['correct']
            naive_sem_total += nr['total']
            hybrid_sem_correct += hr['correct']
            hybrid_sem_total += hr['total']
        else:
            naive_arith_correct += nr['correct']
            naive_arith_total += nr['total']
            hybrid_arith_correct += hr['correct']
            hybrid_arith_total += hr['total']
    
    print("-" * 72)
    
    # Semantic totals
    naive_sem_acc = naive_sem_correct / naive_sem_total * 100 if naive_sem_total > 0 else 0
    hybrid_sem_acc = hybrid_sem_correct / hybrid_sem_total * 100 if hybrid_sem_total > 0 else 0
    sem_delta = hybrid_sem_acc - naive_sem_acc
    sem_delta_str = f"+{sem_delta:.1f}%" if sem_delta > 0 else f"{sem_delta:.1f}%"
    print(f"  {'SEMANTIC TOTAL':<38} {naive_sem_acc:>10.1f}% {hybrid_sem_acc:>10.1f}% {sem_delta_str:>8}")
    
    # Arithmetic totals
    naive_arith_acc = naive_arith_correct / naive_arith_total * 100 if naive_arith_total > 0 else 0
    hybrid_arith_acc = hybrid_arith_correct / hybrid_arith_total * 100 if hybrid_arith_total > 0 else 0
    arith_delta = hybrid_arith_acc - naive_arith_acc
    arith_delta_str = f"+{arith_delta:.1f}%" if arith_delta > 0 else f"{arith_delta:.1f}%"
    print(f"  {'ARITHMETIC TOTAL':<38} {naive_arith_acc:>10.1f}% {hybrid_arith_acc:>10.1f}% {arith_delta_str:>8}")
    
    # Overall totals
    naive_total_correct = naive_sem_correct + naive_arith_correct
    naive_total_total = naive_sem_total + naive_arith_total
    hybrid_total_correct = hybrid_sem_correct + hybrid_arith_correct
    hybrid_total_total = hybrid_sem_total + hybrid_arith_total
    
    naive_total_acc = naive_total_correct / naive_total_total * 100 if naive_total_total > 0 else 0
    hybrid_total_acc = hybrid_total_correct / hybrid_total_total * 100 if hybrid_total_total > 0 else 0
    total_delta = hybrid_total_acc - naive_total_acc
    total_delta_str = f"+{total_delta:.1f}%" if total_delta > 0 else f"{total_delta:.1f}%"
    
    print("-" * 72)
    print(f"  {'OVERALL TOTAL':<38} {naive_total_acc:>10.1f}% {hybrid_total_acc:>10.1f}% {total_delta_str:>8}")
    
    print(f"\nNaive:  {naive_total_correct}/{naive_total_total} correct")
    print(f"Hybrid: {hybrid_total_correct}/{hybrid_total_total} correct")
    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Save results
    results_data = {
        "model": MODEL,
        "samples_per_type": SAMPLES_PER_TYPE,
        "timestamp": datetime.now().isoformat(),
        "results": {
            "naive": {k: dict(v) for k, v in results['naive'].items()},
            "hybrid": {k: dict(v) for k, v in results['hybrid'].items()},
        },
        "totals": {
            "naive": {
                "semantic": {"correct": naive_sem_correct, "total": naive_sem_total, "accuracy": round(naive_sem_acc, 2)},
                "arithmetic": {"correct": naive_arith_correct, "total": naive_arith_total, "accuracy": round(naive_arith_acc, 2)},
                "overall": {"correct": naive_total_correct, "total": naive_total_total, "accuracy": round(naive_total_acc, 2)},
            },
            "hybrid": {
                "semantic": {"correct": hybrid_sem_correct, "total": hybrid_sem_total, "accuracy": round(hybrid_sem_acc, 2)},
                "arithmetic": {"correct": hybrid_arith_correct, "total": hybrid_arith_total, "accuracy": round(hybrid_arith_acc, 2)},
                "overall": {"correct": hybrid_total_correct, "total": hybrid_total_total, "accuracy": round(hybrid_total_acc, 2)},
            }
        }
    }
    
    with open(os.path.join(SCRIPT_DIR, "baseline_comparison_results.json"), "w") as f:
        json.dump(results_data, f, indent=2)
    
    print("\nResults saved to baseline_comparison_results.json")


if __name__ == "__main__":
    run_comparison()
