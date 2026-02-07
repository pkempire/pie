#!/usr/bin/env python3
"""
Test of Time (ToT) Benchmark Runner
Evaluates temporal reasoning on ToT-Semantic and ToT-Arithmetic tasks.
Uses OpenAI API (gpt-4o-mini).
"""

import os
import re
import sys
import json
import time
import ast
from collections import defaultdict
from openai import OpenAI

# --------------- Config ---------------
MODEL = "gpt-4o-mini"
MAX_CHAR_ESTIMATE = 16000  # ~4K tokens
SAMPLES_PER_TYPE = 2
MAX_RESPONSE_TOKENS = 1024

# Get API key from environment
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError(
        "OPENAI_API_KEY environment variable not set. "
        "Please set it before running the benchmark."
    )

client = OpenAI(api_key=api_key)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# --------------- Helpers ---------------

def call_llm(prompt):
    """Send a prompt to the LLM and return the text response."""
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=MAX_RESPONSE_TOKENS,
    )
    return resp.choices[0].message.content.strip()


def extract_json_from_response(text):
    """Try to extract a JSON object from an LLM response."""
    # Try finding JSON = {...} pattern
    m = re.search(r'JSON\s*=\s*(\{.*?\})', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(m.group(1))
            except Exception:
                pass

    # Try finding any {...} block (greedy to capture nested)
    matches = re.findall(r'\{[^{}]*\}', text, re.DOTALL)
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(match)
            except Exception:
                pass

    # Try the whole thing
    try:
        return json.loads(text)
    except Exception:
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


# --------------- Semantic Evaluation ---------------

def evaluate_semantic_answer(predicted_text, label, question_type):
    """Check if the LLM's response matches the semantic label."""
    pred = predicted_text.strip()
    gt = label.strip()

    # Direct match (case-insensitive)
    if pred.lower() == gt.lower():
        return True

    # For timeline: comma-separated lists - order matters
    if question_type == "timeline":
        gt_list = [x.strip().lower() for x in gt.split(",")]
        m = re.findall(r'E\d+', pred, re.IGNORECASE)
        if m:
            pred_list = [x.lower() for x in m]
            if pred_list == gt_list:
                return True
            # Check contiguous subsequence
            if len(pred_list) >= len(gt_list):
                for i in range(len(pred_list) - len(gt_list) + 1):
                    if pred_list[i:i + len(gt_list)] == gt_list:
                        return True
        return False

    # For numeric answers
    if gt.isdigit():
        nums = re.findall(r'\b' + re.escape(gt) + r'\b', pred)
        if nums:
            return True

    # For entity answers (E76, etc.)
    if re.match(r'^E\d+$', gt, re.IGNORECASE):
        if re.search(r'\b' + re.escape(gt) + r'\b', pred, re.IGNORECASE):
            return True

    # For year answers
    if re.match(r'^\d{4}$', gt):
        if gt in pred:
            return True

    return False


# --------------- Arithmetic Evaluation ---------------

def parse_label_str(label_str):
    """Parse the arithmetic label string to dict."""
    try:
        return json.loads(label_str)
    except Exception:
        try:
            return ast.literal_eval(label_str)
        except Exception:
            return {"raw": label_str}


def evaluate_arithmetic_answer(response_text, label_str, question_type):
    """Check if the LLM's response matches the arithmetic label."""
    label = parse_label_str(label_str)
    extracted = extract_json_from_response(response_text)

    if extracted is None:
        # Fallback: try to find the answer directly in text
        if 'answer' in label:
            ans = str(label['answer']).strip().lower()
            if ans in response_text.lower():
                return True
        return False

    # Handle different label formats
    if 'answer' in label:
        gt_ans = normalize(label['answer'])
        pred_ans = normalize(extracted.get('answer', ''))
        if gt_ans == pred_ans:
            return True
        # Try numeric comparison
        try:
            if float(gt_ans) == float(pred_ans):
                return True
        except Exception:
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

    # For multi-key answers (H, M, S or hours, minutes)
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
                except Exception:
                    all_match = False
        if all_match and multi_keys:
            return True

    return False


# --------------- Main Runner ---------------

def run_benchmark():
    print("Loading ToT datasets from local JSON...")
    with open(os.path.join(SCRIPT_DIR, "tot_arithmetic.json")) as f:
        ds_arith = json.load(f)
    with open(os.path.join(SCRIPT_DIR, "tot_semantic.json")) as f:
        ds_sem = json.load(f)

    print("Arithmetic: {} questions".format(len(ds_arith)))
    print("Semantic: {} questions".format(len(ds_sem)))
    print("Model: {}".format(MODEL))
    print("Samples per type: {}".format(SAMPLES_PER_TYPE))
    print("=" * 60)

    results = defaultdict(lambda: {"correct": 0, "total": 0, "errors": []})

    # --- Semantic ---
    print("\n--- ToT-Semantic ---")
    sem_types = sorted(set(r['question_type'] for r in ds_sem))
    for qt in sem_types:
        print("\n  [{}]".format(qt))
        candidates = [r for r in ds_sem if r['question_type'] == qt]
        # Filter to prompts under token budget
        candidates = [r for r in candidates
                      if len(r['prompt'] + '\n' + r['question']) < MAX_CHAR_ESTIMATE]
        if not candidates:
            print("    SKIPPED (all prompts too long)")
            continue

        # Sort by prompt length (shorter first) and take SAMPLES_PER_TYPE
        candidates.sort(key=lambda r: len(r['prompt']))
        selected = candidates[:SAMPLES_PER_TYPE]

        for i, row in enumerate(selected):
            full_prompt = row['prompt'] + "\n" + row['question']
            label = row['label']
            try:
                response = call_llm(full_prompt)
                correct = evaluate_semantic_answer(response, label, qt)
                results["sem_{}".format(qt)]["total"] += 1
                if correct:
                    results["sem_{}".format(qt)]["correct"] += 1
                status = "✓" if correct else "✗"
                resp_short = response.replace('\n', ' ')[:100]
                print("    {:2d}. {}  expected={}  got={}".format(i + 1, status, label, resp_short))
            except Exception as e:
                results["sem_{}".format(qt)]["total"] += 1
                results["sem_{}".format(qt)]["errors"].append(str(e))
                print("    {:2d}. ERROR: {}".format(i + 1, e))
            time.sleep(0.3)

    # --- Arithmetic ---
    print("\n--- ToT-Arithmetic ---")
    arith_types = sorted(set(r['question_type'] for r in ds_arith))
    for qt in arith_types:
        print("\n  [{}]".format(qt))
        candidates = [r for r in ds_arith if r['question_type'] == qt]
        selected = candidates[:SAMPLES_PER_TYPE]

        for i, row in enumerate(selected):
            question = row['question']
            label_str = row['label']
            try:
                response = call_llm(question)
                correct = evaluate_arithmetic_answer(response, label_str, qt)
                results["arith_{}".format(qt)]["total"] += 1
                if correct:
                    results["arith_{}".format(qt)]["correct"] += 1
                status = "✓" if correct else "✗"
                label_short = label_str[:80]
                resp_short = response.replace('\n', ' ')[:100]
                print("    {:2d}. {}  expected={}  got={}".format(i + 1, status, label_short, resp_short))
            except Exception as e:
                results["arith_{}".format(qt)]["total"] += 1
                results["arith_{}".format(qt)]["errors"].append(str(e))
                print("    {:2d}. ERROR: {}".format(i + 1, e))
            time.sleep(0.3)

    # --- Summary ---
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    sem_correct = 0
    sem_total = 0
    arith_correct = 0
    arith_total = 0

    print("\n{:<45} {:>7} {:>7} {:>10}".format('Category', 'Correct', 'Total', 'Accuracy'))
    print("-" * 72)

    for key in sorted(results.keys()):
        r = results[key]
        acc = r['correct'] / r['total'] * 100 if r['total'] > 0 else 0
        print("  {:<43} {:>7} {:>7} {:>9.1f}%".format(key, r['correct'], r['total'], acc))
        if key.startswith("sem_"):
            sem_correct += r['correct']
            sem_total += r['total']
        else:
            arith_correct += r['correct']
            arith_total += r['total']

    print("-" * 72)
    sem_acc = sem_correct / sem_total * 100 if sem_total > 0 else 0
    arith_acc = arith_correct / arith_total * 100 if arith_total > 0 else 0
    total_correct = sem_correct + arith_correct
    total_total = sem_total + arith_total
    total_acc = total_correct / total_total * 100 if total_total > 0 else 0

    print("  {:<43} {:>7} {:>7} {:>9.1f}%".format('ToT-Semantic TOTAL', sem_correct, sem_total, sem_acc))
    print("  {:<43} {:>7} {:>7} {:>9.1f}%".format('ToT-Arithmetic TOTAL', arith_correct, arith_total, arith_acc))
    print("  {:<43} {:>7} {:>7} {:>9.1f}%".format('OVERALL TOTAL', total_correct, total_total, total_acc))

    print("\n\nPaper Reference (from ToT paper - full dataset):")
    print("  GPT-4o:   ~48-55% Semantic, ~60-70% Arithmetic")
    print("  Gemini:   ~40-50% Semantic, ~55-65% Arithmetic")
    print("  GPT-3.5:  ~30-40% Semantic, ~45-55% Arithmetic")
    print("  (Exact numbers vary by question type and graph complexity)")

    # Save results
    results_data = {
        "model": MODEL,
        "samples_per_type": SAMPLES_PER_TYPE,
        "results": {k: dict(v) for k, v in results.items()},
        "totals": {
            "semantic": {"correct": sem_correct, "total": sem_total, "accuracy": round(sem_acc, 2)},
            "arithmetic": {"correct": arith_correct, "total": arith_total, "accuracy": round(arith_acc, 2)},
            "overall": {"correct": total_correct, "total": total_total, "accuracy": round(total_acc, 2)},
        }
    }
    with open(os.path.join(SCRIPT_DIR, "results.json"), "w") as f:
        json.dump(results_data, f, indent=2)
    print("\nResults saved to results.json")


if __name__ == "__main__":
    run_benchmark()
