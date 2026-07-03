#!/usr/bin/env python3
"""
PIE Full Benchmark Suite
========================

Runs all benchmarks in parallel with nice progress bars and logging.

Tasks:
  1. PIE extraction (203 batches, ~6 hours) - sequential (needs sliding window)
  2. LongMemEval full (500 questions) - parallel baselines
  3. LoCoMo full (1,986 questions) - parallel baselines
  4. Test of Time (arithmetic + semantic split) - parallel
  5. Paper baselines (BM25, Contriever) for comparison

Usage:
    python run_full_suite.py                    # Run everything
    python run_full_suite.py --skip-extraction  # Skip PIE extraction (already done)
    python run_full_suite.py --only extraction  # Only run extraction
    python run_full_suite.py --only longmemeval # Only run LongMemEval
    python run_full_suite.py --dry-run          # Show what would run

Outputs:
    logs/YYYYMMDD_HHMMSS/
        ├── master.log            # High-level progress
        ├── extraction.log        # PIE extraction details
        ├── longmemeval.log       # LongMemEval details
        ├── locomo.log            # LoCoMo details
        ├── tot.log               # Test of Time details
        └── results/              # JSON result files
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import queue
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env
_env_path = PROJECT_ROOT / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


# ══════════════════════════════════════════════════════════════════════════════
# PROGRESS BAR
# ══════════════════════════════════════════════════════════════════════════════

class ProgressBar:
    """Thread-safe progress bar with multiple concurrent tasks."""
    
    def __init__(self, total_tasks: int = 5):
        self.tasks: dict[str, dict] = {}
        self.lock = threading.Lock()
        self.total_tasks = total_tasks
        self.start_time = time.time()
        self._stop = False
        self._display_thread: Optional[threading.Thread] = None
    
    def add_task(self, task_id: str, name: str, total: int, status: str = "pending"):
        with self.lock:
            self.tasks[task_id] = {
                "name": name,
                "total": total,
                "current": 0,
                "status": status,  # pending, running, done, error
                "start_time": None,
                "end_time": None,
                "extra": "",
            }
    
    def update(self, task_id: str, current: int = None, status: str = None, extra: str = None):
        with self.lock:
            if task_id not in self.tasks:
                return
            task = self.tasks[task_id]
            if current is not None:
                task["current"] = current
            if status is not None:
                task["status"] = status
                if status == "running" and task["start_time"] is None:
                    task["start_time"] = time.time()
                elif status in ("done", "error"):
                    task["end_time"] = time.time()
            if extra is not None:
                task["extra"] = extra
    
    def increment(self, task_id: str, delta: int = 1):
        with self.lock:
            if task_id in self.tasks:
                self.tasks[task_id]["current"] += delta
    
    def _format_bar(self, current: int, total: int, width: int = 20) -> str:
        if total == 0:
            return "░" * width
        filled = int(width * current / total)
        return "█" * filled + "░" * (width - filled)
    
    def _format_time(self, seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            return f"{seconds/3600:.1f}h"
    
    def _get_eta(self, task: dict) -> str:
        if task["status"] != "running" or task["current"] == 0:
            return "--"
        elapsed = time.time() - task["start_time"]
        rate = task["current"] / elapsed
        remaining = (task["total"] - task["current"]) / rate if rate > 0 else 0
        return self._format_time(remaining)
    
    def render(self) -> str:
        with self.lock:
            lines = []
            lines.append("")
            lines.append("╔══════════════════════════════════════════════════════════════════════════╗")
            lines.append("║                        PIE BENCHMARK SUITE                               ║")
            lines.append("╠══════════════════════════════════════════════════════════════════════════╣")
            
            elapsed = time.time() - self.start_time
            lines.append(f"║  Elapsed: {self._format_time(elapsed):<10}                                              ║")
            lines.append("╠══════════════════════════════════════════════════════════════════════════╣")
            
            for task_id, task in self.tasks.items():
                status_emoji = {
                    "pending": "⏳",
                    "running": "🔄",
                    "done": "✅",
                    "error": "❌",
                }.get(task["status"], "❓")
                
                bar = self._format_bar(task["current"], task["total"])
                pct = 100 * task["current"] / task["total"] if task["total"] > 0 else 0
                eta = self._get_eta(task)
                
                name_str = f"{status_emoji} {task['name'][:18]:<18}"
                progress_str = f"{bar} {pct:5.1f}%"
                count_str = f"{task['current']:>5}/{task['total']:<5}"
                eta_str = f"ETA: {eta:<6}"
                
                line = f"║  {name_str} {progress_str} {count_str} {eta_str}  ║"
                lines.append(line)
                
                if task["extra"]:
                    extra_line = f"║     └─ {task['extra'][:64]:<64}  ║"
                    lines.append(extra_line)
            
            lines.append("╚══════════════════════════════════════════════════════════════════════════╝")
            return "\n".join(lines)
    
    def start_display(self, refresh_rate: float = 0.5):
        """Start background thread to refresh display."""
        def _display_loop():
            while not self._stop:
                # Clear screen and move cursor to top
                print("\033[2J\033[H" + self.render(), flush=True)
                time.sleep(refresh_rate)
        
        self._display_thread = threading.Thread(target=_display_loop, daemon=True)
        self._display_thread.start()
    
    def stop_display(self):
        self._stop = True
        if self._display_thread:
            self._display_thread.join(timeout=1)
        # Print final state
        print(self.render())


# ══════════════════════════════════════════════════════════════════════════════
# LOGGING SETUP
# ══════════════════════════════════════════════════════════════════════════════

def setup_logging(output_dir: Path) -> dict[str, logging.Logger]:
    """Create separate loggers for each task."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    loggers = {}
    
    # Master logger (also goes to console)
    master = logging.getLogger("master")
    master.setLevel(logging.INFO)
    master.handlers = []
    
    fh = logging.FileHandler(output_dir / "master.log")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S"))
    master.addHandler(fh)
    
    loggers["master"] = master
    
    # Task-specific loggers (file only)
    for name in ["extraction", "longmemeval", "locomo", "tot", "baselines"]:
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        logger.handlers = []
        
        fh = logging.FileHandler(output_dir / f"{name}.log")
        fh.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S"))
        logger.addHandler(fh)
        
        loggers[name] = logger
    
    return loggers


# ══════════════════════════════════════════════════════════════════════════════
# TASK RUNNERS
# ══════════════════════════════════════════════════════════════════════════════

def run_extraction(progress: ProgressBar, logger: logging.Logger, output_dir: Path, dry_run: bool = False) -> dict:
    """Run PIE extraction pipeline (203 batches)."""
    task_id = "extraction"
    
    if dry_run:
        progress.update(task_id, status="done", extra="DRY RUN - skipped")
        return {"status": "dry_run"}
    
    progress.update(task_id, status="running")
    logger.info("Starting PIE extraction pipeline")
    
    try:
        from pie.config import PIEConfig
        from pie.ingestion.pipeline import IngestionPipeline
        from pie.core.parser import parse_conversations, group_into_daily_batches
        
        # Load and count batches
        config = PIEConfig(output_dir=output_dir / "extraction")
        conversations = parse_conversations(config.conversations_path, year_min=2025)
        batches = group_into_daily_batches(conversations)
        
        total_batches = len(batches)
        progress.update(task_id, current=0)
        progress.tasks[task_id]["total"] = total_batches
        
        logger.info(f"Found {len(conversations)} conversations in {total_batches} batches")
        
        # Create pipeline with progress callback
        pipeline = IngestionPipeline(config)
        
        # Monkey-patch to report progress
        original_process = pipeline._process_batch
        batch_count = [0]
        
        def progress_wrapper(batch):
            result = original_process(batch)
            batch_count[0] += 1
            progress.update(task_id, current=batch_count[0], 
                          extra=f"{batch.date} | {pipeline.world_model.stats['entities']} entities")
            logger.info(f"[{batch_count[0]}/{total_batches}] {batch.date} - {pipeline.world_model.stats}")
            return result
        
        pipeline._process_batch = progress_wrapper
        
        # Run
        pipeline.run(save_every=5)
        
        progress.update(task_id, status="done", extra=f"{pipeline.world_model.stats['entities']} entities extracted")
        logger.info(f"Extraction complete: {pipeline.world_model.stats}")
        
        return {
            "status": "done",
            "batches": total_batches,
            "stats": pipeline.world_model.stats,
        }
        
    except Exception as e:
        progress.update(task_id, status="error", extra=str(e)[:60])
        logger.exception(f"Extraction failed: {e}")
        return {"status": "error", "error": str(e)}


def run_longmemeval(progress: ProgressBar, logger: logging.Logger, output_dir: Path, 
                   baselines: list[str], dry_run: bool = False) -> dict:
    """Run LongMemEval benchmark (500 questions)."""
    task_id = "longmemeval"
    
    if dry_run:
        progress.update(task_id, status="done", extra="DRY RUN - skipped")
        return {"status": "dry_run"}
    
    progress.update(task_id, status="running")
    logger.info(f"Starting LongMemEval with baselines: {baselines}")
    
    try:
        from pie.core.llm import LLMClient
        from benchmarks.longmemeval.adapter import load_dataset
        from benchmarks.longmemeval.baselines import naive_rag, full_context
        from benchmarks.longmemeval.runner import judge_answer
        from collections import defaultdict
        
        # Load data
        data_path = PROJECT_ROOT / "benchmarks/longmemeval/data/longmemeval_s_cleaned.json"
        dataset = load_dataset(data_path)
        
        total = len(dataset) * len(baselines)
        progress.tasks[task_id]["total"] = total
        progress.update(task_id, current=0)
        
        logger.info(f"Loaded {len(dataset)} questions")
        
        llm = LLMClient()
        results = {}
        processed = 0
        
        baseline_fns = {
            "naive_rag_turn": (naive_rag, {"chunk_by": "turn", "top_k": 10}),
            "naive_rag_session": (naive_rag, {"chunk_by": "session", "top_k": 5}),
            "full_context": (full_context, {}),
        }
        
        for baseline_name in baselines:
            if baseline_name not in baseline_fns:
                logger.warning(f"Unknown baseline: {baseline_name}")
                continue
            
            fn, kwargs = baseline_fns[baseline_name]
            scores_by_type = defaultdict(lambda: {"count": 0, "score": 0.0})
            total_score = 0
            baseline_results = []
            
            for i, item in enumerate(dataset):
                try:
                    result = fn(item, llm=llm, **kwargs)
                    hypothesis = result.hypothesis
                except Exception as e:
                    hypothesis = f"Error: {e}"
                
                try:
                    score, reason = judge_answer(
                        question=item["question"],
                        gold_answer=item["answer"],
                        hypothesis=hypothesis,
                        llm=llm,
                        model="gpt-4o"
                    )
                except Exception as e:
                    score, reason = 0.0, f"Judge error: {e}"
                
                total_score += score
                qtype = item["question_type"]
                scores_by_type[qtype]["count"] += 1
                scores_by_type[qtype]["score"] += score
                
                baseline_results.append({
                    "question_id": item["question_id"],
                    "question_type": qtype,
                    "score": score,
                })
                
                processed += 1
                acc = 100 * total_score / (i + 1)
                progress.update(task_id, current=processed, 
                              extra=f"{baseline_name}: {acc:.1f}% ({i+1}/{len(dataset)})")
                logger.debug(f"{baseline_name} [{i+1}] {qtype}: {score} (running: {acc:.1f}%)")
            
            final_acc = 100 * total_score / len(dataset)
            results[baseline_name] = {
                "accuracy": final_acc,
                "by_type": {t: 100*d["score"]/d["count"] for t, d in scores_by_type.items()},
                "raw": baseline_results,
            }
            logger.info(f"{baseline_name}: {final_acc:.1f}%")
        
        # Save results
        results_path = output_dir / "longmemeval_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        
        progress.update(task_id, status="done", 
                       extra=f"Best: {max(r['accuracy'] for r in results.values()):.1f}%")
        return {"status": "done", "results": results}
        
    except Exception as e:
        progress.update(task_id, status="error", extra=str(e)[:60])
        logger.exception(f"LongMemEval failed: {e}")
        return {"status": "error", "error": str(e)}


def run_locomo(progress: ProgressBar, logger: logging.Logger, output_dir: Path,
               baselines: list[str], dry_run: bool = False) -> dict:
    """Run LoCoMo benchmark (1,986 questions)."""
    task_id = "locomo"
    
    if dry_run:
        progress.update(task_id, status="done", extra="DRY RUN - skipped")
        return {"status": "dry_run"}
    
    progress.update(task_id, status="running")
    logger.info(f"Starting LoCoMo with baselines: {baselines}")
    
    try:
        from pie.core.llm import LLMClient
        from benchmarks.locomo.adapter import load_dataset, flatten_qa
        from benchmarks.locomo.baselines import naive_rag
        from benchmarks.locomo.runner import judge_answer
        from collections import defaultdict
        
        CATEGORY_MAP = {1: "single_hop", 2: "multi_hop", 3: "temporal", 4: "adversarial", 5: "commonsense"}
        
        # Load data
        data_path = PROJECT_ROOT / "benchmarks/locomo/data/locomo10.json"
        data = load_dataset(data_path)
        items = flatten_qa(data)
        
        total = len(items) * len(baselines)
        progress.tasks[task_id]["total"] = total
        progress.update(task_id, current=0)
        
        logger.info(f"Loaded {len(items)} QA pairs from {len(data)} conversations")
        
        llm = LLMClient()
        results = {}
        processed = 0
        
        for baseline_name in baselines:
            chunk_by = "turn" if "turn" in baseline_name else "session"
            scores_by_cat = defaultdict(lambda: {"count": 0, "score": 0.0})
            total_score = 0
            baseline_results = []
            
            for i, item in enumerate(items):
                try:
                    result = naive_rag(item, llm=llm, chunk_by=chunk_by)
                    hypothesis = result.hypothesis
                except Exception as e:
                    hypothesis = f"Error: {e}"
                
                try:
                    score, reason = judge_answer(
                        question=item["question"],
                        gold_answer=item["answer"],
                        hypothesis=hypothesis,
                        llm=llm,
                        model="gpt-4o"
                    )
                except Exception as e:
                    score, reason = 0.0, f"Judge error: {e}"
                
                total_score += score
                cat = CATEGORY_MAP.get(item.get("category", 0), "unknown")
                scores_by_cat[cat]["count"] += 1
                scores_by_cat[cat]["score"] += score
                
                baseline_results.append({
                    "question_id": item.get("question_id", f"q{i}"),
                    "category": cat,
                    "score": score,
                })
                
                processed += 1
                acc = 100 * total_score / (i + 1)
                if (i + 1) % 50 == 0:
                    progress.update(task_id, current=processed,
                                  extra=f"{baseline_name}: {acc:.1f}% ({i+1}/{len(items)})")
                    logger.debug(f"{baseline_name} [{i+1}] {cat}: {score} (running: {acc:.1f}%)")
            
            final_acc = 100 * total_score / len(items)
            results[baseline_name] = {
                "accuracy": final_acc,
                "by_category": {c: 100*d["score"]/d["count"] for c, d in scores_by_cat.items()},
                "raw": baseline_results,
            }
            logger.info(f"{baseline_name}: {final_acc:.1f}%")
        
        # Save results
        results_path = output_dir / "locomo_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        
        progress.update(task_id, status="done",
                       extra=f"Best: {max(r['accuracy'] for r in results.values()):.1f}%")
        return {"status": "done", "results": results}
        
    except Exception as e:
        progress.update(task_id, status="error", extra=str(e)[:60])
        logger.exception(f"LoCoMo failed: {e}")
        return {"status": "error", "error": str(e)}


def run_test_of_time(progress: ProgressBar, logger: logging.Logger, output_dir: Path,
                     dry_run: bool = False) -> dict:
    """Run Test of Time benchmark (arithmetic + semantic split)."""
    task_id = "tot"
    
    if dry_run:
        progress.update(task_id, status="done", extra="DRY RUN - skipped")
        return {"status": "dry_run"}
    
    progress.update(task_id, status="running")
    logger.info("Starting Test of Time benchmark")
    
    try:
        from openai import OpenAI
        import re
        
        client = OpenAI()
        
        # Load datasets
        arith_path = PROJECT_ROOT / "benchmarks/tot/tot_arithmetic.json"
        sem_path = PROJECT_ROOT / "benchmarks/tot/tot_semantic.json"
        
        with open(arith_path) as f:
            ds_arith = json.load(f)
        with open(sem_path) as f:
            ds_sem = json.load(f)
        
        # Sample: 50 per question type for semantic, all for arithmetic (smaller)
        SAMPLES_PER_TYPE = 50
        MAX_CHAR = 16000
        
        # Filter semantic by length and sample
        sem_by_type = {}
        for item in ds_sem:
            qt = item["question_type"]
            if len(item["prompt"] + item["question"]) < MAX_CHAR:
                if qt not in sem_by_type:
                    sem_by_type[qt] = []
                sem_by_type[qt].append(item)
        
        sem_items = []
        for qt, items in sem_by_type.items():
            items.sort(key=lambda x: len(x["prompt"]))
            sem_items.extend(items[:SAMPLES_PER_TYPE])
        
        # Sample arithmetic
        arith_by_type = {}
        for item in ds_arith:
            qt = item["question_type"]
            if qt not in arith_by_type:
                arith_by_type[qt] = []
            arith_by_type[qt].append(item)
        
        arith_items = []
        for qt, items in arith_by_type.items():
            arith_items.extend(items[:SAMPLES_PER_TYPE])
        
        total = len(sem_items) + len(arith_items)
        progress.tasks[task_id]["total"] = total
        progress.update(task_id, current=0)
        
        logger.info(f"Semantic: {len(sem_items)} items, Arithmetic: {len(arith_items)} items")
        
        results = {
            "semantic": {"by_type": {}, "correct": 0, "total": 0},
            "arithmetic": {"by_type": {}, "correct": 0, "total": 0},
        }
        processed = 0
        
        def call_llm(prompt: str) -> str:
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=512,
            )
            return resp.choices[0].message.content.strip()
        
        def check_semantic(response: str, label: str, qt: str) -> bool:
            if response.lower() == label.lower():
                return True
            if qt == "timeline":
                gt_list = [x.strip().lower() for x in label.split(",")]
                found = re.findall(r'E\d+', response, re.IGNORECASE)
                if found and [x.lower() for x in found] == gt_list:
                    return True
            if re.match(r'^E\d+$', label, re.IGNORECASE):
                if re.search(r'\b' + re.escape(label) + r'\b', response, re.IGNORECASE):
                    return True
            return False
        
        def check_arithmetic(response: str, label_str: str) -> bool:
            try:
                label = json.loads(label_str)
            except:
                return label_str.lower() in response.lower()
            
            if "answer" in label:
                ans = str(label["answer"]).lower()
                if ans in response.lower():
                    return True
            return False
        
        # Run semantic
        for item in sem_items:
            qt = item["question_type"]
            prompt = item["prompt"] + "\n" + item["question"]
            try:
                response = call_llm(prompt)
                correct = check_semantic(response, item["label"], qt)
            except Exception as e:
                logger.warning(f"Semantic error: {e}")
                correct = False
            
            if qt not in results["semantic"]["by_type"]:
                results["semantic"]["by_type"][qt] = {"correct": 0, "total": 0}
            results["semantic"]["by_type"][qt]["total"] += 1
            results["semantic"]["total"] += 1
            if correct:
                results["semantic"]["by_type"][qt]["correct"] += 1
                results["semantic"]["correct"] += 1
            
            processed += 1
            if processed % 20 == 0:
                progress.update(task_id, current=processed,
                              extra=f"Semantic: {100*results['semantic']['correct']/results['semantic']['total']:.1f}%")
            
            time.sleep(0.2)  # Rate limit
        
        # Run arithmetic
        for item in arith_items:
            qt = item["question_type"]
            try:
                response = call_llm(item["question"])
                correct = check_arithmetic(response, item["label"])
            except Exception as e:
                logger.warning(f"Arithmetic error: {e}")
                correct = False
            
            if qt not in results["arithmetic"]["by_type"]:
                results["arithmetic"]["by_type"][qt] = {"correct": 0, "total": 0}
            results["arithmetic"]["by_type"][qt]["total"] += 1
            results["arithmetic"]["total"] += 1
            if correct:
                results["arithmetic"]["by_type"][qt]["correct"] += 1
                results["arithmetic"]["correct"] += 1
            
            processed += 1
            if processed % 20 == 0:
                progress.update(task_id, current=processed,
                              extra=f"Arith: {100*results['arithmetic']['correct']/results['arithmetic']['total']:.1f}%")
            
            time.sleep(0.2)
        
        # Compute final accuracies
        sem_acc = 100 * results["semantic"]["correct"] / results["semantic"]["total"]
        arith_acc = 100 * results["arithmetic"]["correct"] / results["arithmetic"]["total"]
        
        results["semantic"]["accuracy"] = sem_acc
        results["arithmetic"]["accuracy"] = arith_acc
        results["overall_accuracy"] = (results["semantic"]["correct"] + results["arithmetic"]["correct"]) / (results["semantic"]["total"] + results["arithmetic"]["total"]) * 100
        
        # Save
        results_path = output_dir / "tot_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"ToT Semantic: {sem_acc:.1f}%, Arithmetic: {arith_acc:.1f}%")
        progress.update(task_id, status="done", extra=f"Sem: {sem_acc:.1f}% | Arith: {arith_acc:.1f}%")
        
        return {"status": "done", "results": results}
        
    except Exception as e:
        progress.update(task_id, status="error", extra=str(e)[:60])
        logger.exception(f"ToT failed: {e}")
        return {"status": "error", "error": str(e)}


def run_paper_baselines(progress: ProgressBar, logger: logging.Logger, output_dir: Path,
                        dry_run: bool = False) -> dict:
    """Implement BM25 and Contriever baselines for fair comparison."""
    task_id = "baselines"
    
    if dry_run:
        progress.update(task_id, status="done", extra="DRY RUN - skipped")
        return {"status": "dry_run"}
    
    progress.update(task_id, status="running")
    logger.info("Running paper baselines (BM25, Contriever)")
    
    try:
        # BM25 baseline using rank_bm25
        try:
            from rank_bm25 import BM25Okapi
            HAS_BM25 = True
        except ImportError:
            logger.warning("rank_bm25 not installed, skipping BM25 baseline")
            HAS_BM25 = False
        
        # Contriever would need sentence-transformers with the facebook/contriever model
        # For now, we'll use a simpler embedding baseline
        
        results = {}
        
        if HAS_BM25:
            # Run BM25 on LongMemEval subset
            from benchmarks.longmemeval.adapter import load_dataset
            from pie.core.llm import LLMClient
            from benchmarks.longmemeval.runner import judge_answer
            from collections import defaultdict
            
            data_path = PROJECT_ROOT / "benchmarks/longmemeval/data/longmemeval_s_cleaned.json"
            dataset = load_dataset(data_path)[:100]  # Subset for baselines
            
            progress.tasks[task_id]["total"] = len(dataset) * 2  # BM25 + embedding
            llm = LLMClient()
            
            # BM25 baseline
            bm25_scores = defaultdict(lambda: {"count": 0, "score": 0.0})
            total_score = 0
            
            for i, item in enumerate(dataset):
                # Build BM25 index from conversations
                conversations = item.get("haystack_conversations", [])
                docs = []
                for conv in conversations:
                    for turn in conv.get("turns", []):
                        docs.append(turn.get("user", "") + " " + turn.get("assistant", ""))
                
                if not docs:
                    continue
                
                tokenized = [d.lower().split() for d in docs]
                bm25 = BM25Okapi(tokenized)
                
                # Retrieve top-k
                query_tokens = item["question"].lower().split()
                scores = bm25.get_scores(query_tokens)
                top_k_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:10]
                context = "\n".join([docs[i] for i in top_k_idx])
                
                # Generate answer
                prompt = f"Context:\n{context}\n\nQuestion: {item['question']}\nAnswer:"
                try:
                    resp = llm.chat([{"role": "user", "content": prompt}], model="gpt-4o")
                    hypothesis = resp["content"]
                except:
                    hypothesis = "Error"
                
                # Judge
                try:
                    score, _ = judge_answer(item["question"], item["answer"], hypothesis, llm, "gpt-4o")
                except:
                    score = 0.0
                
                total_score += score
                bm25_scores[item["question_type"]]["count"] += 1
                bm25_scores[item["question_type"]]["score"] += score
                
                progress.increment(task_id)
                if (i + 1) % 20 == 0:
                    acc = 100 * total_score / (i + 1)
                    progress.update(task_id, extra=f"BM25: {acc:.1f}%")
            
            results["bm25"] = {
                "accuracy": 100 * total_score / len(dataset),
                "by_type": {t: 100*d["score"]/d["count"] for t, d in bm25_scores.items()},
            }
            logger.info(f"BM25: {results['bm25']['accuracy']:.1f}%")
        
        # Save results
        results_path = output_dir / "paper_baselines_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        
        progress.update(task_id, status="done", extra="BM25 complete")
        return {"status": "done", "results": results}
        
    except Exception as e:
        progress.update(task_id, status="error", extra=str(e)[:60])
        logger.exception(f"Paper baselines failed: {e}")
        return {"status": "error", "error": str(e)}


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="PIE Full Benchmark Suite")
    parser.add_argument("--skip-extraction", action="store_true", help="Skip PIE extraction")
    parser.add_argument("--only", type=str, choices=["extraction", "longmemeval", "locomo", "tot", "baselines"],
                       help="Run only this task")
    parser.add_argument("--dry-run", action="store_true", help="Show what would run")
    parser.add_argument("--output", type=Path, default=None, help="Output directory")
    parser.add_argument("--no-progress", action="store_true", help="Disable live progress (for logging)")
    parser.add_argument("--sequential", action="store_true", help="Run benchmarks sequentially (not parallel)")
    parser.add_argument("--baselines", type=str, nargs="+", 
                       default=["naive_rag_turn", "naive_rag_session"],
                       help="Baselines to run (default: naive_rag_turn naive_rag_session)")
    args = parser.parse_args()
    
    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output or (PROJECT_ROOT / "logs" / timestamp)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    loggers = setup_logging(output_dir)
    master_log = loggers["master"]
    
    master_log.info(f"PIE Benchmark Suite starting")
    master_log.info(f"Output: {output_dir}")
    
    # Initialize progress bar
    progress = ProgressBar()
    
    # Define tasks
    tasks_to_run = []
    
    if args.only:
        task_map = {
            "extraction": ("extraction", "PIE Extraction", 203),
            "longmemeval": ("longmemeval", "LongMemEval", 500),
            "locomo": ("locomo", "LoCoMo", 1986),
            "tot": ("tot", "Test of Time", 500),
            "baselines": ("baselines", "Paper Baselines", 200),
        }
        if args.only in task_map:
            tasks_to_run.append(task_map[args.only])
    else:
        if not args.skip_extraction:
            tasks_to_run.append(("extraction", "PIE Extraction", 203))
        tasks_to_run.extend([
            ("longmemeval", "LongMemEval", 500),
            ("locomo", "LoCoMo", 1986),
            ("tot", "Test of Time", 500),
            ("baselines", "Paper Baselines", 200),
        ])
    
    # Add tasks to progress bar
    for task_id, name, total in tasks_to_run:
        progress.add_task(task_id, name, total)
    
    # Start progress display
    if not args.no_progress and not args.dry_run:
        progress.start_display()
    
    results = {}
    
    try:
        # Run tasks
        # Extraction must run first (sequential). Then benchmarks can run in parallel.
        
        task_ids = [t[0] for t in tasks_to_run]
        
        # Phase 1: Run extraction first if needed
        if "extraction" in task_ids:
            results["extraction"] = run_extraction(
                progress, loggers["extraction"], output_dir, args.dry_run
            )
        
        # Phase 2: Run benchmarks in parallel using ThreadPoolExecutor
        parallel_tasks = [t for t in task_ids if t != "extraction"]
        
        if parallel_tasks and not args.dry_run and not args.sequential:
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = {}
                
                for task_id in parallel_tasks:
                    if task_id == "longmemeval":
                        futures[executor.submit(
                            run_longmemeval, progress, loggers["longmemeval"], output_dir,
                            args.baselines, False
                        )] = "longmemeval"
                    elif task_id == "locomo":
                        futures[executor.submit(
                            run_locomo, progress, loggers["locomo"], output_dir,
                            args.baselines, False
                        )] = "locomo"
                    elif task_id == "tot":
                        futures[executor.submit(
                            run_test_of_time, progress, loggers["tot"], output_dir, False
                        )] = "tot"
                    elif task_id == "baselines":
                        futures[executor.submit(
                            run_paper_baselines, progress, loggers["baselines"], output_dir, False
                        )] = "baselines"
                
                for future in as_completed(futures):
                    task_id = futures[future]
                    try:
                        results[task_id] = future.result()
                    except Exception as e:
                        results[task_id] = {"status": "error", "error": str(e)}
                        master_log.error(f"{task_id} failed: {e}")
        else:
            # Sequential fallback for dry run or --sequential flag
            for task_id in parallel_tasks:
                if task_id == "longmemeval":
                    results["longmemeval"] = run_longmemeval(
                        progress, loggers["longmemeval"], output_dir,
                        baselines=args.baselines,
                        dry_run=args.dry_run
                    )
                elif task_id == "locomo":
                    results["locomo"] = run_locomo(
                        progress, loggers["locomo"], output_dir,
                        baselines=args.baselines,
                        dry_run=args.dry_run
                    )
                elif task_id == "tot":
                    results["tot"] = run_test_of_time(
                        progress, loggers["tot"], output_dir, args.dry_run
                    )
                elif task_id == "baselines":
                    results["baselines"] = run_paper_baselines(
                        progress, loggers["baselines"], output_dir, args.dry_run
                    )
        
    finally:
        if not args.no_progress and not args.dry_run:
            progress.stop_display()
    
    # Save final summary
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "results": results,
    }
    
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    # Print final summary
    print("\n" + "=" * 70)
    print("  BENCHMARK SUITE COMPLETE")
    print("=" * 70)
    print(f"  Output: {output_dir}")
    print()
    
    for task_id, result in results.items():
        status = result.get("status", "unknown")
        emoji = "✅" if status == "done" else "❌" if status == "error" else "⏭️"
        
        if "results" in result and isinstance(result["results"], dict):
            if "accuracy" in result["results"]:
                detail = f"{result['results']['accuracy']:.1f}%"
            elif any("accuracy" in v for v in result["results"].values() if isinstance(v, dict)):
                accs = [v["accuracy"] for v in result["results"].values() if isinstance(v, dict) and "accuracy" in v]
                detail = f"Best: {max(accs):.1f}%"
            else:
                detail = status
        else:
            detail = status
        
        print(f"  {emoji} {task_id:<15} {detail}")
    
    print("=" * 70)
    print(f"\nLogs: {output_dir}/")
    print("  master.log        - High-level progress")
    print("  extraction.log    - PIE extraction details")
    print("  longmemeval.log   - LongMemEval Q&A details")
    print("  locomo.log        - LoCoMo Q&A details")
    print("  tot.log           - Test of Time details")


if __name__ == "__main__":
    main()
