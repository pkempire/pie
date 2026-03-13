"""
PIE Cloud Benchmark Suite (Modal)
=================================

Run all benchmarks in the cloud. Close your laptop, come back to results.

Setup:
    pip install modal
    modal token new  # authenticate
    
Run:
    modal run cloud_benchmark.py                    # Full suite
    modal run cloud_benchmark.py --only longmemeval # Just LongMemEval
    modal run cloud_benchmark.py --dry-run          # Preview

Results uploaded to Modal volume, downloadable after.
"""

import modal
import os

# ══════════════════════════════════════════════════════════════════════════════
# Modal App Setup
# ══════════════════════════════════════════════════════════════════════════════

app = modal.App("pie-benchmarks")

# Persistent volume for results
results_volume = modal.Volume.from_name("pie-benchmark-results", create_if_missing=True)

# Docker image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "openai>=1.0",
        "numpy",
        "tqdm",
        "rank_bm25",
        "sentence-transformers",
        "requests",
        "python-dotenv",
    ])
    .run_commands([
        "pip install torch --index-url https://download.pytorch.org/whl/cpu",
    ])
)

# Secrets (set via: modal secret create openai-secret OPENAI_API_KEY=sk-...)
secrets = [modal.Secret.from_name("openai-secret")]


# ══════════════════════════════════════════════════════════════════════════════
# Benchmark Functions
# ══════════════════════════════════════════════════════════════════════════════

@app.function(
    image=image,
    secrets=secrets,
    volumes={"/results": results_volume},
    timeout=6 * 3600,  # 6 hours max
    cpu=4,
    memory=16384,  # 16GB RAM
)
def run_longmemeval(baselines: list[str], limit: int | None = None):
    """Run LongMemEval benchmark."""
    import json
    import time
    from datetime import datetime
    from pathlib import Path
    from collections import defaultdict
    from openai import OpenAI
    
    print(f"Starting LongMemEval with baselines: {baselines}")
    
    # Download dataset
    import urllib.request
    data_url = "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json"
    data_path = Path("/tmp/longmemeval_s_cleaned.json")
    
    if not data_path.exists():
        print("Downloading LongMemEval dataset...")
        urllib.request.urlretrieve(data_url, data_path)
    
    with open(data_path) as f:
        dataset = json.load(f)
    
    if limit:
        dataset = dataset[:limit]
    
    print(f"Loaded {len(dataset)} questions")
    
    client = OpenAI()
    results = {}
    
    for baseline_name in baselines:
        print(f"\n{'='*60}")
        print(f"Running baseline: {baseline_name}")
        print(f"{'='*60}")
        
        scores_by_type = defaultdict(lambda: {"count": 0, "score": 0.0})
        total_score = 0
        baseline_results = []
        
        for i, item in enumerate(dataset):
            # Get context based on baseline
            if baseline_name == "naive_rag":
                context = _naive_rag_context(item, client)
            elif baseline_name == "bm25":
                context = _bm25_context(item)
            elif baseline_name == "full_context":
                context = _full_context(item)
            else:
                context = _naive_rag_context(item, client)
            
            # Generate answer
            try:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "Answer questions based on the provided context. Be concise."},
                        {"role": "user", "content": f"Context:\n{context[:50000]}\n\nQuestion: {item['question']}\n\nAnswer:"}
                    ],
                    max_tokens=300,
                )
                hypothesis = response.choices[0].message.content.strip()
            except Exception as e:
                hypothesis = f"Error: {e}"
            
            # Judge
            try:
                judge_response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "Judge if the hypothesis answers the question correctly given the gold answer. Reply only 1.0 (correct), 0.5 (partial), or 0.0 (wrong)."},
                        {"role": "user", "content": f"Question: {item['question']}\nGold Answer: {item['answer']}\nHypothesis: {hypothesis}\n\nScore:"}
                    ],
                    max_tokens=10,
                )
                score_text = judge_response.choices[0].message.content.strip()
                score = float(score_text) if score_text in ["0.0", "0.5", "1.0"] else 0.0
            except:
                score = 0.0
            
            total_score += score
            qtype = item.get("question_type", "unknown")
            scores_by_type[qtype]["count"] += 1
            scores_by_type[qtype]["score"] += score
            
            baseline_results.append({
                "question_id": item["question_id"],
                "question_type": qtype,
                "score": score,
                "hypothesis": hypothesis[:200],
            })
            
            if (i + 1) % 20 == 0:
                acc = 100 * total_score / (i + 1)
                print(f"  [{i+1}/{len(dataset)}] Running accuracy: {acc:.1f}%")
        
        final_acc = 100 * total_score / len(dataset)
        results[baseline_name] = {
            "accuracy": final_acc,
            "by_type": {t: 100*d["score"]/d["count"] for t, d in scores_by_type.items()},
            "total": len(dataset),
        }
        print(f"\n{baseline_name}: {final_acc:.1f}%")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = Path(f"/results/longmemeval_{timestamp}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    results_volume.commit()
    print(f"\nResults saved to: {results_path}")
    return results


@app.function(
    image=image,
    secrets=secrets,
    volumes={"/results": results_volume},
    timeout=6 * 3600,
    cpu=4,
    memory=16384,
)
def run_locomo(baselines: list[str], limit: int | None = None):
    """Run LoCoMo benchmark."""
    import json
    from datetime import datetime
    from pathlib import Path
    from collections import defaultdict
    from openai import OpenAI
    import urllib.request
    
    print(f"Starting LoCoMo with baselines: {baselines}")
    
    # Download dataset
    data_url = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
    data_path = Path("/tmp/locomo10.json")
    
    if not data_path.exists():
        print("Downloading LoCoMo dataset...")
        urllib.request.urlretrieve(data_url, data_path)
    
    with open(data_path) as f:
        data = json.load(f)
    
    # Flatten QA pairs
    CATEGORY_MAP = {1: "single_hop", 2: "multi_hop", 3: "temporal", 4: "adversarial", 5: "commonsense"}
    items = []
    for conv in data:
        conv_text = "\n".join([f"{t['role']}: {t['content']}" for t in conv.get("conversation", [])])
        for qa in conv.get("qa", conv.get("qa_pairs", [])):
            items.append({
                "question": qa["question"],
                "answer": qa["answer"],
                "category": CATEGORY_MAP.get(qa.get("category", 0), "unknown"),
                "context": conv_text,
            })
    
    if limit:
        items = items[:limit]
    
    print(f"Loaded {len(items)} QA pairs")
    
    client = OpenAI()
    results = {}
    
    for baseline_name in baselines:
        print(f"\n{'='*60}")
        print(f"Running baseline: {baseline_name}")
        print(f"{'='*60}")
        
        scores_by_cat = defaultdict(lambda: {"count": 0, "score": 0.0})
        total_score = 0
        
        for i, item in enumerate(items):
            context = item["context"][:50000]
            
            try:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "Answer questions based on the conversation. Be concise."},
                        {"role": "user", "content": f"Conversation:\n{context}\n\nQuestion: {item['question']}\n\nAnswer:"}
                    ],
                    max_tokens=300,
                )
                hypothesis = response.choices[0].message.content.strip()
            except Exception as e:
                hypothesis = f"Error: {e}"
            
            # Judge
            try:
                judge_response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "Judge if the hypothesis answers correctly. Reply 1.0 (correct), 0.5 (partial), or 0.0 (wrong)."},
                        {"role": "user", "content": f"Question: {item['question']}\nGold: {item['answer']}\nHypothesis: {hypothesis}\n\nScore:"}
                    ],
                    max_tokens=10,
                )
                score_text = judge_response.choices[0].message.content.strip()
                score = float(score_text) if score_text in ["0.0", "0.5", "1.0"] else 0.0
            except:
                score = 0.0
            
            total_score += score
            cat = item["category"]
            scores_by_cat[cat]["count"] += 1
            scores_by_cat[cat]["score"] += score
            
            if (i + 1) % 100 == 0:
                acc = 100 * total_score / (i + 1)
                print(f"  [{i+1}/{len(items)}] Running accuracy: {acc:.1f}%")
        
        final_acc = 100 * total_score / len(items)
        results[baseline_name] = {
            "accuracy": final_acc,
            "by_category": {c: 100*d["score"]/d["count"] for c, d in scores_by_cat.items()},
            "total": len(items),
        }
        print(f"\n{baseline_name}: {final_acc:.1f}%")
    
    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = Path(f"/results/locomo_{timestamp}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    results_volume.commit()
    return results


@app.function(
    image=image,
    secrets=secrets,
    volumes={"/results": results_volume},
    timeout=2 * 3600,
    cpu=2,
    memory=8192,
)
def run_test_of_time(samples_per_type: int = 30):
    """Run Test of Time benchmark."""
    import json
    import time
    from datetime import datetime
    from pathlib import Path
    from collections import defaultdict
    from openai import OpenAI
    import urllib.request
    
    print("Starting Test of Time benchmark")
    
    # Download datasets
    arith_url = "https://raw.githubusercontent.com/google-research/test-of-time/main/data/tot_arithmetic.json"
    sem_url = "https://raw.githubusercontent.com/google-research/test-of-time/main/data/tot_semantic.json"
    
    # Note: These URLs may need adjustment based on actual repo structure
    # For now, we'll use a simplified version
    
    client = OpenAI()
    results = {
        "semantic": {"by_type": {}, "correct": 0, "total": 0},
        "arithmetic": {"by_type": {}, "correct": 0, "total": 0},
    }
    
    # Simplified ToT evaluation
    tot_questions = [
        {"type": "semantic", "subtype": "before_after", "prompt": "E1 was the R1 of E2 from 1990 to 1995. E3 was the R2 of E4 from 1985 to 1992.", "question": "Did E1 start their role before or after E3?", "answer": "after"},
        {"type": "semantic", "subtype": "timeline", "prompt": "E1 was the R1 of E2 from 1990 to 1995. E3 was the R2 of E4 from 1985 to 1992. E5 was the R3 of E6 from 2000 to 2005.", "question": "Order these entities by when they started: E1, E3, E5", "answer": "E3, E1, E5"},
        {"type": "arithmetic", "subtype": "duration", "prompt": "E1 was the R1 of E2 from 1990 to 2000.", "question": "How many years was E1 the R1 of E2?", "answer": "10"},
    ]
    
    for q in tot_questions:
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "user", "content": f"{q['prompt']}\n\nQuestion: {q['question']}"}
                ],
                max_tokens=100,
            )
            answer = response.choices[0].message.content.strip().lower()
            correct = q["answer"].lower() in answer
        except:
            correct = False
        
        qtype = q["type"]
        subtype = q["subtype"]
        
        if subtype not in results[qtype]["by_type"]:
            results[qtype]["by_type"][subtype] = {"correct": 0, "total": 0}
        
        results[qtype]["by_type"][subtype]["total"] += 1
        results[qtype]["total"] += 1
        if correct:
            results[qtype]["by_type"][subtype]["correct"] += 1
            results[qtype]["correct"] += 1
    
    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = Path(f"/results/tot_{timestamp}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    results_volume.commit()
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Helper Functions (run inside Modal)
# ══════════════════════════════════════════════════════════════════════════════

def _naive_rag_context(item: dict, client) -> str:
    """Build context using naive RAG (embedding retrieval)."""
    from openai import OpenAI
    import numpy as np
    
    sessions = item.get("haystack_sessions", [])
    dates = item.get("haystack_dates", [])
    
    # Build chunks
    chunks = []
    for i, session in enumerate(sessions):
        date = dates[i] if i < len(dates) else f"Session {i}"
        text = "\n".join([f"{t.get('role', 'user')}: {t.get('content', '')}" for t in session])
        if text.strip():
            chunks.append({"text": text[:2000], "date": date})
    
    if not chunks:
        return "No context available."
    
    # Embed query
    query_resp = client.embeddings.create(model="text-embedding-3-small", input=item["question"])
    query_emb = np.array(query_resp.data[0].embedding)
    
    # Embed chunks (batch)
    chunk_texts = [c["text"] for c in chunks[:100]]  # Limit for API
    chunk_resp = client.embeddings.create(model="text-embedding-3-small", input=chunk_texts)
    chunk_embs = np.array([d.embedding for d in chunk_resp.data])
    
    # Cosine similarity
    scores = np.dot(chunk_embs, query_emb) / (np.linalg.norm(chunk_embs, axis=1) * np.linalg.norm(query_emb) + 1e-8)
    
    # Top-k
    top_k = 10
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    context_parts = []
    for idx in top_indices:
        context_parts.append(f"[{chunks[idx]['date']}]\n{chunks[idx]['text']}")
    
    return "\n\n".join(context_parts)


def _bm25_context(item: dict) -> str:
    """Build context using BM25."""
    from rank_bm25 import BM25Okapi
    import numpy as np
    
    sessions = item.get("haystack_sessions", [])
    dates = item.get("haystack_dates", [])
    
    chunks = []
    for i, session in enumerate(sessions):
        date = dates[i] if i < len(dates) else f"Session {i}"
        text = "\n".join([f"{t.get('role', 'user')}: {t.get('content', '')}" for t in session])
        if text.strip():
            chunks.append({"text": text[:2000], "date": date})
    
    if not chunks:
        return "No context available."
    
    # Tokenize
    tokenized = [c["text"].lower().split() for c in chunks]
    query_tokens = item["question"].lower().split()
    
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(query_tokens)
    
    top_k = 10
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    context_parts = []
    for idx in top_indices:
        if scores[idx] > 0:
            context_parts.append(f"[{chunks[idx]['date']}]\n{chunks[idx]['text']}")
    
    return "\n\n".join(context_parts) if context_parts else "No relevant context found."


def _full_context(item: dict) -> str:
    """Build full context (all sessions)."""
    sessions = item.get("haystack_sessions", [])
    dates = item.get("haystack_dates", [])
    
    parts = []
    for i, session in enumerate(sessions):
        date = dates[i] if i < len(dates) else f"Session {i}"
        text = "\n".join([f"{t.get('role', 'user')}: {t.get('content', '')}" for t in session])
        if text.strip():
            parts.append(f"[{date}]\n{text}")
    
    return "\n\n".join(parts)[:100000]  # Truncate to ~100k chars


# ══════════════════════════════════════════════════════════════════════════════
# Main Entry Points
# ══════════════════════════════════════════════════════════════════════════════

@app.local_entrypoint()
def main(
    only: str = None,
    baselines: str = "naive_rag,bm25",
    limit: int = None,
    dry_run: bool = False,
):
    """
    Run PIE benchmark suite in the cloud.
    
    Args:
        only: Run only this benchmark (longmemeval, locomo, tot)
        baselines: Comma-separated baseline names
        limit: Limit questions per benchmark (for testing)
        dry_run: Just print what would run
    """
    baseline_list = baselines.split(",")
    
    print("=" * 60)
    print("  PIE CLOUD BENCHMARK SUITE")
    print("=" * 60)
    print(f"  Baselines: {baseline_list}")
    print(f"  Limit: {limit or 'None (full)'}")
    print(f"  Only: {only or 'All benchmarks'}")
    print("=" * 60)
    
    if dry_run:
        print("\nDRY RUN - would execute:")
        if not only or only == "longmemeval":
            print(f"  • LongMemEval: 500 questions × {len(baseline_list)} baselines")
        if not only or only == "locomo":
            print(f"  • LoCoMo: 1986 questions × {len(baseline_list)} baselines")
        if not only or only == "tot":
            print(f"  • Test of Time: semantic + arithmetic")
        return
    
    results = {}
    
    if not only or only == "longmemeval":
        print("\n🚀 Launching LongMemEval...")
        results["longmemeval"] = run_longmemeval.remote(baseline_list, limit)
    
    if not only or only == "locomo":
        print("\n🚀 Launching LoCoMo...")
        results["locomo"] = run_locomo.remote(baseline_list, limit)
    
    if not only or only == "tot":
        print("\n🚀 Launching Test of Time...")
        results["tot"] = run_test_of_time.remote()
    
    # Wait for all to complete
    print("\n⏳ Waiting for results (you can close this terminal)...")
    print("   Check status at: https://modal.com/apps")
    
    final_results = {}
    for name, future in results.items():
        try:
            final_results[name] = future
            print(f"\n✅ {name} complete")
        except Exception as e:
            print(f"\n❌ {name} failed: {e}")
    
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    
    for name, result in final_results.items():
        if isinstance(result, dict):
            for baseline, data in result.items():
                if isinstance(data, dict) and "accuracy" in data:
                    print(f"  {name}/{baseline}: {data['accuracy']:.1f}%")
    
    print("\n📁 Full results saved to Modal volume: pie-benchmark-results")
    print("   Download with: modal volume get pie-benchmark-results .")


# ══════════════════════════════════════════════════════════════════════════════
# Download Results
# ══════════════════════════════════════════════════════════════════════════════

@app.function(volumes={"/results": results_volume})
def list_results():
    """List all saved results."""
    import os
    from pathlib import Path
    
    results_dir = Path("/results")
    files = sorted(results_dir.glob("*.json"))
    
    for f in files:
        size = f.stat().st_size
        print(f"  {f.name} ({size:,} bytes)")
    
    return [str(f) for f in files]


@app.function(volumes={"/results": results_volume})
def get_result(filename: str) -> dict:
    """Get a specific result file."""
    import json
    from pathlib import Path
    
    path = Path(f"/results/{filename}")
    if not path.exists():
        raise FileNotFoundError(f"No result file: {filename}")
    
    with open(path) as f:
        return json.load(f)
