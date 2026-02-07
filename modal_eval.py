"""
PIE Benchmark Runner — Modal-compatible unified evaluation harness.

Runs all retrieval approaches across all benchmarks:
- Approaches: naive_rag, pie_temporal, graph_aware, hybrid
- Benchmarks: LongMemEval, Test of Time, LoCoMo, MSC

Usage:
  # Local (test mode)
  python modal_eval.py --benchmark longmemeval --approach naive_rag --n 10

  # Modal (full run)
  modal run modal_eval.py --benchmark longmemeval --approach naive_rag

  # All benchmarks, all approaches
  modal run modal_eval.py --all
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Literal

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


# ── Configuration ─────────────────────────────────────────────────────────────

APPROACHES = ["naive_rag", "pie_temporal", "graph_aware", "hybrid"]
BENCHMARKS = ["longmemeval", "tot", "locomo", "msc"]

@dataclass
class EvalConfig:
    benchmark: str
    approach: str
    n_questions: int = 0  # 0 = all
    model: str = "gpt-4o-mini"
    output_dir: str = "benchmarks/results"
    world_model_path: str = "output/world_model.json"


# ── Approach Implementations ──────────────────────────────────────────────────

def get_retriever(approach: str, config: EvalConfig):
    """Factory for retrieval approaches."""
    
    if approach == "naive_rag":
        from pie.eval.query_interface import retrieve_entities_by_embedding
        from pie.core.llm import LLMClient
        
        llm = LLMClient()
        with open(config.world_model_path) as f:
            data = json.load(f)
        
        def retrieve(query: str, top_k: int = 10):
            entities = retrieve_entities_by_embedding(query, data, llm, top_k=top_k)
            return [{"id": eid, "name": e.get("name"), "type": e.get("type")} 
                    for eid, e, score in entities]
        return retrieve
    
    elif approach == "pie_temporal":
        # Semantic temporal reformulation
        from pie.core.world_model import WorldModel
        from pie.core.llm import LLMClient
        from pie.eval.query_interface import compile_entity_context
        
        wm = WorldModel(config.world_model_path)
        llm = LLMClient()
        
        def retrieve(query: str, top_k: int = 10):
            # Get entities and compile temporal context
            query_emb = llm.embed_single(query)
            entities = wm.find_by_embedding(query_emb, top_k=top_k)
            
            results = []
            for entity, score in entities:
                # Compile temporal narrative
                transitions = wm.get_transitions(entity.id)
                context = compile_entity_context(
                    entity.__dict__,
                    [t.__dict__ for t in transitions],
                    [],  # relationships
                    {},  # all_entities
                    time.time()
                )
                results.append({
                    "id": entity.id,
                    "name": entity.name,
                    "type": entity.type.value,
                    "temporal_context": context,
                })
            return results
        return retrieve
    
    elif approach == "graph_aware":
        from pie.retrieval.graph_retriever import retrieve_subgraph
        from pie.core.world_model import WorldModel
        from pie.core.llm import LLMClient
        
        wm = WorldModel(config.world_model_path)
        llm = LLMClient()
        
        def retrieve(query: str, top_k: int = 10):
            result = retrieve_subgraph(query, wm, llm, max_entities=top_k)
            entities = []
            for eid in result.entity_ids:
                entity = wm.get_entity(eid)
                if entity:
                    entities.append({
                        "id": entity.id,
                        "name": entity.name,
                        "type": entity.type.value,
                    })
            return entities
        return retrieve
    
    elif approach == "hybrid":
        # Combine embedding + graph traversal
        from pie.retrieval.graph_retriever import retrieve_subgraph
        from pie.eval.query_interface import retrieve_entities_by_embedding
        from pie.core.world_model import WorldModel
        from pie.core.llm import LLMClient
        
        wm = WorldModel(config.world_model_path)
        llm = LLMClient()
        with open(config.world_model_path) as f:
            data = json.load(f)
        
        def retrieve(query: str, top_k: int = 10):
            # Get both
            emb_results = retrieve_entities_by_embedding(query, data, llm, top_k=top_k//2)
            graph_result = retrieve_subgraph(query, wm, llm, max_entities=top_k//2)
            
            # Merge and dedupe
            seen = set()
            entities = []
            
            for eid, e, score in emb_results:
                if eid not in seen:
                    seen.add(eid)
                    entities.append({"id": eid, "name": e.get("name"), "source": "embedding"})
            
            for eid in graph_result.entity_ids:
                if eid not in seen:
                    seen.add(eid)
                    entity = wm.get_entity(eid)
                    if entity:
                        entities.append({"id": eid, "name": entity.name, "source": "graph"})
            
            return entities[:top_k]
        return retrieve
    
    else:
        raise ValueError(f"Unknown approach: {approach}")


# ── Benchmark Runners ─────────────────────────────────────────────────────────

def run_longmemeval(config: EvalConfig, retriever):
    """Run LongMemEval benchmark."""
    from benchmarks.longmemeval.adapter import LongMemEvalAdapter
    from benchmarks.longmemeval.baselines import LongMemEvalBaselines
    
    adapter = LongMemEvalAdapter()
    baselines = LongMemEvalBaselines(model=config.model)
    
    questions = adapter.load_questions()
    if config.n_questions > 0:
        questions = questions[:config.n_questions]
    
    results = []
    correct = 0
    
    for i, q in enumerate(questions):
        query = q["question"]
        expected = q.get("answer", "")
        
        # Retrieve
        entities = retriever(query)
        
        # Build context and answer
        context = "\n".join([f"- {e['name']}" for e in entities])
        answer = baselines.answer_with_context(query, context)
        
        # Score (simplified — real impl uses LLM judge)
        is_correct = expected.lower() in answer.lower() if expected else False
        if is_correct:
            correct += 1
        
        results.append({
            "question": query,
            "expected": expected,
            "answer": answer,
            "correct": is_correct,
            "entities_retrieved": len(entities),
        })
        
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(questions)}] Accuracy: {correct/(i+1)*100:.1f}%")
    
    return {
        "benchmark": "longmemeval",
        "approach": config.approach,
        "accuracy": correct / len(results) if results else 0,
        "total": len(results),
        "correct": correct,
        "results": results,
    }


def run_tot(config: EvalConfig, retriever):
    """Run Test of Time benchmark."""
    # Load ToT data
    tot_path = PROJECT_ROOT / "benchmarks" / "tot" / "tot_semantic.json"
    if not tot_path.exists():
        tot_path = PROJECT_ROOT / "benchmarks" / "test-of-time" / "data" / "tot_semantic" / "test.json"
    
    with open(tot_path) as f:
        data = json.load(f)
    
    questions = data if isinstance(data, list) else data.get("examples", [])
    if config.n_questions > 0:
        questions = questions[:config.n_questions]
    
    from pie.core.llm import LLMClient
    llm = LLMClient()
    
    results = []
    correct = 0
    
    for i, q in enumerate(questions):
        query = q.get("question", q.get("input", ""))
        expected = q.get("answer", q.get("target", ""))
        q_type = q.get("question_type", q.get("type", "unknown"))
        
        # Retrieve
        entities = retriever(query)
        
        # Build context
        context = "\n".join([f"- {e.get('name', e.get('id', 'unknown'))}" for e in entities])
        
        # Answer
        response = llm.chat(
            messages=[
                {"role": "system", "content": "Answer temporal reasoning questions concisely based on the provided context."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"}
            ],
            model=config.model,
            max_tokens=100,
        )
        answer = response["content"].strip()
        
        # Simple match check
        is_correct = expected.lower() in answer.lower() if expected else False
        if is_correct:
            correct += 1
        
        results.append({
            "question": query,
            "expected": expected,
            "answer": answer,
            "correct": is_correct,
            "type": q_type,
        })
        
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(questions)}] Accuracy: {correct/(i+1)*100:.1f}%")
    
    # Breakdown by type
    by_type = {}
    for r in results:
        t = r["type"]
        if t not in by_type:
            by_type[t] = {"correct": 0, "total": 0}
        by_type[t]["total"] += 1
        if r["correct"]:
            by_type[t]["correct"] += 1
    
    for t in by_type:
        by_type[t]["accuracy"] = by_type[t]["correct"] / by_type[t]["total"] if by_type[t]["total"] else 0
    
    return {
        "benchmark": "tot",
        "approach": config.approach,
        "accuracy": correct / len(results) if results else 0,
        "total": len(results),
        "correct": correct,
        "by_type": by_type,
    }


def run_locomo(config: EvalConfig, retriever):
    """Run LoCoMo benchmark."""
    from benchmarks.locomo.adapter import LoCoMoAdapter
    
    adapter = LoCoMoAdapter()
    questions = adapter.load_questions()
    
    if config.n_questions > 0:
        questions = questions[:config.n_questions]
    
    from pie.core.llm import LLMClient
    llm = LLMClient()
    
    results = []
    correct = 0
    
    for i, q in enumerate(questions):
        query = q.get("question", "")
        expected = q.get("answer", "")
        category = q.get("category", "unknown")
        
        entities = retriever(query)
        context = "\n".join([f"- {e.get('name', '')}" for e in entities])
        
        response = llm.chat(
            messages=[
                {"role": "system", "content": "Answer based on the context provided."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"}
            ],
            model=config.model,
            max_tokens=200,
        )
        answer = response["content"].strip()
        
        is_correct = expected.lower() in answer.lower() if expected else False
        if is_correct:
            correct += 1
        
        results.append({
            "question": query,
            "expected": expected,
            "answer": answer,
            "correct": is_correct,
            "category": category,
        })
        
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(questions)}] Accuracy: {correct/(i+1)*100:.1f}%")
    
    return {
        "benchmark": "locomo",
        "approach": config.approach,
        "accuracy": correct / len(results) if results else 0,
        "total": len(results),
        "correct": correct,
    }


def run_msc(config: EvalConfig, retriever):
    """Run MSC (Multi-Session Chat) benchmark."""
    from benchmarks.msc.adapter import MSCAdapter
    
    adapter = MSCAdapter()
    items = adapter.load_items()
    
    if config.n_questions > 0:
        items = items[:config.n_questions]
    
    from pie.core.llm import LLMClient
    llm = LLMClient()
    
    results = []
    total_score = 0
    
    for i, item in enumerate(items):
        query = item.get("query", item.get("question", ""))
        expected = item.get("expected", item.get("persona_fact", ""))
        
        entities = retriever(query)
        context = "\n".join([f"- {e.get('name', '')}" for e in entities])
        
        response = llm.chat(
            messages=[
                {"role": "system", "content": "Generate a response consistent with the persona context."},
                {"role": "user", "content": f"Persona context:\n{context}\n\nUser message: {query}\n\nResponse:"}
            ],
            model=config.model,
            max_tokens=200,
        )
        answer = response["content"].strip()
        
        # Partial credit scoring
        score = 1.0 if expected.lower() in answer.lower() else 0.5 if any(w in answer.lower() for w in expected.lower().split()[:3]) else 0.0
        total_score += score
        
        results.append({
            "query": query,
            "expected": expected,
            "answer": answer,
            "score": score,
        })
        
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(items)}] Score: {total_score/(i+1)*100:.1f}%")
    
    return {
        "benchmark": "msc",
        "approach": config.approach,
        "score": total_score / len(results) if results else 0,
        "total": len(results),
    }


BENCHMARK_RUNNERS = {
    "longmemeval": run_longmemeval,
    "tot": run_tot,
    "locomo": run_locomo,
    "msc": run_msc,
}


# ── Main Entry Point ──────────────────────────────────────────────────────────

def run_eval(config: EvalConfig):
    """Run a single evaluation."""
    print(f"\n{'='*60}")
    print(f"Benchmark: {config.benchmark}")
    print(f"Approach: {config.approach}")
    print(f"Model: {config.model}")
    print(f"{'='*60}\n")
    
    # Get retriever
    retriever = get_retriever(config.approach, config)
    
    # Get runner
    runner = BENCHMARK_RUNNERS.get(config.benchmark)
    if not runner:
        raise ValueError(f"Unknown benchmark: {config.benchmark}")
    
    # Run
    start = time.time()
    result = runner(config, retriever)
    result["elapsed_seconds"] = time.time() - start
    result["timestamp"] = datetime.now().isoformat()
    result["model"] = config.model
    
    # Save
    output_dir = Path(config.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"{config.benchmark}_{config.approach}.json"
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2, default=str)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_file}")
    if "accuracy" in result:
        print(f"Accuracy: {result['accuracy']*100:.1f}%")
    if "score" in result:
        print(f"Score: {result['score']*100:.1f}%")
    print(f"Elapsed: {result['elapsed_seconds']:.1f}s")
    print(f"{'='*60}\n")
    
    return result


def run_all(n_questions: int = 0):
    """Run all benchmarks with all approaches."""
    all_results = {}
    
    for benchmark in BENCHMARKS:
        all_results[benchmark] = {}
        for approach in APPROACHES:
            try:
                config = EvalConfig(
                    benchmark=benchmark,
                    approach=approach,
                    n_questions=n_questions,
                )
                result = run_eval(config)
                all_results[benchmark][approach] = result
            except Exception as e:
                print(f"ERROR: {benchmark}/{approach}: {e}")
                all_results[benchmark][approach] = {"error": str(e)}
    
    # Summary table
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"{'Benchmark':<15} {'naive_rag':<12} {'pie_temporal':<12} {'graph_aware':<12} {'hybrid':<12}")
    print("-"*80)
    
    for benchmark in BENCHMARKS:
        row = f"{benchmark:<15}"
        for approach in APPROACHES:
            r = all_results.get(benchmark, {}).get(approach, {})
            if "error" in r:
                row += f"{'ERROR':<12}"
            elif "accuracy" in r:
                row += f"{r['accuracy']*100:.1f}%{'':<7}"
            elif "score" in r:
                row += f"{r['score']*100:.1f}%{'':<7}"
            else:
                row += f"{'N/A':<12}"
        print(row)
    
    print("="*80)
    
    return all_results


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PIE Benchmark Evaluation")
    parser.add_argument("--benchmark", "-b", choices=BENCHMARKS, help="Benchmark to run")
    parser.add_argument("--approach", "-a", choices=APPROACHES, help="Retrieval approach")
    parser.add_argument("--n", type=int, default=0, help="Number of questions (0=all)")
    parser.add_argument("--model", default="gpt-4o-mini", help="LLM model")
    parser.add_argument("--all", action="store_true", help="Run all benchmarks and approaches")
    parser.add_argument("--output", default="benchmarks/results", help="Output directory")
    
    args = parser.parse_args()
    
    if args.all:
        run_all(n_questions=args.n)
    elif args.benchmark and args.approach:
        config = EvalConfig(
            benchmark=args.benchmark,
            approach=args.approach,
            n_questions=args.n,
            model=args.model,
            output_dir=args.output,
        )
        run_eval(config)
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python modal_eval.py --benchmark tot --approach naive_rag --n 50")
        print("  python modal_eval.py --all --n 20")


if __name__ == "__main__":
    main()


# ── Modal Deployment ──────────────────────────────────────────────────────────

try:
    import modal
    
    # Create Modal app
    app = modal.App("pie-eval")
    
    # Define image with dependencies
    image = modal.Image.debian_slim(python_version="3.11").pip_install(
        "openai>=1.0.0",
        "numpy",
        "requests",
    )
    
    # Mount the project
    project_mount = modal.Mount.from_local_dir(
        str(PROJECT_ROOT),
        remote_path="/app",
        condition=lambda path: not any(x in path for x in ["__pycache__", ".git", "output", ".venv"]),
    )
    
    @app.function(
        image=image,
        mounts=[project_mount],
        secrets=[modal.Secret.from_name("openai-api-key")],
        timeout=3600,
    )
    def modal_run_eval(benchmark: str, approach: str, n_questions: int = 0):
        """Modal function to run evaluation."""
        import sys
        sys.path.insert(0, "/app")
        
        config = EvalConfig(
            benchmark=benchmark,
            approach=approach,
            n_questions=n_questions,
        )
        return run_eval(config)
    
    @app.function(
        image=image,
        mounts=[project_mount],
        secrets=[modal.Secret.from_name("openai-api-key")],
        timeout=14400,  # 4 hours for full run
    )
    def modal_run_all(n_questions: int = 0):
        """Modal function to run all evaluations."""
        import sys
        sys.path.insert(0, "/app")
        return run_all(n_questions=n_questions)
    
    @app.local_entrypoint()
    def modal_main(
        benchmark: str = None,
        approach: str = None,
        n: int = 0,
        all: bool = False,
    ):
        """Modal CLI entrypoint."""
        if all:
            result = modal_run_all.remote(n_questions=n)
        elif benchmark and approach:
            result = modal_run_eval.remote(benchmark, approach, n_questions=n)
        else:
            print("Usage: modal run modal_eval.py --benchmark tot --approach naive_rag")
            print("       modal run modal_eval.py --all")
            return
        
        print(json.dumps(result, indent=2, default=str))

except ImportError:
    # Modal not installed, CLI-only mode
    pass
