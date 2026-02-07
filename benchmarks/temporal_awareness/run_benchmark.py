#!/usr/bin/env python3
"""
Run the Temporal Awareness Negotiation Benchmark.

Usage:
    python run_benchmark.py                                    # Run all conditions
    python run_benchmark.py --condition time_aware             # Single condition
    python run_benchmark.py --model claude-sonnet-4-20250514      # Specific model
    python run_benchmark.py --pie_temporal_injection           # Test PIE approach
    python run_benchmark.py --quick                            # Quick test (10 trials)
"""

import argparse
import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Import benchmark components
from negotiation_benchmark import (
    NegotiationEngine, Condition, NegotiationResult, MockLLMClient
)


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""
    conditions: List[Condition]
    time_limits: List[float]
    turn_limits: List[int]
    scenarios: List[str]
    trials_per_condition: int
    model_name: str
    output_dir: Path


class LLMClient:
    """
    Generic LLM client wrapper.
    Replace this with actual API implementation.
    """
    
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
        
        # Detect provider from model name
        if "gpt" in model_name.lower() or "o1" in model_name.lower():
            self.provider = "openai"
        elif "claude" in model_name.lower():
            self.provider = "anthropic"
        else:
            self.provider = "unknown"
    
    async def chat(self, system: str, messages: List[Dict[str, str]]) -> str:
        """Send a chat request to the LLM."""
        
        if self.provider == "openai":
            return await self._chat_openai(system, messages)
        elif self.provider == "anthropic":
            return await self._chat_anthropic(system, messages)
        else:
            raise ValueError(f"Unknown provider for model: {self.model_name}")
    
    async def _chat_openai(self, system: str, messages: List[Dict[str, str]]) -> str:
        """OpenAI API call."""
        try:
            import openai
            client = openai.AsyncOpenAI(api_key=self.api_key)
            
            full_messages = [{"role": "system", "content": system}] + messages
            
            response = await client.chat.completions.create(
                model=self.model_name,
                messages=full_messages,
                temperature=0.7,
                max_tokens=1000
            )
            return response.choices[0].message.content
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")
    
    async def _chat_anthropic(self, system: str, messages: List[Dict[str, str]]) -> str:
        """Anthropic API call."""
        try:
            import anthropic
            client = anthropic.AsyncAnthropic(api_key=self.api_key)
            
            response = await client.messages.create(
                model=self.model_name,
                max_tokens=1000,
                system=system,
                messages=messages
            )
            return response.content[0].text
        except ImportError:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")


async def run_single_trial(
    config: BenchmarkConfig,
    condition: Condition,
    time_limit: float,
    turn_limit: Optional[int],
    scenario: str,
    trial_num: int,
    llm_client
) -> NegotiationResult:
    """Run a single negotiation trial."""
    
    engine = NegotiationEngine(
        llm_client=llm_client,
        scenario_name=scenario,
        condition=condition,
        time_limit=time_limit,
        turn_limit=turn_limit,
        model_name=config.model_name
    )
    
    result = await engine.run_negotiation()
    return result


async def run_benchmark(config: BenchmarkConfig, use_mock: bool = False) -> Dict[str, Any]:
    """Run the full benchmark suite."""
    
    results = []
    
    # Create output directory
    config.output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Running temporal awareness benchmark")
    print(f"Model: {config.model_name}")
    print(f"Conditions: {[c.value for c in config.conditions]}")
    print(f"Trials per condition: {config.trials_per_condition}")
    print("-" * 50)
    
    for scenario in config.scenarios:
        print(f"\nScenario: {scenario}")
        
        for condition in config.conditions:
            print(f"  Condition: {condition.value}")
            
            # Determine time/turn limits based on condition
            if condition == Condition.TURN_BASED:
                limits = [(None, tl) for tl in config.turn_limits]
            else:
                limits = [(tl, None) for tl in config.time_limits]
            
            for time_limit, turn_limit in limits:
                limit_str = f"{turn_limit} turns" if turn_limit else f"{time_limit}s"
                deals_closed = 0
                
                for trial in range(config.trials_per_condition):
                    # Create client for each trial (could pool in production)
                    if use_mock:
                        client = MockLLMClient(behavior="cooperative")
                    else:
                        client = LLMClient(config.model_name)
                    
                    try:
                        result = await run_single_trial(
                            config, condition, 
                            time_limit or 300, turn_limit,
                            scenario, trial, client
                        )
                        results.append(result)
                        
                        if result.deal_reached:
                            deals_closed += 1
                            
                    except Exception as e:
                        print(f"    Trial {trial+1} error: {e}")
                
                closure_rate = deals_closed / config.trials_per_condition * 100
                print(f"    {limit_str}: {closure_rate:.1f}% deal closure ({deals_closed}/{config.trials_per_condition})")
    
    # Aggregate statistics
    summary = aggregate_results(results, config)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = config.output_dir / f"results_{timestamp}.json"
    
    with open(results_file, "w") as f:
        json.dump({
            "config": {
                "model": config.model_name,
                "conditions": [c.value for c in config.conditions],
                "time_limits": config.time_limits,
                "turn_limits": config.turn_limits,
                "scenarios": config.scenarios,
                "trials_per_condition": config.trials_per_condition
            },
            "summary": summary,
            "results": [r.to_dict() for r in results]
        }, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    return summary


def aggregate_results(results: List[NegotiationResult], config: BenchmarkConfig) -> Dict[str, Any]:
    """Aggregate results into summary statistics."""
    
    summary = {
        "total_trials": len(results),
        "by_condition": {},
        "by_scenario": {},
        "key_comparisons": {}
    }
    
    # Group by condition
    for condition in config.conditions:
        condition_results = [r for r in results if r.condition == condition]
        if condition_results:
            deals = sum(1 for r in condition_results if r.deal_reached)
            summary["by_condition"][condition.value] = {
                "trials": len(condition_results),
                "deals_closed": deals,
                "closure_rate": deals / len(condition_results) * 100,
                "avg_turns": sum(len(r.turns) for r in condition_results) / len(condition_results),
                "avg_joint_payoff": sum(r.joint_payoff or 0 for r in condition_results) / len(condition_results)
            }
    
    # Key comparisons (if we have both conditions)
    control_results = [r for r in results if r.condition == Condition.CONTROL]
    time_aware_results = [r for r in results if r.condition == Condition.TIME_AWARE]
    turn_based_results = [r for r in results if r.condition == Condition.TURN_BASED]
    pie_results = [r for r in results if r.condition == Condition.PIE_TEMPORAL]
    
    if control_results and time_aware_results:
        control_rate = sum(1 for r in control_results if r.deal_reached) / len(control_results) * 100
        time_aware_rate = sum(1 for r in time_aware_results if r.deal_reached) / len(time_aware_results) * 100
        
        summary["key_comparisons"]["control_vs_time_aware"] = {
            "control_closure": control_rate,
            "time_aware_closure": time_aware_rate,
            "improvement": time_aware_rate - control_rate,
            "relative_improvement": ((time_aware_rate - control_rate) / control_rate * 100) if control_rate > 0 else float('inf')
        }
    
    if turn_based_results:
        turn_rate = sum(1 for r in turn_based_results if r.deal_reached) / len(turn_based_results) * 100
        summary["key_comparisons"]["turn_based_closure"] = turn_rate
    
    if pie_results:
        pie_rate = sum(1 for r in pie_results if r.deal_reached) / len(pie_results) * 100
        summary["key_comparisons"]["pie_temporal_closure"] = pie_rate
    
    return summary


def print_summary(summary: Dict[str, Any]):
    """Print a human-readable summary."""
    
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    
    print(f"\nTotal trials: {summary['total_trials']}")
    
    print("\nBy Condition:")
    for condition, stats in summary["by_condition"].items():
        print(f"  {condition}:")
        print(f"    Trials: {stats['trials']}")
        print(f"    Deal closure: {stats['closure_rate']:.1f}%")
        print(f"    Avg turns: {stats['avg_turns']:.1f}")
        print(f"    Avg joint payoff: {stats['avg_joint_payoff']:.0f}")
    
    if summary["key_comparisons"]:
        print("\nKey Comparisons:")
        
        if "control_vs_time_aware" in summary["key_comparisons"]:
            comp = summary["key_comparisons"]["control_vs_time_aware"]
            print(f"  Control → Time-Aware:")
            print(f"    {comp['control_closure']:.1f}% → {comp['time_aware_closure']:.1f}%")
            print(f"    Improvement: +{comp['improvement']:.1f} percentage points")
            if comp['relative_improvement'] != float('inf'):
                print(f"    Relative improvement: +{comp['relative_improvement']:.0f}%")
        
        if "turn_based_closure" in summary["key_comparisons"]:
            print(f"  Turn-based closure: {summary['key_comparisons']['turn_based_closure']:.1f}%")
        
        if "pie_temporal_closure" in summary["key_comparisons"]:
            print(f"  PIE temporal closure: {summary['key_comparisons']['pie_temporal_closure']:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Run temporal awareness benchmark")
    
    parser.add_argument("--condition", type=str, 
                       choices=["control", "time_aware", "urgency", "turn_based", "pie_temporal"],
                       help="Run only this condition")
    parser.add_argument("--model", type=str, default="claude-sonnet-4-20250514",
                       help="Model to use")
    parser.add_argument("--trials", type=int, default=100,
                       help="Trials per condition")
    parser.add_argument("--quick", action="store_true",
                       help="Quick test with 10 trials")
    parser.add_argument("--mock", action="store_true",
                       help="Use mock LLM (for testing)")
    parser.add_argument("--pie_temporal_injection", action="store_true",
                       help="Include PIE temporal injection condition")
    parser.add_argument("--output", type=str, default="./results",
                       help="Output directory")
    
    args = parser.parse_args()
    
    # Build conditions list
    if args.condition:
        conditions = [Condition(args.condition)]
    else:
        conditions = [Condition.CONTROL, Condition.TIME_AWARE, Condition.TURN_BASED]
        if args.pie_temporal_injection:
            conditions.append(Condition.PIE_TEMPORAL)
    
    config = BenchmarkConfig(
        conditions=conditions,
        time_limits=[240.0, 300.0, 360.0],
        turn_limits=[5, 6, 7, 8, 9],
        scenarios=["new_recruit"],  # Add "rubbermind" for full benchmark
        trials_per_condition=10 if args.quick else args.trials,
        model_name=args.model,
        output_dir=Path(args.output)
    )
    
    # Run benchmark
    summary = asyncio.run(run_benchmark(config, use_mock=args.mock))
    
    # Print summary
    print_summary(summary)


if __name__ == "__main__":
    main()
