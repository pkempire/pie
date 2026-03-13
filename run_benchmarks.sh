#!/bin/bash
#
# PIE Benchmark Suite Runner
# ==========================
#
# Quick commands:
#   ./run_benchmarks.sh                    # Full suite (extraction + all benchmarks)
#   ./run_benchmarks.sh --skip-extraction  # Skip extraction, run benchmarks only
#   ./run_benchmarks.sh --only longmemeval # Just LongMemEval
#   ./run_benchmarks.sh --only locomo      # Just LoCoMo
#   ./run_benchmarks.sh --only tot         # Just Test of Time
#   ./run_benchmarks.sh --dry-run          # Preview what would run
#
# For background execution with logging:
#   nohup ./run_benchmarks.sh > benchmark.log 2>&1 &
#   tail -f benchmark.log
#

set -e
cd "$(dirname "$0")"

# Load environment
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Check for API keys
if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY not set"
    exit 1
fi

# Run the benchmark suite
echo "Starting PIE Benchmark Suite..."
echo "Output: logs/$(date +%Y%m%d_%H%M%S)/"
echo ""

python3 run_full_suite.py "$@"
