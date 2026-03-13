#!/bin/bash
#
# PIE Comprehensive Benchmark Suite
# ==================================
#
# Runs all memory providers against all benchmarks.
#
# Usage:
#   ./run_all_benchmarks.sh              # Full suite (local)
#   ./run_all_benchmarks.sh --cloud      # Run in Modal cloud
#   ./run_all_benchmarks.sh --test       # Quick test (5 items each)
#   ./run_all_benchmarks.sh --provider pie --benchmark longmemeval
#

set -e
cd "$(dirname "$0")"

# Load environment
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Check API keys
echo "=============================================="
echo "  PIE BENCHMARK SUITE"
echo "=============================================="
echo ""
echo "API Keys:"
[ -n "$OPENAI_API_KEY" ] && echo "  ✅ OpenAI" || echo "  ❌ OpenAI (REQUIRED)"
[ -n "$MEM0_API_KEY" ] && echo "  ✅ Mem0" || echo "  ⚠️  Mem0 (will use local sim)"
[ -n "$ZEP_API_KEY" ] && echo "  ✅ Zep" || echo "  ⚠️  Zep (will use local sim)"
[ -n "$SUPERMEMORY_API_KEY" ] && echo "  ✅ Supermemory" || echo "  ⚠️  Supermemory (will use local sim)"
echo ""

if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY required"
    exit 1
fi

# Parse args
CLOUD=false
TEST=false
PROVIDER=""
BENCHMARK=""
LIMIT=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --cloud)
            CLOUD=true
            shift
            ;;
        --test)
            TEST=true
            LIMIT="--limit 5"
            shift
            ;;
        --provider|-p)
            PROVIDER="$2"
            shift 2
            ;;
        --benchmark|-b)
            BENCHMARK="$2"
            shift 2
            ;;
        --limit|-n)
            LIMIT="--limit $2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Build command
if [ "$CLOUD" = true ]; then
    echo "Running in Modal cloud..."
    echo "You can close your laptop after this starts."
    echo ""
    
    # Check Modal is installed
    if ! command -v modal &> /dev/null; then
        echo "Installing Modal..."
        pip install modal
        modal token new
    fi
    
    # Set secrets
    echo "Setting Modal secrets..."
    modal secret create openai-secret OPENAI_API_KEY="$OPENAI_API_KEY" 2>/dev/null || true
    [ -n "$MEM0_API_KEY" ] && modal secret create mem0-secret MEM0_API_KEY="$MEM0_API_KEY" 2>/dev/null || true
    [ -n "$ZEP_API_KEY" ] && modal secret create zep-secret ZEP_API_KEY="$ZEP_API_KEY" 2>/dev/null || true
    
    # Run
    MODAL_ARGS=""
    [ -n "$PROVIDER" ] && MODAL_ARGS="$MODAL_ARGS --provider $PROVIDER"
    [ -n "$BENCHMARK" ] && MODAL_ARGS="$MODAL_ARGS --benchmark $BENCHMARK"
    [ -n "$LIMIT" ] && MODAL_ARGS="$MODAL_ARGS $LIMIT"
    
    modal run cloud_benchmark.py $MODAL_ARGS
else
    echo "Running locally..."
    echo ""
    
    CMD="python3 run_memory_benchmark.py"
    [ -n "$PROVIDER" ] && CMD="$CMD --provider $PROVIDER"
    [ -n "$BENCHMARK" ] && CMD="$CMD --benchmark $BENCHMARK"
    [ -n "$LIMIT" ] && CMD="$CMD $LIMIT"
    [ -z "$PROVIDER" ] && [ -z "$BENCHMARK" ] && CMD="$CMD --all"
    
    echo "Command: $CMD"
    echo ""
    
    $CMD
fi

echo ""
echo "=============================================="
echo "  COMPLETE"
echo "=============================================="
