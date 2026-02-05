#!/bin/bash
# Download benchmark datasets for PIE evaluation suite

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== PIE Benchmark Dataset Downloader ==="
echo ""

# LoCoMo
echo "📥 Downloading LoCoMo dataset..."
mkdir -p "$SCRIPT_DIR/locomo/data"
if [ ! -f "$SCRIPT_DIR/locomo/data/locomo10.json" ]; then
    curl -L -o "$SCRIPT_DIR/locomo/data/locomo10.json" \
        "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
    echo "✅ LoCoMo downloaded"
else
    echo "✅ LoCoMo already exists"
fi

# LongMemEval - check if it exists
echo ""
echo "📥 Checking LongMemEval dataset..."
if [ -f "$SCRIPT_DIR/longmemeval/data/longmemeval_s_cleaned.json" ]; then
    echo "✅ LongMemEval already exists"
else
    echo "⚠️  LongMemEval not found"
    echo "   Please download from: https://github.com/xiaowu0162/LongMemEval"
    echo "   Place in: $SCRIPT_DIR/longmemeval/data/longmemeval_s_cleaned.json"
fi

# MSC - provide instructions
echo ""
echo "📥 MSC Dataset Instructions..."
mkdir -p "$SCRIPT_DIR/msc/data"
if [ -f "$SCRIPT_DIR/msc/data/msc_valid.jsonl" ]; then
    echo "✅ MSC already exists"
else
    echo "⚠️  MSC not found"
    echo "   To download, install ParlAI and run:"
    echo "   pip install parlai"
    echo "   parlai display_data -t msc -dt valid > msc_raw.txt"
    echo "   Then convert to JSONL format"
fi

echo ""
echo "=== Dataset Status ==="
echo ""

# Summary
for bench in longmemeval locomo msc; do
    data_dir="$SCRIPT_DIR/$bench/data"
    count=$(find "$data_dir" -name "*.json*" 2>/dev/null | wc -l | tr -d ' ')
    if [ "$count" -gt 0 ]; then
        echo "✅ $bench: $count data file(s)"
    else
        echo "❌ $bench: No data files found"
    fi
done

echo ""
echo "Run 'python -m benchmarks.eval_harness --subset 5' to test setup"
