#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [ ! -d ".venv" ]; then
  uv venv .venv
fi

source .venv/bin/activate
uv pip install -q "manim>=0.20.0" "openai>=2.0.0"

python make_voiceover.py

# Use -q m for a faster 720p preview. Use -q h for 1080p final.
QUALITY="${1:-h}"
manim -q "$QUALITY" --format mp4 memory_strategy_cards.py MemoryStrategyCards

echo
echo "Rendered files:"
find media/videos -name 'MemoryStrategyCards.mp4' -print
