#!/usr/bin/env bash
# Run the full experiment pipeline.
# Usage: bash scripts/run_pipeline.sh
# (or use: pixi run pipeline)

set -euo pipefail

echo "=============================================="
echo "  MVR Option 1: Prompt Engineering Comparison"
echo "=============================================="
echo ""

cd "$(dirname "$0")/.."

echo "Step 1/4: Preparing data..."
python src/01_prepare_data.py
echo ""

echo "Step 2/4: Generating feedback (4 strategies × N answers × 3 runs)..."
python src/02_generate.py
echo ""

echo "Step 3/4: Evaluating feedback (G-Eval, 5 dimensions)..."
python src/03_evaluate.py
echo ""

echo "Step 4/4: Analysing results..."
python src/04_analyse.py
echo ""

echo "=============================================="
echo "  ✅ Pipeline complete!"
echo "  Results: results/"
echo "  Figures: results/figures/"
echo "=============================================="
