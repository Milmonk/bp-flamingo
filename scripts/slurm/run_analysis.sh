#!/bin/bash
# ============================================================================
# BP-FLAMINGO: SLURM Job — Analysis & Visualisation
# ============================================================================
# Usage: sbatch scripts/slurm/run_analysis.sh
#
# Runs:
#   1. Architecture and pipeline flowcharts
#   2. Results comparison charts
#   3. Tokenisation analysis (needs tokenizer download)
# ============================================================================

#SBATCH --job-name=bp-flamingo-analysis
#SBATCH --output=logs/analysis_%j.out
#SBATCH --error=logs/analysis_%j.err
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00

echo "============================================"
echo "  BP-FLAMINGO: Analysis & Visualisation Job"
echo "  Job ID:    $SLURM_JOB_ID"
echo "  Node:      $SLURM_NODELIST"
echo "  Start:     $(date)"
echo "============================================"

cd $HOME/projects/bp-flamingo
source venv/bin/activate
mkdir -p logs outputs/visualizations

# 1. Create all visualizations (flowcharts + charts)
echo ""
echo "=== Creating Visualisations ==="
PYTHONPATH=. python3 src/analysis/attention_analysis.py

# 2. Run tokenization analysis
echo ""
echo "=== Running Tokenisation Analysis ==="
PYTHONPATH=. python3 src/analysis/tokenization_analysis.py

echo ""
echo "--- Output files ---"
ls -la outputs/visualizations/

echo ""
echo "============================================"
echo "  Job finished: $(date)"
echo "============================================"
