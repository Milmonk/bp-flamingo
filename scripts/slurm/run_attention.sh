#!/bin/bash
# ============================================================================
# BP-FLAMINGO: SLURM Job — Attention Visualisation
# ============================================================================
# Usage: sbatch scripts/slurm/run_attention.sh
#
# Extracts cross-attention weights from OpenFlamingo and creates
# heatmap visualisations for selected test images.
# Requires GPU for model inference.
# ============================================================================

#SBATCH --job-name=bp-flamingo-attention
#SBATCH --output=logs/attention_%j.out
#SBATCH --error=logs/attention_%j.err
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00

echo "============================================"
echo "  BP-FLAMINGO: Attention Visualisation Job"
echo "  Job ID:    $SLURM_JOB_ID"
echo "  Node:      $SLURM_NODELIST"
echo "  Start:     $(date)"
echo "============================================"

cd $HOME/projects/bp-flamingo
source venv/bin/activate
mkdir -p logs outputs/visualizations

# Run attention extraction and visualisation
PYTHONPATH=. python3 src/analysis/extract_attention.py

echo ""
echo "--- Generated files ---"
ls -la outputs/visualizations/attention_*

echo ""
echo "============================================"
echo "  Job finished: $(date)"
echo "============================================"
