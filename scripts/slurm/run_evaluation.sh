#!/bin/bash
# ============================================================================
# BP-FLAMINGO: SLURM Job — Evaluation
# ============================================================================
# Usage: sbatch scripts/slurm/run_evaluation.sh
#
# Computes BLEU, chrF, METEOR, BERTScore for all 5 experiments.
# BERTScore requires GPU for reasonable speed.
# ============================================================================

#SBATCH --job-name=bp-flamingo-eval
#SBATCH --output=logs/evaluation_%j.out
#SBATCH --error=logs/evaluation_%j.err
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00

echo "============================================"
echo "  BP-FLAMINGO: Evaluation Job"
echo "  Job ID:    $SLURM_JOB_ID"
echo "  Node:      $SLURM_NODELIST"
echo "  Start:     $(date)"
echo "============================================"

cd $HOME/projects/bp-flamingo
source venv/bin/activate
mkdir -p logs outputs/metrics

# Run full evaluation (all 5 experiments, all metrics including BERTScore)
PYTHONPATH=. python3 src/evaluation/evaluate.py --all

echo ""
echo "--- Output files ---"
ls -la outputs/metrics/

echo ""
echo "============================================"
echo "  Job finished: $(date)"
echo "============================================"
