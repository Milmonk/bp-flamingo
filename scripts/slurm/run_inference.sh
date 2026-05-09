#!/bin/bash
# ============================================================================
# BP-FLAMINGO: SLURM Job — OpenFlamingo Inference
# ============================================================================
# Usage: sbatch scripts/slurm/run_inference.sh
#
# Runs all three inference modes (EN, DE, FR) on a GPU node.
# Modify #SBATCH parameters below to match your HPC configuration.
# ============================================================================

#SBATCH --job-name=bp-flamingo-inference
#SBATCH --output=logs/inference_%j.out
#SBATCH --error=logs/inference_%j.err
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00

# ---- MODIFY THESE IF NEEDED ----
# Uncomment and change partition if yours is different:
# #SBATCH --partition=dgx
# #SBATCH --account=your_account

# For specific GPU type (uncomment if needed):
# #SBATCH --gres=gpu:A100:1
# #SBATCH --gres=gpu:V100:1
# ---------------------------------

echo "============================================"
echo "  BP-FLAMINGO: Inference Job"
echo "  Job ID:    $SLURM_JOB_ID"
echo "  Node:      $SLURM_NODELIST"
echo "  GPU:       $CUDA_VISIBLE_DEVICES"
echo "  Start:     $(date)"
echo "============================================"

# Navigate to project root
cd $HOME/projects/bp-flamingo

# Activate environment
source venv/bin/activate

# Ensure output directories exist
mkdir -p logs outputs/captions

# Print GPU info
python3 -c "
import torch
print(f'  PyTorch:  {torch.__version__}')
print(f'  CUDA:     {torch.version.cuda}')
print(f'  GPU:      {torch.cuda.get_device_name(0)}')
print(f'  Memory:   {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

echo ""

# --- Run all inference experiments ---
# To run a quick test first (10 samples), uncomment:
# PYTHONPATH=. python3 src/inference/run_all_inference.py --max-samples 10

# Full run (all 1000 test samples × 3 languages):
PYTHONPATH=. python3 src/inference/run_all_inference.py

echo ""
echo "============================================"
echo "  Job finished: $(date)"
echo "============================================"
