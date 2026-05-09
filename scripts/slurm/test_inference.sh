#!/bin/bash
# ============================================================================
# BP-FLAMINGO: Quick Inference Test (GPU node)
# ============================================================================
# Usage: sbatch scripts/slurm/test_inference.sh
#
# Runs inference on just 3 samples to verify everything works.
# Check logs/test_inference_<jobid>.out for results.
# ============================================================================

#SBATCH --job-name=bp-flamingo-test
#SBATCH --output=logs/test_inference_%j.out
#SBATCH --error=logs/test_inference_%j.err
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00

echo "============================================"
echo "  BP-FLAMINGO: Quick Inference Test"
echo "  Job ID: $SLURM_JOB_ID"
echo "  Node:   $SLURM_NODELIST"
echo "  Start:  $(date)"
echo "============================================"

cd $HOME/projects/bp-flamingo
source venv/bin/activate
mkdir -p logs outputs/captions

# Print GPU info
python3 -c "
import torch
print(f'  GPU:    {torch.cuda.get_device_name(0)}')
print(f'  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

# Run with just 3 samples — enough to verify model loads and generates
echo ""
echo "--- Testing EN-only mode (3 samples) ---"
PYTHONPATH=. python3 src/inference/flamingo_inference.py \
    --language en \
    --max-samples 3 \
    --suffix test

echo ""
echo "--- Testing Direct DE mode (3 samples) ---"
PYTHONPATH=. python3 src/inference/flamingo_inference.py \
    --language de \
    --max-samples 3 \
    --suffix test

echo ""
echo "--- Testing Direct FR mode (3 samples) ---"
PYTHONPATH=. python3 src/inference/flamingo_inference.py \
    --language fr \
    --max-samples 3 \
    --suffix test

echo ""
echo "--- Checking outputs ---"
echo "Generated files:"
ls -la outputs/captions/*test* 2>/dev/null || echo "  No output files found!"

echo ""
echo "Sample output (EN):"
python3 -c "
import json
try:
    with open('outputs/captions/captions_en_only_test_2016_flickr_test.json') as f:
        data = json.load(f)
    for r in data['results'][:3]:
        print(f'  Image: {r[\"image_filename\"]}')
        print(f'  Generated: {r[\"generated_caption\"]}')
        print(f'  Reference: {r[\"reference_caption\"]}')
        print()
except Exception as e:
    print(f'  Could not read output: {e}')
"

echo ""
echo "============================================"
echo "  Test finished: $(date)"
echo "============================================"
