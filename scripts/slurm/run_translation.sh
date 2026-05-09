#!/bin/bash
# ============================================================================
# BP-FLAMINGO: SLURM Job — MarianMT Translation
# ============================================================================
# Usage: sbatch scripts/slurm/run_translation.sh
#
# Translates EN captions from OpenFlamingo → DE and FR using MarianMT.
# MarianMT is small (~300M params), so this runs much faster than inference.
# ============================================================================

#SBATCH --job-name=bp-flamingo-translate
#SBATCH --output=logs/translation_%j.out
#SBATCH --error=logs/translation_%j.err
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00

echo "============================================"
echo "  BP-FLAMINGO: MarianMT Translation Job"
echo "  Job ID:    $SLURM_JOB_ID"
echo "  Node:      $SLURM_NODELIST"
echo "  Start:     $(date)"
echo "============================================"

cd $HOME/projects/bp-flamingo
source venv/bin/activate
mkdir -p logs outputs/translations

# Translate EN captions → DE and FR
PYTHONPATH=. python3 src/translation/translate.py --all --batch-size 64

echo ""
echo "--- Output files ---"
ls -la outputs/translations/

echo ""
echo "--- Sample translations ---"
python3 -c "
import json
for lang in ['de', 'fr']:
    path = f'outputs/translations/translations_en_{lang}_test_2016_flickr.json'
    try:
        with open(path) as f:
            data = json.load(f)
        print(f'=== EN → {lang.upper()} ({data[\"metadata\"][\"num_samples\"]} samples) ===')
        for r in data['results'][:3]:
            print(f'  EN source:    {r[\"source_caption\"]}')
            print(f'  Translated:   {r[\"translated_caption\"]}')
            print(f'  Reference:    {r[\"reference_caption\"]}')
            print()
    except Exception as e:
        print(f'  Error reading {lang}: {e}')
"

echo ""
echo "============================================"
echo "  Job finished: $(date)"
echo "============================================"
