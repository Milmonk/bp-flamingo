#!/bin/bash
# ============================================================================
# BP-FLAMINGO: Project Setup Script for HPC Perun
# ============================================================================
# Usage: cd ~/bp-flamingo && bash scripts/setup_project.sh
#
# Installation order (to avoid dependency conflicts):
#   1. torch + torchvision (need exact version match)
#   2. requirements.txt (all other dependencies)
#   3. OpenFlamingo from GitHub (PyPI version has strict pinned deps)
# ============================================================================

set -e  # Exit on error

PYTHON_BIN="/usr/bin/python3.11"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "============================================"
echo "  BP-FLAMINGO: Project Setup"
echo "  Project root: $PROJECT_ROOT"
echo "============================================"

# --- 0. Check Python 3.11 exists ---
if [ ! -f "$PYTHON_BIN" ]; then
    echo "ERROR: Python 3.11 not found at $PYTHON_BIN"
    echo "Check available versions: ls /usr/bin/python*"
    exit 1
fi

echo "Using Python: $($PYTHON_BIN --version)"
echo ""

# --- 1. Create project directory structure ---
echo "[1/7] Creating project structure..."

cd "$PROJECT_ROOT"

directories=(
    "configs"
    "data/multi30k/raw"
    "data/multi30k/processed"
    "models/openflamingo"
    "models/marianmt"
    "src/inference"
    "src/translation"
    "src/evaluation"
    "src/utils"
    "outputs/captions"
    "outputs/translations"
    "outputs/metrics"
    "outputs/visualizations"
    "notebooks"
    "docs"
    "scripts/slurm"
    "logs"
)

for dir in "${directories[@]}"; do
    mkdir -p "$dir"
done

touch src/__init__.py
touch src/inference/__init__.py
touch src/translation/__init__.py
touch src/evaluation/__init__.py
touch src/utils/__init__.py

echo "  Done — project structure created."

# --- 2. Create virtual environment with Python 3.11 ---
echo ""
echo "[2/7] Setting up Python 3.11 virtual environment..."

if [ -d "venv" ]; then
    echo "  Removing old venv..."
    rm -rf venv
fi

$PYTHON_BIN -m venv venv
source venv/bin/activate

echo "  Done — venv created with $(python3 --version)"

# --- 3. Install PyTorch + torchvision ---
echo ""
echo "[3/7] Installing PyTorch and torchvision..."

pip install --upgrade pip setuptools wheel --quiet
pip install torch torchvision

echo ""
echo "  Done — PyTorch installed."

# --- 4. Install remaining dependencies from requirements.txt ---
echo ""
echo "[4/7] Installing dependencies from requirements.txt..."

pip install -r requirements.txt

echo ""
echo "  Done — dependencies installed."

# --- 5. Install OpenFlamingo from GitHub ---
echo ""
echo "[5/7] Installing OpenFlamingo from GitHub..."

pip install git+https://github.com/mlfoundations/open_flamingo.git --no-deps
pip install einops_exts --quiet

echo ""
echo "  Done — OpenFlamingo installed from GitHub (+ einops_exts)."

# --- 6. Download NLTK data (needed for METEOR) ---
echo ""
echo "[6/7] Downloading NLTK data..."

python3 -c "
import nltk
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('omw-1.4', quiet=True)
print('  Done — NLTK data downloaded.')
"

# --- 7. Verify installation ---
echo ""
echo "[7/7] Verifying installation..."
echo ""

python3 -c "
import sys
import torch
import transformers
import open_clip
import open_flamingo
import sacrebleu
import nltk
import bert_score
import langdetect
import pandas
import matplotlib

print('  Python version:    ', sys.version.split()[0])
print('  PyTorch version:   ', torch.__version__)
print('  CUDA available:    ', torch.cuda.is_available())
if torch.cuda.is_available():
    print('  GPU:               ', torch.cuda.get_device_name(0))
    print('  GPU memory:        ', f'{torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
else:
    print('  (GPU check skipped — run on compute node via SLURM)')
print('  Transformers:      ', transformers.__version__)
print('  OpenCLIP:          ', open_clip.__version__)
print('  OpenFlamingo:      ', open_flamingo.__version__ if hasattr(open_flamingo, '__version__') else 'installed')
print('  SacreBLEU:         ', sacrebleu.__version__)
print('  BERTScore:         ', bert_score.__version__)
print()
print('  All packages OK!')
"

echo ""
echo "============================================"
echo "  Setup complete!"
echo ""
echo "  To activate environment in future sessions:"
echo "    cd $PROJECT_ROOT"
echo "    source venv/bin/activate"
echo "============================================"
