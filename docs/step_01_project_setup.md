# BP-FLAMINGO: Documentation Log

## Step 1 — Project Setup and Environment Configuration

**Date:** 2026-03-16  
**Status:** Completed  
**HPC:** Perun (login01)  
**Python:** 3.11 (/usr/bin/python3.11)

---

### 1.1 Objective

Prepare a reproducible project structure and Python environment on HPC Perun
for multimodal image captioning experiments using OpenFlamingo, MarianMT,
and the Multi30K dataset.

---

### 1.2 Project Structure

```
bp-flamingo/
├── configs/              # YAML configuration files
│   └── config.yaml       # Central experiment config
├── data/                 # Datasets
│   └── multi30k/
│       ├── raw/          # Original downloaded data
│       └── processed/    # Preprocessed/cleaned data
├── models/               # Downloaded model checkpoints
│   ├── openflamingo/
│   └── marianmt/
├── src/                  # Source code (Python package)
│   ├── inference/        # OpenFlamingo caption generation
│   ├── translation/      # MarianMT EN→DE/FR translation
│   ├── evaluation/       # Metrics computation
│   └── utils/            # Shared utilities (config loader, logging)
├── outputs/              # Experiment outputs
│   ├── captions/         # Generated captions (JSON)
│   ├── translations/     # Translated captions (JSON)
│   ├── metrics/          # Evaluation results (JSON/CSV)
│   └── visualizations/   # Charts, attention maps, flowcharts
├── notebooks/            # Jupyter notebooks for analysis
├── docs/                 # This documentation
├── scripts/              # Shell/SLURM scripts
│   └── slurm/            # HPC job submission scripts
├── logs/                 # Runtime logs
├── requirements.txt      # Python dependencies (excl. torch, OpenFlamingo)
└── venv/                 # Virtual environment (not in git)
```

---

### 1.3 Environment Details

- **HPC:** Perun, login node `login01`
- **OS:** Linux (manylinux)
- **Python:** 3.11 (`/usr/bin/python3.11`)
- **Virtual environment:** venv (standard library)
- **Module system:** Lmod available but no Python modules installed

---

### 1.4 Installation Order and Dependency Resolution

The installation required a specific order due to dependency conflicts.
Three packages needed special handling:

**Step 1 — PyTorch + torchvision:**

```bash
pip install torch torchvision
```

Must be installed together on one line so pip resolves their exact version
match (e.g., torch 2.10.0 ↔ torchvision 0.25.0). Installing them separately
or with loose version ranges in requirements.txt caused pip to fail with
`ResolutionImpossible`.

**Step 2 — requirements.txt (all other dependencies):**

```bash
pip install -r requirements.txt
```

Contains all other packages: open-clip-torch, transformers, sentencepiece,
evaluation metrics, data handling, visualization, etc.
Torch and torchvision are NOT in this file.

**Step 3 — OpenFlamingo from GitHub:**

```bash
pip install git+https://github.com/mlfoundations/open_flamingo.git --no-deps
pip install einops_exts
```

The PyPI version of open-flamingo (2.0.0, 2.0.1) has strict pinned
dependencies (e.g., `sentencepiece==0.1.98`, `torch==2.0.1`) that conflict
with current versions. Installing from GitHub with `--no-deps` avoids these
conflicts. The `einops_exts` package must be installed separately as it was
skipped by `--no-deps`.

Note: pip may show a warning about torch version incompatibility. This is
only a metadata warning from the old PyPI package — the GitHub version of
OpenFlamingo works correctly with newer PyTorch.

---

### 1.5 Issues Encountered and Solutions

| # | Issue | Cause | Solution |
|---|-------|-------|----------|
| 1 | `pip install` failed with `ResolutionImpossible` for torch/torchvision | Default Python was 3.9; PyTorch 2.6+ requires Python 3.10+ | Discovered `/usr/bin/python3.11` on the system, created venv with it |
| 2 | `ResolutionImpossible` for torch + torchvision in requirements.txt | torchvision requires exact torch version; pip cannot resolve both from loose ranges | Moved torch/torchvision out of requirements.txt; install together with `pip install torch torchvision` |
| 3 | `ResolutionImpossible` for open-flamingo + sentencepiece | PyPI open-flamingo pins `sentencepiece==0.1.98`, we need `>=0.1.99` | Install from GitHub with `--no-deps` |
| 4 | `ModuleNotFoundError: No module named 'einops_exts'` | `--no-deps` skipped this small dependency | `pip install einops_exts` |
| 5 | `bash: !': event not found` | Bash interprets `!` inside double quotes as history expansion | Use single quotes: `python3 -c 'import open_flamingo; print("OK")'` |

---

### 1.6 Installed Package Versions

Key packages (as installed on 2026-03-16):

| Package | Version | Notes |
|---------|---------|-------|
| Python | 3.11.x | System-installed at `/usr/bin/python3.11` |
| torch | 2.10.0 | Latest compatible with Python 3.11 |
| torchvision | 0.25.0 | Matches torch 2.10.0 |
| open-flamingo | GitHub HEAD | Installed from `mlfoundations/open_flamingo` |
| open-clip-torch | latest | Vision encoder for OpenFlamingo |
| transformers | latest | HuggingFace, used for MarianMT |
| sentencepiece | latest | Tokenizer for MarianMT |
| sacrebleu | latest | BLEU, chrF metrics |
| nltk | latest | METEOR metric |
| bert-score | latest | BERTScore metric |
| langdetect | latest | Language detection of outputs |

---

### 1.7 Configuration

All experiment parameters are centralized in `configs/config.yaml`:

- **Model selection:** OpenFlamingo 3B (default) or 9B
- **Generation hyperparameters:** beam search, temperature, top-k/p
- **Translation models:** MarianMT EN→DE, EN→FR
- **Evaluation metrics:** BLEU, chrF, METEOR, BERTScore
- **Three experiment modes:**
  - `en_only` — OpenFlamingo → English caption → evaluate vs EN references
  - `direct_multilingual` — OpenFlamingo → DE/FR caption directly
  - `translate_baseline` — OpenFlamingo → EN → MarianMT → DE/FR

---

### 1.8 Setup Instructions (Reproducibility)

To reproduce this environment from scratch on HPC Perun:

```bash
# 1. Copy project to HPC
cd ~/projects  # or your workspace
# place bp-flamingo/ directory here

# 2. Run automated setup
cd bp-flamingo
bash scripts/setup_project.sh

# 3. If einops_exts is missing (check error output):
source venv/bin/activate
pip install einops_exts

# 4. Verify
python3 -c 'import open_flamingo; print("OK")'

# 5. For future sessions:
cd ~/projects/bp-flamingo
source venv/bin/activate
```

---

### 1.9 Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Environment manager | venv | Standard, lightweight, no extra installs needed |
| Python version | 3.11 | Latest available on Perun; supports current PyTorch |
| OpenFlamingo source | GitHub | PyPI version has outdated pinned dependencies |
| Config format | YAML | Human-readable, supports nested structures |
| Output format | JSON/JSONL | Easy to parse, language-agnostic, appendable |
| Dataset | Multi30K | EN/DE/FR references, established benchmark |
| Translation model | MarianMT | Lightweight, good quality for high-resource pairs |
| Default model size | 3B | Fits single GPU; 9B available for comparison |

---

**Next step:** Step 2 — Download and prepare Multi30K dataset