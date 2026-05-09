# BP-FLAMINGO: Documentation Log

## Step 2 — Multi30K Dataset Download and Preparation

**Date:** 2026-03-17  
**Status:** Completed  
**Depends on:** Step 1 (environment setup)

---

### 2.1 Objective

Download and prepare the Multi30K dataset for use in the experimental pipeline.
Convert raw text files and images into a unified JSON format that all pipeline
modules (inference, translation, evaluation) can consume.

---

### 2.2 Dataset Overview: Multi30K

Multi30K is a multilingual extension of the Flickr30K image captioning dataset.
It provides parallel image descriptions in multiple languages.

| Property | Value |
|----------|-------|
| Source | https://github.com/multi30k/dataset |
| Base dataset | Flickr30K (31,014 images) |
| Languages used | English (EN), German (DE), French (FR) |
| Type | Translation Task 1 — human translations of EN captions |
| Splits | train (29,000), val (1,014), test_2016 (1,000) |
| Reference type | Single reference per language per image |

**Why Multi30K?**
- Provides aligned EN/DE/FR captions for the same images
- Established benchmark in multimodal machine translation research
- Translation-based references enable fair comparison of our translate
  baseline (OpenFlamingo EN → MarianMT → target language)
- Covers our target language pairs: EN→DE, EN→FR

---

### 2.3 Data Pipeline

```
[Multi30K GitHub]              [Flickr30K / HuggingFace]
     │                                  │
     │ Text captions (.en/.de/.fr)       │ Images (.jpg)
     │ Image order files (.images)       │
     ▼                                  ▼
  data/multi30k/raw/texts/     data/multi30k/raw/images/
     │                                  │
     └──────────┐    ┌──────────────────┘
                ▼    ▼
        data_processor.py
                │
                ▼
    data/multi30k/processed/
    ├── train.json
    ├── val.json
    ├── test_2016_flickr.json
    └── dataset_summary.json
```

---

### 2.4 Download Process

**Text data:** Downloaded from Multi30K GitHub repository
(`multi30k/dataset/master/data/task1/raw/`). Compressed `.gz` files
decompressed on the fly via `curl | gunzip`.

**Image order files:** Downloaded from the same repository
(`data/task1/image_splits/`). These map each caption line to a Flickr image.

**Images:** Downloaded from HuggingFace (`nlphuji/flickr30k`) as
`flickr30k-images.zip` (~4.4 GB). Required steps:
1. Create HuggingFace account and accept dataset license
2. Generate access token at https://huggingface.co/settings/tokens
3. Login via Python: `from huggingface_hub import login; login(token='...')`
4. Download via Python: `hf_hub_download(repo_id='nlphuji/flickr30k', ...)`
5. Unzip and move images to `data/multi30k/raw/images/`

Note: `huggingface-cli` command was not available in PATH on HPC Perun
despite `huggingface_hub` being installed. All HuggingFace operations were
done via the Python API instead.

---

### 2.5 Processed Data Format

Each JSON file contains a list of entries:

```json
{
    "id": 0,
    "image_filename": "1000092795.jpg",
    "image_path": "data/multi30k/raw/images/1000092795.jpg",
    "image_exists": true,
    "captions": {
        "en": "Two young guys with shaggy hair looking at their hands ...",
        "de": "Zwei junge Männer mit zotteligem Haar ...",
        "fr": "Deux jeunes hommes aux cheveux hirsutes ..."
    }
}
```

---

### 2.6 Dataset Statistics (Verified)

| Split | Samples | Unique images | Images found | EN avg words | DE avg words | FR avg words |
|-------|---------|---------------|--------------|-------------|-------------|-------------|
| train | 29,000 | 29,000 | 29,000 (100%) | 11.9 | 11.1 | 12.5 |
| val | 1,014 | 1,014 | 1,014 (100%) | 12.0 | 11.4 | 12.5 |
| test_2016_flickr | 1,000 | 1,000 | 1,000 (100%) | 11.9 | 10.9 | 12.4 |
| **Total** | **31,014** | **31,014** | **31,014 (100%)** | | | |

Observations:
- French captions are slightly longer on average than English (~12.5 vs ~11.9)
- German captions are slightly shorter (~11.0 vs ~11.9)
- All images successfully matched to all three language captions

---

### 2.7 Data Loader API

The `Multi30KLoader` class provides a unified interface for all modules:

```python
from src.utils.data_loader import Multi30KLoader

loader = Multi30KLoader()

# Load evaluation split
test = loader.load_split("test_2016_flickr", require_images=True)

# Get few-shot examples for OpenFlamingo
few_shot = loader.get_few_shot_examples(n=4, seed=42)

# Load an image
image = loader.load_image(test[0])

# Get all reference captions in German
de_refs = loader.get_references("test_2016_flickr", "de")
```

Important: Run Python scripts from project root with `PYTHONPATH=.` or
use `python3 -m src.utils.data_loader` to ensure module imports work.

---

### 2.8 Issues Encountered and Solutions

| # | Issue | Cause | Solution |
|---|-------|-------|----------|
| 1 | HuggingFace `load_dataset()` failed | `nlphuji/flickr30k` uses deprecated dataset script format | Downloaded `flickr30k-images.zip` directly via `hf_hub_download()` |
| 2 | `pip install huggingface-cli` failed | Not a standalone package; CLI is part of `huggingface_hub` | Used Python API directly: `from huggingface_hub import login` |
| 3 | `huggingface-cli` command not found | Binary not in PATH on HPC Perun | Used `hf_hub_download()` Python API instead |
| 4 | `ModuleNotFoundError: No module named 'src'` | Python doesn't know about project package structure | Run with `PYTHONPATH=.` or `python3 -m src.utils.data_loader` |

---

### 2.9 Execution Instructions (Reproducibility)

```bash
cd ~/projects/bp-flamingo
source venv/bin/activate

# 1. Download text data and image order files
bash scripts/download_multi30k.sh

# 2. Login to HuggingFace (one-time, requires token)
python3 -c "from huggingface_hub import login; login(token='YOUR_TOKEN')"

# 3. Download images (~4.4 GB)
bash scripts/download_images.sh
# OR manually:
#   python3 -c "from huggingface_hub import hf_hub_download; ..."
#   unzip data/multi30k/raw/flickr30k-images.zip ...

# 4. Process into unified JSON
python3 src/utils/data_processor.py

# 5. Verify
PYTHONPATH=. python3 src/utils/data_loader.py
```

---

### 2.10 Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Data format | JSON | Human-readable, easy to inspect and debug |
| Image tracking | `image_exists` field | Pipeline can work even with missing images |
| Caching | In-memory per split | Avoids re-reading JSON for repeated access |
| Few-shot source | Train split | Standard practice; avoids data leakage |
| Test split | test_2016_flickr | Most widely used Multi30K test set |
| Image download | HuggingFace zip | Reliable, versioned, single download |

---

**Next step:** Step 3 — OpenFlamingo Inference Module