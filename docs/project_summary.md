# BP-FLAMINGO: Project Summary

## Multimodal Image Captioning with OpenFlamingo: Multilingual Evaluation

---

### Overview

This project implements and evaluates a multimodal image captioning pipeline
using OpenFlamingo, with a focus on multilingual capabilities. The pipeline
generates image captions in English, German, and French, and compares two
approaches to multilingual caption generation:

1. **Direct multilingual generation** — prompting OpenFlamingo to generate
   in the target language via few-shot examples
2. **Translate baseline** — generating English captions, then translating
   using MarianMT

---

### Pipeline Architecture

```
                    ┌─────────────────────────────────────────────────────┐
                    │              OpenFlamingo 3B                        │
                    │         (ViT-L/14 + MPT-1B)                        │
                    └───────────┬──────────┬──────────┬──────────────────┘
                                │          │          │
                     EN prompt  │  DE prompt│  FR prompt│
                                ▼          ▼          ▼
                          EN captions  DE captions  FR captions
                          (1000)       (1000)       (1000)
                                │
                    ┌───────────┴───────────┐
                    │       MarianMT        │
                    │  EN→DE       EN→FR    │
                    └───┬──────────────┬────┘
                        ▼              ▼
                   DE translations  FR translations
                   (1000)          (1000)
                                │
                    ┌───────────┴───────────┐
                    │     Evaluation         │
                    │  BLEU, chrF, METEOR,   │
                    │  BERTScore, LangDetect │
                    └───────────────────────┘
```

---

### Results

| Experiment | BLEU | chrF | METEOR | BERTScore | Lang % |
|------------|------|------|--------|-----------|--------|
| EN-only | 6.43 | 23.65 | 27.69 | 91.33 | 94.5% |
| Direct DE | 1.11 | 16.67 | 12.01 | 74.40 | 63.7% |
| Direct FR | 2.25 | 16.62 | 15.76 | 75.30 | 82.8% |
| Translate DE | 3.95 | 25.39 | 23.09 | 75.97 | 99.8% |
| Translate FR | 5.67 | 24.97 | 24.47 | 78.28 | 98.8% |

**Main conclusion:** The translate baseline significantly outperforms direct
multilingual generation on all metrics. The improvement is 2.5-3.6x on BLEU
and ~50% on chrF. Language accuracy improves from 64-83% to over 99%.

---

### Technical Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| Vision-language model | OpenFlamingo 3B | GitHub HEAD |
| Vision encoder | ViT-L/14 (CLIP) | OpenAI |
| Language model | MPT-1B | RedPajama 200B |
| Translation | MarianMT (OPUS-MT) | Helsinki-NLP |
| Dataset | Multi30K | Task 1 (translations) |
| Evaluation | sacrebleu, NLTK, bert_score | Latest |
| GPU | NVIDIA H200 | 150.1 GB |
| HPC | Perun | Python 3.11 |

---

### Project Steps

| Step | Description | Status | Time |
|------|-------------|--------|------|
| 1 | Environment setup | ✅ | — |
| 2 | Multi30K dataset preparation | ✅ | ~30 min (download) |
| 3 | OpenFlamingo inference (3 modes) | ✅ | 99.3 min |
| 4 | MarianMT translation (EN→DE, EN→FR) | ✅ | 0.5 min |
| 5 | Evaluation (4 metrics, 5 experiments) | ✅ | 0.4 min |

---

### File Structure

```
bp-flamingo/
├── configs/config.yaml                          # Central configuration
├── data/multi30k/                               # Dataset (31,014 images + captions)
├── src/
│   ├── inference/flamingo_inference.py           # OpenFlamingo wrapper
│   ├── inference/run_all_inference.py            # Run all 3 modes
│   ├── translation/translate.py                  # MarianMT wrapper
│   ├── evaluation/evaluate.py                    # Metrics computation
│   └── utils/                                    # Config, data loader
├── outputs/
│   ├── captions/                                 # 3 JSON files (EN, DE, FR)
│   ├── translations/                             # 2 JSON files (EN→DE, EN→FR)
│   └── metrics/                                  # Evaluation results + CSV
├── scripts/slurm/                                # HPC job scripts
├── docs/                                         # Step-by-step documentation
│   ├── step_01_project_setup.md
│   ├── step_02_dataset_preparation.md
│   ├── step_03_openflamingo_inference.md
│   ├── step_04_translation.md
│   ├── step_05_evaluation.md
│   └── project_summary.md
└── requirements.txt
```

---

### Documentation for Thesis

Each step is documented in `docs/` with:
- Objective and methodology
- Actual results and statistics
- Issues encountered and solutions
- Design decisions with rationale
- Reproducibility instructions

This documentation can be directly used as source material for the
Implementation and Results chapters of the bachelor thesis.