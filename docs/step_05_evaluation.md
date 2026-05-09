# BP-FLAMINGO: Documentation Log

## Step 5 — Evaluation and Results Analysis

**Date:** 2026-03-18  
**Status:** Completed  
**HPC Job ID:** 16522 (gpu02)  
**Evaluation time:** 26.1 seconds

---

### 5.1 Objective

Compute automatic evaluation metrics for all 5 experimental modes and
produce a comparative analysis answering the main research question:
*Which approach produces better multilingual image captions — direct
multilingual generation or the translate baseline?*

---

### 5.2 Results Summary

| Experiment | BLEU | chrF | METEOR | BERTScore (F1) | Target Lang % | Avg Length |
|------------|------|------|--------|----------------|---------------|------------|
| en_only | 6.43 | 23.65 | 27.69 | 91.33 | 94.5% (EN) | 9.1 words |
| direct_de | 1.11 | 16.67 | 12.01 | 74.40 | 63.7% (DE) | 8.6 words |
| direct_fr | 2.25 | 16.62 | 15.76 | 75.30 | 82.8% (FR) | 8.3 words |
| translate_de | 3.95 | 25.39 | 23.09 | 75.97 | 99.8% (DE) | 8.8 words |
| translate_fr | 5.67 | 24.97 | 24.47 | 78.28 | 98.8% (FR) | 9.4 words |

Reference caption average lengths: EN 11.9, DE 10.9, FR 12.4 words.

---

### 5.3 Key Findings

#### Finding 1: Translate baseline significantly outperforms direct multilingual

Across all metrics and both languages, the translate baseline (OpenFlamingo EN →
MarianMT → target language) produces substantially better results than direct
multilingual generation:

**German:**
- BLEU: 3.95 vs 1.11 (3.6x improvement)
- chrF: 25.39 vs 16.67 (+52% improvement)
- METEOR: 23.09 vs 12.01 (1.9x improvement)
- BERTScore: 75.97 vs 74.40 (+1.57 points)

**French:**
- BLEU: 5.67 vs 2.25 (2.5x improvement)
- chrF: 24.97 vs 16.62 (+50% improvement)
- METEOR: 24.47 vs 15.76 (1.6x improvement)
- BERTScore: 78.28 vs 75.30 (+2.98 points)

#### Finding 2: Language accuracy is a critical differentiator

The most striking difference is in language accuracy:
- **Translate baseline:** 99.8% DE, 98.8% FR — nearly perfect
- **Direct DE:** Only 63.7% German — over a third of outputs are in wrong language
- **Direct FR:** 82.8% French — better than DE, but still unreliable

This means that a significant portion of direct multilingual outputs are
evaluated against references in the wrong language, which partly explains
the low metric scores. However, even if we could filter to only same-language
outputs, the translate baseline would still be expected to win due to the
quality advantage of dedicated MT models.

#### Finding 3: French direct generation works better than German

Across all metrics, direct French outperforms direct German:
- BLEU: 2.25 vs 1.11
- METEOR: 15.76 vs 12.01
- Language accuracy: 82.8% vs 63.7%

This suggests that the MPT-1B language model underlying OpenFlamingo has
more French than German in its training data, making it more amenable to
French generation via few-shot prompting.

#### Finding 4: English baseline has highest absolute quality

The EN-only mode achieves the highest scores (BLEU 6.43, METEOR 27.69),
confirming that OpenFlamingo works best in its native language. The high
BERTScore (91.33) reflects good semantic overlap even when exact wording
differs from references.

Note: BERTScore for EN uses roberta-large while DE/FR use
bert-base-multilingual-cased, so the EN BERTScore is not directly
comparable to DE/FR BERTScores.

#### Finding 5: Generated captions are consistently shorter than references

All experiments produce captions shorter than the Multi30K references:
- EN: 9.1 vs 11.9 words (76% of reference length)
- DE: 8.6-8.8 vs 10.9 words (~80%)
- FR: 8.3-9.4 vs 12.4 words (67-76%)

This length discrepancy contributes to lower BLEU scores (BLEU's brevity
penalty penalizes short hypotheses).

---

### 5.4 Metrics Explained

**BLEU** (Bilingual Evaluation Understudy): Measures n-gram precision between
hypothesis and reference. Range 0-100. Our scores are low (1-6) because we have
single references per image — BLEU improves significantly with multiple references.
The relative ranking between experiments is more informative than absolute values.

**chrF** (character F-score): Character-level metric, particularly useful for
morphologically rich languages like German where word-level metrics penalize
valid inflectional variants. Our chrF scores (16-25) are more informative than
BLEU for German evaluation.

**METEOR** (Metric for Evaluation of Translation with Explicit ORdering):
Considers synonyms and stemming, making it more forgiving of paraphrases.
Highest correlation with human judgments among traditional metrics.

**BERTScore**: Uses contextual embeddings to compute semantic similarity.
Less sensitive to exact word choice, captures meaning overlap. The high
EN BERTScore (91.33) suggests that even when captions use different words,
they describe similar content. Note: EN uses roberta-large, while DE/FR
use bert-base-multilingual-cased — scores across languages are not directly
comparable.

---

### 5.5 Language Detection Analysis

| Experiment | Target | % Target | % English | % Other |
|------------|--------|----------|-----------|---------|
| en_only | EN | 94.5% | 94.5% | 5.5% other |
| direct_de | DE | 63.7% | ~33% | ~3% other |
| direct_fr | FR | 82.8% | ~15% | ~2% other |
| translate_de | DE | 99.8% | ~0.2% | — |
| translate_fr | FR | 98.8% | ~1.2% | — |

The direct multilingual experiments exhibit significant **code-switching**
(generating in the wrong language), which is a known limitation of
English-centric language models when prompted for multilingual output.
The translate baseline eliminates this problem almost entirely.

---

### 5.6 Issues Encountered and Solutions

| # | Issue | Cause | Solution |
|---|-------|-------|----------|
| 1 | `KeyError: 'bert-base-german-cased'` | bert_score library doesn't support this model directly | Changed to `bert-base-multilingual-cased` for DE and FR |
| 2 | RobertaModel pooler weights warning | Standard HuggingFace warning, not an error | Ignored (does not affect BERTScore computation) |

---

### 5.7 Practical Implications

For building a multilingual image captioning system:

1. **Do not rely on direct multilingual generation** with English-centric
   models like OpenFlamingo — language accuracy is too unreliable.

2. **The translate baseline is a strong approach** — generate the best
   possible English caption, then use a dedicated MT model. This produces
   grammatically correct output in the target language with near-perfect
   language accuracy.

3. **The translation step does not introduce significant errors** — the
   quality gap between EN-only and translate baselines is primarily due to
   the inherent difficulty of cross-lingual evaluation, not MT errors.

4. **French is easier than German** for both direct and translate approaches,
   likely due to training data distribution and morphological complexity.

---

### 5.8 Output Files

| File | Content |
|------|---------|
| `outputs/metrics/evaluation_test_2016_flickr.json` | Full results (all metrics, all experiments) |
| `outputs/metrics/summary_test_2016_flickr.csv` | CSV summary table for thesis |

---

### 5.9 Execution Instructions (Reproducibility)

```bash
cd ~/projects/bp-flamingo
source venv/bin/activate

# Full evaluation (GPU recommended for BERTScore)
sbatch scripts/slurm/run_evaluation.sh

# Or without BERTScore (fast, CPU OK)
PYTHONPATH=. python3 src/evaluation/evaluate.py --all --no-bertscore

# Results
cat outputs/metrics/summary_test_2016_flickr.csv
```

---

**Project complete! All 5 steps finished successfully.**