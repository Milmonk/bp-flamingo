# BP-FLAMINGO: Documentation Log

## Step 6 — Architecture Analysis and Visualisation

**Date:** 2026-03-18  
**Status:** Ready to execute  
**Depends on:** Steps 1-5 (complete pipeline)

---

### 6.1 Objective

Create structured visualisations and analyses required by the thesis:
1. Architecture flowchart of OpenFlamingo layers
2. Experimental pipeline flowchart
3. Multilingual tokenisation analysis
4. Results comparison charts

These address thesis requirement #5: "Analyse the internal processing pipeline,
including a structured flowchart of Flamingo layers, multilingual tokenisation
mechanisms, and attention visualisation."

---

### 6.2 Outputs Produced

| File | Description |
|------|-------------|
| `flamingo_architecture_flowchart.png` | Detailed diagram of OpenFlamingo-3B architecture showing ViT-L/14, Perceiver Resampler, Gated Cross-Attention, and MPT-1B with frozen/trainable annotations |
| `experimental_pipeline_flowchart.png` | Full experimental pipeline from dataset through all 5 modes to evaluation |
| `results_comparison_chart.png` | Bar charts comparing BLEU, chrF, METEOR, BERTScore across all experiments |
| `language_accuracy_chart.png` | Bar chart showing target language accuracy (%) per experiment |
| `tokenization_analysis.json` | Detailed tokenisation statistics for EN/DE/FR |

---

### 6.3 Architecture Flowchart

The flowchart shows the data flow through OpenFlamingo:

```
Image (224×224)          Text Prompt
      │                       │
      ▼                       ▼
  ViT-L/14 (CLIP)        MPT Tokenizer
  [Frozen]               [vocab: 50,432]
      │                       │
  257×1024                Token IDs
  (patch embeddings)          │
      │                       ▼
      ▼                  Self-Attention
  Perceiver Resampler     [Frozen]
  (6 layers)                  │
  [Trainable]                 ▼
      │              Gated Cross-Attention
  64×1024 ──────────► [Trainable]
  (visual tokens)             │
                              ▼
                       Feed-Forward
                        [Frozen]
                              │
                        × 24 blocks
                              │
                              ▼
                         LM Head
                              │
                              ▼
                      Generated Caption
```

Key insight: Only the Perceiver Resampler and Gated Cross-Attention layers
are trained (1.05B of 2.56B total parameters). The vision encoder and
language model remain frozen from their pre-trained weights.

---

### 6.4 Tokenisation Analysis

Compares how the MPT-1B tokenizer (BPE, vocab size ~50K) handles the
same caption content across English, German, and French:

Key questions answered:
- How many tokens does the same sentence produce in each language?
- What is the tokens-per-word ratio for each language?
- How does tokenization affect prompt length and generation speed?

Expected findings:
- German produces more tokens per word (compound words split into subwords)
- French produces more tokens due to accented characters
- This explains why multilingual few-shot prompts are longer and inference slower

---

### 6.5 Execution Instructions

```bash
cd ~/projects/bp-flamingo
source venv/bin/activate

# Run all analyses via SLURM
sbatch scripts/slurm/run_analysis.sh

# Check outputs
ls -la outputs/visualizations/
```

---

**This step completes the practical component of the thesis.**