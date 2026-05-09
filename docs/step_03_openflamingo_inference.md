# BP-FLAMINGO: Documentation Log

## Step 3 — OpenFlamingo Inference

**Date:** 2026-03-17 to 2026-03-18  
**Status:** Completed  
**Depends on:** Step 1 (environment), Step 2 (dataset)  
**HPC Job ID:** 16505 (full run), gpu09 node

---

### 3.1 Objective

Implement an inference module that loads the OpenFlamingo model and generates
image captions in three modes: English-only, direct German, and direct French.
Evaluate the model's ability to generate multilingual captions via few-shot
prompting.

---

### 3.2 Model: OpenFlamingo-3B

| Property | Value |
|----------|-------|
| Model | OpenFlamingo-3B-vitl-mpt1b |
| HuggingFace ID | `openflamingo/OpenFlamingo-3B-vitl-mpt1b` |
| Vision encoder | ViT-L/14 (OpenAI CLIP) |
| Language model | MPT-1B (RedPajama 200B) |
| Cross-attention | Every layer (`cross_attn_every_n_layers=1`) |
| Total parameters | 2.56B |
| Trainable parameters | 1.05B (Flamingo-specific layers) |
| GPU memory usage | ~8-10 GB (fp32) on NVIDIA H200 (150.1 GB available) |

---

### 3.3 Inference Modes and Prompt Design

**Mode 1: EN-only**
- Few-shot captions: English
- Prompt template: `<image>Output: {caption}<|endofchunk|>`
- Query: `<image>Output:`
- Purpose: Baseline quality of OpenFlamingo

**Mode 2: Direct German**
- Few-shot captions: German (from Multi30K DE references)
- Prompt template: `<image>German description: {de_caption}<|endofchunk|>`
- Query: `<image>German description:`
- Purpose: Test ability to generate German text via few-shot priming

**Mode 3: Direct French**
- Few-shot captions: French (from Multi30K FR references)
- Prompt template: `<image>French description: {fr_caption}<|endofchunk|>`
- Query: `<image>French description:`
- Purpose: Test ability to generate French text via few-shot priming

Key design decision: few-shot examples use captions in the **target language**
(not English). This gives the model the strongest possible signal about the
desired output language.

---

### 3.4 Few-Shot and Generation Configuration

| Parameter | Value |
|-----------|-------|
| Few-shot examples | 4 (from train split) |
| Few-shot seed | 42 (fixed for reproducibility) |
| max_new_tokens | 50 |
| num_beams | 3 |
| temperature | 1.0 |
| top_p | 0.9 (note: inactive with beam search) |
| repetition_penalty | 1.0 |

---

### 3.5 Full Run Results

**Environment:**
- GPU: NVIDIA H200 (150.1 GB)
- PyTorch: 2.10.0+cu128
- CUDA: 12.8

**Timing:**

| Experiment | Samples | Total time | Avg per sample |
|------------|---------|------------|----------------|
| EN-only | 1,000 | 767.4s (12.8 min) | 0.77s |
| Direct DE | 1,000 | 2,551.3s (42.5 min) | 2.55s |
| Direct FR | 1,000 | 2,635.9s (43.9 min) | 2.64s |
| **Total** | **3,000** | **5,955.1s (99.3 min)** | **1.99s** |

Observation: Multilingual modes are ~3.3x slower than EN-only. This is because
the tokenized German/French few-shot prompts are longer (due to compound words
in German and accented characters in French producing more tokens), and the
model generates longer sequences before stopping.

---

### 3.6 Qualitative Analysis (Test Samples)

**EN-only mode — clean English captions:**

| Image | Generated | Reference |
|-------|-----------|-----------|
| 1007129816.jpg | A man wearing a beer hat. | A man in an orange hat starring at something. |
| 1009434119.jpg | A Boston Terrier running down a grassy field. | A Boston Terrier is running on lush green grass in front of a white fence. |
| 101362133.jpg | A man in a white t-shirt and black pants holding a karate stick. | A girl in karate uniform breaking a stick with a front kick. |

Observations: Captions are fluent and relevant, but may describe incorrect
details (e.g., "beer hat" vs "orange hat", "man" vs "girl").

**Direct DE mode — mixed language output:**

| Image | Generated | Language |
|-------|-----------|----------|
| 1007129816.jpg | A man in a rural area wears a beer cap. | English |
| 1009434119.jpg | A Boston Terrier sprints across a lawn. | English |
| 101362133.jpg | Ein Tae Kwon Do-Kämpfer schneidet einen Schläger. | German |

Observations: 1 out of 3 samples generated in German. The model inconsistently
switches between English and German despite German few-shot examples.

**Direct FR mode — predominantly French output:**

| Image | Generated | Language |
|-------|-----------|----------|
| 1007129816.jpg | Un homme à la campagne avec un verre à vin. | French |
| 1009434119.jpg | Un bébé dans le jardin. | French |
| 101362133.jpg | Un jeune homme avec un sabre à la main. | French |

Observations: All 3 samples generated in French. French generation appears
more reliable than German, possibly due to greater French representation in
the MPT-1B training data.

---

### 3.7 Key Findings

1. **English generation works well** — captions are fluent and generally
   describe the image content correctly, though with occasional errors in
   fine-grained details.

2. **Direct multilingual generation is unreliable** — the model was not
   trained for multilingual generation, so results vary significantly:
   - French: Higher success rate for target-language generation
   - German: Frequently falls back to English

3. **Few-shot language matters** — using target-language captions in few-shot
   examples significantly improves the probability of generating in the
   target language compared to English-only few-shot with a translated prompt
   template.

4. **Post-processing is essential** — raw model output frequently contains:
   - Repetitions of the prompt template ("German description: ...")
   - Self-repeating captions
   - These are cleaned by the post-processing pipeline

5. **Speed difference** — multilingual modes are ~3x slower than English,
   likely due to longer tokenized prompts and generated sequences.

These findings strongly motivate the **translate baseline** approach (Step 4):
generate high-quality English captions, then translate using a dedicated
MT model (MarianMT), which should produce more reliable multilingual output.

---

### 3.8 Output Format

Each experiment produces a JSON file in `outputs/captions/`:

```json
{
    "metadata": {
        "model": "openflamingo/OpenFlamingo-3B-vitl-mpt1b",
        "split": "test_2016_flickr",
        "language": "en",
        "num_few_shot": 4,
        "seed": 42,
        "num_samples": 1000,
        "total_time_seconds": 767.4,
        "avg_time_per_sample": 0.7674,
        "generation_config": { ... }
    },
    "results": [
        {
            "id": 0,
            "image_filename": "1007129816.jpg",
            "language": "en",
            "generated_caption": "A man wearing a beer hat.",
            "raw_caption": "A man wearing a beer hat.",
            "reference_caption": "A man in an orange hat starring at something.",
            "reference_en": "A man in an orange hat starring at something.",
            "num_few_shot": 4,
            "seed": 42
        }
    ]
}
```

Output files produced:
- `outputs/captions/captions_en_only_test_2016_flickr.json` (1,000 samples)
- `outputs/captions/captions_direct_de_test_2016_flickr.json` (1,000 samples)
- `outputs/captions/captions_direct_fr_test_2016_flickr.json` (1,000 samples)

The `raw_caption` field preserves the unprocessed model output for error
analysis in the thesis.

---

### 3.9 Issues Encountered and Solutions

| # | Issue | Cause | Solution |
|---|-------|-------|----------|
| 1 | `total_mem` AttributeError | PyTorch 2.10 renamed to `total_memory` | `sed -i 's/total_mem/total_memory/g'` |
| 2 | `all_tied_weights_keys` AttributeError | transformers 4.57 incompatible with MPT model | `pip install transformers==4.44.2` |
| 3 | `vision_x should be of shape (b, T_img, F, C, H, W)` | Missing `frames` dimension in vision tensor | Added `.unsqueeze(2)` for frames=1 dimension |
| 4 | Repetitive captions ("Output: ... Output: ...") | Model doesn't stop at prompt template boundaries | Post-processing: truncate at template strings |
| 5 | Direct multilingual generating English | EN-only few-shot examples, model defaults to English | Changed few-shot to use target language captions |
| 6 | SLURM partition name `gpu` → `GPU` | HPC Perun uses uppercase partition names | `sed -i 's/--partition=gpu/--partition=GPU/'` |
| 7 | tar overwriting fixed files with old versions | Re-extracting archives restored old buggy code | Apply `sed` fixes after extraction, or edit in place |

---

### 3.10 Module Architecture

```
src/inference/
├── flamingo_inference.py    # Core inference class (FlamingoInference)
│   ├── load_model()         # Load model, tokenizer, image processor
│   ├── _build_prompt()      # Construct few-shot prompt (language-aware)
│   ├── _prepare_image()     # Process image through CLIP vision encoder
│   ├── generate_caption()   # Single image → caption (with post-processing)
│   └── run_inference()      # Full split → JSON output
└── run_all_inference.py     # Run all 3 modes sequentially (single model load)

scripts/slurm/
├── test_inference.sh        # Quick test (3 samples × 3 languages, ~1 min)
└── run_inference.sh         # Full run (1000 × 3 languages, ~99 min)
```

---

### 3.11 Execution Instructions (Reproducibility)

```bash
cd ~/projects/bp-flamingo
source venv/bin/activate

# Quick test (3 samples):
sbatch scripts/slurm/test_inference.sh

# Full run (1000 samples × 3 languages):
sbatch scripts/slurm/run_inference.sh

# Monitor:
squeue -u $USER
tail -f logs/inference_<JOBID>.out

# Verify outputs:
ls -la outputs/captions/
python3 -c "
import json
for f in ['en_only', 'direct_de', 'direct_fr']:
    with open(f'outputs/captions/captions_{f}_test_2016_flickr.json') as fh:
        d = json.load(fh)
    print(f'{f}: {d[\"metadata\"][\"num_samples\"]} samples, {d[\"metadata\"][\"avg_time_per_sample\"]:.2f}s/sample')
"
```

---

### 3.12 Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Model size | 3B | Fits single GPU, fast inference, sufficient for comparison |
| Few-shot count | 4 | Standard for Flamingo; balances context length vs. quality |
| Few-shot language | Target language | Maximizes probability of target-language output |
| Beam search | 3 beams | Good quality/speed tradeoff |
| Post-processing | Template + repetition removal | Handles common failure modes |
| Raw caption storage | Yes | Enables error analysis in thesis |
| transformers version | 4.44.2 | Last version compatible with MPT-1B model |

---

**Next step:** Step 4 — MarianMT Translation Module (EN → DE, EN → FR)