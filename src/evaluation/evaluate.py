"""
BP-FLAMINGO: Evaluation Module

Computes automatic evaluation metrics comparing generated/translated captions
against Multi30K reference captions.

Metrics:
  - BLEU (via sacrebleu) — precision-based n-gram overlap
  - chrF (via sacrebleu) — character-level F-score (good for morphological languages)
  - METEOR (via nltk) — considers synonyms and stemming
  - BERTScore (via bert_score) — semantic similarity using contextual embeddings

Additional analysis:
  - Language detection accuracy (langdetect)
  - Caption length statistics

Usage:
    PYTHONPATH=. python3 src/evaluation/evaluate.py --all
    PYTHONPATH=. python3 src/evaluation/evaluate.py --experiment en_only
"""

import json
import time
import argparse
import sys
from pathlib import Path
from typing import Optional
from collections import Counter

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config, get_output_path


# ============================================================================
# Metric computation functions
# ============================================================================

def compute_bleu(hypotheses: list[str], references: list[str]) -> dict:
    """Compute corpus-level BLEU score."""
    import sacrebleu
    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    return {
        "score": round(bleu.score, 2),
        "bp": round(bleu.bp, 4),
        "precisions": [round(p, 2) for p in bleu.precisions],
    }


def compute_chrf(hypotheses: list[str], references: list[str]) -> dict:
    """Compute corpus-level chrF score."""
    import sacrebleu
    chrf = sacrebleu.corpus_chrf(hypotheses, [references])
    return {
        "score": round(chrf.score, 2),
    }


def compute_meteor(hypotheses: list[str], references: list[str]) -> dict:
    """Compute average sentence-level METEOR score."""
    from nltk.translate.meteor_score import meteor_score
    from nltk import word_tokenize

    scores = []
    for hyp, ref in zip(hypotheses, references):
        try:
            hyp_tokens = word_tokenize(hyp.lower())
            ref_tokens = word_tokenize(ref.lower())
            if len(hyp_tokens) == 0 or len(ref_tokens) == 0:
                scores.append(0.0)
            else:
                score = meteor_score([ref_tokens], hyp_tokens)
                scores.append(score)
        except Exception:
            scores.append(0.0)

    return {
        "score": round(np.mean(scores) * 100, 2),
        "std": round(np.std(scores) * 100, 2),
        "min": round(np.min(scores) * 100, 2),
        "max": round(np.max(scores) * 100, 2),
    }


def compute_bertscore(
    hypotheses: list[str],
    references: list[str],
    lang: str = "en",
    model_name: Optional[str] = None,
) -> dict:
    """Compute BERTScore (precision, recall, F1)."""
    from bert_score import score as bert_score_fn

    # Use language-specific model if provided
    if model_name is None:
        model_map = {
            "en": "roberta-large",
            "de": "bert-base-multilingual-cased",
            "fr": "bert-base-multilingual-cased",
        }
        model_name = model_map.get(lang, "bert-base-multilingual-cased")

    P, R, F1 = bert_score_fn(
        hypotheses,
        references,
        model_type=model_name,
        lang=lang,
        verbose=False,
        batch_size=64,
    )

    return {
        "precision": round(P.mean().item() * 100, 2),
        "recall": round(R.mean().item() * 100, 2),
        "f1": round(F1.mean().item() * 100, 2),
        "model": model_name,
    }


def detect_languages(texts: list[str]) -> dict:
    """Detect languages of generated texts and compute statistics."""
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 42  # Reproducibility

    detections = []
    for text in texts:
        try:
            lang = detect(text)
            detections.append(lang)
        except Exception:
            detections.append("unknown")

    # Count language distribution
    counter = Counter(detections)
    total = len(detections)

    distribution = {
        lang: {
            "count": count,
            "percentage": round(count / total * 100, 1),
        }
        for lang, count in counter.most_common()
    }

    return {
        "distribution": distribution,
        "total": total,
        "most_common": counter.most_common(1)[0][0] if counter else "unknown",
    }


def compute_length_stats(
    hypotheses: list[str],
    references: list[str],
) -> dict:
    """Compute caption length statistics."""
    hyp_lengths = [len(h.split()) for h in hypotheses]
    ref_lengths = [len(r.split()) for r in references]

    return {
        "hypothesis_avg": round(np.mean(hyp_lengths), 1),
        "hypothesis_std": round(np.std(hyp_lengths), 1),
        "reference_avg": round(np.mean(ref_lengths), 1),
        "reference_std": round(np.std(ref_lengths), 1),
        "length_ratio": round(np.mean(hyp_lengths) / np.mean(ref_lengths), 3)
            if np.mean(ref_lengths) > 0 else 0,
    }


# ============================================================================
# Experiment evaluation
# ============================================================================

def evaluate_experiment(
    name: str,
    hypotheses: list[str],
    references: list[str],
    eval_language: str,
    config: dict,
    compute_bert: bool = True,
) -> dict:
    """
    Run all metrics for a single experiment.

    Args:
        name: Experiment name (e.g., "en_only", "direct_de").
        hypotheses: List of generated/translated captions.
        references: List of reference captions.
        eval_language: Language code for BERTScore model selection.
        config: Config dictionary.
        compute_bert: Whether to compute BERTScore (slow).

    Returns:
        Dictionary with all metric results.
    """
    print(f"\n  --- Evaluating: {name} ({len(hypotheses)} samples, lang={eval_language}) ---")

    results = {
        "experiment": name,
        "language": eval_language,
        "num_samples": len(hypotheses),
    }

    # BLEU
    print("    Computing BLEU...")
    results["bleu"] = compute_bleu(hypotheses, references)

    # chrF
    print("    Computing chrF...")
    results["chrf"] = compute_chrf(hypotheses, references)

    # METEOR
    print("    Computing METEOR...")
    results["meteor"] = compute_meteor(hypotheses, references)

    # BERTScore
    if compute_bert:
        print(f"    Computing BERTScore (model for '{eval_language}')...")
        bertscore_models = config["evaluation"].get("bertscore_models", {})
        model_name = bertscore_models.get(eval_language)
        results["bertscore"] = compute_bertscore(
            hypotheses, references, lang=eval_language, model_name=model_name
        )

    # Language detection
    print("    Detecting output languages...")
    results["language_detection"] = detect_languages(hypotheses)

    # Length statistics
    results["length_stats"] = compute_length_stats(hypotheses, references)

    # Print summary
    print(f"\n    Results for {name}:")
    print(f"      BLEU:       {results['bleu']['score']}")
    print(f"      chrF:       {results['chrf']['score']}")
    print(f"      METEOR:     {results['meteor']['score']}")
    if compute_bert:
        print(f"      BERTScore:  {results['bertscore']['f1']} (F1)")
    lang_det = results["language_detection"]
    print(f"      Language:   {lang_det['most_common']} ({lang_det['distribution'].get(lang_det['most_common'], {}).get('percentage', 0)}%)")
    print(f"      Avg length: {results['length_stats']['hypothesis_avg']} words (ref: {results['length_stats']['reference_avg']})")

    return results


def load_experiment_data(config: dict, split: str = "test_2016_flickr") -> list[dict]:
    """
    Load all experiment outputs and prepare for evaluation.

    Returns list of experiment dicts, each with:
        name, hypotheses, references, eval_language
    """
    captions_dir = Path(config["paths"]["captions_dir"])
    translations_dir = Path(config["paths"]["translations_dir"])

    experiments = []

    # 1. EN-only
    en_path = captions_dir / f"captions_en_only_{split}.json"
    if en_path.exists():
        with open(en_path) as f:
            data = json.load(f)
        experiments.append({
            "name": "en_only",
            "hypotheses": [r["generated_caption"] for r in data["results"]],
            "references": [r["reference_caption"] for r in data["results"]],
            "eval_language": "en",
        })

    # 2. Direct DE
    de_path = captions_dir / f"captions_direct_de_{split}.json"
    if de_path.exists():
        with open(de_path) as f:
            data = json.load(f)
        experiments.append({
            "name": "direct_de",
            "hypotheses": [r["generated_caption"] for r in data["results"]],
            "references": [r["reference_caption"] for r in data["results"]],
            "eval_language": "de",
        })

    # 3. Direct FR
    fr_path = captions_dir / f"captions_direct_fr_{split}.json"
    if fr_path.exists():
        with open(fr_path) as f:
            data = json.load(f)
        experiments.append({
            "name": "direct_fr",
            "hypotheses": [r["generated_caption"] for r in data["results"]],
            "references": [r["reference_caption"] for r in data["results"]],
            "eval_language": "fr",
        })

    # 4. Translate baseline DE
    trans_de_path = translations_dir / f"translations_en_de_{split}.json"
    if trans_de_path.exists():
        with open(trans_de_path) as f:
            data = json.load(f)
        experiments.append({
            "name": "translate_de",
            "hypotheses": [r["translated_caption"] for r in data["results"]],
            "references": [r["reference_caption"] for r in data["results"]],
            "eval_language": "de",
        })

    # 5. Translate baseline FR
    trans_fr_path = translations_dir / f"translations_en_fr_{split}.json"
    if trans_fr_path.exists():
        with open(trans_fr_path) as f:
            data = json.load(f)
        experiments.append({
            "name": "translate_fr",
            "hypotheses": [r["translated_caption"] for r in data["results"]],
            "references": [r["reference_caption"] for r in data["results"]],
            "eval_language": "fr",
        })

    return experiments


def main():
    parser = argparse.ArgumentParser(description="BP-FLAMINGO Evaluation")
    parser.add_argument("--all", action="store_true",
                        help="Evaluate all experiments")
    parser.add_argument("--experiment", type=str, default=None,
                        choices=["en_only", "direct_de", "direct_fr",
                                 "translate_de", "translate_fr"],
                        help="Evaluate a single experiment")
    parser.add_argument("--split", type=str, default="test_2016_flickr")
    parser.add_argument("--no-bertscore", action="store_true",
                        help="Skip BERTScore computation (faster)")
    parser.add_argument("--device", type=str, default="auto")

    args = parser.parse_args()
    config = load_config()

    print("============================================")
    print("  BP-FLAMINGO: Evaluation")
    print("============================================")

    # Load all experiment data
    experiments = load_experiment_data(config, args.split)
    print(f"\n  Found {len(experiments)} experiments to evaluate:")
    for exp in experiments:
        print(f"    - {exp['name']} ({len(exp['hypotheses'])} samples, lang={exp['eval_language']})")

    # Filter if specific experiment requested
    if args.experiment:
        experiments = [e for e in experiments if e["name"] == args.experiment]
        if not experiments:
            print(f"\n  ERROR: Experiment '{args.experiment}' not found.")
            sys.exit(1)

    # Run evaluation
    compute_bert = not args.no_bertscore
    all_results = []
    total_start = time.time()

    for exp in experiments:
        result = evaluate_experiment(
            name=exp["name"],
            hypotheses=exp["hypotheses"],
            references=exp["references"],
            eval_language=exp["eval_language"],
            config=config,
            compute_bert=compute_bert,
        )
        all_results.append(result)

    total_elapsed = time.time() - total_start

    # ========================================
    # Save results
    # ========================================
    output = {
        "metadata": {
            "split": args.split,
            "num_experiments": len(all_results),
            "metrics_computed": ["bleu", "chrf", "meteor"]
                + (["bertscore"] if compute_bert else []),
            "total_time_seconds": round(total_elapsed, 2),
        },
        "results": all_results,
    }

    output_path = get_output_path(config, "metrics_dir", f"evaluation_{args.split}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n  Saved: {output_path}")

    # ========================================
    # Print comparison table
    # ========================================
    print(f"\n{'='*70}")
    print(f"  EVALUATION SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Experiment':<18} {'BLEU':>7} {'chrF':>7} {'METEOR':>8} ", end="")
    if compute_bert:
        print(f"{'BERTSc':>8} ", end="")
    print(f"{'Lang%':>8}")
    print(f"  {'-'*18} {'-'*7} {'-'*7} {'-'*8} ", end="")
    if compute_bert:
        print(f"{'-'*8} ", end="")
    print(f"{'-'*8}")

    for r in all_results:
        lang_det = r["language_detection"]
        target = r["language"]
        target_pct = lang_det["distribution"].get(target, {}).get("percentage", 0)

        print(f"  {r['experiment']:<18} {r['bleu']['score']:>7.2f} {r['chrf']['score']:>7.2f} {r['meteor']['score']:>8.2f} ", end="")
        if compute_bert:
            print(f"{r['bertscore']['f1']:>8.2f} ", end="")
        print(f"{target_pct:>7.1f}%")

    print(f"{'='*70}")
    print(f"  Total evaluation time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"{'='*70}")

    # Save summary table as CSV for easy use in thesis
    csv_path = get_output_path(config, "metrics_dir", f"summary_{args.split}.csv")
    with open(csv_path, "w") as f:
        headers = ["experiment", "language", "bleu", "chrf", "meteor"]
        if compute_bert:
            headers += ["bertscore_f1"]
        headers += ["target_lang_pct", "avg_length"]
        f.write(",".join(headers) + "\n")

        for r in all_results:
            lang_det = r["language_detection"]
            target_pct = lang_det["distribution"].get(r["language"], {}).get("percentage", 0)
            row = [
                r["experiment"],
                r["language"],
                str(r["bleu"]["score"]),
                str(r["chrf"]["score"]),
                str(r["meteor"]["score"]),
            ]
            if compute_bert:
                row.append(str(r["bertscore"]["f1"]))
            row += [
                str(target_pct),
                str(r["length_stats"]["hypothesis_avg"]),
            ]
            f.write(",".join(row) + "\n")

    print(f"  CSV summary: {csv_path}")


if __name__ == "__main__":
    main()