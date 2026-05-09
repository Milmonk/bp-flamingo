"""
BP-FLAMINGO: Multilingual Tokenisation Analysis

Analyses how the MPT-1B tokeniser handles text in different languages.
Compares token counts, subword patterns, and vocabulary coverage
for English, German, and French captions.

Usage:
    PYTHONPATH=. python3 src/analysis/tokenization_analysis.py
"""

import json
import sys
from pathlib import Path
from collections import Counter

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config
from src.utils.data_loader import Multi30KLoader


def load_tokenizer():
    """Load the MPT-1B tokenizer (same one used by OpenFlamingo)."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "anas-awadalla/mpt-1b-redpajama-200b",
        trust_remote_code=True,
    )
    return tokenizer


def analyse_tokenization(texts: list[str], tokenizer, language: str) -> dict:
    """
    Analyse tokenization statistics for a list of texts.

    Returns dict with token counts, subword ratios, etc.
    """
    all_token_counts = []
    all_word_counts = []
    all_tokens = []
    unknown_tokens = 0
    total_tokens = 0
    continuation_tokens = 0  # Tokens that are subword continuations

    for text in texts:
        words = text.split()
        tokens = tokenizer.tokenize(text)
        token_ids = tokenizer.encode(text)

        all_word_counts.append(len(words))
        all_token_counts.append(len(tokens))
        all_tokens.extend(tokens)
        total_tokens += len(tokens)

        # Count continuation tokens (subwords starting with Ġ or similar)
        for t in tokens:
            if not t.startswith("Ġ") and tokens.index(t) != 0:
                continuation_tokens += 1

    # Token frequency
    token_freq = Counter(all_tokens)

    # Compute statistics
    token_counts = np.array(all_token_counts)
    word_counts = np.array(all_word_counts)
    token_per_word = token_counts / np.maximum(word_counts, 1)

    return {
        "language": language,
        "num_texts": len(texts),
        "avg_words": round(float(np.mean(word_counts)), 2),
        "avg_tokens": round(float(np.mean(token_counts)), 2),
        "std_tokens": round(float(np.std(token_counts)), 2),
        "min_tokens": int(np.min(token_counts)),
        "max_tokens": int(np.max(token_counts)),
        "avg_tokens_per_word": round(float(np.mean(token_per_word)), 3),
        "median_tokens_per_word": round(float(np.median(token_per_word)), 3),
        "total_tokens": total_tokens,
        "unique_tokens": len(token_freq),
        "top_20_tokens": token_freq.most_common(20),
    }


def show_tokenization_examples(texts: list[str], tokenizer, language: str, n: int = 5):
    """Print detailed tokenization for a few example sentences."""
    print(f"\n  === Tokenization Examples [{language.upper()}] ===")

    for text in texts[:n]:
        tokens = tokenizer.tokenize(text)
        token_ids = tokenizer.encode(text)

        print(f"\n  Text:    {text}")
        print(f"  Words:   {len(text.split())}")
        print(f"  Tokens:  {len(tokens)}")
        print(f"  Ratio:   {len(tokens)/max(len(text.split()),1):.2f} tokens/word")
        print(f"  Tokens:  {tokens}")


def analyse_parallel_sentences(loader, tokenizer, split="test_2016_flickr", n=10):
    """
    Compare tokenization of parallel sentences (same meaning in EN/DE/FR).
    """
    data = loader.load_split(split)

    print(f"\n  === Parallel Sentence Tokenization (first {n} samples) ===")
    print(f"  {'ID':<6} {'Lang':<6} {'Words':<8} {'Tokens':<8} {'Ratio':<8} Text")
    print(f"  {'-'*80}")

    for sample in data[:n]:
        for lang in ["en", "de", "fr"]:
            text = sample["captions"][lang]
            tokens = tokenizer.tokenize(text)
            words = text.split()
            ratio = len(tokens) / max(len(words), 1)

            text_short = text[:60] + "..." if len(text) > 60 else text
            print(f"  {sample['id']:<6} {lang.upper():<6} {len(words):<8} {len(tokens):<8} {ratio:<8.2f} {text_short}")
        print()


def main():
    config = load_config()
    loader = Multi30KLoader(config)

    print("============================================")
    print("  BP-FLAMINGO: Tokenisation Analysis")
    print("============================================")

    # Load tokenizer
    print("\n  Loading MPT-1B tokenizer...")
    tokenizer = load_tokenizer()
    vocab_size = tokenizer.vocab_size
    print(f"  Vocabulary size: {vocab_size:,}")

    # Load captions for each language
    test_data = loader.load_split("test_2016_flickr")

    captions = {
        "en": [s["captions"]["en"] for s in test_data],
        "de": [s["captions"]["de"] for s in test_data],
        "fr": [s["captions"]["fr"] for s in test_data],
    }

    # === Tokenization statistics per language ===
    print("\n" + "="*60)
    print("  Tokenisation Statistics (test set, 1000 samples)")
    print("="*60)

    all_stats = {}
    for lang in ["en", "de", "fr"]:
        stats = analyse_tokenization(captions[lang], tokenizer, lang)
        all_stats[lang] = stats

        print(f"\n  [{lang.upper()}]")
        print(f"    Avg words per caption:  {stats['avg_words']}")
        print(f"    Avg tokens per caption: {stats['avg_tokens']} (±{stats['std_tokens']})")
        print(f"    Token range:            {stats['min_tokens']} – {stats['max_tokens']}")
        print(f"    Avg tokens per word:    {stats['avg_tokens_per_word']}")
        print(f"    Unique tokens used:     {stats['unique_tokens']:,}")

    # === Comparison table ===
    print(f"\n  {'':=<60}")
    print(f"  Summary Comparison")
    print(f"  {'':=<60}")
    print(f"  {'Metric':<28} {'EN':>8} {'DE':>8} {'FR':>8}")
    print(f"  {'-'*28} {'-'*8} {'-'*8} {'-'*8}")
    print(f"  {'Avg words/caption':<28} {all_stats['en']['avg_words']:>8} {all_stats['de']['avg_words']:>8} {all_stats['fr']['avg_words']:>8}")
    print(f"  {'Avg tokens/caption':<28} {all_stats['en']['avg_tokens']:>8} {all_stats['de']['avg_tokens']:>8} {all_stats['fr']['avg_tokens']:>8}")
    print(f"  {'Avg tokens/word':<28} {all_stats['en']['avg_tokens_per_word']:>8} {all_stats['de']['avg_tokens_per_word']:>8} {all_stats['fr']['avg_tokens_per_word']:>8}")
    print(f"  {'Unique tokens':<28} {all_stats['en']['unique_tokens']:>8,} {all_stats['de']['unique_tokens']:>8,} {all_stats['fr']['unique_tokens']:>8,}")

    # === Examples ===
    for lang in ["en", "de", "fr"]:
        show_tokenization_examples(captions[lang], tokenizer, lang, n=3)

    # === Parallel sentence comparison ===
    analyse_parallel_sentences(loader, tokenizer, n=5)

    # === Save results ===
    output = {
        "tokenizer": "anas-awadalla/mpt-1b-redpajama-200b",
        "vocab_size": vocab_size,
        "statistics": {},
    }
    for lang, stats in all_stats.items():
        stats_copy = dict(stats)
        stats_copy["top_20_tokens"] = [(t, c) for t, c in stats["top_20_tokens"]]
        output["statistics"][lang] = stats_copy

    output_path = Path(config["paths"]["outputs_dir"]) / "visualizations" / "tokenization_analysis.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n  Saved: {output_path}")
    print("  Done!")


if __name__ == "__main__":
    main()