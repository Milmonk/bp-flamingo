"""
BP-FLAMINGO: MarianMT Translation Module

Translates English captions (from OpenFlamingo) into target languages
using Helsinki-NLP MarianMT models.

Supported language pairs:
  - EN → DE (Helsinki-NLP/opus-mt-en-de)
  - EN → FR (Helsinki-NLP/opus-mt-en-fr)

Usage:
    PYTHONPATH=. python3 src/translation/translate.py
    PYTHONPATH=. python3 src/translation/translate.py --target-lang de
    PYTHONPATH=. python3 src/translation/translate.py --target-lang fr
    PYTHONPATH=. python3 src/translation/translate.py --all
"""

import json
import time
import argparse
import sys
from pathlib import Path
from typing import Optional

import torch
from transformers import MarianMTModel, MarianTokenizer
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config, get_output_path


class MarianTranslator:
    """
    MarianMT translation wrapper.

    Loads a MarianMT model for a specific language pair and translates
    text in batches for efficiency.
    """

    def __init__(
        self,
        source_lang: str = "en",
        target_lang: str = "de",
        config: Optional[dict] = None,
        device: str = "auto",
    ):
        """
        Initialize the translator.

        Args:
            source_lang: Source language code.
            target_lang: Target language code.
            config: Optional config dict.
            device: Device to use ("auto", "cuda", "cpu").
        """
        self.config = config or load_config()
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.trans_config = self.config["translation"]

        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Get model name from config
        pair_key = f"{source_lang}_{target_lang}"
        self.model_name = self.trans_config["models"].get(pair_key)
        if self.model_name is None:
            raise ValueError(
                f"No translation model configured for {source_lang}→{target_lang}. "
                f"Available pairs: {list(self.trans_config['models'].keys())}"
            )

        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Load the MarianMT model and tokenizer."""
        print(f"  Loading MarianMT model: {self.model_name}")
        print(f"  Language pair: {self.source_lang} → {self.target_lang}")
        print(f"  Device: {self.device}")
        start_time = time.time()

        self.tokenizer = MarianTokenizer.from_pretrained(self.model_name)
        self.model = MarianMTModel.from_pretrained(self.model_name)
        self.model = self.model.to(self.device)
        self.model.eval()

        # Model info
        param_count = sum(p.numel() for p in self.model.parameters())
        elapsed = time.time() - start_time
        print(f"  Model loaded in {elapsed:.1f}s")
        print(f"  Parameters: {param_count / 1e6:.0f}M")

    def translate_batch(self, texts: list[str]) -> list[str]:
        """
        Translate a batch of texts.

        Args:
            texts: List of source-language strings.

        Returns:
            List of translated strings.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        gen_config = self.trans_config["generation"]

        # Tokenize
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=gen_config["max_length"],
        ).to(self.device)

        # Generate translations
        with torch.no_grad():
            translated = self.model.generate(
                **inputs,
                max_length=gen_config["max_length"],
                num_beams=gen_config["num_beams"],
            )

        # Decode
        results = self.tokenizer.batch_decode(translated, skip_special_tokens=True)
        return results

    def translate_single(self, text: str) -> str:
        """Translate a single text string."""
        return self.translate_batch([text])[0]


def translate_captions_file(
    input_path: str,
    target_lang: str,
    config: Optional[dict] = None,
    device: str = "auto",
    batch_size: int = 32,
) -> dict:
    """
    Translate all captions in an inference output file.

    Args:
        input_path: Path to OpenFlamingo output JSON (EN captions).
        target_lang: Target language code ("de" or "fr").
        config: Optional config dict.
        device: Device to use.
        batch_size: Number of captions to translate at once.

    Returns:
        Dictionary with translation results and metadata.
    """
    config = config or load_config()

    # Load input captions
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        input_data = json.load(f)

    source_metadata = input_data["metadata"]
    source_results = input_data["results"]

    print(f"\n  Translating {len(source_results)} captions: EN → {target_lang.upper()}")

    # Initialize translator
    translator = MarianTranslator(
        source_lang="en",
        target_lang=target_lang,
        config=config,
        device=device,
    )
    translator.load_model()

    # Extract all English captions
    en_captions = [r["generated_caption"] for r in source_results]

    # Translate in batches
    all_translations = []
    start_time = time.time()

    for i in tqdm(range(0, len(en_captions), batch_size), desc=f"  Translating [{target_lang}]"):
        batch = en_captions[i : i + batch_size]
        translated = translator.translate_batch(batch)
        all_translations.extend(translated)

    elapsed = time.time() - start_time
    avg_time = elapsed / len(all_translations) if all_translations else 0

    print(f"  Completed: {len(all_translations)} translations in {elapsed:.1f}s")
    print(f"  Average: {avg_time:.4f}s per caption")

    # Build output records
    results = []
    for i, source_record in enumerate(source_results):
        record = {
            "id": source_record["id"],
            "image_filename": source_record["image_filename"],
            "source_language": "en",
            "target_language": target_lang,
            "source_caption": source_record["generated_caption"],
            "translated_caption": all_translations[i],
            "reference_caption": "",  # Will be filled from dataset
            "reference_en": source_record.get("reference_en", source_record.get("reference_caption", "")),
        }
        results.append(record)

    # Add reference captions from dataset
    try:
        from src.utils.data_loader import Multi30KLoader
        loader = Multi30KLoader(config)
        split = source_metadata.get("split", "test_2016_flickr")
        test_data = loader.load_split(split)

        # Build lookup by id
        ref_lookup = {s["id"]: s["captions"].get(target_lang, "") for s in test_data}
        for record in results:
            record["reference_caption"] = ref_lookup.get(record["id"], "")
    except Exception as e:
        print(f"  WARNING: Could not load reference captions: {e}")

    # Build output
    output_data = {
        "metadata": {
            "source_model": source_metadata.get("model", "unknown"),
            "translation_model": translator.model_name,
            "source_language": "en",
            "target_language": target_lang,
            "split": source_metadata.get("split", "unknown"),
            "num_samples": len(results),
            "batch_size": batch_size,
            "total_time_seconds": round(elapsed, 2),
            "avg_time_per_caption": round(avg_time, 4),
            "translation_config": translator.trans_config["generation"],
            "source_file": str(input_path.name),
        },
        "results": results,
    }

    # Save
    filename = f"translations_en_{target_lang}_{source_metadata.get('split', 'test')}.json"
    output_path = get_output_path(config, "translations_dir", filename)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"  Saved: {output_path}")

    return output_data


def main():
    parser = argparse.ArgumentParser(description="MarianMT Translation")
    parser.add_argument("--target-lang", type=str, default=None,
                        choices=["de", "fr"],
                        help="Target language (or use --all for both)")
    parser.add_argument("--all", action="store_true",
                        help="Translate to all configured target languages")
    parser.add_argument("--input", type=str, default=None,
                        help="Path to EN captions JSON (auto-detected if not set)")
    parser.add_argument("--split", type=str, default="test_2016_flickr",
                        help="Dataset split name")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Translation batch size")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto, cuda, cpu")

    args = parser.parse_args()

    config = load_config()

    print("============================================")
    print("  BP-FLAMINGO: MarianMT Translation")
    print("============================================")

    # Determine input file
    if args.input:
        input_path = args.input
    else:
        input_path = Path(config["paths"]["captions_dir"]) / f"captions_en_only_{args.split}.json"

    print(f"  Input: {input_path}")

    if not Path(input_path).exists():
        print(f"\n  ERROR: Input file not found: {input_path}")
        print(f"  Run OpenFlamingo inference first (Step 3).")
        sys.exit(1)

    # Determine target languages
    if args.all:
        target_langs = ["de", "fr"]
    elif args.target_lang:
        target_langs = [args.target_lang]
    else:
        target_langs = ["de", "fr"]
        print(f"  No --target-lang specified, translating to: {target_langs}")

    total_start = time.time()

    for lang in target_langs:
        print(f"\n{'='*50}")
        print(f"  Translating: EN → {lang.upper()}")
        print(f"{'='*50}")

        translate_captions_file(
            input_path=input_path,
            target_lang=lang,
            config=config,
            device=args.device,
            batch_size=args.batch_size,
        )

    total_elapsed = time.time() - total_start
    print(f"\n{'='*50}")
    print(f"  All translations completed in {total_elapsed:.1f}s")
    print(f"  ({total_elapsed/60:.1f} minutes)")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()