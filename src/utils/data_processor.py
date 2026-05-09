"""
BP-FLAMINGO: Multi30K Data Processor

Converts raw Multi30K text files + images into a unified JSON format
suitable for the experimental pipeline.

Usage:
    cd ~/bp-flamingo
    source venv/bin/activate
    python3 src/utils/data_processor.py

Output format (per split):
    data/multi30k/processed/{split}.json

Each entry:
{
    "id": 0,
    "image_filename": "1000092795.jpg",
    "image_path": "data/multi30k/raw/images/1000092795.jpg",
    "captions": {
        "en": "Two young guys with shaggy hair looking at their hands ...",
        "de": "Zwei junge Männer mit zotteligem Haar ...",
        "fr": "Deux jeunes hommes aux cheveux hirsutes ..."
    }
}
"""

import json
import os
import sys
from pathlib import Path


def find_project_root() -> Path:
    """Find project root by looking for configs/config.yaml."""
    current = Path(__file__).resolve().parent
    for parent in [current] + list(current.parents):
        if (parent / "configs" / "config.yaml").exists():
            return parent
    raise FileNotFoundError("Could not find project root.")


def load_text_file(filepath: Path) -> list[str]:
    """Load a text file, one sentence per line."""
    with open(filepath, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]
    return lines


def process_split(
    split_name: str,
    text_dir: Path,
    image_dir: Path,
    output_dir: Path,
    languages: list[str] = ["en", "de", "fr"],
) -> dict:
    """
    Process one data split (train/val/test) into unified JSON.

    Returns:
        Dictionary with statistics about the processed split.
    """
    print(f"\n  Processing split: {split_name}")

    # --- Load image filenames ---
    image_list_file = text_dir / f"{split_name}.images"
    if not image_list_file.exists():
        print(f"    WARNING: Image list not found: {image_list_file}")
        print(f"    Skipping split: {split_name}")
        return None

    image_filenames = load_text_file(image_list_file)
    # Some entries may have '#N' suffix (e.g., '1000092795.jpg#0')
    # We want just the filename
    image_filenames = [fname.split("#")[0] for fname in image_filenames]
    print(f"    Image entries: {len(image_filenames)}")

    # --- Load captions for each language ---
    captions = {}
    for lang in languages:
        caption_file = text_dir / f"{split_name}.{lang}"
        if not caption_file.exists():
            print(f"    WARNING: Caption file not found: {caption_file}")
            continue
        captions[lang] = load_text_file(caption_file)
        print(f"    Captions [{lang}]: {len(captions[lang])} lines")

    # --- Verify all have same length ---
    lengths = {lang: len(caps) for lang, caps in captions.items()}
    lengths["images"] = len(image_filenames)

    if len(set(lengths.values())) != 1:
        print(f"    ERROR: Mismatched lengths: {lengths}")
        sys.exit(1)

    n_samples = len(image_filenames)
    print(f"    Verified: all {n_samples} entries aligned across languages.")

    # --- Check image availability ---
    images_found = 0
    images_missing = 0
    for fname in set(image_filenames):
        if (image_dir / fname).exists():
            images_found += 1
        else:
            images_missing += 1

    unique_images = len(set(image_filenames))
    print(f"    Unique images: {unique_images}")
    print(f"    Images found: {images_found}/{unique_images}", end="")
    if images_missing > 0:
        print(f" ({images_missing} missing)")
    else:
        print(" (all present)")

    # --- Build unified dataset ---
    dataset = []
    for idx in range(n_samples):
        entry = {
            "id": idx,
            "image_filename": image_filenames[idx],
            "image_path": str(image_dir / image_filenames[idx]),
            "image_exists": (image_dir / image_filenames[idx]).exists(),
            "captions": {}
        }
        for lang in languages:
            if lang in captions:
                entry["captions"][lang] = captions[lang][idx]

        dataset.append(entry)

    # --- Save to JSON ---
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{split_name}.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"    Saved: {output_file} ({n_samples} entries)")

    # --- Stats ---
    stats = {
        "split": split_name,
        "n_samples": n_samples,
        "n_unique_images": unique_images,
        "n_images_found": images_found,
        "n_images_missing": images_missing,
        "languages": list(captions.keys()),
        "caption_lengths": {},
    }

    # Average caption length per language
    for lang in captions:
        avg_len = sum(len(c.split()) for c in captions[lang]) / len(captions[lang])
        stats["caption_lengths"][lang] = round(avg_len, 1)

    return stats


def main():
    root = find_project_root()
    text_dir = root / "data" / "multi30k" / "raw" / "texts"
    image_dir = root / "data" / "multi30k" / "raw" / "images"
    output_dir = root / "data" / "multi30k" / "processed"

    print("============================================")
    print("  BP-FLAMINGO: Multi30K Data Processor")
    print(f"  Text dir:   {text_dir}")
    print(f"  Image dir:  {image_dir}")
    print(f"  Output dir: {output_dir}")
    print("============================================")

    # Check text data exists
    if not text_dir.exists():
        print("\nERROR: Text data not found. Run download_multi30k.sh first.")
        sys.exit(1)

    # Process all splits
    splits = ["train", "val", "test_2016_flickr"]
    all_stats = []

    for split in splits:
        stats = process_split(split, text_dir, image_dir, output_dir)
        if stats:
            all_stats.append(stats)

    # --- Save dataset summary ---
    summary = {
        "dataset": "Multi30K",
        "description": "Multilingual image captioning dataset (EN, DE, FR)",
        "source": "https://github.com/multi30k/dataset",
        "splits": all_stats,
        "total_samples": sum(s["n_samples"] for s in all_stats),
    }

    summary_file = output_dir / "dataset_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # --- Print summary ---
    print("\n============================================")
    print("  Dataset Summary")
    print("============================================")
    print(f"  Total samples: {summary['total_samples']}")
    print()

    for s in all_stats:
        print(f"  {s['split']}:")
        print(f"    Samples:      {s['n_samples']}")
        print(f"    Languages:    {', '.join(s['languages'])}")
        print(f"    Images found: {s['n_images_found']}/{s['n_unique_images']}")
        print(f"    Avg caption length (words):")
        for lang, avg in s["caption_lengths"].items():
            print(f"      {lang}: {avg}")
        print()

    print(f"  Summary saved: {summary_file}")
    print("============================================")


if __name__ == "__main__":
    main()