"""
BP-FLAMINGO: Multi30K Data Loader

Loads processed Multi30K JSON files and provides easy access to samples.
Used by all pipeline stages (inference, translation, evaluation).

Usage:
    from src.utils.data_loader import Multi30KLoader

    loader = Multi30KLoader()
    
    # Load test split
    test_data = loader.load_split("test_2016_flickr")
    
    # Get a sample
    sample = test_data[0]
    print(sample["image_path"])
    print(sample["captions"]["en"])
    
    # Get few-shot examples from train split
    few_shot = loader.get_few_shot_examples(n=4, seed=42)
    
    # Get only samples with existing images
    valid = loader.load_split("test_2016_flickr", require_images=True)
"""

import json
import random
from pathlib import Path
from typing import Optional
from PIL import Image

from src.utils.config import load_config, find_project_root


class Multi30KLoader:
    """Loader for processed Multi30K dataset."""

    def __init__(self, config: Optional[dict] = None):
        """
        Initialize the data loader.

        Args:
            config: Optional config dict. If None, loads from default location.
        """
        self.config = config or load_config()
        self.root = find_project_root()
        self.processed_dir = self.root / "data" / "multi30k" / "processed"
        self.languages = self.config["dataset"]["languages"]
        self._cache = {}  # Cache loaded splits

    def load_split(
        self,
        split: str,
        require_images: bool = False,
    ) -> list[dict]:
        """
        Load a dataset split from processed JSON.

        Args:
            split: Split name ("train", "val", "test_2016_flickr").
            require_images: If True, only return samples where image file exists.

        Returns:
            List of sample dictionaries.
        """
        cache_key = (split, require_images)
        if cache_key in self._cache:
            return self._cache[cache_key]

        json_path = self.processed_dir / f"{split}.json"
        if not json_path.exists():
            raise FileNotFoundError(
                f"Processed data not found: {json_path}\n"
                f"Run: python3 src/utils/data_processor.py"
            )

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if require_images:
            data = [s for s in data if s.get("image_exists", False)]

        self._cache[cache_key] = data
        return data

    def get_few_shot_examples(
        self,
        n: int = 4,
        seed: int = 42,
        split: str = None,
        require_images: bool = True,
    ) -> list[dict]:
        """
        Get random few-shot examples for in-context learning.

        Args:
            n: Number of examples.
            seed: Random seed for reproducibility.
            split: Which split to sample from. Default: from config.
            require_images: If True, only samples with existing images.

        Returns:
            List of n sample dictionaries.
        """
        if split is None:
            split = self.config["dataset"]["few_shot_split"]

        data = self.load_split(split, require_images=require_images)

        rng = random.Random(seed)
        examples = rng.sample(data, min(n, len(data)))

        return examples

    def load_image(self, sample: dict) -> Optional[Image.Image]:
        """
        Load an image for a given sample.

        Args:
            sample: A sample dictionary from load_split().

        Returns:
            PIL Image or None if image doesn't exist.
        """
        image_path = Path(sample["image_path"])

        # Handle both absolute and relative paths
        if not image_path.is_absolute():
            image_path = self.root / image_path

        if not image_path.exists():
            return None

        return Image.open(image_path).convert("RGB")

    def get_references(
        self,
        split: str,
        language: str,
    ) -> list[str]:
        """
        Get all reference captions for a split in a specific language.

        Args:
            split: Split name.
            language: Language code ("en", "de", "fr").

        Returns:
            List of reference caption strings.
        """
        data = self.load_split(split)
        return [sample["captions"][language] for sample in data]

    def get_summary(self) -> dict:
        """Load and return the dataset summary."""
        summary_path = self.processed_dir / "dataset_summary.json"
        if not summary_path.exists():
            return {"error": "Summary not found. Run data_processor.py first."}

        with open(summary_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def __repr__(self) -> str:
        summary = self.get_summary()
        if "error" in summary:
            return f"Multi30KLoader(not processed yet)"
        return (
            f"Multi30KLoader("
            f"splits={[s['split'] for s in summary['splits']]}, "
            f"total={summary['total_samples']}, "
            f"languages={self.languages})"
        )


if __name__ == "__main__":
    # Quick test
    loader = Multi30KLoader()
    print(loader)
    print()

    summary = loader.get_summary()
    if "error" not in summary:
        for split_info in summary["splits"]:
            split = split_info["split"]
            data = loader.load_split(split)
            print(f"  {split}: {len(data)} samples")
            print(f"    First caption (EN): {data[0]['captions']['en'][:80]}...")
            print(f"    First caption (DE): {data[0]['captions']['de'][:80]}...")
            print(f"    First caption (FR): {data[0]['captions']['fr'][:80]}...")
            print()

        # Test few-shot
        examples = loader.get_few_shot_examples(n=2)
        print(f"  Few-shot examples: {len(examples)} samples")
        for ex in examples:
            print(f"    [{ex['image_filename']}] {ex['captions']['en'][:60]}...")