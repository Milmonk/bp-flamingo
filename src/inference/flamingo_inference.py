"""
BP-FLAMINGO: OpenFlamingo Inference Module

Loads the OpenFlamingo model and generates image captions.
Supports three modes:
  1. EN-only: Generate English captions
  2. Direct multilingual: Generate captions in target language via prompt
  3. (Translation mode is handled by the translation module)

Usage:
    # From project root:
    PYTHONPATH=. python3 src/inference/flamingo_inference.py

    # Or via SLURM:
    sbatch scripts/slurm/run_inference.sh
"""

import json
import time
import argparse
import sys
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from tqdm import tqdm

from open_flamingo import create_model_and_transforms
from huggingface_hub import hf_hub_download

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config, get_output_path
from src.utils.data_loader import Multi30KLoader


class FlamingoInference:
    """
    OpenFlamingo inference wrapper.
    
    Handles model loading, prompt construction, and caption generation
    for both English and multilingual modes.
    """

    def __init__(self, config: Optional[dict] = None, device: str = "auto"):
        """
        Initialize the inference engine.

        Args:
            config: Optional config dict. If None, loads from default.
            device: Device to use ("auto", "cuda", "cpu").
        """
        self.config = config or load_config()
        self.of_config = self.config["openflamingo"]
        
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"  Device: {self.device}")
        if self.device.type == "cuda":
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        self.model = None
        self.image_processor = None
        self.tokenizer = None

    def load_model(self):
        """Load OpenFlamingo model, image processor, and tokenizer."""
        print(f"\n  Loading OpenFlamingo model: {self.of_config['model_name']}")
        start_time = time.time()

        # Create model and transforms
        model, image_processor, tokenizer = create_model_and_transforms(
            clip_vision_encoder_path=self.of_config["vision_encoder"],
            clip_vision_encoder_pretrained=self.of_config["vision_encoder_pretrained"],
            lang_encoder_path="anas-awadalla/mpt-1b-redpajama-200b",
            tokenizer_path="anas-awadalla/mpt-1b-redpajama-200b",
            cross_attn_every_n_layers=1,
        )

        # Download and load checkpoint
        print("  Downloading checkpoint (this may take a while on first run)...")
        checkpoint_path = hf_hub_download(
            repo_id=self.of_config["model_name"],
            filename="checkpoint.pt",
        )
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"), strict=False)

        # Move to device and set eval mode
        model = model.to(self.device)
        model.eval()

        # Configure tokenizer
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        self.model = model
        self.image_processor = image_processor
        self.tokenizer = tokenizer

        elapsed = time.time() - start_time
        print(f"  Model loaded in {elapsed:.1f}s")
        
        # Print model size
        param_count = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {param_count / 1e9:.2f}B")

    def _prepare_image(self, image: Image.Image) -> torch.Tensor:
        """Process a single image for the model."""
        return self.image_processor(image).unsqueeze(0)

    def _build_prompt(
        self,
        few_shot_examples: list[dict],
        language: str = "en",
    ) -> tuple[str, torch.Tensor]:
        """
        Build the few-shot prompt with interleaved images and text.

        Args:
            few_shot_examples: List of sample dicts with captions and images.
            language: Target language for caption generation.

        Returns:
            Tuple of (prompt_text, vision_tensors).
        """
        # Language-specific prompt templates
        prompt_templates = {
            "en": "Output:",
            "de": "German description:",
            "fr": "French description:",
        }
        
        prompt_prefix = prompt_templates.get(language, "Output:")
        
        # For direct multilingual mode, use target language captions in few-shot
        # For EN mode, use English captions
        caption_lang = language if language in ["de", "fr"] else "en"
        
        prompt_parts = []
        vision_tensors = []

        for example in few_shot_examples:
            # Load and process image
            image_path = Path(example["image_path"])
            if not image_path.is_absolute():
                image_path = PROJECT_ROOT / image_path
            
            img = Image.open(image_path).convert("RGB")
            vision_tensors.append(self._prepare_image(img))

            # Use target language caption if available, fall back to English
            caption = example["captions"].get(caption_lang, "")
            if not caption:
                caption = example["captions"].get("en", "")
            prompt_parts.append(f"<image>{prompt_prefix} {caption}<|endofchunk|>")

        # Final query image placeholder
        prompt_parts.append(f"<image>{prompt_prefix}")

        prompt_text = "".join(prompt_parts)
        
        return prompt_text, vision_tensors

    def generate_caption(
        self,
        image: Image.Image,
        few_shot_examples: list[dict],
        language: str = "en",
    ) -> dict:
        """
        Generate a caption for a single image.

        Args:
            image: PIL Image to caption.
            few_shot_examples: List of few-shot example dicts.
            language: Target language ("en", "de", "fr").

        Returns:
            Dict with generated caption and metadata.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Build prompt
        prompt_text, few_shot_vision = self._build_prompt(few_shot_examples, language)

        # Process query image
        query_vision = self._prepare_image(image)

        # Stack all vision inputs: [few_shot_1, few_shot_2, ..., query]
        all_vision = few_shot_vision + [query_vision]
        vision_x = torch.cat(all_vision, dim=0).unsqueeze(0).unsqueeze(2).to(self.device)
        # Shape: [batch=1, n_images, frames=1, channels, height, width]

        # Tokenize prompt
        lang_x = self.tokenizer(
            [prompt_text],
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        # Generate
        gen_config = self.of_config["generation"]
        
        with torch.no_grad():
            generated = self.model.generate(
                vision_x=vision_x,
                lang_x=lang_x["input_ids"],
                attention_mask=lang_x["attention_mask"],
                max_new_tokens=gen_config["max_new_tokens"],
                num_beams=gen_config["num_beams"],
                temperature=gen_config["temperature"],
                top_k=gen_config["top_k"] if gen_config["top_k"] > 0 else None,
                top_p=gen_config["top_p"],
                repetition_penalty=gen_config["repetition_penalty"],
            )

        # Decode — only the newly generated tokens
        input_len = lang_x["input_ids"].shape[1]
        generated_tokens = generated[0, input_len:]
        raw_caption = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        caption = raw_caption

        # Clean up caption
        caption = caption.strip()
        # Remove any trailing special tokens
        for stop_token in ["<|endofchunk|>", "<image>", "\n"]:
            if stop_token in caption:
                caption = caption[:caption.index(stop_token)].strip()

        # Remove prompt template leaking (e.g., "German description:", "Output:")
        for template in ["Output:", "German description:", "French description:"]:
            if template in caption:
                caption = caption[:caption.index(template)].strip()
        
        # Remove repetitions: if caption contains itself, keep first occurrence
        # Split into sentences and check for duplicates
        if len(caption) > 10:
            half = len(caption) // 2
            # Check if the second half is a repeat of the first
            for i in range(half - 5, half + 5):
                if i > 0 and caption[:i].strip() == caption[i:2*i].strip():
                    caption = caption[:i].strip()
                    break
        
        # Remove trailing period duplicates
        while caption.endswith(".."):
            caption = caption[:-1]

        return {
            "caption": caption,
            "raw_caption": raw_caption,
            "language": language,
            "prompt_template": language,
            "num_few_shot": len(few_shot_examples),
            "num_beams": gen_config["num_beams"],
        }

    def run_inference(
        self,
        split: str = "test_2016_flickr",
        language: str = "en",
        num_few_shot: int = 4,
        seed: int = 42,
        max_samples: Optional[int] = None,
        output_suffix: str = "",
    ) -> list[dict]:
        """
        Run inference on a full dataset split.

        Args:
            split: Dataset split to evaluate.
            language: Target language for generation.
            num_few_shot: Number of few-shot examples.
            seed: Random seed for few-shot selection.
            max_samples: Limit number of samples (None = all).
            output_suffix: Optional suffix for output filename.

        Returns:
            List of result dictionaries.
        """
        if self.model is None:
            self.load_model()

        # Load data
        loader = Multi30KLoader(self.config)
        test_data = loader.load_split(split, require_images=True)
        few_shot_examples = loader.get_few_shot_examples(
            n=num_few_shot, seed=seed
        )

        if max_samples:
            test_data = test_data[:max_samples]

        print(f"\n  Running inference:")
        print(f"    Split: {split}")
        print(f"    Language: {language}")
        print(f"    Few-shot examples: {num_few_shot}")
        print(f"    Samples: {len(test_data)}")
        print()

        results = []
        start_time = time.time()

        for sample in tqdm(test_data, desc=f"  Generating [{language}]"):
            # Load image
            image = loader.load_image(sample)
            if image is None:
                print(f"    WARNING: Image not found: {sample['image_filename']}")
                continue

            # Generate caption
            result = self.generate_caption(image, few_shot_examples, language)

            # Build output record
            record = {
                "id": sample["id"],
                "image_filename": sample["image_filename"],
                "language": language,
                "generated_caption": result["caption"],
                "raw_caption": result["raw_caption"],
                "reference_caption": sample["captions"].get(language, ""),
                "reference_en": sample["captions"].get("en", ""),
                "num_few_shot": num_few_shot,
                "seed": seed,
            }
            results.append(record)

        elapsed = time.time() - start_time
        avg_time = elapsed / len(results) if results else 0

        print(f"\n  Completed: {len(results)} captions in {elapsed:.1f}s")
        print(f"  Average: {avg_time:.2f}s per sample")

        # Save results
        mode_name = f"en_only" if language == "en" else f"direct_{language}"
        filename = f"captions_{mode_name}_{split}"
        if output_suffix:
            filename += f"_{output_suffix}"
        filename += ".json"

        output_path = get_output_path(self.config, "captions_dir", filename)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({
                "metadata": {
                    "model": self.of_config["model_name"],
                    "split": split,
                    "language": language,
                    "num_few_shot": num_few_shot,
                    "seed": seed,
                    "num_samples": len(results),
                    "total_time_seconds": round(elapsed, 2),
                    "avg_time_per_sample": round(avg_time, 4),
                    "generation_config": self.of_config["generation"],
                },
                "results": results,
            }, f, ensure_ascii=False, indent=2)

        print(f"  Saved: {output_path}")

        return results


def main():
    parser = argparse.ArgumentParser(description="OpenFlamingo Inference")
    parser.add_argument("--language", type=str, default="en",
                        choices=["en", "de", "fr"],
                        help="Target language for caption generation")
    parser.add_argument("--split", type=str, default="test_2016_flickr",
                        help="Dataset split to evaluate")
    parser.add_argument("--num-few-shot", type=int, default=4,
                        help="Number of few-shot examples")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for few-shot selection")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit number of samples (for testing)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto, cuda, cpu")
    parser.add_argument("--suffix", type=str, default="",
                        help="Optional suffix for output filename")
    
    args = parser.parse_args()

    print("============================================")
    print("  BP-FLAMINGO: OpenFlamingo Inference")
    print("============================================")

    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    engine = FlamingoInference(device=args.device)
    engine.load_model()

    engine.run_inference(
        split=args.split,
        language=args.language,
        num_few_shot=args.num_few_shot,
        seed=args.seed,
        max_samples=args.max_samples,
        output_suffix=args.suffix,
    )


if __name__ == "__main__":
    main()