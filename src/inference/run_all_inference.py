"""
BP-FLAMINGO: Run All Inference Experiments

Runs OpenFlamingo inference in all configured modes:
  1. EN-only: Generate English captions
  2. Direct DE: Generate German captions via prompt
  3. Direct FR: Generate French captions via prompt

Usage:
    PYTHONPATH=. python3 src/inference/run_all_inference.py
    PYTHONPATH=. python3 src/inference/run_all_inference.py --max-samples 10  # quick test
"""

import argparse
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.flamingo_inference import FlamingoInference
from src.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Run all inference experiments")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit samples per experiment (for testing)")
    parser.add_argument("--split", type=str, default="test_2016_flickr",
                        help="Dataset split to evaluate")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-few-shot", type=int, default=4)
    args = parser.parse_args()

    config = load_config()

    print("============================================")
    print("  BP-FLAMINGO: Run All Inference Experiments")
    print("============================================")
    print()

    # Load model once, reuse for all experiments
    engine = FlamingoInference(config=config, device=args.device)
    engine.load_model()

    # Define experiments
    experiments = [
        {"language": "en", "description": "EN-only (English captions)"},
        {"language": "de", "description": "Direct multilingual (German captions)"},
        {"language": "fr", "description": "Direct multilingual (French captions)"},
    ]

    total_start = time.time()

    for i, exp in enumerate(experiments, 1):
        print(f"\n{'='*50}")
        print(f"  Experiment {i}/{len(experiments)}: {exp['description']}")
        print(f"{'='*50}")

        engine.run_inference(
            split=args.split,
            language=exp["language"],
            num_few_shot=args.num_few_shot,
            seed=args.seed,
            max_samples=args.max_samples,
        )

    total_elapsed = time.time() - total_start
    print(f"\n{'='*50}")
    print(f"  All experiments completed in {total_elapsed:.1f}s")
    print(f"  ({total_elapsed/60:.1f} minutes)")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()