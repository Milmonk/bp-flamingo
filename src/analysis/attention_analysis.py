"""
BP-FLAMINGO: Cross-Attention Visualisation

Extracts and visualises cross-modal attention patterns from OpenFlamingo.
Shows which image regions the model attends to when generating each word.

Usage:
    PYTHONPATH=. python3 src/analysis/attention_analysis.py
    
    # Must run on GPU node:
    sbatch scripts/slurm/run_analysis.sh
"""

import json
import sys
import os
from pathlib import Path

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for HPC
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

from open_flamingo import create_model_and_transforms
from huggingface_hub import hf_hub_download

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config
from src.utils.data_loader import Multi30KLoader


class AttentionExtractor:
    """
    Extracts cross-attention weights from OpenFlamingo during generation.
    
    Uses PyTorch hooks to capture attention patterns from the
    cross-attention (gated_cross_attn) layers.
    """

    def __init__(self, config=None, device="auto"):
        self.config = config or load_config()
        self.of_config = self.config["openflamingo"]
        
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = None
        self.image_processor = None
        self.tokenizer = None
        self.attention_maps = {}
        self.hooks = []

    def load_model(self):
        """Load OpenFlamingo model."""
        print(f"  Loading OpenFlamingo model...")

        model, image_processor, tokenizer = create_model_and_transforms(
            clip_vision_encoder_path=self.of_config["vision_encoder"],
            clip_vision_encoder_pretrained=self.of_config["vision_encoder_pretrained"],
            lang_encoder_path="anas-awadalla/mpt-1b-redpajama-200b",
            tokenizer_path="anas-awadalla/mpt-1b-redpajama-200b",
            cross_attn_every_n_layers=1,
        )

        checkpoint_path = hf_hub_download(
            repo_id=self.of_config["model_name"],
            filename="checkpoint.pt",
        )
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"), strict=False)
        model = model.to(self.device)
        model.eval()

        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        self.model = model
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        print(f"  Model loaded on {self.device}")

    def _register_hooks(self):
        """Register forward hooks on cross-attention layers."""
        self.attention_maps = {}
        self.hooks = []

        # Find cross-attention layers in the language model
        for name, module in self.model.lang_encoder.named_modules():
            if "gated_cross_attn" in name and hasattr(module, "attn"):
                layer_name = name

                def make_hook(lname):
                    def hook_fn(module, input, output):
                        # Store the output — attention patterns
                        if isinstance(output, tuple) and len(output) > 1:
                            # Some implementations return (attn_output, attn_weights)
                            self.attention_maps[lname] = output[1].detach().cpu()
                        elif isinstance(output, torch.Tensor):
                            self.attention_maps[lname] = output.detach().cpu()
                    return hook_fn

                hook = module.attn.register_forward_hook(make_hook(layer_name))
                self.hooks.append(hook)

    def _remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def generate_with_attention(
        self,
        image: Image.Image,
        few_shot_examples: list[dict],
        language: str = "en",
    ) -> dict:
        """
        Generate a caption and extract attention information.
        
        Returns dict with caption, tokens, and available attention data.
        """
        prompt_templates = {
            "en": "Output:",
            "de": "German description:",
            "fr": "French description:",
        }
        prefix = prompt_templates.get(language, "Output:")

        # Build few-shot prompt
        prompt_parts = []
        vision_tensors = []

        for ex in few_shot_examples:
            img_path = Path(ex["image_path"])
            if not img_path.is_absolute():
                img_path = PROJECT_ROOT / img_path
            img = Image.open(img_path).convert("RGB")
            vision_tensors.append(self.image_processor(img).unsqueeze(0))

            cap_lang = language if language in ex["captions"] else "en"
            caption = ex["captions"].get(cap_lang, ex["captions"]["en"])
            prompt_parts.append(f"<image>{prefix} {caption}<|endofchunk|>")

        prompt_parts.append(f"<image>{prefix}")
        prompt_text = "".join(prompt_parts)

        # Process query image
        query_vision = self.image_processor(image).unsqueeze(0)
        all_vision = vision_tensors + [query_vision]
        vision_x = torch.cat(all_vision, dim=0).unsqueeze(0).unsqueeze(2).to(self.device)

        # Tokenize
        lang_x = self.tokenizer(
            [prompt_text], return_tensors="pt", padding=True
        ).to(self.device)

        # Register hooks and generate
        self._register_hooks()

        with torch.no_grad():
            generated = self.model.generate(
                vision_x=vision_x,
                lang_x=lang_x["input_ids"],
                attention_mask=lang_x["attention_mask"],
                max_new_tokens=30,
                num_beams=1,  # Greedy for interpretability
            )

        self._remove_hooks()

        # Decode
        input_len = lang_x["input_ids"].shape[1]
        gen_tokens = generated[0, input_len:]
        caption = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

        # Clean caption
        for stop in ["<|endofchunk|>", "<image>", "\n", "Output:", "German description:", "French description:"]:
            if stop in caption:
                caption = caption[:caption.index(stop)].strip()

        # Get individual token strings
        token_strings = [self.tokenizer.decode([t]) for t in gen_tokens.tolist()]

        return {
            "caption": caption,
            "token_strings": token_strings,
            "num_attention_layers": len(self.attention_maps),
            "attention_layer_names": list(self.attention_maps.keys()),
        }


def create_architecture_flowchart(output_dir: Path):
    """
    Create a structured flowchart of the OpenFlamingo architecture.
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 16))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 20)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # Colors
    c_input = "#e3f2fd"
    c_vision = "#bbdefb"
    c_bridge = "#fff9c4"
    c_lang = "#c8e6c9"
    c_cross = "#ffccbc"
    c_output = "#f3e5f5"
    c_border = "#37474f"
    c_arrow = "#455a64"

    box_props = dict(boxstyle="round,pad=0.4", facecolor=c_input, edgecolor=c_border, linewidth=1.5)

    def draw_box(x, y, w, h, text, color, fontsize=11, bold=False):
        props = dict(boxstyle=f"round,pad=0.4", facecolor=color, edgecolor=c_border, linewidth=1.5)
        weight = "bold" if bold else "normal"
        ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
                fontweight=weight, bbox=props, transform=ax.transData)

    def draw_arrow(x1, y1, x2, y2, text=""):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                     arrowprops=dict(arrowstyle="-|>", color=c_arrow, lw=1.5))
        if text:
            mx, my = (x1+x2)/2, (y1+y2)/2
            ax.text(mx + 0.3, my, text, fontsize=8, color="#666", style="italic")

    # Title
    ax.text(5, 19.3, "OpenFlamingo-3B Architecture", ha="center", va="center",
            fontsize=16, fontweight="bold", color="#1a3a5c")
    ax.text(5, 18.8, "Vision-Language Model with Gated Cross-Attention",
            ha="center", va="center", fontsize=10, color="#666")

    # === INPUT LAYER ===
    draw_box(2.5, 17.5, 3, 0.8, "Input Image\n(224 × 224 × 3)", c_input, 10)
    draw_box(7.5, 17.5, 3, 0.8, "Text Prompt\n\"<image>Output:\"", c_input, 10)

    # === VISION ENCODER ===
    draw_arrow(2.5, 17.0, 2.5, 16.3)
    draw_box(2.5, 15.8, 3, 0.8, "ViT-L/14 (CLIP)\nVision Encoder", c_vision, 10, bold=True)
    ax.text(4.8, 15.8, "Frozen", fontsize=8, color="#1565c0", style="italic")

    draw_arrow(2.5, 15.2, 2.5, 14.5)
    ax.text(3.8, 14.8, "257 × 1024\n(patch embeddings)", fontsize=8, color="#666", ha="left")

    # === PERCEIVER RESAMPLER ===
    draw_box(2.5, 13.9, 3, 0.8, "Perceiver Resampler\n(6 layers)", c_bridge, 10, bold=True)
    ax.text(4.8, 13.9, "Trainable", fontsize=8, color="#e65100", style="italic")

    draw_arrow(2.5, 13.3, 2.5, 12.6)
    ax.text(3.8, 12.9, "64 × 1024\n(visual tokens)", fontsize=8, color="#666", ha="left")

    # === TOKENIZER ===
    draw_arrow(7.5, 17.0, 7.5, 16.3)
    draw_box(7.5, 15.8, 3, 0.8, "MPT Tokenizer\n(vocab: 50,432)", c_input, 10, bold=True)

    draw_arrow(7.5, 15.2, 7.5, 14.5)
    ax.text(6.2, 14.8, "Token IDs\n+ embeddings", fontsize=8, color="#666", ha="right")

    # === LANGUAGE MODEL BLOCKS ===
    # Block 1
    draw_box(7.5, 13.9, 3, 0.6, "Self-Attention Layer", c_lang, 10)
    draw_arrow(7.5, 13.3, 7.5, 12.8)

    # Cross-attention
    draw_box(5, 12.2, 4.5, 0.8, "Gated Cross-Attention\n(visual tokens ↔ text tokens)", c_cross, 10, bold=True)
    ax.text(8, 12.2, "Trainable", fontsize=8, color="#e65100", style="italic")

    # Arrow from Perceiver to Cross-Attention
    draw_arrow(2.5, 12.2, 3.0, 12.2, "visual\ntokens")

    draw_arrow(5, 11.5, 5, 10.8)

    # Feed-forward
    draw_box(5, 10.3, 3, 0.6, "Feed-Forward Network", c_lang, 10)
    ax.text(7.2, 10.3, "Frozen", fontsize=8, color="#1565c0", style="italic")

    # Repeat indicator
    draw_arrow(5, 9.7, 5, 9.2)
    ax.text(5, 8.8, "× 24 transformer blocks\n(cross-attn every layer)", ha="center",
            fontsize=9, color="#666", style="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff", edgecolor="#ccc", linewidth=1, linestyle="dashed"))

    draw_arrow(5, 8.3, 5, 7.6)

    # === LM HEAD ===
    draw_box(5, 7.1, 3, 0.6, "LM Head (Linear)", c_lang, 10)

    draw_arrow(5, 6.5, 5, 5.8)

    # === OUTPUT ===
    draw_box(5, 5.3, 4, 0.8, "Generated Caption\n\"A man wearing a beer hat.\"", c_output, 10, bold=True)

    # === LEGEND ===
    legend_y = 3.5
    ax.text(1.2, legend_y + 0.8, "Legend:", fontsize=10, fontweight="bold", color="#333")

    legend_items = [
        (c_vision, "Vision encoder (frozen CLIP)"),
        (c_bridge, "Perceiver Resampler (trainable)"),
        (c_cross, "Gated Cross-Attention (trainable)"),
        (c_lang, "Language model layers (frozen MPT-1B)"),
    ]
    for i, (color, label) in enumerate(legend_items):
        y = legend_y - i * 0.5
        ax.add_patch(plt.Rectangle((1.2, y - 0.15), 0.4, 0.3,
                                    facecolor=color, edgecolor=c_border, linewidth=1))
        ax.text(1.9, y, label, fontsize=9, va="center")

    # Training info
    ax.text(1.2, 1.5, "Training strategy:", fontsize=9, fontweight="bold", color="#333")
    ax.text(1.2, 1.0, "Only Perceiver Resampler and Gated Cross-Attention layers are trained.",
            fontsize=9, color="#555")
    ax.text(1.2, 0.6, "Vision encoder (CLIP) and language model (MPT-1B) remain frozen.",
            fontsize=9, color="#555")
    ax.text(1.2, 0.2, "Total: 2.56B parameters, 1.05B trainable.",
            fontsize=9, color="#555")

    plt.tight_layout()
    output_path = output_dir / "flamingo_architecture_flowchart.png"
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {output_path}")


def create_pipeline_flowchart(output_dir: Path):
    """
    Create a flowchart showing the full experimental pipeline.
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    c_data = "#e3f2fd"
    c_model = "#c8e6c9"
    c_eval = "#fff9c4"
    c_result = "#f3e5f5"
    c_border = "#37474f"
    c_arrow = "#455a64"

    def draw_box(x, y, text, color, fontsize=9, bold=False):
        props = dict(boxstyle="round,pad=0.4", facecolor=color, edgecolor=c_border, linewidth=1.5)
        weight = "bold" if bold else "normal"
        ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
                fontweight=weight, bbox=props)

    def arrow(x1, y1, x2, y2, text=""):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                     arrowprops=dict(arrowstyle="-|>", color=c_arrow, lw=1.5))
        if text:
            mx, my = (x1+x2)/2 + 0.15, (y1+y2)/2 + 0.15
            ax.text(mx, my, text, fontsize=7, color="#666", style="italic")

    # Title
    ax.text(7, 7.5, "Experimental Pipeline: Multilingual Image Captioning", ha="center",
            fontsize=14, fontweight="bold", color="#1a3a5c")

    # Dataset
    draw_box(1.5, 5.5, "Multi30K\nDataset\n(1000 images)", c_data, bold=True)
    draw_box(1.5, 3.5, "Few-shot\nexamples\n(4 from train)", c_data)

    # OpenFlamingo
    draw_box(5, 6.2, "OpenFlamingo 3B", c_model, 10, bold=True)

    # Three modes from OpenFlamingo
    draw_box(5, 5, "EN-only\n(1000 captions)", c_model)
    draw_box(5, 3.5, "Direct DE\n(1000 captions)", c_model)
    draw_box(5, 2.0, "Direct FR\n(1000 captions)", c_model)

    # MarianMT
    draw_box(8.5, 5, "MarianMT\nEN→DE\n(1000 translations)", c_model)
    draw_box(8.5, 3.5, "MarianMT\nEN→FR\n(1000 translations)", c_model)

    # Evaluation
    draw_box(11.5, 4.2, "Evaluation\nBLEU, chrF\nMETEOR\nBERTScore", c_eval, 9, bold=True)

    # Results
    draw_box(11.5, 2.0, "Results\nComparison\nTable + CSV", c_result, 9, bold=True)

    # Arrows
    arrow(2.5, 5.5, 3.8, 5.5)
    arrow(2.5, 3.5, 3.8, 3.5)
    arrow(1.5, 5.0, 1.5, 4.0)

    arrow(5, 5.8, 5, 5.5)
    arrow(5, 5.8, 5, 4.0)
    arrow(5, 5.8, 5, 2.5)

    # EN-only to MarianMT
    arrow(6.2, 5.0, 7.3, 5.0, "EN captions")
    arrow(6.2, 5.0, 7.3, 3.7)

    # All to evaluation
    arrow(6.2, 5.0, 10.2, 4.5)
    arrow(6.2, 3.5, 10.2, 4.2)
    arrow(6.2, 2.0, 10.2, 3.8)
    arrow(9.7, 5.0, 10.2, 4.5)
    arrow(9.7, 3.5, 10.2, 4.2)

    # Eval to results
    arrow(11.5, 3.5, 11.5, 2.7)

    plt.tight_layout()
    output_path = output_dir / "experimental_pipeline_flowchart.png"
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {output_path}")


def create_results_chart(output_dir: Path):
    """Create a bar chart comparing all experiments across metrics."""

    experiments = ["EN-only", "Direct DE", "Direct FR", "Translate DE", "Translate FR"]
    bleu = [6.43, 1.11, 2.25, 3.95, 5.67]
    chrf = [23.65, 16.67, 16.62, 25.39, 24.97]
    meteor = [27.69, 12.01, 15.76, 23.09, 24.47]
    bertscore = [91.33, 74.40, 75.30, 75.97, 78.28]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Evaluation Results: All Experiments", fontsize=16, fontweight="bold", color="#1a3a5c")

    colors = ["#2196F3", "#ef5350", "#ff7043", "#66bb6a", "#26a69a"]

    metrics = [
        (axes[0, 0], "BLEU Score", bleu),
        (axes[0, 1], "chrF Score", chrf),
        (axes[1, 0], "METEOR Score", meteor),
        (axes[1, 1], "BERTScore (F1)", bertscore),
    ]

    for ax, title, values in metrics:
        bars = ax.bar(experiments, values, color=colors, edgecolor="white", linewidth=1.5)
        ax.set_title(title, fontsize=13, fontweight="bold", color="#333")
        ax.set_ylabel("Score", fontsize=10)
        ax.tick_params(axis="x", rotation=25, labelsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)

        # Value labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    output_path = output_dir / "results_comparison_chart.png"
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {output_path}")


def create_language_accuracy_chart(output_dir: Path):
    """Create a bar chart showing language detection accuracy."""

    experiments = ["EN-only", "Direct DE", "Direct FR", "Translate DE", "Translate FR"]
    target_pct = [94.5, 63.7, 82.8, 99.8, 98.8]
    other_pct = [5.5, 36.3, 17.2, 0.2, 1.2]

    fig, ax = plt.subplots(figsize=(10, 5))

    colors_target = ["#2196F3", "#66bb6a", "#66bb6a", "#26a69a", "#26a69a"]
    colors_other = ["#e0e0e0", "#ef5350", "#ff7043", "#e0e0e0", "#e0e0e0"]

    x = np.arange(len(experiments))
    bars1 = ax.bar(x, target_pct, color=colors_target, edgecolor="white", linewidth=1.5, label="Target language")
    bars2 = ax.bar(x, other_pct, bottom=target_pct, color=colors_other, edgecolor="white", linewidth=1.5, label="Wrong language")

    ax.set_title("Language Accuracy: % of Outputs in Target Language", fontsize=14, fontweight="bold", color="#1a3a5c")
    ax.set_ylabel("Percentage (%)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(experiments, fontsize=10)
    ax.set_ylim(0, 110)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    for bar, val in zip(bars1, target_pct):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 5,
                f"{val}%", ha="center", va="top", fontsize=11, fontweight="bold", color="white")

    plt.tight_layout()
    output_path = output_dir / "language_accuracy_chart.png"
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    config = load_config()
    output_dir = Path(config["paths"]["outputs_dir"]) / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("============================================")
    print("  BP-FLAMINGO: Analysis & Visualisation")
    print("============================================")

    # 1. Architecture flowchart (no GPU needed)
    print("\n  [1/4] Creating architecture flowchart...")
    create_architecture_flowchart(output_dir)

    # 2. Pipeline flowchart (no GPU needed)
    print("\n  [2/4] Creating pipeline flowchart...")
    create_pipeline_flowchart(output_dir)

    # 3. Results comparison chart (no GPU needed)
    print("\n  [3/4] Creating results comparison chart...")
    create_results_chart(output_dir)

    # 4. Language accuracy chart (no GPU needed)
    print("\n  [4/4] Creating language accuracy chart...")
    create_language_accuracy_chart(output_dir)

    print(f"\n  All visualisations saved to: {output_dir}")
    print("  Done!")


if __name__ == "__main__":
    main()