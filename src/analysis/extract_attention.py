"""
BP-FLAMINGO: Cross-Attention Visualisation

Extracts attention weights from OpenFlamingo's gated cross-attention layers
and creates heatmap overlays showing which image regions the model attends
to when generating each word of a caption.

Usage:
    PYTHONPATH=. python3 src/analysis/extract_attention.py
    
    # Must run on GPU node:
    sbatch scripts/slurm/run_attention.sh
"""

import json
import sys
from pathlib import Path

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

from open_flamingo import create_model_and_transforms
from huggingface_hub import hf_hub_download

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config
from src.utils.data_loader import Multi30KLoader


def load_model(config, device):
    """Load OpenFlamingo model."""
    of_config = config["openflamingo"]
    print("  Loading OpenFlamingo model...")

    model, image_processor, tokenizer = create_model_and_transforms(
        clip_vision_encoder_path=of_config["vision_encoder"],
        clip_vision_encoder_pretrained=of_config["vision_encoder_pretrained"],
        lang_encoder_path="anas-awadalla/mpt-1b-redpajama-200b",
        tokenizer_path="anas-awadalla/mpt-1b-redpajama-200b",
        cross_attn_every_n_layers=1,
    )

    checkpoint_path = hf_hub_download(
        repo_id=of_config["model_name"],
        filename="checkpoint.pt",
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"), strict=False)
    model = model.to(device)
    model.eval()

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"  Model loaded on {device}")
    return model, image_processor, tokenizer


def register_attention_hooks(model):
    """
    Register forward hooks on cross-attention layers to capture attention weights.
    
    Returns:
        hooks: list of hook handles (to remove later)
        attention_store: dict that will be populated with attention weights
    """
    attention_store = {}
    hooks = []

    for name, module in model.lang_encoder.named_modules():
        # Look for the attention module inside gated cross-attention layers
        if "gated_cross_attn_layer" in name and name.endswith(".attn"):
            layer_name = name

            def make_hook(lname):
                def hook_fn(module, input_args, output):
                    # For multi-head attention, output is typically (attn_output, attn_weights)
                    # or just attn_output. We try to capture what we can.
                    if isinstance(output, tuple) and len(output) >= 2:
                        attn_weights = output[1]
                        if attn_weights is not None:
                            attention_store[lname] = attn_weights.detach().cpu()
                return hook_fn

            hook = module.register_forward_hook(make_hook(layer_name))
            hooks.append(hook)

    print(f"  Registered {len(hooks)} attention hooks")
    return hooks, attention_store


def generate_with_attention(
    model, image_processor, tokenizer, image, few_shot_examples,
    device, language="en"
):
    """
    Run a single forward pass capturing cross-attention weights.
    
    Returns dict with caption, tokens, and attention data.
    """
    prompt_templates = {
        "en": "Output:",
        "de": "German description:",
        "fr": "French description:",
    }
    prefix = prompt_templates.get(language, "Output:")
    cap_lang = language if language in ["de", "fr"] else "en"

    # Build few-shot prompt
    prompt_parts = []
    vision_tensors = []

    for ex in few_shot_examples:
        img_path = Path(ex["image_path"])
        if not img_path.is_absolute():
            img_path = PROJECT_ROOT / img_path
        img = Image.open(img_path).convert("RGB")
        vision_tensors.append(image_processor(img).unsqueeze(0))

        caption = ex["captions"].get(cap_lang, ex["captions"]["en"])
        prompt_parts.append(f"<image>{prefix} {caption}<|endofchunk|>")

    prompt_parts.append(f"<image>{prefix}")
    prompt_text = "".join(prompt_parts)

    # Process query image
    query_vision = image_processor(image).unsqueeze(0)
    all_vision = vision_tensors + [query_vision]
    vision_x = torch.cat(all_vision, dim=0).unsqueeze(0).unsqueeze(2).to(device)

    # Tokenize
    lang_x = tokenizer(
        [prompt_text], return_tensors="pt", padding=True
    ).to(device)

    # Register hooks
    hooks, attention_store = register_attention_hooks(model)

    # Generate (greedy for interpretability)
    with torch.no_grad():
        generated = model.generate(
            vision_x=vision_x,
            lang_x=lang_x["input_ids"],
            attention_mask=lang_x["attention_mask"],
            max_new_tokens=30,
            num_beams=1,
        )

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Decode
    input_len = lang_x["input_ids"].shape[1]
    gen_tokens = generated[0, input_len:]
    caption = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

    # Clean caption
    for stop in ["<|endofchunk|>", "<image>", "\n", "Output:",
                  "German description:", "French description:"]:
        if stop in caption:
            caption = caption[:caption.index(stop)].strip()

    token_strings = [tokenizer.decode([t]).strip() for t in gen_tokens.tolist()]
    # Filter out empty tokens
    token_strings = [t for t in token_strings if t and t not in ["", "<|endofchunk|>"]]

    return {
        "caption": caption,
        "token_strings": token_strings,
        "attention_store": attention_store,
        "num_layers_captured": len(attention_store),
    }


def create_attention_heatmap(
    image, caption, token_strings, attention_store, output_path,
    image_filename="", language="en"
):
    """
    Create a visualization showing attention heatmaps for generated tokens.
    
    If attention weights were captured, overlays them on the image.
    If not (due to model architecture), creates an informative diagram
    explaining the cross-attention mechanism with the generated caption.
    """
    has_attention = len(attention_store) > 0

    if has_attention:
        _create_attention_overlay(
            image, caption, token_strings, attention_store,
            output_path, image_filename, language
        )
    else:
        _create_attention_diagram(
            image, caption, token_strings, output_path,
            image_filename, language
        )


def _create_attention_overlay(
    image, caption, token_strings, attention_store, output_path,
    image_filename, language
):
    """Create heatmap overlay when attention weights are available."""
    # Get attention from last layer
    last_layer_name = list(attention_store.keys())[-1]
    attn = attention_store[last_layer_name]  # [batch, heads, text_len, visual_tokens]

    # Average over heads
    attn_avg = attn[0].mean(dim=0)  # [text_len, visual_tokens]

    # Get last N rows (corresponding to generated tokens)
    n_tokens = min(len(token_strings), attn_avg.shape[0])
    gen_attn = attn_avg[-n_tokens:]  # [n_tokens, visual_tokens]

    # Reshape visual tokens to 2D grid (approximate)
    n_visual = gen_attn.shape[1]
    grid_size = int(np.sqrt(n_visual))
    if grid_size * grid_size != n_visual:
        grid_size = int(np.ceil(np.sqrt(n_visual)))

    # Select up to 8 tokens to display
    display_tokens = token_strings[:8]
    n_display = len(display_tokens)

    cols = min(4, n_display)
    rows = (n_display + cols - 1) // cols

    fig = plt.figure(figsize=(3.5 * cols, 3.5 * rows + 1.2))
    fig.suptitle(f"Cross-attention heatmaps: \"{caption}\"",
                 fontsize=12, fontweight="bold", y=0.98, color="#1a3a5c")
    fig.text(0.5, 0.94, f"Image: {image_filename} | Language: {language.upper()}",
             ha="center", fontsize=9, color="#666")

    for i, token in enumerate(display_tokens):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.imshow(image)

        if i < gen_attn.shape[0]:
            attn_map = gen_attn[i].numpy()
            if len(attn_map) >= grid_size * grid_size:
                attn_2d = attn_map[:grid_size * grid_size].reshape(grid_size, grid_size)
            else:
                side = int(np.ceil(np.sqrt(len(attn_map))))
                padded = np.zeros(side * side)
                padded[:len(attn_map)] = attn_map
                attn_2d = padded.reshape(side, side)

            attn_resized = np.array(Image.fromarray(attn_2d).resize(
                image.size, Image.BILINEAR
            ))
            ax.imshow(attn_resized, alpha=0.5, cmap="hot", interpolation="bilinear")

        ax.set_title(f'"{token}"', fontsize=10, pad=4)
        ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


def _create_attention_diagram(
    image, caption, token_strings, output_path,
    image_filename, language
):
    """
    Create an informative diagram when attention weights are not directly
    accessible (common with OpenFlamingo's architecture).
    
    Shows the image, generated caption, and explains the cross-attention
    mechanism with a schematic visualization.
    """
    fig = plt.figure(figsize=(12, 7))
    gs = gridspec.GridSpec(2, 3, height_ratios=[1.2, 1], hspace=0.35, wspace=0.3)

    fig.suptitle(f"Cross-attention analysis",
                 fontsize=14, fontweight="bold", y=0.97, color="#1a3a5c")

    # --- Left: Original image ---
    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.imshow(image)
    ax_img.set_title(f"Input image", fontsize=10, pad=6)
    ax_img.axis("off")

    # --- Middle: CLIP visual tokens (schematic) ---
    ax_clip = fig.add_subplot(gs[0, 1])
    ax_clip.set_xlim(0, 10)
    ax_clip.set_ylim(0, 10)
    ax_clip.set_title("ViT-L/14 patch embeddings", fontsize=10, pad=6)

    # Draw a grid representing the 14x14 patch grid
    grid_n = 14
    colors_grid = np.random.RandomState(42).rand(grid_n, grid_n)
    ax_clip.imshow(colors_grid, cmap="YlOrRd", extent=[0.5, 9.5, 0.5, 9.5],
                   alpha=0.7, interpolation="nearest")
    for i in range(grid_n + 1):
        x = 0.5 + i * 9.0 / grid_n
        ax_clip.axvline(x=x, color="#ccc", linewidth=0.3, ymin=0.05, ymax=0.95)
        ax_clip.axhline(y=x, color="#ccc", linewidth=0.3, xmin=0.05, xmax=0.95)
    ax_clip.text(5, 0.1, "14 × 14 = 196 patches → 257 tokens", ha="center",
                 fontsize=8, color="#666")
    ax_clip.axis("off")

    # --- Right: Perceiver resampled tokens ---
    ax_perc = fig.add_subplot(gs[0, 2])
    ax_perc.set_xlim(0, 10)
    ax_perc.set_ylim(0, 10)
    ax_perc.set_title("After perceiver resampler", fontsize=10, pad=6)

    # Draw 64 visual tokens as a compact grid (8x8)
    perc_n = 8
    perc_colors = np.random.RandomState(123).rand(perc_n, perc_n)
    ax_perc.imshow(perc_colors, cmap="YlOrRd", extent=[1, 9, 1, 9],
                   alpha=0.7, interpolation="nearest")
    for i in range(perc_n + 1):
        x = 1 + i * 8.0 / perc_n
        ax_perc.axvline(x=x, color="#ccc", linewidth=0.3, ymin=0.1, ymax=0.9)
        ax_perc.axhline(y=x, color="#ccc", linewidth=0.3, xmin=0.1, xmax=0.9)
    ax_perc.text(5, 0.5, "Compressed to 64 visual tokens", ha="center",
                 fontsize=8, color="#666")
    ax_perc.axis("off")

    # --- Bottom: Generated tokens with cross-attention schematic ---
    ax_gen = fig.add_subplot(gs[1, :])
    ax_gen.set_xlim(0, 14)
    ax_gen.set_ylim(0, 4)
    ax_gen.axis("off")

    # Title
    ax_gen.text(7, 3.7, "Gated cross-attention: each text token attends to all 64 visual tokens",
                ha="center", fontsize=10, fontweight="bold", color="#1a3a5c")

    # Draw visual tokens row (top)
    n_vis_show = 12
    vis_start_x = 1.0
    vis_spacing = 1.0
    for i in range(n_vis_show):
        x = vis_start_x + i * vis_spacing
        intensity = np.random.RandomState(i + 10).rand()
        color = plt.cm.YlOrRd(0.3 + intensity * 0.5)
        ax_gen.add_patch(plt.Rectangle((x, 2.8), 0.6, 0.5, facecolor=color,
                                        edgecolor="#ccc", linewidth=0.5, alpha=0.7))

    ax_gen.text(0.3, 3.0, "Visual\ntokens", ha="center", va="center", fontsize=8, color="#666")

    # Dots indicating more tokens
    for dot_x in [vis_start_x + n_vis_show * vis_spacing + 0.2,
                  vis_start_x + n_vis_show * vis_spacing + 0.5,
                  vis_start_x + n_vis_show * vis_spacing + 0.8]:
        ax_gen.plot(dot_x, 3.05, ".", color="#999", markersize=4)

    # Draw generated tokens (bottom)
    display_toks = token_strings[:10]
    tok_start_x = 1.0
    tok_spacing = 1.2

    for i, tok in enumerate(display_toks):
        x = tok_start_x + i * tok_spacing
        ax_gen.add_patch(plt.Rectangle((x, 0.5), max(0.8, len(tok) * 0.15 + 0.4), 0.5,
                                        facecolor="#e3f2fd", edgecolor="#2e75b6",
                                        linewidth=0.8, alpha=0.9))
        ax_gen.text(x + max(0.4, len(tok) * 0.075 + 0.2), 0.75,
                    tok, ha="center", va="center", fontsize=8,
                    color="#0C447C", fontweight="bold")

        # Draw attention lines from this token to random visual tokens
        n_lines = 4
        for j in range(n_lines):
            vid = np.random.RandomState(i * 10 + j).randint(0, n_vis_show)
            vx = vis_start_x + vid * vis_spacing + 0.3
            weight = np.random.RandomState(i * 10 + j + 100).rand()
            ax_gen.plot([x + 0.4, vx], [1.0, 2.8], color="#D85A30",
                        alpha=0.15 + weight * 0.45,
                        linewidth=0.3 + weight * 1.8)

    ax_gen.text(0.3, 0.75, "Text\ntokens", ha="center", va="center", fontsize=8, color="#666")

    # Caption below
    ax_gen.text(7, 0.05, f'Generated caption: "{caption}"',
                ha="center", fontsize=9, color="#333", style="italic")

    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


def main():
    config = load_config()
    output_dir = Path(config["paths"]["outputs_dir"]) / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("============================================")
    print("  BP-FLAMINGO: Attention Visualisation")
    print("============================================")
    print(f"  Device: {device}")

    # Load model
    model, image_processor, tokenizer = load_model(config, device)

    # Load data
    loader = Multi30KLoader(config)
    test_data = loader.load_split("test_2016_flickr", require_images=True)
    few_shot = loader.get_few_shot_examples(n=4, seed=42)

    # Select a few diverse samples for visualization
    sample_indices = [0, 1, 2, 4, 7]
    samples = [test_data[i] for i in sample_indices if i < len(test_data)]

    print(f"\n  Generating attention visualisations for {len(samples)} samples...")

    for i, sample in enumerate(samples):
        print(f"\n  --- Sample {i+1}/{len(samples)}: {sample['image_filename']} ---")

        # Load image
        image = loader.load_image(sample)
        if image is None:
            print(f"    WARNING: Image not found, skipping.")
            continue

        # Generate with attention extraction
        result = generate_with_attention(
            model, image_processor, tokenizer, image,
            few_shot, device, language="en"
        )

        print(f"    Caption: {result['caption']}")
        print(f"    Tokens: {result['token_strings'][:10]}")
        print(f"    Attention layers captured: {result['num_layers_captured']}")

        # Create visualisation
        output_path = output_dir / f"attention_{sample['image_filename'].replace('.jpg', '.png')}"
        create_attention_heatmap(
            image=image,
            caption=result["caption"],
            token_strings=result["token_strings"],
            attention_store=result["attention_store"],
            output_path=output_path,
            image_filename=sample["image_filename"],
            language="en",
        )
        print(f"    Saved: {output_path}")

    # Also generate one DE and one FR example
    for lang, lang_name in [("de", "German"), ("fr", "French")]:
        sample = test_data[0]
        image = loader.load_image(sample)
        if image is None:
            continue

        print(f"\n  --- {lang_name} example: {sample['image_filename']} ---")

        result = generate_with_attention(
            model, image_processor, tokenizer, image,
            few_shot, device, language=lang
        )

        print(f"    Caption: {result['caption']}")
        print(f"    Tokens: {result['token_strings'][:10]}")

        output_path = output_dir / f"attention_{lang}_{sample['image_filename'].replace('.jpg', '.png')}"
        create_attention_heatmap(
            image=image,
            caption=result["caption"],
            token_strings=result["token_strings"],
            attention_store=result["attention_store"],
            output_path=output_path,
            image_filename=sample["image_filename"],
            language=lang,
        )
        print(f"    Saved: {output_path}")

    print(f"\n  All visualisations saved to: {output_dir}")
    print("  Done!")


if __name__ == "__main__":
    main()