#!/bin/bash
# ============================================================================
# BP-FLAMINGO: Download Multi30K Dataset
# ============================================================================
# Usage: cd ~/bp-flamingo && bash scripts/download_multi30k.sh
#
# Downloads:
#   1. Multi30K text data (EN, DE, FR captions) from GitHub
#   2. Flickr30K images from HuggingFace
#
# Output structure:
#   data/multi30k/raw/
#   ├── texts/          # Raw caption text files
#   │   ├── train.en, train.de, train.fr
#   │   ├── val.en, val.de, val.fr
#   │   └── test_2016_flickr.en, .de, .fr
#   └── images/         # Flickr30K images (downloaded via HuggingFace)
# ============================================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data/multi30k/raw"
TEXT_DIR="$DATA_DIR/texts"
IMAGE_DIR="$DATA_DIR/images"

echo "============================================"
echo "  BP-FLAMINGO: Download Multi30K Dataset"
echo "============================================"
echo ""

# --- 1. Download text data from Multi30K GitHub ---
echo "[1/3] Downloading Multi30K text data..."

mkdir -p "$TEXT_DIR"

BASE_URL="https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw"

# Define files to download: split.language
declare -A FILES=(
    # Training set
    ["train.en"]="$BASE_URL/train.en.gz"
    ["train.de"]="$BASE_URL/train.de.gz"
    ["train.fr"]="$BASE_URL/train.fr.gz"
    # Validation set
    ["val.en"]="$BASE_URL/val.en.gz"
    ["val.de"]="$BASE_URL/val.de.gz"
    ["val.fr"]="$BASE_URL/val.fr.gz"
    # Test set 2016
    ["test_2016_flickr.en"]="$BASE_URL/test_2016_flickr.en.gz"
    ["test_2016_flickr.de"]="$BASE_URL/test_2016_flickr.de.gz"
    ["test_2016_flickr.fr"]="$BASE_URL/test_2016_flickr.fr.gz"
)

for filename in "${!FILES[@]}"; do
    url="${FILES[$filename]}"
    output="$TEXT_DIR/$filename"

    if [ -f "$output" ]; then
        echo "  Already exists: $filename"
    else
        echo "  Downloading: $filename ..."
        curl -sL "$url" | gunzip > "$output"
        echo "  Saved: $filename ($(wc -l < "$output") lines)"
    fi
done

echo "  Done — text data downloaded."

# --- 2. Download image list (to map image IDs) ---
echo ""
echo "[2/3] Downloading image order files..."

IMAGE_SPLITS_URL="https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/image_splits"

for split in train val test_2016_flickr; do
    output="$TEXT_DIR/${split}.images"
    url="$IMAGE_SPLITS_URL/${split}.txt"

    if [ -f "$output" ]; then
        echo "  Already exists: ${split}.images"
    else
        echo "  Downloading: ${split}.images ..."
        curl -sL "$url" > "$output"
        echo "  Saved: ${split}.images ($(wc -l < "$output") entries)"
    fi
done

echo "  Done — image order files downloaded."

# --- 3. Download Flickr30K images via Python/HuggingFace ---
echo ""
echo "[3/3] Downloading Flickr30K images..."
echo "  (This may take a while on first run)"

mkdir -p "$IMAGE_DIR"

# Activate venv if not already active
if [ -z "$VIRTUAL_ENV" ]; then
    source "$PROJECT_ROOT/venv/bin/activate"
fi

python3 << 'PYTHON_SCRIPT'
import os
import sys
import json

DATA_DIR = os.environ.get("DATA_DIR", "data/multi30k/raw")
IMAGE_DIR = os.path.join(DATA_DIR, "images")
TEXT_DIR = os.path.join(DATA_DIR, "texts")

# Collect all unique image filenames we need
needed_images = set()
for split in ["train", "val", "test_2016_flickr"]:
    img_file = os.path.join(TEXT_DIR, f"{split}.images")
    if os.path.exists(img_file):
        with open(img_file) as f:
            for line in f:
                fname = line.strip()
                if fname:
                    needed_images.add(fname)

print(f"  Total unique images needed: {len(needed_images)}")

# Check how many we already have
existing = set(os.listdir(IMAGE_DIR)) if os.path.exists(IMAGE_DIR) else set()
missing = needed_images - existing

if len(missing) == 0:
    print("  All images already downloaded!")
    sys.exit(0)

print(f"  Images to download: {len(missing)}")
print(f"  Already have: {len(existing)}")
print()

# Try downloading from HuggingFace datasets
try:
    from datasets import load_dataset
    from PIL import Image
    from tqdm import tqdm

    print("  Loading Flickr30K from HuggingFace...")
    print("  (You may need to accept the dataset license on HuggingFace)")
    print("  (Visit: https://huggingface.co/datasets/nlphuji/flickr30k)")
    print()

    ds = load_dataset("nlphuji/flickr30k", split="test")

    saved = 0
    for item in tqdm(ds, desc="  Saving images"):
        # The dataset uses 'image' field and 'filename' or 'img_id'
        img = item.get("image")
        # Try different field names for the filename
        fname = item.get("filename", None)
        if fname is None:
            img_id = item.get("img_id", item.get("image_id", None))
            if img_id is not None:
                fname = f"{img_id}.jpg"

        if fname and fname in missing and img is not None:
            img.save(os.path.join(IMAGE_DIR, fname))
            saved += 1

    print(f"\n  Saved {saved} images from HuggingFace.")

    if saved < len(missing):
        print(f"  WARNING: {len(missing) - saved} images still missing.")
        print("  You may need to download Flickr30K images manually.")

except Exception as e:
    print(f"  HuggingFace download failed: {e}")
    print()
    print("  === MANUAL DOWNLOAD INSTRUCTIONS ===")
    print("  The Flickr30K images require manual download:")
    print("  1. Visit: https://huggingface.co/datasets/nlphuji/flickr30k")
    print("  2. Accept the license agreement")
    print("  3. Login: huggingface-cli login")
    print("  4. Re-run this script")
    print()
    print("  Alternative: Download from the original Flickr30K source:")
    print("  https://shannon.cs.illinois.edu/DenotationGraph/")
    print("  and place images in: " + IMAGE_DIR)
    sys.exit(1)

PYTHON_SCRIPT

echo ""
echo "============================================"
echo "  Multi30K download complete!"
echo "  Text data: $TEXT_DIR"
echo "  Images:    $IMAGE_DIR"
echo "============================================"
