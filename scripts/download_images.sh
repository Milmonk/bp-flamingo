#!/bin/bash
# ============================================================================
# BP-FLAMINGO: Download Flickr30K Images
# ============================================================================
# Usage: cd ~/bp-flamingo && bash scripts/download_images.sh
#
# Downloads flickr30k-images.zip (~4.4 GB) from HuggingFace and extracts
# images to data/multi30k/raw/images/
#
# Prerequisites:
#   1. HuggingFace account: https://huggingface.co/join
#   2. Accept dataset license: https://huggingface.co/datasets/nlphuji/flickr30k
#   3. Login: huggingface-cli login
# ============================================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data/multi30k/raw"
IMAGE_DIR="$DATA_DIR/images"
ZIP_FILE="$DATA_DIR/flickr30k-images.zip"

echo "============================================"
echo "  BP-FLAMINGO: Download Flickr30K Images"
echo "============================================"
echo ""

# Activate venv if not already active
if [ -z "$VIRTUAL_ENV" ]; then
    source "$PROJECT_ROOT/venv/bin/activate"
fi

# Check if images already exist
if [ -d "$IMAGE_DIR" ] && [ "$(ls -1 "$IMAGE_DIR"/*.jpg 2>/dev/null | wc -l)" -gt 30000 ]; then
    echo "Images already downloaded (~$(ls -1 "$IMAGE_DIR"/*.jpg | wc -l) files)."
    echo "Skipping download."
    exit 0
fi

mkdir -p "$IMAGE_DIR"

# --- Step 1: Download the zip file ---
echo "[1/2] Downloading flickr30k-images.zip (~4.4 GB)..."
echo "  This will take a while depending on your connection speed."
echo ""

if [ -f "$ZIP_FILE" ]; then
    echo "  Zip file already exists, skipping download."
else
    huggingface-cli download nlphuji/flickr30k flickr30k-images.zip \
        --repo-type dataset \
        --local-dir "$DATA_DIR"

    echo ""
    echo "  Download complete."
fi

# --- Step 2: Extract images ---
echo ""
echo "[2/2] Extracting images..."

# Extract - the zip contains a flickr30k-images/ folder
cd "$DATA_DIR"
unzip -q -o flickr30k-images.zip -d "$DATA_DIR/tmp_extract"

# Move images to our target directory
if [ -d "$DATA_DIR/tmp_extract/flickr30k-images" ]; then
    mv "$DATA_DIR/tmp_extract/flickr30k-images/"*.jpg "$IMAGE_DIR/" 2>/dev/null || true
    mv "$DATA_DIR/tmp_extract/flickr30k-images/"*.png "$IMAGE_DIR/" 2>/dev/null || true
elif [ -d "$DATA_DIR/tmp_extract/flickr30k_images" ]; then
    mv "$DATA_DIR/tmp_extract/flickr30k_images/"*.jpg "$IMAGE_DIR/" 2>/dev/null || true
    mv "$DATA_DIR/tmp_extract/flickr30k_images/"*.png "$IMAGE_DIR/" 2>/dev/null || true
else
    # Images might be directly in tmp_extract
    mv "$DATA_DIR/tmp_extract/"*.jpg "$IMAGE_DIR/" 2>/dev/null || true
fi

# Cleanup
rm -rf "$DATA_DIR/tmp_extract"

# Optionally remove zip to save space
# rm -f "$ZIP_FILE"

IMAGE_COUNT=$(ls -1 "$IMAGE_DIR"/*.jpg 2>/dev/null | wc -l)

echo ""
echo "============================================"
echo "  Done! Extracted $IMAGE_COUNT images."
echo "  Location: $IMAGE_DIR"
echo ""
echo "  (Optional) Remove zip to save ~4.4 GB:"
echo "  rm $ZIP_FILE"
echo "============================================"
