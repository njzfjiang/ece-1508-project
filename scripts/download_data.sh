#!/bin/bash
# scripts/download_data.sh
# Data download and preparation script

set -e

# Resolve all project paths relative to this script, not the caller's
# current working directory. This allows the script to run from anywhere.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

echo "======================================================"
echo "  DarkDriving Dataset Download & Setup"
echo "======================================================"
echo "Project root: $PROJECT_ROOT"
echo ""

# Define paths
RAW_DIR="data/raw/darkdriving_lle"
DATASET_DIR="$RAW_DIR"
PROCESSED_DIR="data/processed"
REQUIRED_SUBDIRS=("train/day" "train/night" "test/day" "test/night")

# Create directories
mkdir -p data/raw
mkdir -p data/processed

# Check that the expected dataset structure exists
missing_data=false
for subdir in "${REQUIRED_SUBDIRS[@]}"; do
    if [ ! -d "$DATASET_DIR/$subdir" ]; then
        missing_data=true
        break
    fi
done

if [ "$missing_data" = false ]; then
    echo "✅ Dataset found at $DATASET_DIR"
    echo ""
    OVERWRITE_ARGS=()
    if [ -d "$PROCESSED_DIR/day2night" ] || [ -d "$PROCESSED_DIR/splits" ]; then
        read -p "Processed output already exists. Rebuild it? [y/N] " -n 1 -r
        echo ""
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Skipping preprocessing. Exiting."
            exit 0
        fi
        OVERWRITE_ARGS=(--overwrite)
    fi
else
    echo "❌ Expected dataset structure not found at $DATASET_DIR"
    echo ""
    echo "Please manually download the dataset:"
    echo ""
    echo "  🔗 https://github.com/DriveMindLab/DarkDriving-ICRA-2026"
    echo ""
    echo "  1. Download DarkDriving_lle from the OneDrive link"
    echo "  2. Extract to: $RAW_DIR"
    echo "     (e.g., unzip DarkDriving_lle.zip -d $RAW_DIR)"
    echo ""
    echo "⚠️  OneDrive links cannot be automated with wget/curl."
    echo "   Please complete the manual download first."
    echo ""
    exit 1
fi

# Run preprocessing
echo ""
echo "📦 Running preprocessing..."

COPY_ARGS=()
case "$(uname -s 2>/dev/null || true)" in
    MINGW*|MSYS*|CYGWIN*) COPY_ARGS=(--copy_mode) ;;
esac
if [ "${COPY_MODE:-0}" = "1" ]; then
    COPY_ARGS=(--copy_mode)
fi

python scripts/prepare_fewshot_splits.py \
    --raw_dir "$DATASET_DIR" \
    --output_dir "$PROCESSED_DIR" \
    --shot_levels 10 20 50 \
    --num_seeds 3 \
    --val_split 0.1 \
    "${COPY_ARGS[@]}" \
    "${OVERWRITE_ARGS[@]}"

echo ""
echo "✅ Data preparation complete!"
echo ""
echo "📁 Processed data: $PROCESSED_DIR/day2night/"
echo "📁 Few-shot splits: $PROCESSED_DIR/splits/fewshot/"
echo ""
echo "Next, set up the pinned official training code and validate a run:"
echo "  bash scripts/setup_img2img_turbo.sh"
echo "  python src/train/run_experiment.py --model pix2pix --shots 10 --seeds 1 --dry-run"
