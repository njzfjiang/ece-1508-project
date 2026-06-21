#!/bin/bash
# Run all 18 experiments sequentially (2 models x 3 shots x 3 seeds).

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:-0}"
EXTRA_ARGS=()
if [ "${DRY_RUN:-0}" = "1" ]; then
    EXTRA_ARGS+=(--dry-run)
fi

echo "======================================================"
echo "  Running All Few-Shot Experiments"
echo "  pix2pix-turbo (paired) vs CycleGAN-Turbo (unpaired)"
echo "  Shots: 10, 20, 50  |  Seeds: 1, 2, 3"
echo "  GPU: $GPU_ID"
echo "======================================================"

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "Warning: nvidia-smi was not found; training still requires a CUDA GPU."
fi

"$PYTHON_BIN" src/train/run_experiment.py \
    --model all \
    --shots 10 20 50 \
    --seeds 1 2 3 \
    --gpu "$GPU_ID" \
    "${EXTRA_ARGS[@]}" \
    "$@"

echo "All experiments completed."
