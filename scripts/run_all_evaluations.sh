#!/bin/bash
# Run formal held-out evaluations for all few-shot experiments.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:-0}"
CONFIG_PATH="${CONFIG_PATH:-configs/base.yaml}"
EXTRA_ARGS=()
if [ "${USE_FP16:-0}" = "1" ]; then
    EXTRA_ARGS+=(--use-fp16)
fi

echo "======================================================"
echo "  Running Formal Held-Out Evaluations"
echo "  Metrics: CMMD, SSIM, LPIPS, CLIP Vision cosine"
echo "  Models: ${MODELS:-pix2pix}"
echo "  Shots: ${SHOTS:-10 20 50}  |  Seeds: ${SEEDS:-1 2 3}"
echo "  GPU: $GPU_ID"
echo "======================================================"

for model in ${MODELS:-pix2pix}; do
    for shot in ${SHOTS:-10 20 50}; do
        for seed in ${SEEDS:-1 2 3}; do
            echo
            echo "Evaluating ${model}: ${shot}-shot, seed ${seed}"
            "$PYTHON_BIN" src/eval/run_evaluation.py \
                --model "$model" \
                --shot "$shot" \
                --seed "$seed" \
                --config "$CONFIG_PATH" \
                --gpu "$GPU_ID" \
                "${EXTRA_ARGS[@]}" \
                "$@"
        done
    done
done

"$PYTHON_BIN" src/eval/summarize_evaluations.py --config "$CONFIG_PATH"

echo "All evaluations completed."
