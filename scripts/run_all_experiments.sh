#!/bin/bash
# Script to run all experiments

set -e

echo "Running all experiments..."

# CycleGAN experiments
echo "Running CycleGAN experiments..."
python -m src.train.run_experiment --config configs/cyclegan/config.yaml --model cyclegan

# SD-Turbo experiments
echo "Running SD-Turbo experiments..."
python -m src.train.run_experiment --config configs/sdturbo/config.yaml --model sdturbo

echo "All experiments complete!"
