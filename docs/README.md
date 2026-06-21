# ECE 1508 Project

This is the main README for the project.

## Overview

This project implements and evaluates different generative models for image-to-image translation tasks.

## Project Structure

- `data/`: Dataset directory (raw, processed, and splits)
- `notebooks/`: Jupyter notebooks for EDA and visualization
- `src/`: Source code for models, training, and evaluation
- `configs/`: Configuration files for different models
- `scripts/`: Utility scripts for data preparation and experiment execution
- `results/`: Results and figures from experiments
- `docs/`: Documentation

## Setup

1. Install dependencies: `pip install -r requirements.txt`
2. Set up the environment: `conda env create -f environment.yaml`
3. Download datasets: `bash scripts/download_data.sh`

## Running Experiments

To run experiments:

```bash
bash scripts/run_all_experiments.sh
```

Or run specific models:

```bash
python -m src.train.run_experiment --config configs/cyclegan/config.yaml --model cyclegan
python -m src.train.run_experiment --config configs/sdturbo/config.yaml --model sdturbo
```

## Evaluation

To evaluate trained models:

```bash
python -m src.eval.evaluate --model path/to/model --data data/processed --output results/evaluation
```

To aggregate results:

```bash
python -m src.eval.aggregate --results_dir results --output results/aggregated_results.json
```
