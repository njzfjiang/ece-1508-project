# ece-1508-project
## Few-Shot Day-to-Night Translation: Cycle Alignment vs. Example Imitation

This is the main README for the project.

## Overview

This project investigates few-shot day-to-night image translation by comparing two state-of-the-art single-step conditional models built on the same backbone: CycleGAN-Turbo and SD-Turbo+LoRA. Both models originate from the img2img-turbo framework (Parmar et al., 2024) and share an identical generator architecture—integrating SD-Turbo with LoRA adapters, skip connections, and Zero-Convs into a unified end-to-end network. The key difference lies in their learning paradigm：
CycleGAN-Turbo adopts a cycle-consistency objective with unpaired data，learning the underlying "day ↔ night" mapping by ensuring that a translated night image can be translated back to the original day image. This approach focuses on domain‑level alignment and has been shown to outperform existing GAN‑based and diffusion‑based methods for various scene translation tasks, including day‑to‑night conversion.
SD-Turbo+LoRA (in its paired variant, pix2pix-turbo) follows a direct imitation paradigm, training on explicit (day, night) pairs to replicate the example transformation. The same generator architecture is shared, differing only in the learning objective and data format.
We ask: Under extreme data scarcity (10, 20, and 50 training pairs), which learning paradigm—cycle alignment or example imitation—enables a model to more effectively learn the day-to-night mapping? Where is the breaking point at which generation quality begins to degrade?


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