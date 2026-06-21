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

## Data Download

### DarkDriving Dataset (ICRA 2026)

The dataset is not automatically downloaded due to license and file size.

**Manual Download Steps:**
1. Go to the official repository: https://github.com/DriveMindLab/DarkDriving-ICRA-2026
2. Download **DarkDriving_lle** from the [OneDrive link](https://onedrive.live.com/?id=64492CF1FC56CDDE%21s07d39562e06943cbb357c24a9708a0cb&cid=64492CF1FC56CDDE&redeem=aHR0cHM6Ly8xZHJ2Lm1zL2YvYy82NDQ5MmNmMWZjNTZjZGRlL0lnQmlsZE1IYWVETFE3Tlh3a3FYQ0tETEFiVnJrN3N5RjBsaElJdTNQU1ZKVVBVP2U9c01KUDJU) provided in the README
3. Extract the archive to `data/raw/`:
   ```bash
   # After downloading darkdriving_lle.zip
   unzip darkdriving_lle.zip -d data/raw/
   ```

## Data Preparation

Before running experiments, you need to prepare the dataset and generate few-shot splits:

You can run the helper script from any working directory; it resolves paths relative to the repository:

```bash
bash scripts/download_data.sh
```

If processed output already exists, the script asks before rebuilding it.

### Step 1: Prepare Full Dataset

Organize your raw DarkDriving dataset and run the preprocessing script:

```bash
python scripts/prepare_fewshot_splits.py \
  --raw_dir data/raw/darkdriving_lle \
  --output_dir data/processed \
  --shot_levels 10 20 50 \
  --num_seeds 3 \
  --val_split 0.1 \
  --copy_mode
```

`--copy_mode` is recommended on Windows. Omit it on systems where symlinks are available.

**Parameters:**
- `--raw_dir`: Path to raw dataset (should contain `train/day`, `train/night`, `test/day`, `test/night`)
- `--output_dir`: Output directory for processed data (default: `./data/processed`)
- `--shot_levels`: Few-shot levels to generate (default: [10, 20, 50])
- `--num_seeds`: Number of random seeds per shot level (default: 3)
- `--copy_mode`: Use this flag to copy files instead of symlinking
- `--val_split`: Validation split ratio (default: 0.1)
- `--overwrite`: Safely remove existing `day2night/` and `splits/` outputs before rebuilding

Seed directory names match the actual random seeds: `seed1` uses random seed `1`, and so on.

**Output Structure:**
```
data/processed/
├── day2night/
│   ├── train/day/
│   ├── train/night/
│   ├── val/day/
│   ├── val/night/
│   ├── test/day/
│   └── test/night/
└── splits/fewshot/
    ├── 10shot/seed1/split.json
    ├── 10shot/seed2/split.json
    ├── 20shot/seed1/split.json
    └── ...
```

### Step 2: Verify Data Preparation

Check if the preprocessing was successful:
```bash
ls -la data/processed/day2night/train/
ls -la data/processed/splits/fewshot/
```
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
