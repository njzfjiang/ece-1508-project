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

Python 3.10 and CUDA 11.8 are the reproducible baseline. Conda is recommended:

```bash
conda env create -f environment.yaml
conda activate ece-1508
bash scripts/setup_img2img_turbo.sh
```

The setup script pins the official `img2img-turbo` checkout to commit
`86f54146590ffb4543c8cf85b5a36657da670924`. Set `SKIP_INSTALL=1` if the
environment is already installed.

For a pip-based environment, first install a compatible PyTorch 2.0.1/CUDA
build, then run `python -m pip install -r requirements.txt`.

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

The data pipeline has two deliberate layers:

```text
data/raw/darkdriving_lle
    -> prepare_fewshot_splits.py
data/processed/day2night + data/processed/splits
    -> prepare_img2img_turbo_data.py
data/processed/img2img_turbo/<shot>shot/seed<seed>
```

The first layer is the canonical, model-independent dataset and reproducible
split definition. The second is a lightweight adapter for the official
img2img-turbo loaders. Do not replace the first layer with model-specific
preprocessing.

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

### Step 3: Prepare Official Training Views

The experiment launcher creates these views automatically. To create all of
them ahead of time:

```bash
python scripts/prepare_img2img_turbo_data.py
```

Each `shot/seed` view contains the upstream `train_A`, `train_B`, `test_A`,
`test_B`, prompt JSON, and fixed-prompt files. Files use hardlinks when
possible, so the views do not duplicate image contents.

The same adapter supports both models; no separate CycleGAN data script is
needed. Both models receive the same selected day and night samples for a fair
few-shot comparison. The official CycleGAN-Turbo loader samples the two
domains independently during training, making its batches unpaired.

`src/data/dataset.py` remains available as the project-native paired loader for
EDA, evaluation, and custom experiments. Official training does not use it.

## Running Experiments

Validate all 18 commands and dataset views without starting GPU training:

```bash
DRY_RUN=1 bash scripts/run_all_experiments.sh
```

Run all experiments sequentially on GPU 0:

```bash
bash scripts/run_all_experiments.sh
```

Select another GPU with `GPU_ID=1`. To run a specific model:

```bash
python src/train/run_experiment.py --model pix2pix --shots 10 --seeds 1 --gpu 0
python src/train/run_experiment.py --model cyclegan --shots 20 --seeds 2 --gpu 0
```

All active experiment settings live in `configs/base.yaml`. The
`pix2pix_turbo` and `cyclegan_turbo` sections contain model-specific options.

## Evaluation

To evaluate trained models:

```bash
python -m src.eval.evaluate --model path/to/model --data data/processed --output results/evaluation
```

To aggregate results:

```bash
python -m src.eval.aggregate --results_dir results --output results/aggregated_results.json
```
