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

## Setup

Python 3.10 and CUDA 11.8 are the reproducible baseline. Conda is recommended:

```bash
conda env create -f environment.yaml
conda activate ece-1508
python scripts/setup.py
```

The setup script clones the official `img2img-turbo` repository, pins it to
commit `86f54146590ffb4543c8cf85b5a36657da670924`, installs the required
dependencies, and prepares the few-shot training views. Run `python scripts/setup.py --skip-install` if the environment is already installed. To override the default settings, specify custom values using the `--shots` and `--seeds` command-line arguments.

**Output Structure:**
```
data/
├── raw/
│   └── darkdriving_lle/
│       ├── train/
│       │   ├── day/
│       │   └── night/
│       └── test/
│           ├── day/
│           └── night/
│
├── processed/
│
│   ├── test/                              # global fixed test set
│   │   ├── test_A/                        # night images
│   │   ├── test_B/                        # day images
│   │   ├── fixed_prompt_a.txt             # fixed prompts for unpaired model
│   │   ├── fixed_prompt_b.txt             
│   │   └── test_prompts.json              # fixed prompts for paired model
│
│   ├── 10shot/
│   │   ├── seed1/
│   │   │   ├── train_A/
│   │   │   ├── train_B/
│   │   │   ├── train_prompts.json
│   │   │   ├── fixed_prompt_a.txt
│   │   │   ├── fixed_prompt_b.txt
│   │   │   ├── test_A/                     # validation dataset
│   │   │   ├── test_B/
│   │   │   └── ...
│   │   ├── seed2/
│   │   └── seed3/
│   │
│   ├── 20shot/
│   └── 50shot/
...
```

## Running Experiments

This script launches training and evaluation for both models using the default experimental configuration of 10, 20, and 50 shots with seeds 1, 2, and 3. To override the default settings, specify custom values using the `--shots` and `--seeds` command-line arguments.
```bash
python scripts/run_experiments.py
```

## Evaluation

Formal evaluation is a post-training step over the held-out paired test set in
`data/processed/day2night/test/{day,night}`. It does not use the upstream
training-time eval folders as final results.

Run all formal evaluations after the training checkpoints exist:

```bash
bash scripts/run_all_evaluations.sh
```

The runner computes per-sample SSIM, LPIPS, and CLIP Vision cosine similarity,
then computes run-level CMMD from CLIP image embeddings. Outputs are written
under `results/evaluation/<model>/<shot>shot/seed<seed>/`:

- `generated/`: generated night images aligned by filename
- `per_sample_metrics.csv`: per-pair SSIM, LPIPS, and CLIP similarity
- `summary.json`: aggregate means/stds plus CMMD
- `manifest.json`: checkpoint, split, metric, and filename metadata

To evaluate an existing generated-image directory without running model
inference:

```bash
python src/eval/run_evaluation.py \
  --model pix2pix \
  --shot 10 \
  --seed 1 \
  --checkpoint results/pix2pix/10shot/seed1/checkpoints/model_5001.pkl \
  --generated-dir results/evaluation/pix2pix/10shot/seed1/generated
```

The breaking-point analysis aggregates all `summary.json` files and writes
`results/evaluation/summary.csv` plus `results/evaluation/breaking_points.json`:

```bash
python src/eval/analyze_breaking_point.py
```

Higher is better for SSIM and CLIP similarity; lower is better for LPIPS and
CMMD. A breaking point is flagged at the lowest shot level where the lower-shot
run significantly degrades against the next higher shot level and also shows
increased variance across seeds and test samples.
