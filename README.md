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

**Auto Download:**
Run below code to download the processed data
```
bash scripts/download_dark_driving.sh
```

## Setup

Python 3.10 and CUDA 11.8 are the reproducible baseline. Conda is recommended:

```bash
conda env create -f environment.yaml
conda activate ece-1508
python scripts/setup.py --skip-install
```

The setup script clones the official `img2img-turbo` repository, pins it to
commit `86f54146590ffb4543c8cf85b5a36657da670924`, installs the required
compatibility patches, and prepares the few-shot training views. The Conda
environment already installs `requirements.txt`, so `--skip-install` avoids a
duplicate pip install. To prepare only selected views, pass `--shots` and
`--seeds`.

When a preprocessed subset was uploaded instead of the full raw dataset, set up
only the pinned and patched vendor tree:

```bash
python scripts/setup.py --skip-install --skip-prepare
```

Both setup modes idempotently apply the FP16 safety fix, the upstream
training-loop fix that makes `max_train_steps` stop exactly, and optimizer
parameter de-duplication. Training is launched through the active Python
interpreter, so a copied environment cannot silently reuse another environment's
`accelerate` console script.

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
│   │   ├── test_A/                        # day images (model input)
│   │   ├── test_B/                        # night images (target)
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

This script launches config-driven few-shot training using 10, 20, and 50 shots
with seeds 1, 2, and 3. Pix2pix remains the default so an existing command does
not unexpectedly start the more memory-intensive CycleGAN job. Select one or
both models with `--models`; override the experiment grid with `--shots` and
`--seeds`.

```bash
python scripts/run_experiments.py

python scripts/run_experiments.py \
  --models pix2pix cyclegan
```

`configs/base.yaml` controls both upstream commands, including batch size,
workers, learning rate, training/checkpoint steps, precision, xformers, gradient
checkpointing, LoRA ranks, loss weights, and image preparation. CycleGAN skips
its expensive training-time FID/DINO path by default because formal evaluation
is separate, backpropagates the two translation directions sequentially to
reduce peak activation memory, and checkpoints every 2000 steps because its
checkpoint files are substantially larger. When `logging.use_wandb` is false,
the launchers set `WANDB_MODE=disabled` while retaining the upstream-supported
`wandb` reporter.

For a cheap first VPS smoke test, use the dedicated config and one run:

```bash
python scripts/run_experiments.py \
  --config configs/smoke.yaml \
  --shots 10 \
  --seeds 1
```

The run must stop at exactly two steps and write
`outputs/smoke/pix2pix_turbo/train/10shot/seed1/checkpoints/model_2.pkl`.

Validate the CycleGAN command without using the GPU, then run its conservative
256x256 smoke configuration:

```bash
python scripts/run_experiments.py \
  --models cyclegan \
  --config configs/smoke.yaml \
  --shots 10 \
  --seeds 1 \
  --dry-run-cyclegan

python scripts/run_experiments.py \
  --models cyclegan \
  --config configs/smoke.yaml \
  --shots 10 \
  --seeds 1
```

The compatibility entrypoint `scripts/train_cyclegan_10shot.py` delegates to
the same launcher; it no longer maintains a separate hard-coded training path.
The CycleGAN smoke checkpoint is written to
`outputs/smoke/cyclegan_turbo/train/10shot/seed1/checkpoints/model_2.pkl`.

The launcher refuses to start when that run directory already contains checkpoints,
because automatic resume is not supported. Move or remove an earlier smoke run before
rerunning it; this also prevents evaluation from selecting a stale checkpoint.

## Evaluation

Formal evaluation is a post-training step over the held-out paired test set in
`data/processed/test/{test_A,test_B}`, where `test_A` contains day inputs and
`test_B` contains filename-aligned night targets. It does not use upstream
training-time validation folders as final results.

Run all formal evaluations after the training checkpoints exist:

```bash
bash scripts/run_all_evaluations.sh
```

The wrapper evaluates pix2pix by default. Evaluate both trained models with
`MODELS="pix2pix cyclegan" bash scripts/run_all_evaluations.sh`.

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
  --generated-dir results/evaluation/pix2pix/10shot/seed1/generated
```

The summarizer treats each seed as one independent run and writes per-run and
cross-seed tables:

```bash
python src/eval/summarize_evaluations.py
```

Outputs are `results/evaluation/runs.csv`, `aggregate.csv`, and `aggregate.json`.
Higher is better for SSIM and CLIP similarity; lower is better for LPIPS and
CMMD. With only three seeds, the project reports descriptive mean and standard
deviation rather than an overstated significance-based breaking point.
