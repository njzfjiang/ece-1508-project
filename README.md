# ece-1508-project

Few-shot day-to-night image translation on the DarkDriving dataset. This
project compares CycleGAN-Turbo and SD-Turbo+LoRA under 10-, 20-, and
50-shot settings to study how unpaired cycle-consistent learning and paired
supervision behave under limited data.

## Project Layout

- `configs/`: YAML configuration files used by the scripts
- `data/`: raw and processed dataset splits
- `docs/`: experiment notes and supporting documentation
- `external/`: vendored `img2img-turbo` source
- `notebooks/`: analysis of the experiment, including EDA and smoke-test notebooks
- `outputs/`: training checkpoints and related artifacts
- `results/`: generated samples and evaluation outputs
- `scripts/`: top-level experiment and utility entrypoints
- `src/`: training and evaluation code

## Setup

### DarkDriving Dataset (ICRA 2026)

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

The project is set up for Python 3.10 and CUDA 11.8.

```bash
conda env create -f environment.yaml
conda activate ece-1508
python scripts/setup.py
```

The setup script prepares the vendored `img2img-turbo` code and the processed
few-shot splits. It also applies the repository's replayable pix2pix and
CycleGAN compatibility and memory patches. If you already have the processed
data and only want to patch the vendored training tree, use:

```bash
python scripts/setup.py --skip-install --skip-prepare
```

## Data

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

## Full Experiment Pipeline

The main orchestration entrypoint is `scripts/run_experiment.py`.
It iterates over every requested model, shot count, and seed, then runs:

1. training
2. sample generation
3. evaluation
4. optional cross-run summarization

By default it runs both `pix2pix` and `cyclegan` for `10`, `20`, and `50`
shots with seeds `1`, `2`, and `3`.

This is the complete 18-run grid, so use explicit models, shots, and seeds for
pilots rather than starting the default command accidentally:

```bash
python scripts/run_experiment.py \
  --models cyclegan \
  --shots 10 \
  --seeds 1 \
  --config configs/base.yaml \
  --test-samples 200 \
  --generate-summary
```

Useful flags:

- `--models pix2pix cyclegan`: choose which pipelines to run
- `--shots 10 20 50`: choose the shot counts to iterate over
- `--seeds 1 2 3`: choose the random seeds to iterate over
- `--skip-training`: skip the training stage
- `--skip-generation`: skip sample generation
- `--skip-evaluation`: skip evaluation
- `--generate-summary`: write the aggregated summary after evaluation
- `--test-samples N`: limit the number of test pairs used for generation and evaluation
- `--prompt "..."`: override the generation prompt
- `--metrics ssim lpips clip_similarity cmmd`: override the evaluation metrics
- `--use-fp16`: enable fp16 during generation when supported

## Training Only

The training scripts live under `src/train/` and are invoked by the orchestrator.
The paired pipeline is handled by `src/train/model_paired.py`, and the unpaired
pipeline is handled by `src/train/model_unpaired.py`.

Run selected training launchers without generation or evaluation:

```bash
python scripts/train_models.py \
  --models pix2pix cyclegan \
  --shots 10 \
  --seeds 1 \
  --config configs/smoke.yaml
```

`configs/base.yaml` uses batch size 1 for the validated 512-pixel runs.
CycleGAN additionally uses FP16, gradient checkpointing, zero dataloader
workers, and less frequent checkpoints. Its smoke configuration uses
256-pixel preprocessing.

The formal comparison uses a common budget of 4,000 optimizer steps for both
models. Each run writes a structured `losses.csv` beside its checkpoints, so
training curves do not depend on terminal scrollback or W&B availability.

The default training output root is controlled by `configs/base.yaml`:

- `pix2pix_turbo.output_dir`: paired training checkpoints
- `cyclegan_turbo.output_dir`: unpaired training checkpoints

CycleGAN checkpoints store the three U-Net LoRA adapters and the separately
optimized base `conv_in` layer. The loader remains compatible with older
checkpoints that do not contain `sd_unet_conv_in`, although that missing
trained layer cannot be reconstructed retroactively.

## Generation And Evaluation

The generation script is `scripts/generate_samples.py` and the evaluation script
is `scripts/evaluate_samples.py`. Both loop over the requested models, shots,
and seeds, and write their outputs into the `results/` tree.

The configured per-model checkpoint roots are used automatically:

```bash
python scripts/generate_samples.py \
  --models pix2pix \
  --shots 10 \
  --seeds 1 \
  --config configs/base.yaml \
  --test-samples 200

python scripts/evaluate_samples.py \
  --models pix2pix \
  --shots 10 \
  --seeds 1 \
  --config configs/base.yaml \
  --test-samples 200 \
  --generate-summary
```

Generation writes a manifest containing the checkpoint, checkpoint step,
prompt, test root, and filenames. Evaluation carries that manifest into its
result metadata. `--test-samples` must be identical across generation and
evaluation; the full orchestrator forwards it to both stages.

Generation output is written under:

```text
results/generated/<model>/<shot>shot/seed<seed>/
```

Evaluation output is written under:

```text
results/evaluation/<model>/<shot>shot/seed<seed>/
```

Evaluation computes the metrics defined in `configs/base.yaml` by default:

- SSIM
- LPIPS
- CLIP image similarity
- CMMD

## Summarization

When enabled, the evaluation workflow also writes aggregate summaries across
seeds. The summarization output lives under `results/evaluation/` and includes
CSV and JSON tables for per-run and cross-seed reporting.

## Configuration

The default configuration is `configs/base.yaml`. It controls:

- dataset roots and image size
- training batch size, learning rate, checkpoint cadence, and precision
- model-specific LoRA and loss weights
- evaluation metrics and output paths
- logging behavior

`configs/smoke.yaml` can be used for a quick test run with a smaller setup.

## Tests

The CPU-only launcher and evaluation tests do not download model weights:

```bash
python -m pytest -q
```
