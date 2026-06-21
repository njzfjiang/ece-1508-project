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

## Data Preparation

Before running experiments, you need to prepare the dataset and generate few-shot splits:

The helper script can be launched from any working directory because it resolves all paths relative to the repository:

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

## Evaluation

To evaluate trained models:

```bash
python -m src.eval.evaluate --model path/to/model --data data/processed --output results/evaluation
```

To aggregate results:

```bash
python -m src.eval.aggregate --results_dir results --output results/aggregated_results.json
```

## Troubleshooting

**Issue: "Split file not found" error**
- Ensure you have run `prepare_fewshot_splits.py` with the correct shot levels
- Check that `data/processed/splits/fewshot/` contains the expected directories

**Issue: "Day/night filenames do not match exactly" warning**
- The script pairs images by identical filename and ignores unmatched files

**Issue: Symlinks not working (Windows)**
- Use the `--copy_mode` flag when running `prepare_fewshot_splits.py`

**Issue: Processed output already exists**
- Re-run with `--overwrite` to rebuild the managed processed-data directories safely
