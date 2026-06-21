# Experiment Log

This file logs all experiments conducted for this project.

## Data Preparation Log

Before starting any experiments, record your data preparation steps here.

### Data Preparation Checklist

- [ ] Raw dataset located at: `data/raw/darkdriving_lle/`
- [ ] Verify directory structure:
  - [ ] `data/raw/darkdriving_lle/train/day/` (images present)
  - [ ] `data/raw/darkdriving_lle/train/night/` (images present)
  - [ ] `data/raw/darkdriving_lle/test/day/` (images present)
  - [ ] `data/raw/darkdriving_lle/test/night/` (images present)
- [ ] Run preprocessing script: `python scripts/prepare_fewshot_splits.py`
- [ ] Verify output:
  - [ ] `data/processed/day2night/` contains train/val/test splits
  - [ ] `data/processed/splits/fewshot/` contains shot splits

### Data Preparation Results

| Date | Script | Shot Levels | Num Seeds | Status | Notes |
|------|--------|-------------|-----------|--------|-------|
| | prepare_fewshot_splits.py | 10, 20, 50 | 3 | | |

## Format

- **Date**: YYYY-MM-DD
- **Model**: Model name
- **Configuration**: Configuration file used
- **Shot Level**: Few-shot level (e.g., 10-shot, 20-shot)
- **Seed**: Random seed used
- **Results**: Key metrics and findings

## Experiments

### CycleGAN Experiments

#### Experiment 1: CycleGAN Baseline (Full Dataset)
- **Date**: 
- **Model**: CycleGAN
- **Configuration**: configs/cyclegan/config.yaml
- **Shot Level**: Full training set
- **Seed**: N/A
- **Results**: 
  - FID: 
  - Inception Score: 
  - Training Time: 
  - Notes: 

#### Experiment 2: CycleGAN Few-Shot Learning
- **Date**: 
- **Model**: CycleGAN
- **Configuration**: configs/cyclegan/config.yaml
- **Shot Level**: 10-shot
- **Seed**: 1
- **Results**: 
  - FID: 
  - Inception Score: 
  - Training Time: 
  - Notes: 

### SD-Turbo Experiments

#### Experiment 1: SD-Turbo Baseline (Full Dataset)
- **Date**: 
- **Model**: SD-Turbo
- **Configuration**: configs/sdturbo/config.yaml
- **Shot Level**: Full training set
- **Seed**: N/A
- **Results**: 
  - FID: 
  - Inception Score: 
  - Inference Time: 
  - Notes: 

#### Experiment 2: SD-Turbo Few-Shot Learning
- **Date**: 
- **Model**: SD-Turbo
- **Configuration**: configs/sdturbo/config.yaml
- **Shot Level**: 10-shot
- **Seed**: 1
- **Results**: 
  - FID: 
  - Inception Score: 
  - Inference Time: 
  - Notes: 

## Running Experiments

### Command Template

```bash
# Run with full dataset
python -m src.train.run_experiment --config configs/cyclegan/config.yaml --model cyclegan

# Run with few-shot splits
python -m src.train.run_experiment --config configs/cyclegan/config.yaml --model cyclegan --shot 10 --seed 1
```

### Notes

- Always verify data preparation is complete before running experiments
- Keep track of random seeds for reproducibility
- Log training time and computational resources used
- Save model checkpoints regularly
- Document any issues or observations during training
