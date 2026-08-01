# Experiment Log

This log records completed day-to-night experiments, configuration decisions,
and the remaining formal evaluation work. Validation results must not be
reported as held-out test results.

## Data contract

- Task direction: day to night (`A=day`, `B=night`).
- Training/validation root: `data/processed/{shot}shot/seed{seed}/`.
- Formal held-out test root: `data/processed/test/`.
- Prompt: `a driving scene during the night`.
- Metrics: SSIM (higher), LPIPS (lower), CLIP similarity (higher), and CMMD
  (lower).

### Verified local splits (2026-07-29)

| Shots | Seeds | Train A/B per seed | Validation A/B per seed | Status |
|---:|:---:|---:|---:|:---:|
| 10 | 1, 2, 3 | 10 / 10 | 100 / 100 | Verified |
| 20 | 1, 2, 3 | 20 / 20 | 100 / 100 | Verified |
| 50 | 1, 2, 3 | 50 / 50 | 100 / 100 | Verified |

The held-out test split contains 3,632 filename-aligned day/night pairs. Split
creation is reproducible with:

```bash
python scripts/prepare_fewshot_splits.py --shots 10 20 50 --seeds 1 2 3
```

## Current formal configuration

Source: `configs/base.yaml` as reviewed on 2026-07-29.

| Setting | Pix2Pix-Turbo | CycleGAN-Turbo |
|---|---:|---:|
| Base model | `stabilityai/sd-turbo` | `stabilityai/sd-turbo` |
| Resolution | 512 | 512 |
| Batch size | 1 | 1 |
| Learning rate | 1e-5 | 1e-5 |
| Maximum steps | 2,000 | 2,000 |
| U-Net LoRA rank | 8 | 4 |
| VAE LoRA rank | 4 | 4 |
| Precision | FP32 | FP16 |
| Gradient checkpointing | No | Yes |
| Training preprocessing | `resized_crop_512` | `resize_512x512` |
| Evaluation preprocessing | `resize_512x512` | `resize_512x512` |
| Checkpoint interval | 500 | 2,000 |

The model-specific precision and memory settings differ because the CycleGAN
training graph contains two translation directions. Both models use the same
step budget, resolution, batch size, learning rate, data splits, and evaluation
protocol. The separate `lora_rank_experiment` section remains a historical
controlled ablation at 1,000 steps and LR 5e-6; it is not read by the standard
training launcher.

## Completed validation experiments

All tables below use 10-shot, seed 1, 100 filename-aligned validation pairs,
512x512 generation, generation seed 0, and the four metrics listed above unless
stated otherwise.

### CycleGAN U-Net LoRA-rank pilot

Date: 2026-07-29. Fixed settings: 1,000 steps, LR 5e-6, VAE rank 4, identity
weights 1.0, FP16.

| U-Net rank | SSIM | LPIPS | CLIP similarity | CMMD |
|---:|---:|---:|---:|---:|
| 4 | 0.4279 | 0.7619 | 0.8781 | 0.0764 |
| 128 | 0.4888 | 0.7180 | 0.8925 | 0.0499 |

Decision: rank 128 improves the measured validation metrics at 1,000 steps but
showed signs of excessive visual change/overfitting. Retain U-Net rank 4 as the
lower-capacity formal baseline. Do not treat rank 128 as the default based on a
single seed.

Artifacts: `tmp/lora_rank_experiment/`.

### CycleGAN learning-rate and identity pilots

Date: 2026-07-29. Fixed settings: U-Net/VAE ranks 4/4, 1,000 steps, FP16.

| Learning rate | Identity weights | SSIM | LPIPS | CLIP similarity | CMMD |
|---:|---:|---:|---:|---:|---:|
| 5e-6 | 1.0 | 0.4279 | 0.7619 | 0.8781 | 0.0764 |
| 1e-5 | 1.0 | 0.6301 | 0.5688 | 0.8967 | 0.0450 |
| 5e-6 | 0.5 | 0.4372 | 0.7528 | 0.8844 | 0.0739 |

Decision: doubling the learning rate produced the dominant improvement. Halving
the identity weights yielded only a small gain. Use LR 1e-5 and identity weights
1.0 in the formal configuration. Qualitative review noted that LR 1e-5 can make
lighting changes more aggressive, so retain qualitative examples alongside the
metrics.

Artifacts: `tmp/cyclegan_lr1e-5_idt1/` and
`tmp/cyclegan_lr5e-6_idt0p5/`.

### Pix2Pix-Turbo checkpoint sweep

Date: 2026-07-29. Fixed settings: U-Net/VAE ranks 8/4, LR 1e-5. Generation used
FP16 inference and `resize_512x512`; training used FP32 and
`resized_crop_512`.

| Step | SSIM | LPIPS | CLIP similarity | CMMD |
|---:|---:|---:|---:|---:|
| 500 | 0.5819 | 0.5400 | 0.8892 | 0.0696 |
| 1,000 | 0.6879 | 0.5047 | 0.9185 | 0.0362 |
| 1,500 | 0.7096 | 0.5011 | 0.9184 | 0.0418 |
| 2,000 | 0.7089 | 0.4995 | 0.9316 | 0.0215 |

Decision: the main convergence elbow is at 1,000 steps. The 2,000-step
checkpoint is retained for the formal run because SSIM/LPIPS remain stable while
CLIP similarity and CMMD improve. Runs beyond 2,000 steps are not currently
justified.

Artifacts: `tmp/validation_evaluation/pix2pix_step*/` and
`tmp/validation_generated/pix2pix_step*/`.

### Superseded CycleGAN 10,000-step pilot

An earlier 10-shot, seed-1 run showed a large gain through approximately
4,000-6,000 steps and little improvement afterward. That run predates the final
checkpoint patch that preserves the additional U-Net input-convolution weights,
so its checkpoints and convergence curve are not directly comparable to the
current configuration. It is retained only as historical evidence that 10,000
steps were unnecessary and must not be used as a formal result.

## Formal experiment grid

Planned grid: two models x three shot levels x three seeds = 18 training runs.

```bash
python scripts/run_experiment.py \
  --models pix2pix cyclegan \
  --shots 10 20 50 \
  --seeds 1 2 3 \
  --config configs/base.yaml
```

Run generation and evaluation on the independent test root after all selected
checkpoints are available. The formal evaluation must use 200 held-out samples
as configured in `base.yaml` (or explicitly document a change to all 3,632
pairs), not the per-seed validation directories used above.

## Status

- [x] Data direction and split contract verified.
- [x] Training and evaluation smoke tests completed for both models.
- [x] CycleGAN rank, learning-rate, and identity pilots completed.
- [x] Pix2Pix checkpoint sweep completed.
- [x] Formal baseline configuration selected.
- [x] Complete the 18-run training grid with `configs/5090.yaml`.
- [x] Generate held-out test outputs with a fixed generation seed.
- [x] Evaluate all model/shot/seed runs and generate aggregate CSV/JSON files.
- [ ] Perform qualitative failure-case review using the same selected
  checkpoints.
