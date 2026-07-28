# CycleGAN-Turbo U-Net LoRA-rank experiment

The runner reads the `lora_rank_experiment` section from a project YAML file
and trains each configured U-Net LoRA rank sequentially. The controlled pilot
in `configs/base.yaml` compares ranks 4 and 128 at 10-shot, seed 1, 512 pixels,
and 1,000 steps while holding the other training settings fixed.

Prepare the pinned and patched vendor tree first:

```bash
python scripts/setup.py --skip-install --skip-prepare
```

Run the controlled pilot sequentially:

```bash
python lora_rank_experiment/run_lora_rank_experiment.py \
  --config configs/base.yaml
```

If xFormers is incompatible with the selected GPU, add:

```text
--no-xformers
```

Outputs are isolated by rank:

```text
outputs/lora_rank_experiment/
├── experiment_config.yaml
├── unet_rank_4/
│   ├── resolved_config.yaml
│   ├── command.txt
│   ├── validation/generated/
│   ├── validation/evaluation/summary.json
│   └── ...
├── unet_rank_128/
└── rank_results.json
```

The saved previews use one fixed validation input and fixed latent RNG. They
are diagnostic artifacts only; select a rank using aggregate validation
metrics rather than the preview alone. The images committed in this directory
are illustrative historical artifacts whose complete original run settings
were not recorded, so they are not formal quantitative results.

Each completed rank is automatically generated and scored on the split-local
validation images. To run only the 512-pixel memory smoke without evaluation:

```bash
python lora_rank_experiment/run_lora_rank_experiment.py \
  --config configs/smoke.yaml \
  --skip-evaluation
```
