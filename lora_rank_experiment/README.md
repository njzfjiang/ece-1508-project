# CycleGAN-Turbo U-Net LoRA-rank experiment

The runner reads the `lora_rank_experiment` section from a project YAML file
and trains each configured U-Net LoRA rank sequentially. The formal
configuration is in `configs/base.yaml`.

Prepare the pinned and patched vendor tree first:

```bash
python scripts/setup.py --skip-install --skip-prepare
```

Run the formal experiment sequentially:

```bash
python lora_rank_experiment/run_lora_rank_experiment.py \
  --config configs/base.yaml
```

If xFormers is incompatible with the selected GPU, add:

```text
--no-xformers
```

Formal outputs are isolated by rank:

```text
outputs/lora_rank_experiment/
├── unet_rank_16/
├── unet_rank_32/
├── unet_rank_64/
└── unet_rank_128/
```
