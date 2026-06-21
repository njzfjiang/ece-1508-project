# Project Documentation

The root [README](../README.md) is the single source of truth for environment
setup, data preparation, Colab smoke testing, and experiment commands.

Additional project records:

- [Experiment log](experiment_log.md): experiment metadata, seeds, resources,
  and results.
- [`configs/base.yaml`](../configs/base.yaml): active shared and model-specific
  training configuration.
- [`src/eval/metrics.py`](../src/eval/metrics.py): reusable paired-image metric
  implementations. The formal CMMD-based evaluation runner is still pending.

Keeping operational instructions in the root README avoids maintaining two
copies that drift apart.
