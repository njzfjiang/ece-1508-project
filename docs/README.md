# Project Documentation

The root [README](../README.md) is the single source of truth for environment
setup, data preparation, Colab smoke testing, and experiment commands.

Additional project records:

- [Experiment log](experiment_log.md): experiment metadata, seeds, resources,
  and results.
- [`configs/base.yaml`](../configs/base.yaml): active shared and model-specific
  training configuration.
- [`src/eval/metrics.py`](../src/eval/metrics.py): reusable paired-image and
  CMMD metric implementations.
- [`src/eval/run_evaluation.py`](../src/eval/run_evaluation.py): formal
  held-out evaluation runner.
- [`src/eval/analyze_breaking_point.py`](../src/eval/analyze_breaking_point.py):
  aggregate metric analysis and breaking-point detection.

Keeping operational instructions in the root README avoids maintaining two
copies that drift apart.
