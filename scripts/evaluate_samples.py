#!/usr/bin/env python
"""Evaluate generated samples on the held-out filename-aligned test set."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.evaluate import evaluate_generated_pairs
from src.eval.metrics import CMMDCalculator, MetricsCalculator
from src.eval.summarize_evaluations import summarize
from src.eval.utils import find_pairs, resolve_path

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "base.yaml"
SUPPORTED_METRICS = {"ssim", "lpips", "clip_similarity", "cmmd"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        default=["pix2pix", "cyclegan"],
        choices=["pix2pix", "cyclegan"],
    )
    parser.add_argument("--shots", nargs="+", type=int, default=[10, 20, 50])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--generated-root", type=Path)
    parser.add_argument("--test-root", type=Path)
    parser.add_argument("--test-samples", type=int)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--metrics", nargs="+")
    parser.add_argument("--generate-summary", action="store_true")
    return parser.parse_args()


def _generation_metadata(generated_dir: Path) -> dict:
    path = generated_dir / "generation_manifest.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    args = parse_args()
    config = OmegaConf.load(args.config.resolve())
    metrics = list(
        args.metrics
        or OmegaConf.select(
            config,
            "eval.metrics",
            default=["ssim", "lpips", "clip_similarity", "cmmd"],
        )
    )
    unknown = set(metrics) - SUPPORTED_METRICS
    if unknown:
        raise ValueError(f"Unsupported metrics: {sorted(unknown)}")

    test_root = (
        args.test_root.resolve()
        if args.test_root is not None
        else resolve_path(
            PROJECT_ROOT,
            str(OmegaConf.select(config, "eval.test_root", default="data/processed/test")),
        )
    )
    test_samples = args.test_samples
    if test_samples is None:
        test_samples = int(OmegaConf.select(config, "eval.test_samples", default=200))
    pairs = find_pairs(test_root, limit=test_samples)
    generated_root = (
        resolve_path(PROJECT_ROOT, args.generated_root)
        if args.generated_root is not None
        else resolve_path(
            PROJECT_ROOT,
            str(
                OmegaConf.select(
                    config,
                    "eval.generated_dir",
                    default="results/generated",
                )
            ),
        )
    )
    output_root = (
        resolve_path(PROJECT_ROOT, args.output_root)
        if args.output_root is not None
        else resolve_path(
            PROJECT_ROOT,
            str(OmegaConf.select(config, "eval.output_dir", default="results/evaluation")),
        )
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model_name = str(
        OmegaConf.select(config, "eval.clip_model", default="ViT-B/32")
    )
    calculator = MetricsCalculator(
        device=device,
        lpips_backbone=str(
            OmegaConf.select(config, "eval.lpips_backbone", default="alex")
        ),
        clip_model_name=clip_model_name,
        requested_metrics=set(metrics),
    )
    cmmd_calculator = None
    if "cmmd" in metrics:
        cmmd_calculator = CMMDCalculator(
            device=device,
            clip_model_name=str(
                OmegaConf.select(
                    config,
                    "eval.cmmd_clip_model",
                    default="ViT-L/14@336px",
                )
            ),
            batch_size=int(
                OmegaConf.select(config, "eval.cmmd_batch_size", default=32)
            ),
            sigma=float(OmegaConf.select(config, "eval.cmmd_sigma", default=10.0)),
            scale=float(OmegaConf.select(config, "eval.cmmd_scale", default=1000.0)),
        )

    for model in args.models:
        for shot in args.shots:
            for seed in args.seeds:
                generated_dir = (
                    generated_root / model / f"{shot}shot" / f"seed{seed}"
                )
                output_dir = output_root / model / f"{shot}shot" / f"seed{seed}"
                generation = _generation_metadata(generated_dir)
                for key, expected in (("model", model), ("shot", shot), ("seed", seed)):
                    if key in generation and generation[key] != expected:
                        raise ValueError(
                            f"Generation manifest mismatch for {key}: "
                            f"{generation[key]!r} != {expected!r}"
                        )
                expected_filenames = [day.name for day, _ in pairs]
                if generation.get("filenames") not in (None, expected_filenames):
                    raise ValueError(
                        "Generation manifest filenames do not match this evaluation set"
                    )
                if generation.get("test_root") not in (
                    None,
                    str(test_root.resolve()),
                ):
                    raise ValueError(
                        "Generation manifest test_root does not match this evaluation"
                    )
                metadata = {
                    "model": model,
                    "shot": shot,
                    "seed": seed,
                    "task": "day_to_night",
                    "test_root": str(test_root.resolve()),
                    "metrics": metrics,
                    "metric_configuration": {
                        "clip_similarity_model": (
                            clip_model_name if "clip_similarity" in metrics else None
                        ),
                        "cmmd": (
                            cmmd_calculator.configuration
                            if cmmd_calculator is not None
                            else None
                        ),
                    },
                    "generation": generation or None,
                }
                print(f"Evaluating {model}, {shot}-shot, seed {seed}")
                result = evaluate_generated_pairs(
                    pairs=pairs,
                    generated_dir=generated_dir,
                    output_dir=output_dir,
                    metrics_calculator=calculator,
                    cmmd_calculator=cmmd_calculator,
                    requested_metrics=metrics,
                    metadata=metadata,
                )
                print(json.dumps(result["metrics"], indent=2))

    if args.generate_summary:
        summarize(config=args.config, evaluation_root=output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
