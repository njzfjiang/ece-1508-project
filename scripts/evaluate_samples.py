import argparse
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf

from src.eval.utils import find_pairs
from src.eval.metrics import MetricsCalculator
from src.eval.evaluate import evaluate_generated_pairs
from src.eval.summarize_evaluations import summarize

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "base.yaml"


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--models",
        nargs="+",
        default=["pix2pix", "cyclegan"],
        choices=["pix2pix", "cyclegan"],
    )

    parser.add_argument(
        "--shots",
        nargs="+",
        type=int,
        default=[10, 20, 50],
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[1, 2, 3],
    )

    parser.add_argument(
        "--generated-root",
        type=Path,
        default=Path("results/generated"),
    )

    parser.add_argument(
        "--test-root",
        type=Path,
        default=Path("data/processed/test"),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/evaluation"),
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
    )

    parser.add_argument(
        "--metrics",
        nargs="+",
    )

    parser.add_argument(
        "--generate-summary",
        action="store_true",
    )

    return parser.parse_args()


def main():

    args = parse_args()

    config = OmegaConf.load(args.config)

    metrics = args.metrics or OmegaConf.select(
        config,
        "eval.metrics",
        default=["ssim", "lpips", "clip_similarity", "cmmd"],
    )
    metrics = list(metrics)

    pairs = find_pairs(args.test_root)

    calculator = MetricsCalculator(
        device="cuda" if torch.cuda.is_available() else "cpu",
        requested_metrics=set(metrics),
    )

    for model in args.models:
        for shot in args.shots:
            for seed in args.seeds:

                print(f"Evaluating {model}, {shot}-shot, seed {seed}")

                generated_dir = (
                    args.generated_root
                    / model
                    / f"{shot}shot"
                    / f"seed{seed}"
                )

                output_dir = (
                    args.output_root
                    / model
                    / f"{shot}shot"
                    / f"seed{seed}"
                )

                metadata = {
                    "model": model,
                    "shot": shot,
                    "seed": seed,
                    "task": "day_to_night",
                }

                result = evaluate_generated_pairs(
                    pairs=pairs,
                    generated_dir=generated_dir,
                    output_dir=output_dir,
                    metrics_calculator=calculator,
                    requested_metrics=metrics,
                    metadata=metadata,
                )

                print(json.dumps(result["metrics"], indent=2))

    if args.generate_summary:
        summarize(
            config=args.config,
            evaluation_root=args.output_root,
        )


if __name__ == "__main__":
    main()