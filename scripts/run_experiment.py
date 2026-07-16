#!/usr/bin/env python

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
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
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
    )

    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        help="Exact model checkpoint root; only valid with one selected model",
    )

    parser.add_argument(
        "--test-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "test",
    )

    parser.add_argument("--test-samples", type=int)

    parser.add_argument(
        "--prompt",
        default="a driving scene during the night",
    )

    parser.add_argument("--metrics", nargs="+")

    parser.add_argument("--use-fp16", action="store_true")
    parser.add_argument(
        "--gpus",
        nargs="+",
        type=int,
        help="Run independent training jobs concurrently on these GPU indices",
    )

    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--generate-summary", action="store_true")

    return parser.parse_args()


def run(cmd):

    print("\n[CMD]")
    print(" ".join(map(str, cmd)))
    subprocess.run(cmd, check=True)


def main():

    args = parse_args()

    if args.checkpoint_root is not None and len(args.models) != 1:
        raise ValueError("--checkpoint-root requires exactly one selected model")
    if args.test_samples is not None and args.test_samples <= 0:
        raise ValueError("--test-samples must be positive")

    print("\n==================== Starting Experiment ====================")

    # Training
    if not args.skip_training:
        training_cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts/train_models.py"),
            "--models",
            *args.models,
            "--shots",
            *map(str, args.shots),
            "--seeds",
            *map(str, args.seeds),
            "--config",
            str(args.config.resolve()),
        ]
        if args.gpus:
            training_cmd += ["--gpus", *map(str, args.gpus)]
        run(training_cmd)

    # Generate samples
    if not args.skip_generation:

        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts/generate_samples.py"),
            "--models",
            *args.models,
            "--shots",
            *map(str, args.shots),
            "--seeds",
            *map(str, args.seeds),
            "--test-root",
            str(args.test_root),
            "--config",
            str(args.config.resolve()),
            "--prompt",
            args.prompt,
        ]

        if args.checkpoint_root is not None:
            cmd += ["--checkpoint-root", str(args.checkpoint_root)]

        if args.test_samples:
            cmd += [
                "--test-samples",
                str(args.test_samples),
            ]

        if args.use_fp16:
            cmd.append("--use-fp16")

        run(cmd)

    # Evaluate samples
    if not args.skip_evaluation:

        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts/evaluate_samples.py"),
            "--models",
            *args.models,
            "--shots",
            *map(str, args.shots),
            "--seeds",
            *map(str, args.seeds),
            "--test-root",
            str(args.test_root),
            "--config",
            str(args.config.resolve()),
        ]

        if args.test_samples:
            cmd += ["--test-samples", str(args.test_samples)]

        if args.metrics:
            cmd += [
                "--metrics",
                *args.metrics,
            ]
            
        if args.generate_summary:
            cmd.append("--generate-summary")
            


        run(cmd)

    print("\n==================== Experiment Completed ====================")


if __name__ == "__main__":
    main()
