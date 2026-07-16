#!/usr/bin/env python

import argparse
from pathlib import Path

from src.eval.utils import find_pairs, find_checkpoint
from src.eval.generate import generate

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/base.yaml"


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

    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)

    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-root", type=Path, default=Path("outputs"))

    parser.add_argument(
        "--test-root",
        type=Path,
        default=Path("data/processed/test"),
    )

    parser.add_argument("--output", type=Path)

    parser.add_argument(
        "--prompt",
        default="a driving scene during the night",
    )

    parser.add_argument("--test-samples", type=int)

    parser.add_argument("--use-fp16", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()
    pairs = find_pairs(args.test_root, limit=args.test_samples)
    
    print("Starting sample generation...")

    for model in args.models:
        for shot in args.shots:
            for seed in args.seeds:
                checkpoint = find_checkpoint(
                    root=args.checkpoint_root,
                    shot=shot,
                    seed=seed,
                    checkpoint=args.checkpoint,
                )

                output = (
                    args.output
                    or Path("results/generated") / model / f"{shot}shot" / f"seed{seed}"
                )

                generate(
                    model=model,
                    checkpoint=checkpoint,
                    pairs=pairs,
                    out=output,
                    prompt=args.prompt,
                    fp16=args.use_fp16,
                )

                print(f"Generated samples saved to {output}")
    print("Sample generation completed.")

if __name__ == "__main__":
    main()
