#!/usr/bin/env python
"""Generate held-out day-to-night samples for trained checkpoints."""

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

from src.eval.generate import generate
from src.eval.utils import checkpoint_step, find_checkpoint, find_pairs, resolve_path

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "base.yaml"
DEFAULT_CHECKPOINT_ROOTS = {
    "pix2pix": "outputs/pix2pix_turbo/train",
    "cyclegan": "outputs/cyclegan_turbo/train",
}


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
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        help="Exact model training root; only valid when one model is selected",
    )
    parser.add_argument("--test-root", type=Path)
    parser.add_argument("--output", type=Path, help="Exact single-run output directory")
    parser.add_argument(
        "--prompt", default="a driving scene during the night"
    )
    parser.add_argument("--test-samples", type=int)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--use-fp16", action="store_true")
    parser.add_argument(
        "--generation-seed",
        type=int,
        default=0,
        help="Fixed base seed used for filename-stable latent sampling",
    )
    return parser.parse_args()


def _single_run(args: argparse.Namespace) -> bool:
    return len(args.models) == len(args.shots) == len(args.seeds) == 1


def main() -> int:
    args = parse_args()
    if args.generation_seed < 0:
        raise ValueError("--generation-seed must be non-negative")
    config = OmegaConf.load(args.config.resolve())
    if (args.checkpoint is not None or args.output is not None) and not _single_run(args):
        raise ValueError("--checkpoint and --output require one model, shot, and seed")
    if args.checkpoint_root is not None and len(args.models) != 1:
        raise ValueError("--checkpoint-root requires exactly one model")

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
    generated_root = resolve_path(
        PROJECT_ROOT,
        str(
            OmegaConf.select(
                config,
                "eval.generated_dir",
                default="results/generated",
            )
        ),
    )
    if not torch.cuda.is_available():
        raise RuntimeError("Checkpoint generation requires CUDA")
    torch.cuda.set_device(args.gpu)

    for model in args.models:
        configured_root = OmegaConf.select(
            config,
            f"eval.checkpoint_roots.{model}",
            default=DEFAULT_CHECKPOINT_ROOTS[model],
        )
        checkpoint_root = (
            args.checkpoint_root.resolve()
            if args.checkpoint_root is not None
            else resolve_path(PROJECT_ROOT, str(configured_root))
        )
        pix2pix_image_prep = str(
            OmegaConf.select(
                config,
                "pix2pix_turbo.test_image_prep",
                default="resize_512x512",
            )
        )
        cyclegan_image_prep = str(
            OmegaConf.select(
                config,
                "cyclegan_turbo.val_image_prep",
                default="resize_512x512",
            )
        )
        for shot in args.shots:
            for seed in args.seeds:
                checkpoint = find_checkpoint(
                    root=checkpoint_root,
                    shot=shot,
                    seed=seed,
                    checkpoint=args.checkpoint,
                )
                output = (
                    args.output.resolve()
                    if args.output is not None
                    else generated_root
                    / model
                    / f"{shot}shot"
                    / f"seed{seed}"
                )
                print(f"Generating {model}, {shot}-shot, seed {seed}")
                generate(
                    model=model,
                    checkpoint=checkpoint,
                    pairs=pairs,
                    out=output,
                    prompt=args.prompt,
                    fp16=args.use_fp16,
                    pix2pix_image_prep=pix2pix_image_prep,
                    cyclegan_image_prep=cyclegan_image_prep,
                    seed=args.generation_seed,
                )
                manifest = {
                    "model": model,
                    "shot": shot,
                    "seed": seed,
                    "checkpoint": str(checkpoint),
                    "checkpoint_step": checkpoint_step(checkpoint),
                    "test_root": str(test_root.resolve()),
                    "test_samples": len(pairs),
                    "prompt": args.prompt,
                    "use_fp16": args.use_fp16,
                    "image_prep": (
                        pix2pix_image_prep
                        if model == "pix2pix"
                        else cyclegan_image_prep
                    ),
                    "generation_seed": args.generation_seed,
                    "filenames": [day.name for day, _ in pairs],
                }
                (output / "generation_manifest.json").write_text(
                    json.dumps(manifest, indent=2), encoding="utf-8"
                )
                print(f"Generated samples saved to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
