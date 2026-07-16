"""Run selected few-shot training launchers."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["pix2pix", "cyclegan"],
        default=["pix2pix", "cyclegan"],
    )
    parser.add_argument(
        "--shots",
        nargs="+",
        type=int,
        default=[10, 20, 50],
        help="List of shot counts",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[1, 2, 3],
        help="List of seeds",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "base.yaml",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scripts = {
        "pix2pix": PROJECT_ROOT / "src" / "train" / "model_paired.py",
        "cyclegan": PROJECT_ROOT / "src" / "train" / "model_unpaired.py",
    }
    for model in args.models:
        command = [
            sys.executable,
            str(scripts[model]),
            "--shots",
            *map(str, args.shots),
            "--seeds",
            *map(str, args.seeds),
            "--config",
            str(args.config.resolve()),
        ]
        print("[CMD]", " ".join(command))
        subprocess.run(command, check=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
