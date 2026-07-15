#!/usr/bin/env python3
"""Compatibility entrypoint for the config-driven CycleGAN-Turbo launcher.

Examples:

    python scripts/train_cyclegan_10shot.py --config configs/smoke.yaml --dry-run
    python scripts/train_cyclegan_10shot.py --shots 20 --seed 1

All training logic lives in ``src/train/model_unpaired.py`` so standalone and
multi-model experiment runs share the same config and checkpoint contract.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.train.model_unpaired import main


def compatibility_main() -> int:
    """Preserve the legacy one-run defaults while delegating implementation."""
    arguments = sys.argv[1:]
    if "--shots" not in arguments:
        arguments = ["--shots", "10", *arguments]
    if "--seed" not in arguments and "--seeds" not in arguments:
        arguments = ["--seed", "1", *arguments]
    return main(arguments)


if __name__ == "__main__":
    raise SystemExit(compatibility_main())
