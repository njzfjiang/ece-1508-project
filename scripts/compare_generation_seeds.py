#!/usr/bin/env python
"""Compare repeated generations of the same inputs across inference seeds."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "directories",
        nargs="+",
        type=Path,
        help="Direct output directories for two or more inference seeds",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _image_paths(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Generation directory not found: {directory}")
    return {
        path.name: path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def compare_directories(directories: list[Path]) -> dict:
    if len(directories) < 2:
        raise ValueError("At least two generation directories are required")

    indexed = [_image_paths(directory) for directory in directories]
    common = sorted(set.intersection(*(set(paths) for paths in indexed)))
    if not common:
        raise ValueError("The generation directories have no common image filenames")

    records = []
    for filename in common:
        paths = [images[filename] for images in indexed]
        hashes = [_sha256(path) for path in paths]
        arrays = [_rgb(path) for path in paths]
        if len({array.shape for array in arrays}) != 1:
            raise ValueError(f"Image dimensions differ across seeds for {filename}")

        pairwise_mae = [
            float(np.mean(np.abs(arrays[left] - arrays[right])))
            for left, right in itertools.combinations(range(len(arrays)), 2)
        ]
        records.append(
            {
                "filename": filename,
                "exactly_identical": len(set(hashes)) == 1,
                "pairwise_pixel_mae_mean": float(np.mean(pairwise_mae)),
                "pairwise_pixel_mae_max": float(np.max(pairwise_mae)),
            }
        )

    mae_values = [record["pairwise_pixel_mae_mean"] for record in records]
    identical = sum(record["exactly_identical"] for record in records)
    return {
        "directories": [str(directory.resolve()) for directory in directories],
        "num_common_images": len(records),
        "exactly_identical_images": identical,
        "exactly_identical_fraction": identical / len(records),
        "pairwise_pixel_mae_mean": float(np.mean(mae_values)),
        "pairwise_pixel_mae_max": float(
            max(record["pairwise_pixel_mae_max"] for record in records)
        ),
        "per_image": records,
    }


def main() -> int:
    args = parse_args()
    summary = compare_directories(args.directories)
    rendered = json.dumps(summary, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"Wrote {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
