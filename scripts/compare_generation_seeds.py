#!/usr/bin/env python
"""Compare repeated generations of the same inputs across inference seeds."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from typing import Callable

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
    parser.add_argument(
        "--with-lpips",
        action="store_true",
        help="Also compute pairwise AlexNet LPIPS between inference seeds",
    )
    parser.add_argument("--device", default="cuda")
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


def compare_directories(
    directories: list[Path],
    lpips_fn: Callable[[np.ndarray, np.ndarray], float] | None = None,
) -> dict:
    if len(directories) < 2:
        raise ValueError("At least two generation directories are required")

    indexed = [_image_paths(directory) for directory in directories]
    common = sorted(set.intersection(*(set(paths) for paths in indexed)))
    if not common:
        raise ValueError("The generation directories have no common image filenames")

    records = []
    all_pairwise_mae = []
    all_pairwise_psnr = []
    all_pairwise_p95 = []
    all_unchanged_fraction = []
    all_pairwise_lpips = []
    for filename in common:
        paths = [images[filename] for images in indexed]
        hashes = [_sha256(path) for path in paths]
        arrays = [_rgb(path) for path in paths]
        if len({array.shape for array in arrays}) != 1:
            raise ValueError(f"Image dimensions differ across seeds for {filename}")

        pairwise_mae = []
        pairwise_psnr = []
        pairwise_p95 = []
        unchanged_fraction = []
        pairwise_lpips = []
        for left, right in itertools.combinations(range(len(arrays)), 2):
            delta = np.abs(arrays[left] - arrays[right])
            mse = float(np.mean((arrays[left] - arrays[right]) ** 2))
            pairwise_mae.append(float(np.mean(delta)))
            pairwise_p95.append(float(np.percentile(delta, 95)))
            unchanged_fraction.append(float(np.mean(delta == 0)))
            if mse > 0:
                pairwise_psnr.append(float(10 * np.log10(1.0 / mse)))
            if lpips_fn is not None:
                pairwise_lpips.append(lpips_fn(arrays[left], arrays[right]))

        all_pairwise_mae.extend(pairwise_mae)
        all_pairwise_psnr.extend(pairwise_psnr)
        all_pairwise_p95.extend(pairwise_p95)
        all_unchanged_fraction.extend(unchanged_fraction)
        all_pairwise_lpips.extend(pairwise_lpips)
        records.append(
            {
                "filename": filename,
                "exactly_identical": len(set(hashes)) == 1,
                "pairwise_pixel_mae_mean": float(np.mean(pairwise_mae)),
                "pairwise_pixel_mae_max": float(np.max(pairwise_mae)),
                "pairwise_psnr_mean_db": (
                    float(np.mean(pairwise_psnr)) if pairwise_psnr else None
                ),
                "pixel_abs_delta_p95_8bit": float(np.mean(pairwise_p95) * 255),
                "unchanged_channel_fraction": float(np.mean(unchanged_fraction)),
                **(
                    {"pairwise_lpips_mean": float(np.mean(pairwise_lpips))}
                    if pairwise_lpips
                    else {}
                ),
            }
        )

    identical = sum(record["exactly_identical"] for record in records)
    return {
        "directories": [str(directory.resolve()) for directory in directories],
        "num_common_images": len(records),
        "exactly_identical_images": identical,
        "exactly_identical_fraction": identical / len(records),
        "pairwise_pixel_mae_mean": float(np.mean(all_pairwise_mae)),
        "pairwise_pixel_mae_mean_8bit": float(np.mean(all_pairwise_mae) * 255),
        "pairwise_pixel_mae_max": float(
            max(record["pairwise_pixel_mae_max"] for record in records)
        ),
        "pairwise_psnr_mean_db": (
            float(np.mean(all_pairwise_psnr)) if all_pairwise_psnr else None
        ),
        "pixel_abs_delta_p95_8bit": float(np.mean(all_pairwise_p95) * 255),
        "unchanged_channel_fraction": float(np.mean(all_unchanged_fraction)),
        **(
            {"pairwise_lpips_mean": float(np.mean(all_pairwise_lpips))}
            if all_pairwise_lpips
            else {}
        ),
        "per_image": records,
    }


def main() -> int:
    args = parse_args()
    lpips_fn = None
    if args.with_lpips:
        import torch
        import lpips

        device = args.device
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        model = lpips.LPIPS(net="alex").to(device).eval()

        def compute_lpips(first: np.ndarray, second: np.ndarray) -> float:
            tensors = []
            for array in (first, second):
                tensor = torch.from_numpy(
                    np.ascontiguousarray(array.transpose(2, 0, 1))
                ).unsqueeze(0)
                tensors.append((tensor.to(device) * 2) - 1)
            with torch.inference_mode():
                return float(model(tensors[0], tensors[1]).item())

        lpips_fn = compute_lpips

    summary = compare_directories(args.directories, lpips_fn=lpips_fn)
    rendered = json.dumps(summary, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"Wrote {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
