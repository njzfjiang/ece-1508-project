"""Evaluate filename-aligned generated and target images."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from src.eval.metrics import CMMDCalculator, MetricsCalculator
from src.eval.utils import image_to_tensor, summarize


def evaluate_generated_pairs(
    pairs: list[tuple[Path, Path]],
    generated_dir: Path,
    output_dir: Path,
    metrics_calculator: MetricsCalculator,
    cmmd_calculator: CMMDCalculator | None,
    requested_metrics: list[str],
    metadata: dict,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    cmmd_generated_paths: list[Path] = []
    cmmd_target_paths: list[Path] = []

    for day_path, night_path in pairs:
        generated_path = generated_dir / day_path.name
        if not generated_path.is_file():
            raise FileNotFoundError(f"Generated image missing: {generated_path}")
        generated = image_to_tensor(generated_path)
        target = image_to_tensor(night_path)
        if generated.shape != target.shape:
            raise ValueError(
                f"Shape mismatch for {day_path.name}: "
                f"generated={tuple(generated.shape)}, target={tuple(target.shape)}"
            )

        row: dict[str, float | str] = {
            "filename": day_path.name,
            "day_path": str(day_path.resolve()),
            "night_path": str(night_path.resolve()),
            "generated_path": str(generated_path.resolve()),
        }
        if "ssim" in requested_metrics:
            row["ssim"] = metrics_calculator.compute_ssim(generated, target)
        if "lpips" in requested_metrics:
            row["lpips"] = metrics_calculator.compute_lpips(generated, target)
        if "clip_similarity" in requested_metrics:
            generated_feature = metrics_calculator.extract_clip_features(generated)[0]
            target_feature = metrics_calculator.extract_clip_features(target)[0]
            row["clip_similarity"] = float(
                np.dot(generated_feature, target_feature)
            )
        if "cmmd" in requested_metrics:
            cmmd_generated_paths.append(generated_path)
            cmmd_target_paths.append(night_path)
        rows.append(row)

    summary = summarize(rows, requested_metrics)
    if "cmmd" in requested_metrics:
        if cmmd_calculator is None:
            raise ValueError("CMMD was requested but no CMMD calculator was provided")
        summary["cmmd"] = cmmd_calculator.compute_from_paths(
            cmmd_generated_paths,
            cmmd_target_paths,
        )

    csv_path = output_dir / "per_sample_metrics.csv"
    fieldnames = ["filename", "day_path", "night_path", "generated_path"] + [
        metric
        for metric in ("ssim", "lpips", "clip_similarity")
        if metric in requested_metrics
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    result = {
        "metadata": metadata,
        "num_samples": len(rows),
        "metrics": summary,
        "per_sample_metrics": str(csv_path.resolve()),
        "generated_dir": str(generated_dir.resolve()),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                **metadata,
                "filenames": [day.name for day, _ in pairs],
                "generated_dir": str(generated_dir.resolve()),
                "per_sample_metrics": str(csv_path.resolve()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return result
