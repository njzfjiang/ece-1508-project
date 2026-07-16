"""Evaluate filename-aligned generated and target images."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from src.eval.metrics import MetricsCalculator
from src.eval.utils import image_to_tensor, summarize


def evaluate_generated_pairs(
    pairs: list[tuple[Path, Path]],
    generated_dir: Path,
    output_dir: Path,
    metrics_calculator: MetricsCalculator,
    requested_metrics: list[str],
    metadata: dict,
    cmmd_sigma: float = 1.0,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    generated_features: list[np.ndarray] = []
    target_features: list[np.ndarray] = []
    needs_clip = bool({"clip_similarity", "cmmd"} & set(requested_metrics))

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
        if needs_clip:
            generated_feature = metrics_calculator.extract_clip_features(generated)[0]
            target_feature = metrics_calculator.extract_clip_features(target)[0]
            generated_features.append(generated_feature)
            target_features.append(target_feature)
            if "clip_similarity" in requested_metrics:
                row["clip_similarity"] = float(
                    np.dot(generated_feature, target_feature)
                )
        rows.append(row)

    summary = summarize(rows, requested_metrics)
    if "cmmd" in requested_metrics:
        summary["cmmd"] = metrics_calculator.compute_cmmd_from_features(
            np.stack(generated_features),
            np.stack(target_features),
            sigma=cmmd_sigma,
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
