import csv
import json
from pathlib import Path
import argparse
import numpy as np

from src.eval.metrics import MetricsCalculator
from .utils import image_to_tensor, summarize


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

    rows = []
    generated_features = []
    target_features = []

    for day_path, night_path in pairs:
        generated_path = generated_dir / day_path.name

        if not generated_path.is_file():
            raise FileNotFoundError(f"Generated image missing: {generated_path}")

        generated = image_to_tensor(generated_path)
        target = image_to_tensor(night_path)

        if generated.shape != target.shape:
            raise ValueError(f"Shape mismatch {generated.shape} vs {target.shape}")

        row: dict[str, float | str] = {
            "filename": day_path.name,
        }

        if "ssim" in requested_metrics:
            row["ssim"] = float(metrics_calculator.compute_ssim(generated, target))

        if "lpips" in requested_metrics:
            row["lpips"] = float(metrics_calculator.compute_lpips(generated, target))

        if "cmmd" in requested_metrics:
            generated_features.append(
                metrics_calculator.extract_clip_features(generated)
            )
            target_features.append(metrics_calculator.extract_clip_features(target))
            if "clip_similarity" in requested_metrics:
                row["clip_similarity"] = float(
                    metrics_calculator.clip_similarity_from_features(
                        generated_features[-1], target_features[-1]
                    )
                )
        else:
            if "clip_similarity" in requested_metrics:
                row["clip_similarity"] = float(
                    metrics_calculator.compute_clip_similarity(generated, target)
                )

        rows.append(row)

    summary = summarize(rows, requested_metrics)
    if "cmmd" in requested_metrics:
        summary["cmmd"] = float(
            metrics_calculator.compute_cmmd_from_features(
                np.stack(generated_features),
                np.stack(target_features),
                sigma=cmmd_sigma,
            )
        )

    csv_path = output_dir / "per_sample_metrics.csv"
    fieldnames = list(rows[0].keys())

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["pix2pix", "cyclegan"])
    parser.add_argument("--shot", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--generated-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--test-root", type=Path)
    parser.add_argument("--test-samples", type=int)
    parser.add_argument("--metrics", nargs="+")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--use-fp16", action="store_true")
    return parser.parse_args()


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "base.yaml"
DEFAULT_PROMPT = "a driving scene during the night"
