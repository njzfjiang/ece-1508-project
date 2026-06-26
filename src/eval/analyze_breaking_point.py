#!/usr/bin/env python
"""Aggregate formal evaluation outputs and estimate few-shot breaking points."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "base.yaml"
METRIC_DIRECTIONS = {
    "ssim": "higher",
    "clip_similarity": "higher",
    "lpips": "lower",
    "cmmd": "lower",
}


def load_runs(evaluation_root: Path) -> list[dict]:
    runs = []
    for summary_path in sorted(evaluation_root.glob("*/*shot/seed*/summary.json")):
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        metadata = payload["metadata"]
        run = {
            "model": metadata["model"],
            "shot": int(metadata["shot"]),
            "seed": int(metadata["seed"]),
            "summary_path": summary_path,
            "num_samples": int(payload["num_samples"]),
            "metrics": payload["metrics"],
            "samples": _load_sample_metrics(Path(payload["per_sample_metrics"])),
        }
        runs.append(run)
    if not runs:
        raise FileNotFoundError(f"No summary.json files found under {evaluation_root}")
    return runs


def write_summary_csv(runs: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metric_names = sorted({metric for run in runs for metric in run["metrics"]})
    metric_is_mapping = {
        metric: any(isinstance(run["metrics"].get(metric), dict) for run in runs)
        for metric in metric_names
    }
    fieldnames = ["model", "shot", "seed", "num_samples"]
    for metric in metric_names:
        if metric_is_mapping[metric]:
            fieldnames.extend([f"{metric}_mean", f"{metric}_std"])
        else:
            fieldnames.append(metric)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for run in sorted(runs, key=lambda item: (item["model"], item["shot"], item["seed"])):
            row = {
                "model": run["model"],
                "shot": run["shot"],
                "seed": run["seed"],
                "num_samples": run["num_samples"],
            }
            for metric, value in run["metrics"].items():
                if isinstance(value, dict):
                    row[f"{metric}_mean"] = value["mean"]
                    row[f"{metric}_std"] = value["std"]
                else:
                    row[metric] = value
            writer.writerow(row)


def analyze_breaking_points(
    runs: list[dict],
    alpha: float,
    bootstrap_iterations: int,
    variance_ratio_min: float,
) -> dict:
    grouped = defaultdict(list)
    for run in runs:
        grouped[run["model"]].append(run)

    results = {}
    for model, model_runs in grouped.items():
        shots = sorted({run["shot"] for run in model_runs})
        comparisons = []
        breaking_by_metric = {}
        for low, high in zip(shots, shots[1:]):
            low_runs = [run for run in model_runs if run["shot"] == low]
            high_runs = [run for run in model_runs if run["shot"] == high]
            for metric in sorted(_available_metrics(low_runs, high_runs)):
                direction = METRIC_DIRECTIONS.get(metric, "higher")
                low_values = _metric_values(low_runs, metric)
                high_values = _metric_values(high_runs, metric)
                if not low_values or not high_values:
                    continue
                low_mean = _mean(low_values)
                high_mean = _mean(high_values)
                low_variance = _variance(low_values)
                high_variance = _variance(high_values)
                p_value = _bootstrap_degradation_p_value(
                    low_values,
                    high_values,
                    direction=direction,
                    iterations=bootstrap_iterations,
                    seed=low * 1000 + high,
                )
                mean_degraded = _is_degraded(low_mean, high_mean, direction)
                variance_ratio = (
                    low_variance / high_variance if high_variance > 0 else float("inf")
                )
                variance_increased = low_variance > high_variance and (
                    variance_ratio >= variance_ratio_min or high_variance == 0
                )
                unstable = mean_degraded and p_value < alpha and variance_increased
                comparisons.append(
                    {
                        "metric": metric,
                        "lower_shot": low,
                        "higher_shot": high,
                        "lower_mean": low_mean,
                        "higher_mean": high_mean,
                        "lower_variance": low_variance,
                        "higher_variance": high_variance,
                        "variance_ratio": variance_ratio,
                        "p_value": p_value,
                        "mean_degraded": mean_degraded,
                        "variance_increased": variance_increased,
                        "unstable": unstable,
                    }
                )
                if unstable:
                    breaking_by_metric[metric] = min(
                        breaking_by_metric.get(metric, low), low
                    )

        unstable_shots = list(breaking_by_metric.values())
        results[model] = {
            "breaking_point": min(unstable_shots) if unstable_shots else None,
            "breaking_point_by_metric": breaking_by_metric,
            "comparisons": comparisons,
        }
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--evaluation-root", type=Path)
    parser.add_argument("--summary-csv", type=Path)
    parser.add_argument("--breaking-points", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = OmegaConf.load(args.config.resolve())
    alpha = float(OmegaConf.select(config, "eval.breaking_point_alpha", default=0.05))
    iterations = int(OmegaConf.select(config, "eval.bootstrap_iterations", default=1000))
    variance_ratio_min = float(
        OmegaConf.select(config, "eval.variance_ratio_min", default=1.25)
    )
    evaluation_root = (
        args.evaluation_root
        or PROJECT_ROOT / str(OmegaConf.select(config, "eval.output_dir", default="results/evaluation"))
    ).resolve()
    runs = load_runs(evaluation_root)
    summary_csv = args.summary_csv or evaluation_root / "summary.csv"
    breaking_points = args.breaking_points or evaluation_root / "breaking_points.json"

    write_summary_csv(runs, summary_csv)
    result = analyze_breaking_points(
        runs,
        alpha=alpha,
        bootstrap_iterations=iterations,
        variance_ratio_min=variance_ratio_min,
    )
    breaking_points.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote {summary_csv}")
    print(f"Wrote {breaking_points}")
    return 0


def _load_sample_metrics(path: Path) -> list[dict[str, float]]:
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.is_file():
        return []
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            parsed = {}
            for key, value in row.items():
                if key in METRIC_DIRECTIONS and value not in ("", None):
                    parsed[key] = float(value)
            rows.append(parsed)
    return rows


def _available_metrics(low_runs: list[dict], high_runs: list[dict]) -> set[str]:
    low = {metric for run in low_runs for metric in run["metrics"]}
    high = {metric for run in high_runs for metric in run["metrics"]}
    return low & high


def _metric_values(runs: list[dict], metric: str) -> list[float]:
    values = []
    if metric == "cmmd":
        for run in runs:
            if metric in run["metrics"]:
                values.append(float(run["metrics"][metric]))
        return values

    for run in runs:
        sample_values = [row[metric] for row in run["samples"] if metric in row]
        values.extend(sample_values)
    if values:
        return values
    for run in runs:
        metric_summary = run["metrics"].get(metric)
        if isinstance(metric_summary, dict):
            values.append(float(metric_summary["mean"]))
    return values


def _bootstrap_degradation_p_value(
    low_values: list[float],
    high_values: list[float],
    direction: str,
    iterations: int,
    seed: int,
) -> float:
    rng = random.Random(seed)
    count_not_degraded = 0
    for _ in range(max(iterations, 1)):
        low_sample = [rng.choice(low_values) for _ in low_values]
        high_sample = [rng.choice(high_values) for _ in high_values]
        low_mean = _mean(low_sample)
        high_mean = _mean(high_sample)
        if not _is_degraded(low_mean, high_mean, direction):
            count_not_degraded += 1
    return count_not_degraded / max(iterations, 1)


def _is_degraded(low_mean: float, high_mean: float, direction: str) -> bool:
    if direction == "lower":
        return low_mean > high_mean
    return low_mean < high_mean


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _variance(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    return sum((value - mean) ** 2 for value in values) / (len(values) - 1)


if __name__ == "__main__":
    raise SystemExit(main())
