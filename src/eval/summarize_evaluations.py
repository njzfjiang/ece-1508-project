#!/usr/bin/env python
"""Summarize evaluation runs across seeds without pseudo-replicated tests."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "base.yaml"


def load_runs(evaluation_root: Path) -> list[dict]:
    runs = []
    for summary_path in sorted(evaluation_root.glob("*/*shot/seed*/summary.json")):
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        metadata = payload["metadata"]
        runs.append(
            {
                "model": metadata["model"],
                "shot": int(metadata["shot"]),
                "seed": int(metadata["seed"]),
                "num_samples": int(payload["num_samples"]),
                "metrics": payload["metrics"],
                "summary_path": str(summary_path.resolve()),
            }
        )
    if not runs:
        raise FileNotFoundError(f"No summary.json files found under {evaluation_root}")
    return runs


def run_metric_value(run: dict, metric: str) -> float | None:
    value = run["metrics"].get(metric)
    if value is None:
        return None
    if isinstance(value, dict):
        return float(value["mean"])
    return float(value)


def aggregate_runs(runs: list[dict]) -> list[dict]:
    """Aggregate one value per seed/run for every model, shot, and metric."""
    grouped: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    seeds: dict[tuple[str, int, str], set[int]] = defaultdict(set)
    for run in runs:
        for metric in run["metrics"]:
            value = run_metric_value(run, metric)
            if value is None:
                continue
            key = (run["model"], run["shot"], metric)
            grouped[key].append(value)
            seeds[key].add(run["seed"])

    rows = []
    for (model, shot, metric), values in sorted(grouped.items()):
        mean = sum(values) / len(values)
        variance = (
            sum((value - mean) ** 2 for value in values) / (len(values) - 1)
            if len(values) > 1
            else 0.0
        )
        rows.append(
            {
                "model": model,
                "shot": shot,
                "metric": metric,
                "num_seeds": len(seeds[(model, shot, metric)]),
                "mean": mean,
                "std_across_seeds": variance**0.5,
                "min": min(values),
                "max": max(values),
            }
        )
    return rows


def _write_csv(output_path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_run_csv(runs: list[dict], output_path: Path) -> None:
    metric_names = sorted({metric for run in runs for metric in run["metrics"]})
    fieldnames = ["model", "shot", "seed", "num_samples", *metric_names]
    rows = []
    for run in sorted(
        runs, key=lambda item: (item["model"], item["shot"], item["seed"])
    ):
        row = {
            "model": run["model"],
            "shot": run["shot"],
            "seed": run["seed"],
            "num_samples": run["num_samples"],
        }
        for metric in metric_names:
            row[metric] = run_metric_value(run, metric)
        rows.append(row)
    _write_csv(output_path, fieldnames, rows)


def write_aggregate_outputs(rows: list[dict], csv_path: Path, json_path: Path) -> None:
    fieldnames = [
        "model",
        "shot",
        "metric",
        "num_seeds",
        "mean",
        "std_across_seeds",
        "min",
        "max",
    ]
    _write_csv(csv_path, fieldnames, rows)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--evaluation-root", type=Path)
    parser.add_argument("--runs-csv", type=Path)
    parser.add_argument("--aggregate-csv", type=Path)
    parser.add_argument("--aggregate-json", type=Path)
    return parser.parse_args()


def summarize(
    config: Path = DEFAULT_CONFIG,
    evaluation_root: Path | None = None,
    runs_csv: Path | None = None,
    aggregate_csv: Path | None = None,
    aggregate_json: Path | None = None,
):
    loaded_config = OmegaConf.load(config.resolve())

    evaluation_root = (
        evaluation_root
        or PROJECT_ROOT
        / str(
            OmegaConf.select(
                loaded_config, "eval.output_dir", default="results/evaluation"
            )
        )
    ).resolve()

    runs = load_runs(evaluation_root)
    aggregate = aggregate_runs(runs)

    runs_csv = (runs_csv or evaluation_root / "runs.csv").resolve()
    aggregate_csv = (aggregate_csv or evaluation_root / "aggregate.csv").resolve()
    aggregate_json = (aggregate_json or evaluation_root / "aggregate.json").resolve()

    write_run_csv(runs, runs_csv)
    write_aggregate_outputs(aggregate, aggregate_csv, aggregate_json)

    print(f"Wrote {runs_csv}")
    print(f"Wrote {aggregate_csv}")
    print(f"Wrote {aggregate_json}")


def main():
    args = parse_args()

    summarize(
        config=args.config,
        evaluation_root=args.evaluation_root,
        runs_csv=args.runs_csv,
        aggregate_csv=args.aggregate_csv,
        aggregate_json=args.aggregate_json,
    )


if __name__ == "__main__":
    raise SystemExit(main())
