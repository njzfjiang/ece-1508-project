#!/usr/bin/env python
"""Select reproducible qualitative examples from per-sample evaluation CSVs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


METRICS = ("ssim", "lpips", "clip_similarity")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics-root",
        type=Path,
        default=Path("results/full_grid_5090/evaluation"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/qualitative_selection"),
    )
    parser.add_argument(
        "--models", nargs="+", default=["pix2pix", "cyclegan"]
    )
    parser.add_argument("--shots", nargs="+", type=int, default=[5, 10, 20, 50])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    return parser.parse_args()


def _read_run(path: Path) -> dict[str, dict[str, float]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing per-sample metrics: {path}")
    rows: dict[str, dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"filename", *METRICS}
        missing_columns = required - set(reader.fieldnames or ())
        if missing_columns:
            raise ValueError(f"{path} is missing columns: {sorted(missing_columns)}")
        for row in reader:
            filename = row["filename"]
            if filename in rows:
                raise ValueError(f"Duplicate filename in {path}: {filename}")
            values = {metric: float(row[metric]) for metric in METRICS}
            if not all(math.isfinite(value) for value in values.values()):
                raise ValueError(f"Non-finite metric for {filename} in {path}")
            rows[filename] = values
    if not rows:
        raise ValueError(f"No per-sample rows in {path}")
    return rows


def load_runs(
    root: Path,
    models: list[str],
    shots: list[int],
    seeds: list[int],
) -> dict[tuple[str, int, int], dict[str, dict[str, float]]]:
    runs = {}
    for model in models:
        for shot in shots:
            for seed in seeds:
                path = (
                    root
                    / model
                    / f"{shot}shot"
                    / f"seed{seed}"
                    / "per_sample_metrics.csv"
                )
                rows = _read_run(path)
                runs[(model, shot, seed)] = rows
    return runs


def filename_coverage(
    runs: dict[tuple[str, int, int], dict[str, dict[str, float]]]
) -> dict[str, object]:
    filename_sets = [set(run) for run in runs.values()]
    common = set.intersection(*filename_sets)
    union = set.union(*filename_sets)
    if not common:
        raise ValueError("Evaluation runs have no filenames in common")
    return {
        "run_count": len(runs),
        "per_run_counts": sorted({len(names) for names in filename_sets}),
        "common_filename_count": len(common),
        "union_filename_count": len(union),
        "all_runs_use_identical_filenames": all(
            names == filename_sets[0] for names in filename_sets[1:]
        ),
        "common_filenames": sorted(common),
    }


def _z_scores(values: dict[str, float]) -> dict[str, float]:
    mean = statistics.fmean(values.values())
    std = statistics.stdev(values.values()) if len(values) > 1 else 0.0
    if std == 0:
        return {name: 0.0 for name in values}
    return {name: (value - mean) / std for name, value in values.items()}


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _closest(
    scores: dict[str, float], target: float, excluded: set[str]
) -> str:
    candidates = [name for name in scores if name not in excluded]
    if not candidates:
        raise ValueError("Not enough unique filenames for qualitative selection")
    return min(candidates, key=lambda name: (abs(scores[name] - target), name))


def select_samples(
    runs: dict[tuple[str, int, int], dict[str, dict[str, float]]],
    seeds: list[int],
) -> list[dict[str, object]]:
    required = {
        ("pix2pix", shot, seed) for shot in (5, 10) for seed in seeds
    } | {
        ("cyclegan", shot, seed) for shot in (10, 20) for seed in seeds
    }
    missing = required - set(runs)
    if missing:
        raise ValueError(f"Breaking-point runs are missing: {sorted(missing)}")

    coverage = filename_coverage(runs)
    filenames = list(coverage["common_filenames"])
    aggregate = {
        metric: {
            name: statistics.fmean(
                run[name][metric] for run in runs.values()
            )
            for name in filenames
        }
        for metric in METRICS
    }
    z_ssim = _z_scores(aggregate["ssim"])
    z_lpips = _z_scores(aggregate["lpips"])
    z_clip = _z_scores(aggregate["clip_similarity"])
    difficulty = {
        name: z_lpips[name] - z_ssim[name] - z_clip[name]
        for name in filenames
    }

    pix_degradation = {}
    cycle_degradation = {}
    pix_consistency = {}
    cycle_consistency = {}
    for name in filenames:
        pix_diffs = [
            runs[("pix2pix", 5, seed)][name]["lpips"]
            - runs[("pix2pix", 10, seed)][name]["lpips"]
            for seed in seeds
        ]
        cycle_diffs = [
            runs[("cyclegan", 20, seed)][name]["clip_similarity"]
            - runs[("cyclegan", 10, seed)][name]["clip_similarity"]
            for seed in seeds
        ]
        pix_degradation[name] = statistics.fmean(pix_diffs)
        cycle_degradation[name] = statistics.fmean(cycle_diffs)
        pix_consistency[name] = sum(value > 0 for value in pix_diffs) / len(seeds)
        cycle_consistency[name] = sum(value > 0 for value in cycle_diffs) / len(seeds)

    z_pix = _z_scores(pix_degradation)
    z_cycle = _z_scores(cycle_degradation)
    breaking_score = {
        name: (
            z_pix[name]
            + z_cycle[name]
            + 0.5 * pix_consistency[name]
            + 0.5 * cycle_consistency[name]
        )
        for name in filenames
    }

    selected: list[dict[str, object]] = []
    used: set[str] = set()
    typical_target = _percentile(list(difficulty.values()), 0.50)
    challenging_target = _percentile(list(difficulty.values()), 0.90)
    typical = _closest(difficulty, typical_target, used)
    used.add(typical)
    challenging = _closest(difficulty, challenging_target, used)
    used.add(challenging)

    consistent_breaking = [
        name
        for name in filenames
        if name not in used
        and pix_consistency[name] >= 2 / 3
        and cycle_consistency[name] >= 2 / 3
    ]
    breaking_candidates = consistent_breaking or [
        name for name in filenames if name not in used
    ]
    breaking = max(breaking_candidates, key=lambda name: (breaking_score[name], name))

    for role, name in (
        ("typical", typical),
        ("challenging", challenging),
        ("breaking_sensitive", breaking),
    ):
        selected.append(
            {
                "role": role,
                "filename": name,
                "difficulty_score": difficulty[name],
                "mean_ssim": aggregate["ssim"][name],
                "mean_lpips": aggregate["lpips"][name],
                "mean_clip_similarity": aggregate["clip_similarity"][name],
                "pix2pix_5_to_10_lpips_degradation": pix_degradation[name],
                "pix2pix_consistent_seeds": pix_consistency[name],
                "cyclegan_10_to_20_clip_degradation": cycle_degradation[name],
                "cyclegan_consistent_seeds": cycle_consistency[name],
            }
        )
    return selected


def write_selection(
    output_dir: Path,
    selected: list[dict[str, object]],
    metadata: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    filenames = [str(row["filename"]) for row in selected]
    (output_dir / "selected_filenames.txt").write_text(
        "\n".join(filenames) + "\n", encoding="utf-8"
    )
    (output_dir / "selection.json").write_text(
        json.dumps({**metadata, "selections": selected}, indent=2),
        encoding="utf-8",
    )
    with (output_dir / "selection.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)


def main() -> int:
    args = parse_args()
    metrics_root = args.metrics_root.resolve()
    output_dir = args.output_dir.resolve()
    runs = load_runs(metrics_root, args.models, args.shots, args.seeds)
    coverage = filename_coverage(runs)
    if not coverage["all_runs_use_identical_filenames"]:
        print(
            "WARNING: evaluation runs use different filename subsets; "
            f"selection is restricted to their {coverage['common_filename_count']} "
            "common filenames."
        )
    selected = select_samples(runs, args.seeds)
    write_selection(
        output_dir,
        selected,
        {
            "metrics_root": str(metrics_root),
            "models": args.models,
            "shots": args.shots,
            "seeds": args.seeds,
            "filename_coverage": coverage,
            "selection_rules": {
                "typical": "closest to median cross-run standardized difficulty",
                "challenging": "closest to 90th-percentile cross-run standardized difficulty",
                "breaking_sensitive": (
                    "largest combined pix2pix 5-to-10 LPIPS and CycleGAN "
                    "10-to-20 CLIP degradation, preferring >=2/3 seed consistency"
                ),
            },
        },
    )
    for row in selected:
        print(f"{row['role']}: {row['filename']}")
    print(f"Wrote qualitative selection to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
