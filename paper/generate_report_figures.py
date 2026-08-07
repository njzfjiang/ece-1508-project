"""Generate publication figures from the formal full-grid evaluation summary."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel


ROOT = Path(__file__).resolve().parents[1]
RUNS_PATH = ROOT / "docs" / "results_5090" / "runs.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "figures"

MODEL_ORDER = ("pix2pix", "cyclegan")
MODEL_LABELS = {
    "pix2pix": "pix2pix-turbo",
    "cyclegan": "CycleGAN-Turbo",
}
MODEL_COLORS = {
    "pix2pix": "#2C7FB8",
    "cyclegan": "#E76F51",
}
METRICS = {
    "ssim": ("SSIM", True),
    "lpips": ("LPIPS", False),
    "clip_similarity": ("CLIP similarity", True),
    "cmmd": (r"CMMD ($\sigma=10$)", False),
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 7.5,
            "axes.titlesize": 8.5,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.grid": True,
            "grid.alpha": 0.28,
            "grid.linewidth": 0.5,
        }
    )


def plot_performance(runs: pd.DataFrame, output_dir: Path) -> None:
    shots = sorted(runs["shot"].unique())
    figure, axes = plt.subplots(2, 2, figsize=(6.35, 4.25), constrained_layout=True)

    for axis, (metric, (label, higher_is_better)) in zip(axes.flat, METRICS.items()):
        for model in MODEL_ORDER:
            subset = runs[runs["model"].eq(model)]
            grouped = subset.groupby("shot")[metric]
            means = grouped.mean().reindex(shots)
            standard_deviations = grouped.std(ddof=1).reindex(shots)

            axis.errorbar(
                shots,
                means,
                yerr=standard_deviations,
                color=MODEL_COLORS[model],
                marker="o",
                markersize=3.5,
                capsize=2.5,
                linewidth=1.4,
                label=MODEL_LABELS[model],
            )
            for shot in shots:
                values = subset.loc[subset["shot"].eq(shot), metric]
                axis.scatter(
                    np.full(len(values), shot),
                    values,
                    color=MODEL_COLORS[model],
                    alpha=0.30,
                    s=9,
                    linewidths=0,
                )

        direction = "higher is better" if higher_is_better else "lower is better"
        axis.set_title(f"{label} ({direction})")
        axis.set_xlabel("Training examples per domain")
        axis.set_ylabel(label)
        axis.set_xticks(shots)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False)
    figure.suptitle("Held-out performance by shot level (mean ± SD across seeds)", fontsize=9.5)
    figure.savefig(output_dir / "performance_by_shot.png", bbox_inches="tight")
    plt.close(figure)


def plot_instability(runs: pd.DataFrame, output_dir: Path) -> None:
    shots = sorted(runs["shot"].unique())
    comparisons = list(zip(shots[:-1], shots[1:]))
    metrics = tuple(METRICS)
    figure, axes = plt.subplots(2, 2, figsize=(6.35, 4.0), constrained_layout=True)

    for axis, metric in zip(axes.flat, metrics):
        label, higher_is_better = METRICS[metric]
        sign = 1.0 if higher_is_better else -1.0

        for model in MODEL_ORDER:
            pivot = runs[runs["model"].eq(model)].pivot(
                index="seed", columns="shot", values=metric
            )
            ratios = []
            raw_candidates = []
            for low_shot, high_shot in comparisons:
                low = sign * pivot[low_shot]
                high = sign * pivot[high_shot]
                high_variance = high.var(ddof=1)
                ratio = low.var(ddof=1) / high_variance if high_variance > 0 else np.inf
                test = ttest_rel(high, low, alternative="greater")
                degradation = float((high - low).mean())
                ratios.append(ratio)
                raw_candidates.append(degradation > 0 and test.pvalue < 0.05 and ratio > 1)

            x_values = np.arange(len(comparisons))
            axis.plot(
                x_values,
                ratios,
                color=MODEL_COLORS[model],
                marker="o",
                markersize=3.5,
                linewidth=1.4,
                label=MODEL_LABELS[model],
            )
            candidate_indices = np.flatnonzero(raw_candidates)
            if len(candidate_indices):
                axis.scatter(
                    candidate_indices,
                    np.asarray(ratios)[candidate_indices],
                    marker="*",
                    s=48,
                    color=MODEL_COLORS[model],
                    edgecolor="black",
                    linewidth=0.5,
                    zorder=4,
                )

        axis.axhline(1.0, color="black", linestyle="--", linewidth=0.8)
        axis.set_yscale("log")
        axis.set_title(label)
        axis.set_xticks(
            np.arange(len(comparisons)),
            [f"{low} vs {high}" for low, high in comparisons],
        )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False)
    figure.supylabel("Seed variance ratio (lower / higher)", fontsize=7.5)
    figure.savefig(output_dir / "instability_by_shot.png", bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=RUNS_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_style()
    runs = pd.read_csv(args.runs)
    expected = {(model, shot, seed) for model in MODEL_ORDER for shot in (5, 10, 20, 50) for seed in (1, 2, 3)}
    observed = set(runs[["model", "shot", "seed"]].itertuples(index=False, name=None))
    if observed != expected:
        raise ValueError("Formal runs.csv does not contain the expected 24-run grid")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_performance(runs, args.output_dir)
    plot_instability(runs, args.output_dir)
    print(f"Wrote report figures to {args.output_dir}")


if __name__ == "__main__":
    main()
