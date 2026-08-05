import csv

from scripts.select_qualitative_samples import (
    filename_coverage,
    load_runs,
    select_samples,
)


def _write_run(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("filename", "ssim", "lpips", "clip_similarity"),
        )
        writer.writeheader()
        writer.writerows(rows)


def test_qualitative_selection_is_unique_and_reproducible(tmp_path):
    models = ["pix2pix", "cyclegan"]
    shots = [5, 10, 20, 50]
    seeds = [1, 2, 3]
    names = ["a.png", "b.png", "c.png", "d.png", "e.png"]
    for model in models:
        for shot in shots:
            for seed in seeds:
                rows = []
                for index, name in enumerate(names):
                    rows.append(
                        {
                            "filename": name,
                            "ssim": 0.8 - index * 0.05 + shot * 0.0001,
                            "lpips": (
                                0.2
                                + index * 0.05
                                + (0.05 if model == "pix2pix" and shot == 5 else 0)
                            ),
                            "clip_similarity": (
                                0.9
                                - index * 0.01
                                - (0.03 if model == "cyclegan" and shot == 10 else 0)
                            ),
                        }
                    )
                _write_run(
                    tmp_path
                    / model
                    / f"{shot}shot"
                    / f"seed{seed}"
                    / "per_sample_metrics.csv",
                    rows,
                )

    runs = load_runs(tmp_path, models, shots, seeds)
    first = select_samples(runs, seeds)
    second = select_samples(runs, seeds)

    assert first == second
    assert [row["role"] for row in first] == [
        "typical",
        "challenging",
        "breaking_sensitive",
    ]
    assert len({row["filename"] for row in first}) == 3
    assert filename_coverage(runs)["all_runs_use_identical_filenames"] is True


def test_qualitative_selection_uses_common_filename_intersection(tmp_path):
    models = ["pix2pix", "cyclegan"]
    shots = [5, 10, 20, 50]
    seeds = [1, 2, 3]
    common = ["a.png", "b.png", "c.png"]
    for model in models:
        for shot in shots:
            for seed in seeds:
                extra = "early.png" if shot == 5 else "later.png"
                rows = [
                    {
                        "filename": name,
                        "ssim": 0.8 - index * 0.1,
                        "lpips": 0.2 + index * 0.1 + (0.02 if shot == 5 else 0),
                        "clip_similarity": 0.9 - index * 0.02,
                    }
                    for index, name in enumerate([*common, extra])
                ]
                _write_run(
                    tmp_path
                    / model
                    / f"{shot}shot"
                    / f"seed{seed}"
                    / "per_sample_metrics.csv",
                    rows,
                )

    runs = load_runs(tmp_path, models, shots, seeds)
    coverage = filename_coverage(runs)
    selected = select_samples(runs, seeds)

    assert coverage["common_filename_count"] == 3
    assert coverage["union_filename_count"] == 5
    assert coverage["all_runs_use_identical_filenames"] is False
    assert {row["filename"] for row in selected} == set(common)
