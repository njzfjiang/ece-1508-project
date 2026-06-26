import csv
import json
from pathlib import Path

from PIL import Image

from src.eval.analyze_breaking_point import analyze_breaking_points, load_runs
from src.eval.metrics import MetricsCalculator
from src.eval.run_evaluation import (
    evaluate_generated_pairs,
    find_test_pairs,
    resolve_checkpoint,
)


def write_image(path: Path, color):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=color).save(path)


def make_test_pairs(root: Path, names=("frame_001.jpg", "frame_002.jpg")):
    for name in names:
        write_image(root / "day2night" / "test" / "day" / name, "navy")
        write_image(root / "day2night" / "test" / "night" / name, "black")


class FakeMetrics:
    def compute_ssim(self, generated, target):
        return 0.75

    def compute_lpips(self, generated, target):
        return 0.25

    def compute_clip_similarity(self, generated, target):
        return 0.5

    def extract_clip_features(self, image):
        mean_value = float(image.mean().item())
        return [[mean_value, 1.0 - mean_value]]

    def compute_cmmd_from_features(self, generated_features, target_features):
        return MetricsCalculator.compute_cmmd_from_features(
            generated_features, target_features
        )


def test_cmmd_from_features_is_zero_for_matching_distributions():
    features = [[1.0, 0.0], [0.0, 1.0]]

    assert MetricsCalculator.compute_cmmd_from_features(features, features) == 0.0


def test_resolve_checkpoint_uses_latest_numeric_step(tmp_path):
    checkpoint_dir = tmp_path / "pix2pix" / "10shot" / "seed1" / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    for step in (1, 2001, 501):
        checkpoint_dir.joinpath(f"model_{step}.pkl").touch()
    checkpoint_dir.joinpath("model_latest.pkl").touch()

    checkpoint = resolve_checkpoint(tmp_path, "pix2pix", 10, 1)

    assert checkpoint.name == "model_2001.pkl"


def test_evaluation_preserves_pairing_and_writes_outputs(tmp_path):
    processed = tmp_path / "processed"
    make_test_pairs(processed)
    generated = tmp_path / "generated"
    for name in ("frame_001.jpg", "frame_002.jpg"):
        write_image(generated / name, "gray")
    pairs = find_test_pairs(processed)

    result = evaluate_generated_pairs(
        pairs=pairs,
        generated_dir=generated,
        output_dir=tmp_path / "eval",
        metrics_calculator=FakeMetrics(),
        requested_metrics=["ssim", "lpips", "clip_similarity", "cmmd"],
        metadata={"model": "pix2pix", "shot": 10, "seed": 1},
    )

    per_sample = Path(result["per_sample_metrics"])
    assert per_sample.is_file()
    assert (tmp_path / "eval" / "summary.json").is_file()
    assert (tmp_path / "eval" / "manifest.json").is_file()
    with per_sample.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["filename"] for row in rows] == ["frame_001.jpg", "frame_002.jpg"]
    assert result["metrics"]["ssim"]["mean"] == 0.75
    assert "cmmd" in result["metrics"]


def test_breaking_point_detects_degradation_and_variance(tmp_path):
    root = tmp_path / "evaluation"
    for shot, values in {
        10: [0.1, 0.4, 0.0, 0.5],
        20: [0.8, 0.82, 0.81, 0.83],
        50: [0.84, 0.85, 0.86, 0.87],
    }.items():
        sample_csv = root / "pix2pix" / f"{shot}shot" / "seed1" / "per_sample_metrics.csv"
        sample_csv.parent.mkdir(parents=True)
        with sample_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["filename", "ssim"])
            writer.writeheader()
            for index, value in enumerate(values):
                writer.writerow({"filename": f"{index}.jpg", "ssim": value})
        mean = sum(values) / len(values)
        summary = {
            "metadata": {"model": "pix2pix", "shot": shot, "seed": 1},
            "num_samples": len(values),
            "metrics": {"ssim": {"mean": mean, "std": 0.0}},
            "per_sample_metrics": str(sample_csv),
        }
        sample_csv.with_name("summary.json").write_text(
            json.dumps(summary), encoding="utf-8"
        )

    runs = load_runs(root)
    result = analyze_breaking_points(
        runs,
        alpha=0.05,
        bootstrap_iterations=200,
        variance_ratio_min=1.25,
    )

    assert result["pix2pix"]["breaking_point"] == 10
    assert result["pix2pix"]["breaking_point_by_metric"]["ssim"] == 10
