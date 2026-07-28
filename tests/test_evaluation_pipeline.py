import json
from pathlib import Path

import numpy as np
from PIL import Image

from src.eval.evaluate import evaluate_generated_pairs
from src.eval.utils import find_checkpoint, find_pairs, image_to_tensor


class FakeMetrics:
    def compute_ssim(self, generated, target):
        return 0.75

    def extract_clip_features(self, image):
        mean = float(image.mean())
        feature = np.asarray([[mean + 1.0, 1.0]], dtype=np.float32)
        return feature / np.linalg.norm(feature, axis=1, keepdims=True)

    def compute_cmmd_from_features(self, generated, target, sigma):
        assert generated.ndim == 2
        assert target.ndim == 2
        assert sigma == 2.0
        return 0.125


def _write_image(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), (value, value, value)).save(path)


def test_find_pairs_is_strict_and_honors_limit(tmp_path):
    _write_image(tmp_path / "test_A" / "a.png", 10)
    _write_image(tmp_path / "test_B" / "a.png", 20)
    _write_image(tmp_path / "test_A" / "b.png", 30)
    _write_image(tmp_path / "test_B" / "b.png", 40)

    pairs = find_pairs(tmp_path, limit=1)
    assert [pair[0].name for pair in pairs] == ["a.png"]

    (tmp_path / "test_B" / "b.png").unlink()
    try:
        find_pairs(tmp_path)
    except ValueError as error:
        assert "not filename-aligned" in str(error)
    else:
        raise AssertionError("misaligned held-out data should fail")


def test_image_to_tensor_accepts_path(tmp_path):
    path = tmp_path / "image.png"
    _write_image(path, 128)
    assert tuple(image_to_tensor(path).shape) == (3, 16, 16)


def test_latest_checkpoint_is_selected_numerically(tmp_path):
    folder = tmp_path / "10shot" / "seed1" / "checkpoints"
    folder.mkdir(parents=True)
    (folder / "model_500.pkl").touch()
    (folder / "model_10000.pkl").touch()
    assert find_checkpoint(tmp_path, 10, 1).name == "model_10000.pkl"


def test_evaluation_writes_reproducible_outputs(tmp_path):
    day = tmp_path / "test" / "test_A" / "a.png"
    night = tmp_path / "test" / "test_B" / "a.png"
    generated = tmp_path / "generated" / "a.png"
    _write_image(day, 20)
    _write_image(night, 40)
    _write_image(generated, 35)
    output = tmp_path / "evaluation"

    result = evaluate_generated_pairs(
        pairs=[(day, night)],
        generated_dir=generated.parent,
        output_dir=output,
        metrics_calculator=FakeMetrics(),
        requested_metrics=["ssim", "clip_similarity", "cmmd"],
        metadata={"model": "pix2pix", "checkpoint": "model_2.pkl"},
        cmmd_sigma=2.0,
    )

    assert result["metrics"]["ssim"]["mean"] == 0.75
    assert result["metrics"]["cmmd"] == 0.125
    assert (output / "per_sample_metrics.csv").is_file()
    saved = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert saved["metadata"]["checkpoint"] == "model_2.pkl"
