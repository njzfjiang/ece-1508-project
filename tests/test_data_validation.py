import json
from pathlib import Path

import pytest
from PIL import Image

from scripts.prepare_fewshot_splits import filter_decodable_pairs
from src.train.data_validation import validate_dataset


def _image(path: Path, color: str = "red") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color).save(path)


def _dataset(root: Path) -> Path:
    for folder in ("train_A", "train_B", "test_A", "test_B"):
        _image(root / folder / "sample.jpg")
    (root / "fixed_prompt_a.txt").write_text("day\n", encoding="utf-8")
    (root / "fixed_prompt_b.txt").write_text("night\n", encoding="utf-8")
    for split in ("train", "test"):
        (root / f"{split}_prompts.json").write_text(
            json.dumps({"sample.jpg": "night"}), encoding="utf-8"
        )
    return root


def test_validate_dataset_accepts_aligned_decodable_split(tmp_path):
    validate_dataset(_dataset(tmp_path / "split"), expected_shots=1)


def test_validate_dataset_rejects_prompt_extra(tmp_path):
    root = _dataset(tmp_path / "split")
    (root / "train_prompts.json").write_text(
        json.dumps({"sample.jpg": "night", "extra.jpg": "night"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Prompt manifest does not match images"):
        validate_dataset(root, expected_shots=1)


def test_validate_dataset_rejects_truncated_image(tmp_path):
    root = _dataset(tmp_path / "split")
    image_path = root / "train_B" / "sample.jpg"
    image_path.write_bytes(image_path.read_bytes()[:20])

    with pytest.raises(ValueError, match="Corrupted or unreadable image"):
        validate_dataset(root, expected_shots=1)


def test_prepare_filter_removes_entire_corrupted_pair(tmp_path):
    good_day = tmp_path / "day" / "good.jpg"
    good_night = tmp_path / "night" / "good.jpg"
    bad_day = tmp_path / "day" / "bad.jpg"
    bad_night = tmp_path / "night" / "bad.jpg"
    _image(good_day)
    _image(good_night)
    _image(bad_day)
    bad_night.parent.mkdir(parents=True, exist_ok=True)
    bad_night.write_bytes(b"not an image")

    result = filter_decodable_pairs(
        [(bad_day, bad_night), (good_day, good_night)], "training"
    )

    assert result == [(good_day, good_night)]
