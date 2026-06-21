import json
import tarfile
from pathlib import Path

from PIL import Image

from scripts.package_darkdriving_smoke import package_smoke_dataset, sha256_file


def write_image(path: Path, color: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=color).save(path)


def make_processed_dataset(root: Path):
    train_names = ["train_1.jpg", "train_2.jpg", "train_3.jpg"]
    test_names = [f"test_{index}.jpg" for index in range(5)]
    for name in train_names:
        write_image(root / "day2night" / "train" / "day" / name, "navy")
        write_image(root / "day2night" / "train" / "night" / name, "black")
    for name in test_names:
        write_image(root / "day2night" / "test" / "day" / name, "blue")
        write_image(root / "day2night" / "test" / "night" / name, "gray")

    split = root / "splits" / "fewshot" / "2shot" / "seed1" / "split.json"
    split.parent.mkdir(parents=True)
    split.write_text(
        json.dumps(
            {
                "shot": 2,
                "seed": 1,
                "train_day": ["train_1.jpg", "train_3.jpg"],
                "train_night": ["train_1.jpg", "train_3.jpg"],
            }
        ),
        encoding="utf-8",
    )


def test_package_has_canonical_layout_and_checksum(tmp_path):
    processed = tmp_path / "processed"
    make_processed_dataset(processed)
    output = tmp_path / "darkdriving_smoke.tar.gz"

    archive, manifest = package_smoke_dataset(
        processed_root=processed,
        output=output,
        shot=2,
        seed=1,
        test_pairs=3,
        test_seed=42,
    )

    assert manifest["train_pairs"] == 2
    assert manifest["test_pairs"] == 3
    assert sha256_file(archive) == manifest["sha256"]
    assert archive.with_name(archive.name + ".sha256").is_file()

    with tarfile.open(archive, "r:gz") as tar:
        names = set(tar.getnames())
        assert "data/processed/smoke_manifest.json" in names
        assert "data/processed/splits/fewshot/2shot/seed1/split.json" in names
        train_day = [
            name
            for name in names
            if name.startswith("data/processed/day2night/train/day/")
            and name.endswith(".jpg")
        ]
        test_night = [
            name
            for name in names
            if name.startswith("data/processed/day2night/test/night/")
            and name.endswith(".jpg")
        ]
        assert len(train_day) == 2
        assert len(test_night) == 3


def test_test_sampling_is_deterministic(tmp_path):
    processed = tmp_path / "processed"
    make_processed_dataset(processed)

    _, first = package_smoke_dataset(
        processed, tmp_path / "first.tar.gz", shot=2, seed=1, test_pairs=3
    )
    _, second = package_smoke_dataset(
        processed, tmp_path / "second.tar.gz", shot=2, seed=1, test_pairs=3
    )

    assert first["test_filenames"] == second["test_filenames"]
