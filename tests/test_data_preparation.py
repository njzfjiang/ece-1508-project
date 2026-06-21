import json
from pathlib import Path

import pytest

from scripts.prepare_fewshot_splits import (
    ensure_output_is_empty,
    generate_fewshot_splits,
    reset_output,
)


def make_pairs(root: Path, count: int):
    day_dir = root / "day"
    night_dir = root / "night"
    day_dir.mkdir(parents=True)
    night_dir.mkdir(parents=True)

    pairs = []
    for index in range(count):
        name = f"frame_{index:03d}.jpg"
        day_path = day_dir / name
        night_path = night_dir / name
        day_path.touch()
        night_path.touch()
        pairs.append((day_path, night_path))
    return pairs


def test_seed_directory_matches_random_seed(tmp_path):
    pairs = make_pairs(tmp_path / "raw", 10)
    output_dir = tmp_path / "processed"

    generate_fewshot_splits(pairs, output_dir, shot_levels=[3], num_seeds=2)

    for seed in (1, 2):
        split_file = (
            output_dir / "splits" / "fewshot" / "3shot" / f"seed{seed}" / "split.json"
        )
        split_data = json.loads(split_file.read_text())
        assert split_data["shot"] == 3
        assert split_data["seed"] == seed
        assert len(split_data["train_day"]) == 3
        assert len(split_data["train_night"]) == 3


def test_existing_output_requires_explicit_reset(tmp_path):
    output_dir = tmp_path / "processed"
    managed_file = output_dir / "day2night" / "train" / "day" / "old.jpg"
    managed_file.parent.mkdir(parents=True)
    managed_file.touch()

    with pytest.raises(FileExistsError, match="--overwrite"):
        ensure_output_is_empty(output_dir)

    reset_output(output_dir)
    ensure_output_is_empty(output_dir)
    assert not (output_dir / "day2night").exists()
