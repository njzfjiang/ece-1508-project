from pathlib import Path
import sys

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.compare_generation_seeds import compare_directories


def _save(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((4, 4, 3), value, dtype=np.uint8)).save(path)


def test_compare_generation_seeds_detects_exact_and_changed_outputs(tmp_path):
    directories = [tmp_path / f"seed{seed}" for seed in range(3)]
    for directory in directories:
        _save(directory / "same.png", 10)
    _save(directories[0] / "changed.png", 0)
    _save(directories[1] / "changed.png", 10)
    _save(directories[2] / "changed.png", 20)

    summary = compare_directories(directories)

    assert summary["num_common_images"] == 2
    assert summary["exactly_identical_images"] == 1
    assert summary["exactly_identical_fraction"] == 0.5
    assert summary["pairwise_pixel_mae_mean"] > 0
    assert summary["pairwise_pixel_mae_mean_8bit"] > 0
    assert summary["pairwise_psnr_mean_db"] is not None
    assert 0 < summary["unchanged_channel_fraction"] < 1
