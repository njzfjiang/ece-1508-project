"""Shared path, image, and aggregation helpers for evaluation."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from PIL import Image
from torchvision import transforms

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def resolve_path(project_root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else project_root / candidate


def checkpoint_step(path: Path) -> int:
    match = re.fullmatch(r"model_(\d+)\.pkl", path.name)
    return int(match.group(1)) if match else -1


def find_checkpoint(
    root: Path,
    shot: int,
    seed: int,
    checkpoint: Path | None = None,
) -> Path:
    if checkpoint is not None:
        selected = checkpoint.resolve()
        if not selected.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {selected}")
        return selected

    folder = root / f"{shot}shot" / f"seed{seed}" / "checkpoints"
    if not folder.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {folder}")
    files = [path for path in folder.glob("model_*.pkl") if checkpoint_step(path) >= 0]
    if not files:
        raise FileNotFoundError(f"No model_<step>.pkl checkpoints in {folder}")
    return max(files, key=checkpoint_step).resolve()


def _images_by_name(directory: Path) -> dict[str, Path]:
    return {
        path.name: path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }


def find_pairs(root: Path, limit: int | None = None) -> list[tuple[Path, Path]]:
    day_dir = root / "test_A"
    night_dir = root / "test_B"
    seed = 42
    if not day_dir.is_dir() or not night_dir.is_dir():
        raise FileNotFoundError(
            f"Held-out test directories not found: {day_dir} / {night_dir}"
        )
    day = _images_by_name(day_dir)
    night = _images_by_name(night_dir)
    missing_night = sorted(day.keys() - night.keys())
    missing_day = sorted(night.keys() - day.keys())
    if missing_day or missing_night:
        raise ValueError(
            "Held-out test set is not filename-aligned: "
            f"missing day={missing_day[:3]}, missing night={missing_night[:3]}"
        )
    names = sorted(day)
    rng = np.random.default_rng(seed)
    rng.shuffle(names)
    
    if not names:
        raise ValueError(f"No paired test images found under {root}")
    if limit is not None:
        if limit <= 0:
            raise ValueError("test_samples must be positive")
        names = names[:limit]
    
    return [(day[name], night[name]) for name in names]


def load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def resize8(image: Image.Image) -> Image.Image:
    width = image.width - image.width % 8
    height = image.height - image.height % 8
    if width <= 0 or height <= 0:
        raise ValueError(f"Image is too small for img2img-turbo: {image.size}")
    return image.resize((width, height), Image.Resampling.LANCZOS)


def image_to_tensor(path: Path):
    return transforms.ToTensor()(load_rgb(path))


def summarize(rows: list[dict], metrics: list[str]) -> dict:
    if not rows:
        raise ValueError("Cannot summarize an empty evaluation")
    result = {}
    for name in ("ssim", "lpips", "clip_similarity"):
        if name not in metrics:
            continue
        values = np.asarray([float(row[name]) for row in rows], dtype=np.float64)
        result[name] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return result
