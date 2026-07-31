"""Read-only validation for filename-aligned few-shot datasets."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def image_files(directory: Path) -> dict[str, Path]:
    return {
        path.name: path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }


def _load_prompt_keys(path: Path) -> set[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing prompt manifest: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Prompt manifest must contain a JSON object: {path}")
    return set(payload)


def _assert_same_names(
    left: dict[str, Path],
    right: dict[str, Path],
    left_name: str,
    right_name: str,
) -> None:
    if left.keys() != right.keys():
        raise ValueError(
            f"Dataset domains are not filename-aligned: "
            f"missing from {right_name}={sorted(left.keys() - right.keys())[:3]}, "
            f"missing from {left_name}={sorted(right.keys() - left.keys())[:3]}"
        )


def _assert_decodable(images: dict[str, Path]) -> None:
    for path in images.values():
        try:
            with Image.open(path) as image:
                image.load()
        except Exception as exc:
            raise ValueError(f"Corrupted or unreadable image: {path}: {exc}") from exc


def validate_dataset(dataset: Path, expected_shots: int) -> None:
    """Validate a split without changing any image or manifest."""
    for prompt_name in ("fixed_prompt_a.txt", "fixed_prompt_b.txt"):
        prompt_path = dataset / prompt_name
        if not prompt_path.is_file():
            raise FileNotFoundError(f"Missing domain prompt: {prompt_path}")
        if not prompt_path.read_text(encoding="utf-8").strip():
            raise ValueError(f"Domain prompt is empty: {prompt_path}")

    folders: dict[str, dict[str, Path]] = {}
    for folder_name in ("train_A", "train_B", "test_A", "test_B"):
        folder = dataset / folder_name
        if not folder.is_dir():
            raise FileNotFoundError(f"Missing image folder: {folder}")
        folders[folder_name] = image_files(folder)

    _assert_same_names(
        folders["train_A"], folders["train_B"], "train_A", "train_B"
    )
    _assert_same_names(folders["test_A"], folders["test_B"], "test_A", "test_B")

    train_names = set(folders["train_A"])
    test_names = set(folders["test_A"])
    if len(train_names) != expected_shots:
        raise ValueError(
            f"Expected {expected_shots} paired training images; "
            f"found {len(train_names)}"
        )
    if not test_names:
        raise ValueError("test_A and test_B must contain validation images")

    for manifest_name, expected_names in (
        ("train_prompts.json", train_names),
        ("test_prompts.json", test_names),
    ):
        manifest_path = dataset / manifest_name
        prompt_names = _load_prompt_keys(manifest_path)
        if prompt_names != expected_names:
            raise ValueError(
                f"Prompt manifest does not match images: {manifest_path}; "
                f"missing={sorted(expected_names - prompt_names)[:3]}, "
                f"extra={sorted(prompt_names - expected_names)[:3]}"
            )

    for images in folders.values():
        _assert_decodable(images)
