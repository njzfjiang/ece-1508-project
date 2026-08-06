#!/usr/bin/env python3
"""Derive a smaller nested split from an existing processed split."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

try:
    from scripts.prepare_fewshot_splits import IMAGE_EXTENSIONS, link_or_copy
except ImportError:
    from prepare_fewshot_splits import IMAGE_EXTENSIONS, link_or_copy

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.train.data_validation import validate_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--source-shot", type=int, default=10)
    parser.add_argument("--target-shot", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument(
        "--mode", choices=["auto", "hardlink", "symlink", "copy"], default="auto"
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _materialize_files(source: Path, destination: Path, mode: str) -> None:
    files = sorted(
        path for path in source.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not files:
        raise ValueError(f"No images found in {source}")
    for path in files:
        link_or_copy(path, destination / path.name, mode)


def derive_seed_split(
    data_root: Path,
    source_shot: int,
    target_shot: int,
    seed: int,
    mode: str,
    overwrite: bool,
) -> Path:
    source = data_root / f"{source_shot}shot" / f"seed{seed}"
    destination = data_root / f"{target_shot}shot" / f"seed{seed}"
    if not source.is_dir():
        raise FileNotFoundError(f"Missing source split: {source}")
    if destination.exists():
        if not overwrite:
            raise FileExistsError(
                f"Destination already exists: {destination}. "
                "Move it to a backup location or pass --overwrite explicitly."
            )
        shutil.rmtree(destination)

    prompt_path = source / "train_prompts.json"
    prompts = json.loads(prompt_path.read_text(encoding="utf-8"))
    ordered_names = list(prompts)
    if len(ordered_names) != source_shot:
        raise ValueError(
            f"Expected {source_shot} ordered prompt entries in {prompt_path}; "
            f"found {len(ordered_names)}"
        )
    selected_names = ordered_names[:target_shot]

    for domain in ("train_A", "train_B"):
        for name in selected_names:
            link_or_copy(source / domain / name, destination / domain / name, mode)
    selected_prompts = {name: prompts[name] for name in selected_names}
    (destination / "train_prompts.json").write_text(
        json.dumps(selected_prompts, indent=2), encoding="utf-8"
    )

    for name in ("fixed_prompt_a.txt", "fixed_prompt_b.txt", "test_prompts.json"):
        shutil.copy2(source / name, destination / name)
    for domain in ("test_A", "test_B"):
        _materialize_files(source / domain, destination / domain, mode)

    manifest = {
        "nested": True,
        "source_split": str(source.resolve()),
        "source_shot": source_shot,
        "target_shot": target_shot,
        "seed": seed,
        "filenames": selected_names,
    }
    (destination / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    validate_dataset(destination, expected_shots=target_shot)
    return destination


def main() -> int:
    args = parse_args()
    if args.target_shot <= 0 or args.source_shot <= args.target_shot:
        raise ValueError("Require 0 < target-shot < source-shot")
    for seed in args.seeds:
        destination = derive_seed_split(
            data_root=args.data_root.resolve(),
            source_shot=args.source_shot,
            target_shot=args.target_shot,
            seed=seed,
            mode=args.mode,
            overwrite=args.overwrite,
        )
        print(f"Built nested split: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
