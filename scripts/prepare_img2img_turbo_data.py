"""Create img2img-turbo-compatible few-shot dataset views."""

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Iterable

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def link_or_copy(source: Path, destination: Path, mode: str = "auto") -> str:
    """Materialize a file using the cheapest supported method."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        return "existing"

    methods = [mode] if mode != "auto" else ["hardlink", "symlink", "copy"]
    errors = []
    for method in methods:
        try:
            if method == "hardlink":
                os.link(source, destination)
            elif method == "symlink":
                destination.symlink_to(source.resolve())
            elif method == "copy":
                shutil.copy2(source, destination)
            else:
                raise ValueError(f"Unsupported materialization mode: {method}")
            return method
        except OSError as exc:
            errors.append(f"{method}: {exc}")

    raise OSError(
        f"Could not materialize {source} at {destination}: {'; '.join(errors)}"
    )


def matching_images(day_dir: Path, night_dir: Path) -> list[tuple[Path, Path]]:
    day_files = {
        path.name: path
        for path in day_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }
    night_files = {
        path.name: path
        for path in night_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }
    names = sorted(day_files.keys() & night_files.keys())
    if not names:
        raise ValueError(f"No matching images found in {day_dir} and {night_dir}")
    return [(day_files[name], night_files[name]) for name in names]


def materialize_pairs(
    pairs: Iterable[tuple[Path, Path]],
    day_destination: Path,
    night_destination: Path,
    mode: str,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for day_source, night_source in pairs:
        for source, destination_dir in (
            (day_source, day_destination),
            (night_source, night_destination),
        ):
            method = link_or_copy(source, destination_dir / source.name, mode)
            counts[method] = counts.get(method, 0) + 1
    return counts


def load_training_pairs(processed_root: Path, shot: int, seed: int):
    split_file = (
        processed_root
        / "splits"
        / "fewshot"
        / f"{shot}shot"
        / f"seed{seed}"
        / "split.json"
    )
    if not split_file.is_file():
        raise FileNotFoundError(f"Few-shot split not found: {split_file}")

    split_data = json.loads(split_file.read_text(encoding="utf-8"))
    if split_data.get("shot") != shot or split_data.get("seed") != seed:
        raise ValueError(f"Split metadata does not match {shot}-shot seed {seed}")

    day_names = split_data.get("train_day", [])
    night_names = split_data.get("train_night", [])
    if len(day_names) != shot or len(night_names) != shot:
        raise ValueError(f"Expected {shot} day/night entries in {split_file}")

    train_day = processed_root / "day2night" / "train" / "day"
    train_night = processed_root / "day2night" / "train" / "night"
    pairs = []
    for day_name, night_name in zip(day_names, night_names):
        if day_name != night_name:
            raise ValueError(
                "Official pix2pix-turbo requires paired files to share a filename, "
                f"received {day_name!r} and {night_name!r}"
            )
        pair = (train_day / day_name, train_night / night_name)
        if not pair[0].is_file() or not pair[1].is_file():
            raise FileNotFoundError(f"Processed training pair not found: {pair}")
        pairs.append(pair)
    return pairs


def prepare_dataset_view(
    processed_root: Path,
    output_root: Path,
    shot: int,
    seed: int,
    source_prompt: str = "a driving scene during the day",
    target_prompt: str = "a driving scene at night",
    mode: str = "auto",
    overwrite: bool = False,
) -> Path:
    """Build one dataset accepted by both official training scripts."""
    processed_root = processed_root.resolve()
    output_root = output_root.resolve()
    dataset_dir = output_root / f"{shot}shot" / f"seed{seed}"

    if dataset_dir.exists():
        if not overwrite:
            manifest = dataset_dir / "manifest.json"
            if manifest.is_file():
                metadata = json.loads(manifest.read_text(encoding="utf-8"))
                if metadata.get("shot") == shot and metadata.get("seed") == seed:
                    return dataset_dir
            raise FileExistsError(
                f"Dataset view already exists but is incomplete: {dataset_dir}. "
                "Use --overwrite to rebuild it."
            )
        if output_root not in dataset_dir.parents:
            raise ValueError(
                f"Refusing to remove path outside output root: {dataset_dir}"
            )
        shutil.rmtree(dataset_dir)

    train_pairs = load_training_pairs(processed_root, shot, seed)
    test_pairs = matching_images(
        processed_root / "day2night" / "test" / "day",
        processed_root / "day2night" / "test" / "night",
    )

    train_counts = materialize_pairs(
        train_pairs,
        dataset_dir / "train_A",
        dataset_dir / "train_B",
        mode,
    )
    test_counts = materialize_pairs(
        test_pairs,
        dataset_dir / "test_A",
        dataset_dir / "test_B",
        mode,
    )

    train_prompts = {day.name: target_prompt for day, _ in train_pairs}
    test_prompts = {day.name: target_prompt for day, _ in test_pairs}
    (dataset_dir / "train_prompts.json").write_text(
        json.dumps(train_prompts, indent=2), encoding="utf-8"
    )
    (dataset_dir / "test_prompts.json").write_text(
        json.dumps(test_prompts, indent=2), encoding="utf-8"
    )
    (dataset_dir / "fixed_prompt_a.txt").write_text(
        source_prompt + "\n", encoding="utf-8"
    )
    (dataset_dir / "fixed_prompt_b.txt").write_text(
        target_prompt + "\n", encoding="utf-8"
    )
    (dataset_dir / "manifest.json").write_text(
        json.dumps(
            {
                "shot": shot,
                "seed": seed,
                "train_pairs": len(train_pairs),
                "test_pairs": len(test_pairs),
                "source_prompt": source_prompt,
                "target_prompt": target_prompt,
                "train_materialization": train_counts,
                "test_materialization": test_counts,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return dataset_dir


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_root", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("data/processed/img2img_turbo"),
    )
    parser.add_argument("--shots", type=int, nargs="+", default=[10, 20, 50])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument(
        "--mode",
        choices=["auto", "hardlink", "symlink", "copy"],
        default="auto",
    )
    parser.add_argument("--source_prompt", default="a driving scene during the day")
    parser.add_argument("--target_prompt", default="a driving scene at night")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    for shot in args.shots:
        for seed in args.seeds:
            dataset_dir = prepare_dataset_view(
                processed_root=args.processed_root,
                output_root=args.output_root,
                shot=shot,
                seed=seed,
                source_prompt=args.source_prompt,
                target_prompt=args.target_prompt,
                mode=args.mode,
                overwrite=args.overwrite,
            )
            print(f"Prepared {shot}-shot seed {seed}: {dataset_dir}")


if __name__ == "__main__":
    main()
