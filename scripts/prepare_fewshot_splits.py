# scripts/prepare_fewshot_splits.py
"""
DarkDriving Dataset Preprocessing Script

Raw DarkDriving dataset structure (after download):
  DarkDriving/
    train/
      day/
        frame_0001.jpg
        frame_0002.jpg
        ...
      night/
        frame_0001.jpg
        frame_0002.jpg
        ...
    test/
      day/
        ...
      night/
        ...

Output structure (for this project):
  data/processed/
    day2night/
      train/
        day/     # symlinks or copies of train/day
        night/   # symlinks or copies of train/night
      test/
        day/
        night/
      val/
        day/
        night/
    splits/
      fewshot/
        10shot/seed1/
          split.json
          train_day.txt
          train_night.txt
        ...
"""

import json
import random
import shutil
import argparse
from pathlib import Path
from typing import List, Tuple


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess DarkDriving dataset")
    parser.add_argument(
        "--raw_dir",
        type=str,
        required=True,
        help="Path to raw DarkDriving dataset (containing train/ and test/)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/processed",
        help="Output directory for processed data",
    )
    parser.add_argument(
        "--shot_levels",
        type=int,
        nargs="+",
        default=[10, 20, 50],
        help="Few-shot levels to generate",
    )
    parser.add_argument(
        "--num_seeds", type=int, default=3, help="Number of random seeds per shot level"
    )
    parser.add_argument(
        "--copy_mode",
        action="store_true",
        help="Copy files instead of symlinking (use if symlinks not supported)",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.1,
        help="Validation split ratio from training set",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove existing processed dataset and split outputs before rebuilding",
    )
    return parser.parse_args()


def reset_output(output_dir: Path):
    """Remove only the directories managed by this script."""
    for managed_dir in (output_dir / "day2night", output_dir / "splits"):
        if managed_dir.exists():
            shutil.rmtree(managed_dir)


def ensure_output_is_empty(output_dir: Path):
    """Refuse to mix a new split with files from an earlier preprocessing run."""
    existing = [
        path
        for managed_dir in (output_dir / "day2night", output_dir / "splits")
        if managed_dir.exists()
        for path in managed_dir.iterdir()
    ]
    if existing:
        raise FileExistsError(
            f"Processed output already exists under {output_dir}. "
            "Re-run with --overwrite to rebuild it safely."
        )


def create_directory_structure(base_dir: Path):
    """Create the standard directory structure for the project."""
    dirs = [
        base_dir / "day2night" / "train" / "day",
        base_dir / "day2night" / "train" / "night",
        base_dir / "day2night" / "val" / "day",
        base_dir / "day2night" / "val" / "night",
        base_dir / "day2night" / "test" / "day",
        base_dir / "day2night" / "test" / "night",
        base_dir / "splits" / "fewshot",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def get_image_pairs(day_dir: Path, night_dir: Path) -> List[Tuple[Path, Path]]:
    """
    Get paired (day, night) images by matching filename.
    """
    if not day_dir.exists() or not night_dir.exists():
        raise FileNotFoundError(
            f"Day or night directory not found: {day_dir} or {night_dir}"
        )

    day_files = {
        f.name: f
        for f in day_dir.iterdir()
        if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
    }
    night_files = {
        f.name: f
        for f in night_dir.iterdir()
        if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
    }

    if not day_files or not night_files:
        raise ValueError(f"No image files found in {day_dir} or {night_dir}")

    shared_names = sorted(day_files.keys() & night_files.keys())
    if len(shared_names) != len(day_files) or len(shared_names) != len(night_files):
        print("\n⚠️  WARNING: Day/night filenames do not match exactly!")
        print(f"   Day images: {len(day_files)}")
        print(f"   Night images: {len(night_files)}")
        print(f"   Using {len(shared_names)} filename-matched pairs\n")

    if not shared_names:
        raise ValueError(
            f"No matching day/night filenames found in {day_dir} and {night_dir}"
        )

    return [(day_files[name], night_files[name]) for name in shared_names]


def copy_or_link(src: Path, dst: Path, copy_mode: bool = False):
    """Copy or symlink a file."""
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy_mode:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


def prepare_full_dataset(
    raw_dir: Path, output_dir: Path, copy_mode: bool = False, val_split: float = 0.1
):
    """
    Prepare the full day2night dataset from raw DarkDriving.
    Splits train into train/val.
    """
    raw_train_day = raw_dir / "train" / "day"
    raw_train_night = raw_dir / "train" / "night"
    raw_test_day = raw_dir / "test" / "day"
    raw_test_night = raw_dir / "test" / "night"

    # Get all training pairs
    train_pairs = get_image_pairs(raw_train_day, raw_train_night)
    print(f"Found {len(train_pairs)} training pairs")

    # Shuffle and split train/val
    random.seed(42)
    random.shuffle(train_pairs)
    val_size = int(len(train_pairs) * val_split)
    val_pairs = train_pairs[:val_size]
    train_pairs = train_pairs[val_size:]
    print(f"Train: {len(train_pairs)}, Val: {len(val_pairs)}")

    # Copy/link training pairs
    for day_path, night_path in train_pairs:
        copy_or_link(
            day_path,
            output_dir / "day2night" / "train" / "day" / day_path.name,
            copy_mode,
        )
        copy_or_link(
            night_path,
            output_dir / "day2night" / "train" / "night" / night_path.name,
            copy_mode,
        )

    # Copy/link validation pairs
    for day_path, night_path in val_pairs:
        copy_or_link(
            day_path,
            output_dir / "day2night" / "val" / "day" / day_path.name,
            copy_mode,
        )
        copy_or_link(
            night_path,
            output_dir / "day2night" / "val" / "night" / night_path.name,
            copy_mode,
        )

    # Copy/link test pairs
    test_pairs = get_image_pairs(raw_test_day, raw_test_night)
    print(f"Found {len(test_pairs)} test pairs")
    for day_path, night_path in test_pairs:
        copy_or_link(
            day_path,
            output_dir / "day2night" / "test" / "day" / day_path.name,
            copy_mode,
        )
        copy_or_link(
            night_path,
            output_dir / "day2night" / "test" / "night" / night_path.name,
            copy_mode,
        )

    return train_pairs, val_pairs, test_pairs


def generate_fewshot_splits(
    train_pairs: List[Tuple[Path, Path]],
    output_dir: Path,
    shot_levels: List[int],
    num_seeds: int = 3,
):
    """
    Generate few-shot splits for each shot level and seed.
    Each split is saved as a JSON file containing day and night file lists.
    """
    splits_dir = output_dir / "splits" / "fewshot"

    for shot in shot_levels:
        if shot > len(train_pairs):
            raise ValueError(
                f"Requested {shot}-shot split, but only {len(train_pairs)} training pairs are available"
            )

        for seed in range(1, num_seeds + 1):
            rng = random.Random(seed)
            # Sample shot pairs from training set
            sampled_indices = rng.sample(range(len(train_pairs)), shot)
            sampled_pairs = [train_pairs[i] for i in sampled_indices]

            # Create split directory
            split_dir = splits_dir / f"{shot}shot" / f"seed{seed}"
            split_dir.mkdir(parents=True, exist_ok=True)

            # Save as JSON
            split_data = {
                "shot": shot,
                "seed": seed,
                "train_day": [str(p[0].name) for p in sampled_pairs],
                "train_night": [str(p[1].name) for p in sampled_pairs],
                "full_paths": {
                    "day": [str(p[0]) for p in sampled_pairs],
                    "night": [str(p[1]) for p in sampled_pairs],
                },
            }

            with open(split_dir / "split.json", "w") as f:
                json.dump(split_data, f, indent=2)

            # Also save as simple text lists (for compatibility with some loaders)
            with open(split_dir / "train_day.txt", "w") as f:
                f.write("\n".join([str(p[0].name) for p in sampled_pairs]))
            with open(split_dir / "train_night.txt", "w") as f:
                f.write("\n".join([str(p[1].name) for p in sampled_pairs]))

            print(f"Generated {shot}shot/seed{seed} with {len(sampled_pairs)} pairs")


def generate_split_summary(output_dir: Path, shot_levels: List[int], num_seeds: int):
    """Generate a summary JSON for all splits."""
    summary = {
        "dataset": "DarkDriving",
        "shot_levels": shot_levels,
        "num_seeds": num_seeds,
        "splits": {},
    }

    for shot in shot_levels:
        summary["splits"][f"{shot}shot"] = {}
        for seed in range(1, num_seeds + 1):
            split_file = (
                output_dir
                / "splits"
                / "fewshot"
                / f"{shot}shot"
                / f"seed{seed}"
                / "split.json"
            )
            if split_file.exists():
                with open(split_file) as f:
                    summary["splits"][f"{shot}shot"][f"seed{seed}"] = json.load(f)

    with open(output_dir / "splits" / "split_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    args = parse_args()

    if not 0 <= args.val_split < 1:
        raise ValueError("--val_split must be in the range [0, 1)")
    if args.num_seeds < 1:
        raise ValueError("--num_seeds must be at least 1")
    if any(shot < 1 for shot in args.shot_levels):
        raise ValueError("--shot_levels values must all be positive")

    # Validate input directory
    raw_dir = Path(args.raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw dataset directory not found: {raw_dir}")

    required_subdirs = ["train/day", "train/night", "test/day", "test/night"]
    for subdir in required_subdirs:
        subdir_path = raw_dir / subdir
        if not subdir_path.exists():
            raise FileNotFoundError(
                f"Required directory not found: {subdir_path}\n"
                f"Expected structure: raw_dir/train/day, raw_dir/train/night, etc."
            )

    output_dir = Path(args.output_dir)
    if args.overwrite:
        reset_output(output_dir)
    else:
        ensure_output_is_empty(output_dir)

    print(f"Raw dataset: {raw_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Shot levels: {args.shot_levels}")
    print(f"Number of seeds per level: {args.num_seeds}")
    print()

    # Create directory structure
    create_directory_structure(output_dir)

    # Prepare full dataset
    train_pairs, val_pairs, test_pairs = prepare_full_dataset(
        raw_dir, output_dir, copy_mode=args.copy_mode, val_split=args.val_split
    )

    # Generate few-shot splits
    generate_fewshot_splits(
        train_pairs, output_dir, shot_levels=args.shot_levels, num_seeds=args.num_seeds
    )

    # Generate summary
    generate_split_summary(output_dir, args.shot_levels, args.num_seeds)

    print("\n" + "=" * 50)
    print("Preprocessing complete!")
    print(f"Processed data: {output_dir / 'day2night'}")
    print(f"Splits: {output_dir / 'splits' / 'fewshot'}")
    print("=" * 50)


if __name__ == "__main__":
    main()
