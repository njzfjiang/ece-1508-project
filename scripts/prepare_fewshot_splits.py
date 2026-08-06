"""Prepare DarkDriving dataset for img2img-turbo (day to night)."""

import argparse
import json
import os
import random
import shutil
from pathlib import Path
from typing import List, Tuple

from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def parse_args():
    """
    Parse command-line arguments.

    Available arguments:
        --raw_dir: Path to the raw DarkDriving dataset.
        --output_root: Path to the output directory for few-shot splits.
        --test_dir: Path to the output directory for the test set.
        --shots: List of shot sizes for few-shot splits.
        --seeds: List of random seeds for few-shot splits.
        --mode: Method for linking or copying files (auto, hardlink, symlink, copy).
        --overwrite: Whether to overwrite existing output directories.
        --source_prompt: Prompt for the source images (day).
        --target_prompt: Prompt for the target images (night).
    """
    p = argparse.ArgumentParser()

    p.add_argument("--raw_dir", type=Path, required=True)
    p.add_argument(
        "--output_root", type=Path, default=Path("data/processed/img2img_turbo")
    )
    p.add_argument("--test_dir", type=Path, default=Path("data/processed/test"))

    p.add_argument("--shots", type=int, nargs="+", default=[10, 20, 50])
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])

    p.add_argument(
        "--mode", choices=["auto", "hardlink", "symlink", "copy"], default="auto"
    )
    p.add_argument("--overwrite", action="store_true")

    p.add_argument("--source_prompt", default="a driving scene during the day")
    p.add_argument("--target_prompt", default="a driving scene during the night")

    return p.parse_args()


def link_or_copy(src: Path, dst: Path, mode="auto") -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)

    if not src.exists():
        raise FileNotFoundError(src)

    if dst.exists() or dst.is_symlink():
        return "existing"

    methods = [mode] if mode != "auto" else ["hardlink", "symlink", "copy"]
    errors = []

    for m in methods:
        try:
            if m == "hardlink":
                os.link(src, dst)
            elif m == "symlink":
                dst.symlink_to(src.resolve())
            else:
                shutil.copy2(src, dst)
            return m
        except OSError as e:
            errors.append(str(e))

    raise OSError("; ".join(errors))


def get_pairs(day_dir: Path, night_dir: Path) -> List[Tuple[Path, Path]]:
    if not day_dir.is_dir() or not night_dir.is_dir():
        raise FileNotFoundError(f"Missing: {day_dir} or {night_dir}")

    day = {f.name: f for f in day_dir.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS}
    night = {
        f.name: f for f in night_dir.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS
    }

    missing_night = sorted(day.keys() - night.keys())
    missing_day = sorted(night.keys() - day.keys())
    if missing_day or missing_night:
        raise ValueError(
            "Dataset is not filename-aligned: "
            f"missing day={missing_day[:3]}, missing night={missing_night[:3]}"
        )
    names = sorted(day)
    if not names:
        raise ValueError(f"No paired images found in {day_dir} and {night_dir}")
    return [(day[n], night[n]) for n in names]


def filter_decodable_pairs(
    pairs: List[Tuple[Path, Path]], label: str
) -> List[Tuple[Path, Path]]:
    """Remove corrupt source pairs before any deterministic sampling."""
    valid = []
    for day, night in pairs:
        try:
            for path in (day, night):
                with Image.open(path) as image:
                    image.load()
        except Exception as exc:
            print(f"Skipping corrupted {label} pair {day.name}: {exc}")
            continue
        valid.append((day, night))
    if not valid:
        raise ValueError(f"No decodable paired images found in {label} data")
    print(f"Validated {len(valid)}/{len(pairs)} {label} pairs")
    return valid


def materialize(pairs, day_dst, night_dst, mode):
    for d, n in pairs:
        link_or_copy(d, day_dst / d.name, mode)
        link_or_copy(n, night_dst / n.name, mode)


def save_prompts(pairs, path: Path, prompt: str):
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {day.name: prompt for day, _ in pairs}
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def add_prompts(pairs, path: Path, src_prompt: str, tgt_prompt: str):
    save_prompts(pairs, path, tgt_prompt)

    path.parent.joinpath("fixed_prompt_a.txt").write_text(src_prompt + "\n")
    path.parent.joinpath("fixed_prompt_b.txt").write_text(tgt_prompt + "\n")


def build_test(pairs, out_dir, src_prompt, tgt_prompt, mode):
    materialize(pairs, out_dir / "test_A", out_dir / "test_B", mode)
    add_prompts(pairs, out_dir / "test_prompts.json", src_prompt, tgt_prompt)


def build_train(sampled, out, src_prompt, tgt_prompt, mode, overwrite):
    if out.exists():
        if overwrite:
            shutil.rmtree(out)
        else:
            return False

    materialize(sampled, out / "train_A", out / "train_B", mode)
    add_prompts(sampled, out / "train_prompts.json", src_prompt, tgt_prompt)

    return True


def main():
    args = parse_args()

    train_day = args.raw_dir / "train" / "day"
    train_night = args.raw_dir / "train" / "night"
    test_day = args.raw_dir / "test" / "day"
    test_night = args.raw_dir / "test" / "night"

    train_pairs = filter_decodable_pairs(
        get_pairs(train_day, train_night), "training"
    )
    test_pairs = filter_decodable_pairs(get_pairs(test_day, test_night), "test")

    if args.test_dir.exists() and args.overwrite:
        shutil.rmtree(args.test_dir)

    build_test(
        test_pairs,
        args.test_dir,
        args.source_prompt,
        args.target_prompt,
        args.mode,
    )

    rng = random.Random(42)
    val_pairs = rng.sample(train_pairs, min(100, len(train_pairs)))
    remaining_train = [p for p in train_pairs if p not in val_pairs]

    shots = sorted(set(args.shots))
    if not shots or shots[0] <= 0:
        raise ValueError("Shot sizes must be positive")
    if shots[-1] > len(remaining_train):
        raise ValueError(f"{shots[-1]}-shot exceeds dataset size")

    for seed in args.seeds:
        # Draw one ordered maximum-size sample per seed. Smaller splits are
        # prefixes, making the few-shot grid explicitly nested by construction.
        ordered_sample = random.Random(seed).sample(remaining_train, shots[-1])
        for shot in shots:
            out_dir = args.output_root / f"{shot}shot" / f"seed{seed}"

            created = build_train(
                ordered_sample[:shot],
                out_dir,
                args.source_prompt,
                args.target_prompt,
                args.mode,
                args.overwrite,
            )

            build_test(
                val_pairs,
                out_dir,
                args.source_prompt,
                args.target_prompt,
                args.mode,
            )

            print(f"{'Built' if created else 'Skipped'} {shot}-shot seed {seed}")


if __name__ == "__main__":
    main()
