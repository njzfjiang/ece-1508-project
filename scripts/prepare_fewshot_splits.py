"""Prepare DarkDriving data and create img2img-turbo few-shot training views + shared test set."""

import argparse
import json
import os
import random
import shutil
from pathlib import Path
from typing import List, Tuple

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=Path, required=True)
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("data/processed/img2img_turbo"),
    )
    parser.add_argument(
        "--test_dir",
        type=Path,
        default=Path("data/processed/test"),
    )
    parser.add_argument("--shots", type=int, nargs="+", default=[10, 20, 50])
    parser.add_argument("--num_seeds", type=int, default=3)
    parser.add_argument(
        "--mode",
        choices=["auto", "hardlink", "symlink", "copy"],
        default="auto",
    )
    parser.add_argument("--source_prompt", default="a driving scene during the day")
    parser.add_argument("--target_prompt", default="a driving scene at night")
    parser.add_argument("--overwrite", action="store_true")

    return parser.parse_args()


def link_or_copy(source: Path, destination: Path, mode: str = "auto") -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)

    if not source.exists():
        raise FileNotFoundError(f"Missing source file: {source}")
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
            return method
        except OSError as exc:
            errors.append(f"{method}: {exc}")

    raise OSError(f"Failed to materialize {source}: {'; '.join(errors)}")


def get_image_pairs(day_dir: Path, night_dir: Path) -> List[Tuple[Path, Path]]:
    if not day_dir.is_dir() or not night_dir.is_dir():
        raise FileNotFoundError(f"Missing directory: {day_dir} or {night_dir}")

    day_files = {
        p.name: p
        for p in day_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    }
    night_files = {
        p.name: p
        for p in night_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    }

    names = sorted(day_files.keys() & night_files.keys())
    if not names:
        raise ValueError("No matched day/night pairs found")
    return [(day_files[n], night_files[n]) for n in names]


def materialize_pairs(pairs, day_dst, night_dst, mode):
    for d, n in pairs:
        for src, dst_root in ((d, day_dst), (n, night_dst)):
            link_or_copy(src, dst_root / src.name, mode)


def save_prompts(pairs, out_path, prompt):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = {d.name: prompt for d, _ in pairs}
    out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def build_global_test(test_pairs, test_dir, mode):
    materialize_pairs(
        test_pairs,
        test_dir / "test_A",
        test_dir / "test_B",
        mode,
    )

    (test_dir / "test_prompts.json").write_text(
        json.dumps(
            {p.name: "a driving scene at night" for p, _ in test_pairs}, indent=2
        ),
        encoding="utf-8",
    )


def build_train_split(
    train_pairs,
    output_root,
    shot,
    seed,
    source_prompt,
    target_prompt,
    mode,
    overwrite,
):
    dataset_dir = output_root / f"{shot}shot" / f"seed{seed}"

    if dataset_dir.exists():
        if overwrite:
            shutil.rmtree(dataset_dir)
        else:
            return False

    sampled = random.Random(seed).sample(train_pairs, shot)

    materialize_pairs(
        sampled,
        dataset_dir / "train_A",
        dataset_dir / "train_B",
        mode,
    )

    save_prompts(sampled, dataset_dir / "train_prompts.json", target_prompt)

    (dataset_dir / "fixed_prompt_a.txt").write_text(source_prompt + "\n")
    (dataset_dir / "fixed_prompt_b.txt").write_text(target_prompt + "\n")

    (dataset_dir / "manifest.json").write_text(
        json.dumps(
            {
                "shot": shot,
                "seed": seed,
                "train_pairs": len(sampled),
                "source_prompt": source_prompt,
                "target_prompt": target_prompt,
                "train_day": [d.name for d, _ in sampled],
                "train_night": [n.name for _, n in sampled],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return True


def main():
    args = parse_args()

    train_day = args.raw_dir / "train" / "day"
    train_night = args.raw_dir / "train" / "night"
    test_day = args.raw_dir / "test" / "day"
    test_night = args.raw_dir / "test" / "night"

    train_pairs = get_image_pairs(train_day, train_night)
    test_pairs = get_image_pairs(test_day, test_night)

    if args.test_dir.exists():
        if args.overwrite:
            shutil.rmtree(args.test_dir)
        else:
            print(f"Skipped existing global test set: {args.test_dir}")

    if not args.test_dir.exists():
        args.test_dir.mkdir(parents=True, exist_ok=True)
        build_global_test(test_pairs, args.test_dir, args.mode)
        print(f"Saved global test set to: {args.test_dir}")

    for shot in args.shots:
        if shot > len(train_pairs):
            raise ValueError(f"{shot}-shot exceeds dataset size")

        for seed in range(1, args.num_seeds + 1):
            created = build_train_split(
                train_pairs,
                args.output_root,
                shot,
                seed,
                args.source_prompt,
                args.target_prompt,
                args.mode,
                args.overwrite,
            )
            if created:
                print(f"Built {shot}-shot seed {seed}")
            else:
                print(f"Skipped existing {shot}-shot seed {seed}")


if __name__ == "__main__":
    main()
