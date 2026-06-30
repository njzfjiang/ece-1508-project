"""Prepare DarkDriving dataset for img2img-turbo (night → day)."""

import argparse
import json
import os
import random
import shutil
from pathlib import Path
from typing import List, Tuple

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--raw_dir", type=Path, required=True)
    p.add_argument("--output_root", type=Path, default=Path("data/processed/img2img_turbo"))
    p.add_argument("--test_dir", type=Path, default=Path("data/processed/test"))

    p.add_argument("--shots", type=int, nargs="+", default=[10, 20, 50])
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])

    p.add_argument("--mode", choices=["auto", "hardlink", "symlink", "copy"], default="auto")
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
    night = {f.name: f for f in night_dir.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS}

    names = sorted(day.keys() & night.keys())
    return [(day[n], night[n]) for n in names]


def materialize(pairs, day_dst, night_dst, mode):
    for d, n in pairs:
        link_or_copy(d, day_dst / d.name, mode)
        link_or_copy(n, night_dst / n.name, mode)


def save_prompts(pairs, path: Path, prompt: str, base_dir: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    
    data = { path.name: prompt for _, path in pairs }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def add_prompts(pairs, path: Path, src_prompt: str, tgt_prompt: str, base_dir: Path):
    save_prompts(pairs, path, tgt_prompt, base_dir)

    path.parent.joinpath("fixed_prompt_a.txt").write_text(src_prompt + "\n")
    path.parent.joinpath("fixed_prompt_b.txt").write_text(tgt_prompt + "\n")


def build_test(pairs, out_dir, src_prompt, tgt_prompt, mode):
    materialize(pairs, out_dir / "test_A", out_dir / "test_B", mode)
    add_prompts(pairs, out_dir / "test_prompts.json", src_prompt, tgt_prompt, out_dir)


def build_train(pairs, out, shot, seed, src_prompt, tgt_prompt, mode, overwrite):
    if out.exists():
        if overwrite:
            shutil.rmtree(out)
        else:
            return False

    sampled = random.Random(seed).sample(pairs, shot)

    materialize(sampled, out / "train_A", out / "train_B", mode)
    add_prompts(sampled, out / "train_prompts.json", src_prompt, tgt_prompt, out)

    return True


def main():
    args = parse_args()

    train_day = args.raw_dir / "train" / "day"
    train_night = args.raw_dir / "train" / "night"
    test_day = args.raw_dir / "test" / "day"
    test_night = args.raw_dir / "test" / "night"

    train_pairs = get_pairs(train_day, train_night)
    test_pairs = get_pairs(test_day, test_night)

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
    

    for shot in args.shots:
        if shot > len(remaining_train):
            raise ValueError(f"{shot}-shot exceeds dataset size")

        for seed in args.seeds:
            out_dir = args.output_root / f"{shot}shot" / f"seed{seed}"
            
            created = build_train(
                remaining_train,
                out_dir,
                shot,
                seed,
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