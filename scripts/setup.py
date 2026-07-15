"""Set up the pinned img2img-turbo vendor tree and optional dataset views."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "data" / "raw" / "darkdriving_lle"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
EXTERNAL_DIR = PROJECT_ROOT / "external" / "img2img-turbo"
UPSTREAM_URL = "https://github.com/GaParmar/img2img-turbo.git"
UPSTREAM_COMMIT = "86f54146590ffb4543c8cf85b5a36657da670924"
PATCHES = [
    PROJECT_ROOT / "patches" / "img2img-turbo-training-loop.patch",
    PROJECT_ROOT / "patches" / "img2img-turbo-pix2pix-fp16.patch",
    PROJECT_ROOT / "patches" / "img2img-turbo-optimizer-params.patch",
    PROJECT_ROOT / "patches" / "img2img-turbo-cyclegan-training.patch",
    PROJECT_ROOT / "patches" / "img2img-turbo-cyclegan-memory.patch",
]
REQUIRED_SUBDIRS = ["train/day", "train/night", "test/day", "test/night"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Skip dependency installation",
    )
    parser.add_argument(
        "--skip-prepare",
        action="store_true",
        help="Skip raw-dataset validation and few-shot split preparation",
    )
    parser.add_argument("--shots", nargs="+", type=int, default=[10, 20, 50])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    return parser.parse_args()


def run(command: list[str], check: bool = True) -> subprocess.CompletedProcess:
    print(f"[CMD] {' '.join(map(str, command))}")
    return subprocess.run(command, check=check)


def command_succeeds(command: list[str]) -> bool:
    return (
        subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode
        == 0
    )


def confirm(prompt: str) -> bool:
    try:
        return input(prompt).strip().lower() in ["y", "yes"]
    except EOFError:
        return False


def check_dataset() -> None:
    print("\nChecking dataset...")
    if not DATASET_DIR.is_dir():
        raise FileNotFoundError(f"Dataset not found: {DATASET_DIR}")
    for subdir in REQUIRED_SUBDIRS:
        path = DATASET_DIR / subdir
        if not path.is_dir():
            raise FileNotFoundError(f"Missing required folder: {path}")


def setup_repo() -> None:
    print("\nSetting up img2img-turbo...")
    EXTERNAL_DIR.parent.mkdir(parents=True, exist_ok=True)
    if not EXTERNAL_DIR.exists():
        run(["git", "clone", UPSTREAM_URL, str(EXTERNAL_DIR)])
    run(["git", "-C", str(EXTERNAL_DIR), "fetch"])
    run(["git", "-C", str(EXTERNAL_DIR), "checkout", UPSTREAM_COMMIT])
    for patch_path in PATCHES:
        apply_patch(patch_path)


def apply_patch(patch_path: Path) -> None:
    if not patch_path.is_file():
        raise FileNotFoundError(f"Compatibility patch not found: {patch_path}")
    base = ["git", "-C", str(EXTERNAL_DIR), "apply"]
    if command_succeeds([*base, "--reverse", "--check", str(patch_path)]):
        print(f"Patch already applied: {patch_path.name}")
        return
    if not command_succeeds([*base, "--check", str(patch_path)]):
        run([*base, "--check", str(patch_path)])
        raise RuntimeError(f"Patch cannot be applied cleanly: {patch_path}")
    run([*base, str(patch_path)])
    print(f"Applied patch: {patch_path.name}")


def install_deps(skip: bool = False) -> None:
    print("\nInstalling dependencies...")
    if skip:
        print("Skipping dependency installation.")
        return
    requirements = PROJECT_ROOT / "requirements.txt"
    if not requirements.is_file():
        raise FileNotFoundError(f"Requirements not found: {requirements}")
    run([sys.executable, "-m", "pip", "install", "-r", str(requirements)])


def prepare_splits(shots: list[int], seeds: list[int]) -> None:
    print("\nPreparing few-shot splits...")
    script = PROJECT_ROOT / "scripts" / "prepare_fewshot_splits.py"
    if not script.is_file():
        raise FileNotFoundError(f"Missing script: {script}")

    overwrite = PROCESSED_DIR.exists() and confirm(
        "Processed data exists. Rebuild? [y/N]: "
    )
    command = [
        sys.executable,
        str(script),
        "--raw_dir",
        str(DATASET_DIR),
        "--output_root",
        str(PROCESSED_DIR),
        "--shots",
        *map(str, shots),
        "--seeds",
        *map(str, seeds),
        "--mode",
        "auto",
    ]
    if overwrite:
        command.append("--overwrite")
    run(command)


def main() -> int:
    args = parse_args()
    os.chdir(PROJECT_ROOT)
    print("======================================")
    print(" img2img-turbo project setup")
    print("======================================")

    setup_repo()
    install_deps(args.skip_install)
    if args.skip_prepare:
        print("Skipping raw-dataset validation and split preparation.")
    else:
        check_dataset()
        prepare_splits(args.shots, args.seeds)

    print("\nSetup complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
