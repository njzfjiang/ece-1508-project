import subprocess
import sys
import os
from pathlib import Path
import argparse

# Config
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "data" / "raw" / "darkdriving_lle"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
EXTERNAL_DIR = PROJECT_ROOT / "external" / "img2img-turbo"

UPSTREAM_URL = "https://github.com/GaParmar/img2img-turbo.git"
UPSTREAM_COMMIT = "86f54146590ffb4543c8cf85b5a36657da670924"

REQUIRED_SUBDIRS = [
    "train/day",
    "train/night",
    "test/day",
    "test/night",
]

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Skip dependency installation step",
    )
    return parser.parse_args()

# Helpers
def run(cmd, check=True):
    print(f"[CMD] {' '.join(map(str, cmd))}")
    subprocess.run(cmd, check=check)


def confirm(prompt: str) -> bool:
    try:
        return input(prompt).strip().lower() in ["y", "yes"]
    except EOFError:
        return False

# Start of the main setup functions
def check_dataset():
    print("\n[1/4] Checking dataset...")

    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_DIR}")

    for sub in REQUIRED_SUBDIRS:
        path = DATASET_DIR / sub
        if not path.exists():
            raise FileNotFoundError(f"Missing required folder: {path}")


def setup_repo():
    print("\n[2/4] Setting up img2img-turbo repo...")

    EXTERNAL_DIR.parent.mkdir(parents=True, exist_ok=True)

    if not EXTERNAL_DIR.exists():
        run(["git", "clone", UPSTREAM_URL, str(EXTERNAL_DIR)])

    run(["git", "-C", str(EXTERNAL_DIR), "fetch"])
    run(["git", "-C", str(EXTERNAL_DIR), "checkout", UPSTREAM_COMMIT])


def install_deps(skip=False):
    print("\n[3/4] Installing dependencies...")

    if skip:
        print("Skipping dependency installation.")
        return

    req = PROJECT_ROOT / "requirements.txt"
    if req.exists():
        run([sys.executable, "-m", "pip", "install", "-r", str(req)])
    else:
        print("Requirements.txt not found, skipping install")


def prepare_splits():
    print("\n[4/4] Preparing few-shot splits...")

    script = PROJECT_ROOT / "scripts" / "prepare_fewshot_splits.py"

    if not script.exists():
        raise FileNotFoundError(f"Missing script: {script}")

    overwrite = False
    if PROCESSED_DIR.exists():
        overwrite = confirm("Processed data exists. Rebuild? [y/N]: ")

    cmd = [
        sys.executable,
        str(script),
        "--raw_dir",
        str(DATASET_DIR),
        "--output_root",
        str(PROCESSED_DIR),
        "--shots",
        "10",
        "20",
        "50",
        "--num_seeds",
        "3",
        "--mode",
        "auto",
    ]

    if overwrite:
        cmd.append("--overwrite")

    run(cmd)

def main():
    args = parse_args()
    print("======================================")
    print(" img2img-turbo full pipeline setup")
    print("======================================")

    os.chdir(PROJECT_ROOT)

    check_dataset()
    setup_repo()
    install_deps(args.skip_install)
    prepare_splits()

    print("\n======================================")
    print(" DONE ")
    print("======================================")


if __name__ == "__main__":
    main()