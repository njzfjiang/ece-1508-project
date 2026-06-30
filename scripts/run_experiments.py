import subprocess
import sys
from pathlib import Path
import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--shots",
        nargs="+",
        type=int,
        default=[10, 20, 50],
        help="List of shot counts",
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[1, 2, 3],
        help="List of seeds",
    )

    return parser.parse_args()


def run_script(script_path, shots, seeds):
    cmd = [
        sys.executable,
        str(script_path),
        "--shots", *map(str, shots),
        "--seeds", *map(str, seeds),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    print(f"\nRunning: {script_path}")
    print("STDOUT:\n", result.stdout)
    print("STDERR:\n", result.stderr)

    if result.returncode != 0:
        raise RuntimeError(f"{script_path} failed (exit {result.returncode})")


PROJECT_ROOT = Path(__file__).resolve().parents[1]  # adjust if needed

if __name__ == "__main__":
    args = parse_args()

    run_script(
        PROJECT_ROOT / "src" / "train" / "model_paired.py",
        shots=args.shots,
        seeds=args.seeds,
    )

    run_script(
        PROJECT_ROOT / "src" / "train" / "model_unpaired.py",
        shots=args.shots,
        seeds=args.seeds,
    )