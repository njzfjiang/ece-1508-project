import subprocess
import sys
from pathlib import Path

def run_script(script_path, seeds, shoots):
    cmd = [
        sys.executable,
        str(script_path),
        "--shots", *map(str, shoots),
        "--seeds", *map(str, seeds),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    print(f"\nRunning: {script_path}")
    print("STDOUT:\n", result.stdout)
    print("STDERR:\n", result.stderr)

    if result.returncode != 0:
        print(f"Failed: {script_path} (exit {result.returncode})")


PROJECT_ROOT = Path(".")

if __name__ == "__main__":
    run_script(
        PROJECT_ROOT / "src" / "train" / "model_paired.py",
        shoots=[10, 20, 30],
        seeds=[1, 2, 3]
    )

    run_script(
        PROJECT_ROOT / "src" / "train" / "model_unpaired.py",
        shoots=[10, 20, 30],
        seeds=[1, 2, 3]
    )