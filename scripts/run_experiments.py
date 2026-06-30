import subprocess
import sys

def run_script(script_name):
    result = subprocess.run(
        [sys.executable, script_name],
        capture_output=True,
        text=True
    )

    print(f"Running: {script_name}")
    print("STDOUT:\n", result.stdout)
    print("STDERR:\n", result.stderr)

    if result.returncode != 0:
        print(f"{script_name} failed with exit code {result.returncode}")

if __name__ == "__main__":
    run_script("model_paired.py")
    run_script("model_unpaired.py")