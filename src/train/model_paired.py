import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run(cmd):
    print("\n[CMD]", " ".join(cmd))
    subprocess.run(cmd, check=True)

def main():

    model_path = PROJECT_ROOT / "src" / "train" / "night2day.pkl"
    dataset_folder = PROJECT_ROOT / "data" / "processed"
    output_base = PROJECT_ROOT / "output" / "pix2pix_turbo"

    shots = [10, 20, 50]
    seeds = [1, 2, 3]

    for shot in shots:
        for seed in seeds:
            dataset_dir = dataset_folder / f"{shot}shot" / f"seed{seed}"
            if not dataset_dir.exists():
                print(f"Dataset for shot={shot}, seed={seed} does not exist. Skipping.")
                continue
            output_dir = output_base / f"{shot}shot_seed{seed}"

            cmd = [
                "accelerate",
                "launch",
                "src/train_pix2pix_turbo.py",
                "--pretrained_model_name_or_path",
                str(model_path),
                "--output_dir",
                str(output_dir),
                "--dataset_folder",
                str(dataset_folder),
                "--resolution",
                "512",
                "--train_batch_size",
                "2",
                "--enable_xformers_memory_efficient_attention",
                "--viz_freq",
                "25",
                "--track_val_fid",
                "--report_to",
                "wandb",
                "--tracker_project_name",
                "pix2pix_turbo_darkdriving",
            ]

            print(f"\n===== Running shot={shot}, seed={seed} =====")
            run(cmd)

    print("\nALL EXPERIMENTS DONE")


if __name__ == "__main__":
    main()