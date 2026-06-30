import argparse
import subprocess
from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shots",
        nargs="+",
        type=int,
        default=[10, 20, 50],
        help="List of shot counts for few-shot training",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[1, 2, 3],
        help="List of random seeds for few-shot training",
    )
    return parser.parse_args()


def load_config(config_path: Path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def run(cmd):
    print("\n[CMD]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def train_model(shots, seeds, dataset_folder, output_dir, script_path):
    cfg = load_config(PROJECT_ROOT / "configs" / "base.yaml")["training"]

    for shot in shots:
        for seed in seeds:
            dataset_dir = dataset_folder / f"{shot}shot" / f"seed{seed}"

            if not dataset_dir.exists():
                print(f"Missing dataset {shot}-{seed}, skipping")
                continue

            run_output_dir = output_dir / f"{shot}shot" / f"seed{seed}"
            run_output_dir.mkdir(parents=True, exist_ok=True)

            cmd = [
                "accelerate",
                "launch",
                str(script_path),
                "--pretrained_model_name_or_path",
                str(cfg["model"]),
                "--output_dir",
                str(run_output_dir),
                "--dataset_folder",
                str(dataset_dir),
                "--resolution",
                str(cfg["resolution"]),
                "--train_batch_size",
                str(cfg["batch_size"]),
                "--enable_xformers_memory_efficient_attention",
                "--viz_freq",
                str(cfg["viz_freq"]),
                "--track_val_fid",
                "--seed",
                str(seed),
                "--report_to",
                "wandb",
                "--tracker_project_name",
                "pix2pix_turbo_darkdriving",
            ]

            print(f"Training {shot}-{seed}")
            run(cmd)


def main():
    args = parse_args()
    dataset_folder = PROJECT_ROOT / "data" / "processed"
    output_base = PROJECT_ROOT / "outputs" / "pix2pix_turbo"
    pix2pix_turbo_repo = PROJECT_ROOT / "external" / "img2img-turbo"
    shots = args.shots
    seeds = args.seeds

    # train
    validate_output_dir = output_base / "train"
    train_script_path = pix2pix_turbo_repo / "src" / "train_pix2pix_turbo.py"
    train_model(shots, seeds, dataset_folder, validate_output_dir, train_script_path)
    print(
        "\nModel training completed check the outputs/pix2pix_turbo/train directory for results."
    )


if __name__ == "__main__":
    main()
