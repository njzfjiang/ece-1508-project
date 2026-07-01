"""Launch pix2pix-turbo few-shot training runs from the project config."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "base.yaml"
REQUIRED_VENDOR_MARKERS = (
    "max_train_steps is the authoritative stopping condition",
    "Keep trainable parameters in FP32",
)


def parse_args() -> argparse.Namespace:
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
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def load_config(config_path: Path) -> dict:
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    for section in ("data", "training", "pix2pix_turbo", "logging"):
        if section not in config:
            raise KeyError(f"Missing required config section: {section}")
    return config


def build_train_command(
    script_path: Path,
    dataset_dir: Path,
    output_dir: Path,
    seed: int,
    config: dict,
) -> list[str]:
    training = config["training"]
    pix2pix = config["pix2pix_turbo"]
    command = [
        "accelerate",
        "launch",
        str(script_path),
        "--pretrained_model_name_or_path",
        str(training["model"]),
        "--output_dir",
        str(output_dir),
        "--dataset_folder",
        str(dataset_dir),
        "--resolution",
        str(training["resolution"]),
        "--train_batch_size",
        str(training["batch_size"]),
        "--dataloader_num_workers",
        str(training["num_workers"]),
        "--max_train_steps",
        str(training["max_train_steps"]),
        "--checkpointing_steps",
        str(training["checkpointing_steps"]),
        "--mixed_precision",
        str(training["mixed_precision"]),
        "--learning_rate",
        str(training["learning_rate"]),
        "--viz_freq",
        str(training["viz_freq"]),
        "--eval_freq",
        str(training["eval_freq"]),
        "--num_samples_eval",
        str(training["num_samples_eval"]),
        "--seed",
        str(seed),
        "--lora_rank_unet",
        str(pix2pix["lora_rank_unet"]),
        "--lora_rank_vae",
        str(pix2pix["lora_rank_vae"]),
        "--lambda_l2",
        str(pix2pix["lambda_l2"]),
        "--lambda_lpips",
        str(pix2pix["lambda_lpips"]),
        "--lambda_clipsim",
        str(pix2pix["lambda_clipsim"]),
        "--lambda_gan",
        str(pix2pix["lambda_gan"]),
        "--train_image_prep",
        str(pix2pix["train_image_prep"]),
        "--test_image_prep",
        str(pix2pix["test_image_prep"]),
        "--report_to",
        "wandb",
        "--tracker_project_name",
        str(pix2pix["tracker_project_name"]),
    ]
    if training["enable_xformers"]:
        command.append("--enable_xformers_memory_efficient_attention")
    if training["gradient_checkpointing"]:
        command.append("--gradient_checkpointing")
    if pix2pix["track_val_fid"]:
        command.append("--track_val_fid")
    return command


def training_environment(config: dict) -> dict[str, str]:
    environment = os.environ.copy()
    if not config["logging"]["use_wandb"]:
        environment["WANDB_MODE"] = "disabled"
    return environment


def validate_vendor_script(script_path: Path) -> None:
    source = script_path.read_text(encoding="utf-8")
    missing = [marker for marker in REQUIRED_VENDOR_MARKERS if marker not in source]
    if missing:
        raise RuntimeError(
            "Vendored img2img-turbo is missing required compatibility patches. "
            "Run: python scripts/setup.py --skip-install --skip-prepare. "
            f"Missing markers: {missing}"
        )


def train_model(
    shots: list[int],
    seeds: list[int],
    dataset_root: Path,
    output_root: Path,
    script_path: Path,
    config: dict,
) -> None:
    for shot in shots:
        for seed in seeds:
            dataset_dir = dataset_root / f"{shot}shot" / f"seed{seed}"
            if not dataset_dir.is_dir():
                raise FileNotFoundError(f"Missing dataset: {dataset_dir}")

            output_dir = output_root / f"{shot}shot" / f"seed{seed}"
            existing_checkpoints = list(
                (output_dir / "checkpoints").glob("model_*.pkl")
            )
            if existing_checkpoints:
                raise FileExistsError(
                    "Refusing to mix a new run with existing checkpoints in "
                    f"{output_dir}. Move the old run or choose another output_dir."
                )
            output_dir.mkdir(parents=True, exist_ok=True)
            command = build_train_command(
                script_path=script_path,
                dataset_dir=dataset_dir,
                output_dir=output_dir,
                seed=seed,
                config=config,
            )
            print(f"\nTraining pix2pix: {shot}-shot, seed {seed}")
            print("[CMD]", " ".join(command))
            subprocess.run(
                command,
                check=True,
                env=training_environment(config),
            )


def main() -> int:
    args = parse_args()
    config = load_config(args.config.resolve())
    dataset_root = _project_path(str(config["data"]["root"]))
    output_root = _project_path(str(config["pix2pix_turbo"]["output_dir"]))
    script_path = (
        PROJECT_ROOT / "external" / "img2img-turbo" / "src" / "train_pix2pix_turbo.py"
    )
    if not script_path.is_file():
        raise FileNotFoundError(f"Vendored training script not found: {script_path}")
    validate_vendor_script(script_path)

    train_model(
        shots=args.shots,
        seeds=args.seeds,
        dataset_root=dataset_root,
        output_root=output_root,
        script_path=script_path,
        config=config,
    )
    print(f"\nPix2pix training completed. Checkpoints: {output_root}")
    return 0


def _project_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


if __name__ == "__main__":
    raise SystemExit(main())
