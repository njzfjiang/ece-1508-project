"""Launch CycleGAN-Turbo few-shot training runs from project configuration."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "base.yaml"
TRAIN_SCRIPT = (
    PROJECT_ROOT / "external" / "img2img-turbo" / "src" / "train_cyclegan_turbo.py"
)
MODEL_SCRIPT = PROJECT_ROOT / "external" / "img2img-turbo" / "src" / "cyclegan_turbo.py"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

REQUIRED_TRAINER_MARKERS = (
    "max_train_steps is the authoritative stopping condition for CycleGAN-Turbo",
    "Skip training-time FID initialization when formal evaluation is external",
    "Backward each direction before constructing the next graph",
    'sd["sd_unet_conv_in"] = base_conv_in.state_dict()',
    'os.path.join(args.output_dir, "losses.csv")',
    "if global_step == 0 and args.preview_steps:",
)
REQUIRED_MODEL_MARKERS = (
    "Keep pretrained VAE weights frozen",
    "LoRA-wrapped skip convolutions expose adapter parameters twice",
    'base_conv_in.load_state_dict(sd["sd_unet_conv_in"])',
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shots", nargs="+", type=int, default=[10, 20, 50])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--steps", type=int, help="Override max_train_steps")
    parser.add_argument("--output", type=Path, help="Override CycleGAN output root")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_config(config_path: Path) -> dict:
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    for section in ("data", "training", "cyclegan_turbo", "logging"):
        if section not in config:
            raise KeyError(f"Missing required config section: {section}")
    return config


def runtime_value(config: dict, name: str, default=None):
    cyclegan = config["cyclegan_turbo"]
    training = config["training"]
    value = cyclegan.get(name, training.get(name, default))
    if value is None:
        raise KeyError(f"Missing CycleGAN runtime setting: {name}")
    return value


def build_train_command(
    script_path: Path,
    dataset_dir: Path,
    output_dir: Path,
    seed: int,
    config: dict,
    max_train_steps: int | None = None,
) -> list[str]:
    cyclegan = config["cyclegan_turbo"]
    precision = str(runtime_value(config, "mixed_precision", "no"))
    steps = int(
        max_train_steps
        if max_train_steps is not None
        else runtime_value(config, "max_train_steps")
    )
    command = [
        sys.executable,
        "-m",
        "accelerate.commands.launch",
        "--num_processes",
        "1",
        "--num_machines",
        "1",
        "--mixed_precision",
        precision,
        "--dynamo_backend",
        "no",
        str(script_path),
        "--pretrained_model_name_or_path",
        str(runtime_value(config, "model")),
        "--dataset_folder",
        str(dataset_dir),
        "--output_dir",
        str(output_dir),
        "--train_batch_size",
        str(runtime_value(config, "batch_size")),
        "--dataloader_num_workers",
        str(runtime_value(config, "num_workers")),
        "--max_train_steps",
        str(steps),
        "--checkpointing_steps",
        str(runtime_value(config, "checkpointing_steps")),
        "--preview_steps",
        str(cyclegan.get("preview_steps", 500)),
        "--gradient_accumulation_steps",
        str(runtime_value(config, "gradient_accumulation_steps", 1)),
        "--learning_rate",
        str(runtime_value(config, "learning_rate")),
        "--seed",
        str(seed),
        "--lora_rank_unet",
        str(cyclegan["lora_rank_unet"]),
        "--lora_rank_vae",
        str(cyclegan["lora_rank_vae"]),
        "--lambda_cycle",
        str(cyclegan["lambda_cycle"]),
        "--lambda_cycle_lpips",
        str(cyclegan["lambda_cycle_lpips"]),
        "--lambda_idt",
        str(cyclegan["lambda_identity"]),
        "--lambda_idt_lpips",
        str(cyclegan["lambda_identity_lpips"]),
        "--lambda_gan",
        str(cyclegan["lambda_gan"]),
        "--train_img_prep",
        str(cyclegan["train_image_prep"]),
        "--val_img_prep",
        str(cyclegan["val_image_prep"]),
        "--viz_freq",
        str(runtime_value(config, "viz_freq")),
        "--validation_steps",
        str(cyclegan.get("validation_steps", 500)),
        "--validation_num_images",
        str(cyclegan.get("validation_num_images", 1)),
        "--report_to",
        "wandb",
        "--tracker_project_name",
        str(cyclegan.get("tracker_project_name", "cyclegan_turbo_darkdriving")),
    ]
    if bool(cyclegan.get("skip_training_validation", True)):
        command.append("--skip_training_validation")
    if bool(runtime_value(config, "enable_xformers", False)):
        command.append("--enable_xformers_memory_efficient_attention")
    if bool(runtime_value(config, "gradient_checkpointing", False)):
        command.append("--gradient_checkpointing")
    if bool(cyclegan.get("allow_tf32", False)):
        command.append("--allow_tf32")
    return command


def training_environment(config: dict, gpu: int) -> dict[str, str]:
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    if not config["logging"]["use_wandb"]:
        environment["WANDB_MODE"] = "disabled"
    return environment


def image_files(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def validate_dataset(dataset: Path, expected_shots: int) -> None:
    """Validate the unpaired few-shot folder contract before using the GPU."""
    for prompt_name in ("fixed_prompt_a.txt", "fixed_prompt_b.txt"):
        prompt_path = dataset / prompt_name
        if not prompt_path.is_file():
            raise FileNotFoundError(f"Missing domain prompt: {prompt_path}")
        if not prompt_path.read_text(encoding="utf-8").strip():
            raise ValueError(f"Domain prompt is empty: {prompt_path}")

    counts: dict[str, int] = {}
    for folder_name in ("train_A", "train_B", "test_A", "test_B"):
        folder = dataset / folder_name
        if not folder.is_dir():
            raise FileNotFoundError(f"Missing image folder: {folder}")
        counts[folder_name] = len(image_files(folder))

    if (
        counts["train_A"] != expected_shots
        or counts["train_B"] != expected_shots
    ):
        raise ValueError(
            f"Expected {expected_shots} images in each training domain; "
            f"found train_A={counts['train_A']} and train_B={counts['train_B']}"
        )
    if counts["test_A"] == 0 or counts["test_B"] == 0:
        raise ValueError("test_A and test_B must contain validation images")


def validate_vendor_script(trainer: Path) -> None:
    model = trainer.with_name("cyclegan_turbo.py")
    for path, markers in (
        (trainer, REQUIRED_TRAINER_MARKERS),
        (model, REQUIRED_MODEL_MARKERS),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Vendored CycleGAN file not found: {path}")
        source = path.read_text(encoding="utf-8")
        missing = [marker for marker in markers if marker not in source]
        if missing:
            raise RuntimeError(
                "Vendored CycleGAN-Turbo is missing required compatibility patches. "
                "Run: python scripts/setup.py --skip-install --skip-prepare. "
                f"Missing markers in {path.name}: {missing}"
            )


def validate_vendor() -> None:
    validate_vendor_script(TRAIN_SCRIPT)


def _project_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def main() -> int:
    args = parse_args()
    config = load_config(args.config.resolve())
    if any(shot <= 0 for shot in args.shots):
        raise ValueError("shots must be positive")
    if any(seed <= 0 for seed in args.seeds):
        raise ValueError("seeds must be positive")
    if args.steps is not None and args.steps <= 0:
        raise ValueError("max_train_steps must be positive")
    preview_steps = int(config["cyclegan_turbo"].get("preview_steps", 100))
    if preview_steps < 0:
        raise ValueError("cyclegan_turbo.preview_steps cannot be negative")

    dataset_root = _project_path(config["data"]["root"])
    output_root = (
        args.output.resolve()
        if args.output is not None
        else _project_path(config["cyclegan_turbo"]["output_dir"])
    )
    if not args.dry_run:
        validate_vendor()

    expected_steps = int(
        args.steps
        if args.steps is not None
        else runtime_value(config, "max_train_steps")
    )
    for shot in args.shots:
        for seed in args.seeds:
            dataset_dir = dataset_root / f"{shot}shot" / f"seed{seed}"
            if not args.dry_run:
                validate_dataset(dataset_dir, expected_shots=shot)
            output_dir = output_root / f"{shot}shot" / f"seed{seed}"
            existing = list((output_dir / "checkpoints").glob("model_*.pkl"))
            if existing:
                raise FileExistsError(
                    f"Refusing to mix a new run with checkpoints in {output_dir}"
                )
            command = build_train_command(
                TRAIN_SCRIPT,
                dataset_dir,
                output_dir,
                seed,
                config,
                max_train_steps=args.steps,
            )
            print(f"\nTraining CycleGAN: {shot}-shot, seed {seed}")
            print("[CMD]", " ".join(command))
            if args.dry_run:
                continue
            output_dir.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                command,
                check=True,
                env=training_environment(config, args.gpu),
            )
            expected = output_dir / "checkpoints" / f"model_{expected_steps}.pkl"
            if not expected.is_file():
                raise RuntimeError(f"Training ended without final checkpoint: {expected}")

    print(f"\nCycleGAN training completed. Checkpoints: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
