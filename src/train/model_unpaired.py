"""Launch config-driven CycleGAN-Turbo few-shot training runs."""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "base.yaml"
UPSTREAM_TRAINER = (
    PROJECT_ROOT / "external" / "img2img-turbo" / "src" / "train_cyclegan_turbo.py"
)
UPSTREAM_MODEL = (
    PROJECT_ROOT / "external" / "img2img-turbo" / "src" / "cyclegan_turbo.py"
)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
REQUIRED_TRAINER_MARKERS = (
    "max_train_steps is the authoritative stopping condition for CycleGAN-Turbo",
    "args.skip_training_validation",
    "Backward each direction before constructing the next graph",
)
REQUIRED_MODEL_MARKERS = (
    "LoRA-wrapped skip convolutions expose adapter parameters twice above",
    "Keep pretrained VAE weights frozen",
)


@dataclass(frozen=True)
class CycleGANSettings:
    pretrained_model: str
    resolution: int
    batch_size: int
    num_workers: int
    max_train_steps: int
    checkpointing_steps: int
    mixed_precision: str
    learning_rate: float
    enable_xformers: bool
    gradient_checkpointing: bool
    allow_tf32: bool
    viz_freq: int
    validation_steps: int
    validation_num_images: int
    skip_training_validation: bool
    tracker_project_name: str
    lora_rank_unet: int
    lora_rank_vae: int
    lambda_cycle: float
    lambda_cycle_lpips: float
    lambda_identity: float
    lambda_identity_lpips: float
    lambda_gan: float
    train_image_prep: str
    val_image_prep: str


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shots", nargs="+", type=int, default=[10, 20, 50])
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    seed_group.add_argument("--seed", type=int, help="Alias for one --seeds value")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--dataset", type=Path, help="Single-run dataset override")
    parser.add_argument("--output", type=Path, help="Single-run output override")
    parser.add_argument("--steps", type=int, help="Override max_train_steps")
    parser.add_argument("--resolution", type=int, choices=[256, 512])
    parser.add_argument("--precision", choices=["no", "fp16", "bf16"])
    parser.add_argument("--no-xformers", action="store_true")
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def load_config(config_path: Path) -> dict:
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    for section in ("data", "training", "cyclegan_turbo", "logging"):
        if section not in config:
            raise KeyError(f"Missing required config section: {section}")
    return config


def settings_from_config(config: dict, args: argparse.Namespace) -> CycleGANSettings:
    training = config["training"]
    cyclegan = config["cyclegan_turbo"]

    def value(name: str, default=None):
        return cyclegan.get(name, training.get(name, default))

    resolution = args.resolution or int(value("resolution"))
    image_prep = f"resize_{resolution}x{resolution}"
    return CycleGANSettings(
        pretrained_model=str(training["model"]),
        resolution=resolution,
        batch_size=int(value("batch_size")),
        num_workers=int(value("num_workers")),
        max_train_steps=(
            args.steps if args.steps is not None else int(value("max_train_steps"))
        ),
        checkpointing_steps=int(value("checkpointing_steps")),
        mixed_precision=args.precision or str(value("mixed_precision")),
        learning_rate=float(value("learning_rate")),
        enable_xformers=bool(value("enable_xformers")) and not args.no_xformers,
        gradient_checkpointing=bool(value("gradient_checkpointing"))
        and not args.no_gradient_checkpointing,
        allow_tf32=bool(cyclegan.get("allow_tf32", False)),
        viz_freq=int(value("viz_freq")),
        validation_steps=int(cyclegan.get("validation_steps", 500)),
        validation_num_images=int(cyclegan.get("validation_num_images", 1)),
        skip_training_validation=bool(cyclegan.get("skip_training_validation", True)),
        tracker_project_name=str(
            cyclegan.get("tracker_project_name", "cyclegan_turbo_darkdriving")
        ),
        lora_rank_unet=int(cyclegan["lora_rank_unet"]),
        lora_rank_vae=int(cyclegan["lora_rank_vae"]),
        lambda_cycle=float(cyclegan["lambda_cycle"]),
        lambda_cycle_lpips=float(cyclegan["lambda_cycle_lpips"]),
        lambda_identity=float(cyclegan["lambda_identity"]),
        lambda_identity_lpips=float(cyclegan["lambda_identity_lpips"]),
        lambda_gan=float(cyclegan["lambda_gan"]),
        train_image_prep=(
            image_prep
            if args.resolution
            else str(cyclegan.get("train_image_prep", image_prep))
        ),
        val_image_prep=(
            image_prep
            if args.resolution
            else str(cyclegan.get("val_image_prep", image_prep))
        ),
    )


def build_train_command(
    trainer: Path,
    dataset: Path,
    output: Path,
    seed: int,
    settings: CycleGANSettings,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "accelerate.commands.launch",
        "--num_processes",
        "1",
        "--num_machines",
        "1",
        "--mixed_precision",
        settings.mixed_precision,
        "--dynamo_backend",
        "no",
        str(trainer),
        "--pretrained_model_name_or_path",
        settings.pretrained_model,
        "--dataset_folder",
        str(dataset),
        "--output_dir",
        str(output),
        "--train_batch_size",
        str(settings.batch_size),
        "--dataloader_num_workers",
        str(settings.num_workers),
        "--max_train_steps",
        str(settings.max_train_steps),
        "--checkpointing_steps",
        str(settings.checkpointing_steps),
        "--gradient_accumulation_steps",
        "1",
        "--learning_rate",
        str(settings.learning_rate),
        "--seed",
        str(seed),
        "--lora_rank_unet",
        str(settings.lora_rank_unet),
        "--lora_rank_vae",
        str(settings.lora_rank_vae),
        "--lambda_cycle",
        str(settings.lambda_cycle),
        "--lambda_cycle_lpips",
        str(settings.lambda_cycle_lpips),
        "--lambda_idt",
        str(settings.lambda_identity),
        "--lambda_idt_lpips",
        str(settings.lambda_identity_lpips),
        "--lambda_gan",
        str(settings.lambda_gan),
        "--train_img_prep",
        settings.train_image_prep,
        "--val_img_prep",
        settings.val_image_prep,
        "--viz_freq",
        str(settings.viz_freq),
        "--validation_steps",
        str(settings.validation_steps),
        "--validation_num_images",
        str(settings.validation_num_images),
        "--report_to",
        "wandb",
        "--tracker_project_name",
        settings.tracker_project_name,
    ]
    if settings.skip_training_validation:
        command.append("--skip_training_validation")
    if settings.enable_xformers:
        command.append("--enable_xformers_memory_efficient_attention")
    if settings.gradient_checkpointing:
        command.append("--gradient_checkpointing")
    if settings.allow_tf32:
        command.append("--allow_tf32")
    return command


def validate_upstream() -> None:
    _validate_markers(UPSTREAM_TRAINER, REQUIRED_TRAINER_MARKERS)
    _validate_markers(UPSTREAM_MODEL, REQUIRED_MODEL_MARKERS)


def _validate_markers(path: Path, markers: tuple[str, ...]) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"Vendored CycleGAN file not found: {path}. "
            "Run: python scripts/setup.py --skip-install --skip-prepare"
        )
    source = path.read_text(encoding="utf-8")
    missing = [marker for marker in markers if marker not in source]
    if missing:
        raise RuntimeError(
            "Vendored CycleGAN-Turbo is missing compatibility patches. "
            "Run: python scripts/setup.py --skip-install --skip-prepare. "
            f"Missing markers in {path.name}: {missing}"
        )


def validate_dataset(dataset: Path, expected_images: int) -> None:
    for prompt_name in ("fixed_prompt_a.txt", "fixed_prompt_b.txt"):
        prompt_path = dataset / prompt_name
        if not prompt_path.is_file():
            raise FileNotFoundError(f"Missing domain prompt: {prompt_path}")

    counts = {}
    for folder_name in ("train_A", "train_B", "test_A", "test_B"):
        folder = dataset / folder_name
        if not folder.is_dir():
            raise FileNotFoundError(f"Missing image folder: {folder}")
        counts[folder_name] = sum(
            path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            for path in folder.iterdir()
        )
    if counts["train_A"] != expected_images or counts["train_B"] != expected_images:
        raise ValueError(
            f"Expected {expected_images} images in each training domain; "
            f"found train_A={counts['train_A']} and train_B={counts['train_B']}"
        )
    if counts["test_A"] == 0 or counts["test_B"] == 0:
        raise ValueError("test_A and test_B must contain validation images")


def training_environment(config: dict, gpu: int) -> dict[str, str]:
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    if not config["logging"]["use_wandb"]:
        environment["WANDB_MODE"] = "disabled"
    return environment


def train_model(
    shots: list[int],
    seeds: list[int],
    dataset_root: Path,
    output_root: Path,
    settings: CycleGANSettings,
    config: dict,
    gpu: int,
    dry_run: bool = False,
    dataset_override: Path | None = None,
    output_override: Path | None = None,
) -> None:
    if (dataset_override or output_override) and (len(shots) != 1 or len(seeds) != 1):
        raise ValueError("--dataset/--output overrides require one shot and one seed")

    for shot in shots:
        if shot <= 0:
            raise ValueError("Shot counts must be positive")
        for seed in seeds:
            if seed <= 0:
                raise ValueError("Seeds must be positive")
            dataset = (
                dataset_override or dataset_root / f"{shot}shot" / f"seed{seed}"
            ).resolve()
            output = (
                output_override or output_root / f"{shot}shot" / f"seed{seed}"
            ).resolve()
            validate_dataset(dataset, shot)

            checkpoint_dir = output / "checkpoints"
            if list(checkpoint_dir.glob("model_*.pkl")):
                raise FileExistsError(
                    f"Existing checkpoints found in {checkpoint_dir}; move the old "
                    "run or choose another output directory."
                )

            command = build_train_command(
                UPSTREAM_TRAINER, dataset, output, seed, settings
            )
            print(f"\nTraining CycleGAN: {shot}-shot, seed {seed}")
            print("[CMD]", " ".join(command))
            if dry_run:
                continue

            if importlib.util.find_spec("accelerate") is None:
                raise ModuleNotFoundError(
                    "accelerate is not installed in the active Python environment"
                )
            output.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                command,
                check=True,
                env=training_environment(config, gpu),
            )
            expected = checkpoint_dir / f"model_{settings.max_train_steps}.pkl"
            if not expected.is_file():
                raise FileNotFoundError(
                    f"Training exited without the expected checkpoint: {expected}"
                )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    seeds = [args.seed] if args.seed is not None else args.seeds
    config = load_config(args.config.resolve())
    settings = settings_from_config(config, args)
    if settings.max_train_steps <= 0:
        raise ValueError("max_train_steps must be positive")
    validate_upstream()

    train_model(
        shots=args.shots,
        seeds=seeds,
        dataset_root=_project_path(str(config["data"]["root"])),
        output_root=_project_path(str(config["cyclegan_turbo"]["output_dir"])),
        settings=settings,
        config=config,
        gpu=args.gpu,
        dry_run=args.dry_run,
        dataset_override=args.dataset.resolve() if args.dataset else None,
        output_override=args.output.resolve() if args.output else None,
    )
    print(
        "\nCycleGAN dry run complete."
        if args.dry_run
        else "\nCycleGAN training complete."
    )
    return 0


def _project_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


if __name__ == "__main__":
    raise SystemExit(main())
