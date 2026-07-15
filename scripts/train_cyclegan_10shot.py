#!/usr/bin/env python3
"""Run one understandable few-shot CycleGAN-Turbo training job.

This script does not implement CycleGAN-Turbo itself. It launches the official
trainer cloned by ``scripts/setup.py`` and starts from the pretrained
``stabilityai/sd-turbo`` weights.

Start by printing the command without loading anything onto the GPU:

    python scripts/train_cyclegan_10shot.py --dry-run

Then start the conservative 256x256, batch-size-1 pilot:

    python scripts/train_cyclegan_10shot.py

Select a different prepared split with ``--shots`` and ``--seed``:

    python scripts/train_cyclegan_10shot.py --shots 20 --seed 1
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_TRAINER = (
    PROJECT_ROOT
    / "external"
    / "img2img-turbo"
    / "src"
    / "train_cyclegan_turbo.py"
)
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "processed"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "standalone" / "cyclegan_turbo"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}


@dataclass(frozen=True)
class TrainingSettings:
    """The small set of values worth changing for the first local run."""

    pretrained_model: str = "stabilityai/sd-turbo"
    expected_images_per_domain: int = 10
    resolution: int = 256
    batch_size: int = 1
    steps: int = 100
    checkpoint_every: int = 500
    preview_every: int = 100
    learning_rate: float = 5e-6
    precision: str = "fp16"
    lora_rank_unet: int = 4
    lora_rank_vae: int = 4
    lambda_cycle: float = 1.0
    lambda_cycle_lpips: float = 10.0
    lambda_identity: float = 1.0
    lambda_identity_lpips: float = 1.0
    lambda_gan: float = 0.5
    seed: int = 1
    resume_from_checkpoint: Path | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        help="Override data/processed/<shots>shot/seed<seed>",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Override outputs/standalone/cyclegan_turbo/<shots>shot/seed<seed>",
    )
    parser.add_argument("--shots", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=500,
        help="Save inference and resumable training checkpoints every N steps",
    )
    parser.add_argument(
        "--preview-every",
        type=int,
        default=100,
        help="Save a fixed held-out A-to-B preview every N steps; 0 disables",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-6,
        help="Learning rate for both generator and discriminator optimizers",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=Path,
        help=(
            "Resume from model_<step>.pkl or training_state_<step>.pt; "
            "--steps is the final total step"
        ),
    )
    parser.add_argument("--resolution", type=int, choices=[256, 512], default=256)
    parser.add_argument(
        "--lora-rank-unet",
        type=int,
        default=4,
        help="LoRA rank for the SD-Turbo U-Net adapters (default: 4)",
    )
    parser.add_argument(
        "--lora-rank-vae",
        type=int,
        default=4,
        help="LoRA rank for the SD-Turbo VAE adapters (default: 4)",
    )
    parser.add_argument(
        "--precision", choices=["no", "fp16", "bf16"], default="fp16"
    )
    parser.add_argument(
        "--no-xformers",
        action="store_true",
        help="Disable xFormers memory-efficient attention",
    )
    parser.add_argument(
        "--no-gradient-checkpointing",
        action="store_true",
        help="Use more VRAM to avoid recomputing activations",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the command without starting training",
    )
    return parser.parse_args()


def image_files(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def validate_dataset(dataset: Path, expected_images: int = 10) -> None:
    """Check the folder contract expected by the upstream unpaired loader."""
    for prompt_name in ("fixed_prompt_a.txt", "fixed_prompt_b.txt"):
        prompt_path = dataset / prompt_name
        if not prompt_path.is_file():
            raise FileNotFoundError(f"Missing domain prompt: {prompt_path}")

    counts = {}
    for folder_name in ("train_A", "train_B", "test_A", "test_B"):
        folder = dataset / folder_name
        if not folder.is_dir():
            raise FileNotFoundError(f"Missing image folder: {folder}")
        counts[folder_name] = len(image_files(folder))

    if counts["train_A"] != expected_images or counts["train_B"] != expected_images:
        raise ValueError(
            f"Expected {expected_images} images in each training domain; "
            f"found train_A={counts['train_A']} and train_B={counts['train_B']}"
        )
    if counts["test_A"] == 0 or counts["test_B"] == 0:
        raise ValueError("test_A and test_B must contain validation images")


def validate_upstream_trainer(trainer: Path) -> None:
    if not trainer.is_file():
        raise FileNotFoundError(
            f"Upstream trainer not found: {trainer}\n"
            "Run: python scripts/setup.py --skip-install --skip-prepare"
        )
    model = trainer.with_name("cyclegan_turbo.py")
    required = {
        trainer: (
            "args.skip_training_validation",
            'sd["sd_unet_conv_in"] = base_conv_in.state_dict()',
            "args.resume_from_checkpoint",
            'os.path.join(args.output_dir, "losses.csv")',
        ),
        model: ('base_conv_in.load_state_dict(sd["sd_unet_conv_in"])',),
    }
    missing = []
    for path, markers in required.items():
        if not path.is_file():
            missing.append(str(path))
            continue
        source = path.read_text(encoding="utf-8")
        missing.extend(marker for marker in markers if marker not in source)
    if missing:
        raise RuntimeError(
            "The upstream CycleGAN trainer has not received this project's "
            "current compatibility patch. Re-run scripts/setup.py. "
            f"Missing markers: {missing}"
        )


def build_command(
    trainer: Path,
    dataset: Path,
    output: Path,
    settings: TrainingSettings,
    use_xformers: bool = True,
    use_gradient_checkpointing: bool = True,
) -> list[str]:
    """Translate the readable settings into upstream command-line arguments."""
    image_prep = f"resize_{settings.resolution}x{settings.resolution}"
    command = [
        sys.executable,
        "-m",
        "accelerate.commands.launch",
        "--num_processes",
        "1",
        "--mixed_precision",
        settings.precision,
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
        "0",
        "--max_train_steps",
        str(settings.steps),
        "--checkpointing_steps",
        str(settings.checkpoint_every),
        "--preview_steps",
        str(settings.preview_every),
        "--gradient_accumulation_steps",
        "1",
        "--learning_rate",
        str(settings.learning_rate),
        "--seed",
        str(settings.seed),
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
        image_prep,
        "--val_img_prep",
        image_prep,
        "--viz_freq",
        "25",
        "--validation_steps",
        str(settings.steps + 1),
        "--validation_num_images",
        "1",
        "--report_to",
        "wandb",
        "--tracker_project_name",
        f"cyclegan_turbo_{settings.expected_images_per_domain}shot_local",
        # Formal evaluation is a separate repository step. Skipping validation
        # here also avoids loading the FID and DINO models into GPU memory.
        "--skip_training_validation",
    ]
    if use_xformers:
        command.append("--enable_xformers_memory_efficient_attention")
    if use_gradient_checkpointing:
        command.append("--gradient_checkpointing")
    if settings.resume_from_checkpoint is not None:
        command.extend(
            ["--resume_from_checkpoint", str(settings.resume_from_checkpoint)]
        )
    return command


def print_run_summary(
    dataset: Path,
    output: Path,
    gpu: int,
    settings: TrainingSettings,
) -> None:
    print("CycleGAN-Turbo local pilot")
    print(f"  pretrained base: {settings.pretrained_model}")
    print(f"  dataset:         {dataset}")
    print(f"  images/domain:   {settings.expected_images_per_domain}")
    print(f"  resolution:      {settings.resolution}x{settings.resolution}")
    print(f"  batch size:      {settings.batch_size}")
    print(f"  optimizer steps: {settings.steps}")
    print(f"  checkpoint every: {settings.checkpoint_every} steps")
    print(f"  preview every:   {settings.preview_every or 'disabled'}")
    print(f"  learning rate:   {settings.learning_rate}")
    print(f"  precision:       {settings.precision}")
    print(f"  physical GPU:    {gpu}")
    print(f"  output:          {output}")


def gpu_device_description(gpu: int) -> str:
    """Return the physical NVIDIA GPU name and VRAM without loading PyTorch."""
    command = [
        "nvidia-smi",
        f"--id={gpu}",
        "--query-gpu=name,memory.total",
        "--format=csv,noheader",
    ]
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return "unavailable (nvidia-smi was not found)"
    except subprocess.CalledProcessError as error:
        detail = error.stderr.strip() or "nvidia-smi could not access the device"
        return f"unavailable ({detail})"
    return result.stdout.strip() or "unavailable (empty nvidia-smi response)"


def main() -> int:
    args = parse_args()
    if args.shots <= 0:
        raise ValueError("--shots must be positive")
    if args.seed <= 0:
        raise ValueError("--seed must be positive")
    if args.lora_rank_unet <= 0 or args.lora_rank_vae <= 0:
        raise ValueError("LoRA ranks must be positive")
    if args.checkpoint_every <= 0:
        raise ValueError("--checkpoint-every must be positive")
    if args.preview_every < 0:
        raise ValueError("--preview-every cannot be negative")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive")

    split_name = f"{args.shots}shot"
    seed_name = f"seed{args.seed}"
    dataset = (
        args.dataset or DEFAULT_DATA_ROOT / split_name / seed_name
    ).resolve()
    output = (
        args.output or DEFAULT_OUTPUT_ROOT / split_name / seed_name
    ).resolve()
    settings = TrainingSettings(
        expected_images_per_domain=args.shots,
        resolution=args.resolution,
        steps=args.steps,
        checkpoint_every=args.checkpoint_every,
        preview_every=args.preview_every,
        learning_rate=args.learning_rate,
        precision=args.precision,
        lora_rank_unet=args.lora_rank_unet,
        lora_rank_vae=args.lora_rank_vae,
        seed=args.seed,
        resume_from_checkpoint=(
            args.resume_from_checkpoint.resolve()
            if args.resume_from_checkpoint is not None
            else None
        ),
    )

    if settings.steps <= 0:
        raise ValueError("--steps must be positive")
    validate_upstream_trainer(UPSTREAM_TRAINER)
    validate_dataset(dataset, settings.expected_images_per_domain)

    checkpoint_dir = output / "checkpoints"
    existing_checkpoints = list(checkpoint_dir.glob("model_*.pkl"))
    if settings.resume_from_checkpoint is not None:
        if not settings.resume_from_checkpoint.is_file():
            raise FileNotFoundError(
                f"Resume checkpoint not found: {settings.resume_from_checkpoint}"
            )
    elif existing_checkpoints:
        raise FileExistsError(
            f"Existing checkpoints found in {checkpoint_dir}; choose a new --output "
            "directory or pass --resume-from-checkpoint."
        )

    command = build_command(
        trainer=UPSTREAM_TRAINER,
        dataset=dataset,
        output=output,
        settings=settings,
        use_xformers=not args.no_xformers,
        use_gradient_checkpointing=not args.no_gradient_checkpointing,
    )
    print_run_summary(dataset, output, args.gpu, settings)
    print(f"  GPU device:      {gpu_device_description(args.gpu)}")
    print("\nCommand:\n", " ".join(command))
    if args.dry_run:
        print("\nDry run complete; the GPU was not used.")
        return 0

    if importlib.util.find_spec("accelerate") is None:
        raise ModuleNotFoundError(
            "accelerate is not installed in this Python environment. "
            "Activate the project virtual environment and install requirements.txt."
        )

    output.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    environment["WANDB_MODE"] = "disabled"
    subprocess.run(command, check=True, env=environment)

    expected_checkpoint = checkpoint_dir / f"model_{settings.steps}.pkl"
    if not expected_checkpoint.is_file():
        raise FileNotFoundError(
            f"Training exited without the expected checkpoint: {expected_checkpoint}"
        )
    print(f"\nTraining complete: {expected_checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
