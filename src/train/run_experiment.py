#!/usr/bin/env python
"""Launch few-shot experiments through the pinned official img2img-turbo code."""

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.prepare_img2img_turbo_data import prepare_dataset_view

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "base.yaml"
DEFAULT_EXTERNAL_ROOT = PROJECT_ROOT / "external" / "img2img-turbo"
PINNED_IMG2IMG_TURBO_COMMIT = "86f54146590ffb4543c8cf85b5a36657da670924"


def config_path(config: DictConfig, key: str, default=None):
    value = OmegaConf.select(config, key, default=default)
    return value


def validate_external_repo(external_root: Path) -> None:
    required = [
        external_root / "src" / "train_pix2pix_turbo.py",
        external_root / "src" / "train_cyclegan_turbo.py",
        external_root / "src" / "my_utils" / "training_utils.py",
    ]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Official img2img-turbo checkout is missing. Run "
            "`bash scripts/setup_img2img_turbo.sh` first. Missing: "
            + ", ".join(str(path) for path in missing)
        )
    if (external_root / ".git").exists() and shutil.which("git"):
        result = subprocess.run(
            ["git", "-C", str(external_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        revision = result.stdout.strip()
        if result.returncode == 0 and revision != PINNED_IMG2IMG_TURBO_COMMIT:
            raise ValueError(
                f"img2img-turbo is at {revision}, expected "
                f"{PINNED_IMG2IMG_TURBO_COMMIT}. Re-run the setup script."
            )


def build_training_command(
    model: str,
    shot: int,
    seed: int,
    config: DictConfig,
    dataset_folder: Path,
    output_dir: Path,
) -> list[str]:
    """Map project configuration to the official upstream CLI."""
    common = [
        "accelerate",
        "launch",
        "--num_processes",
        "1",
    ]
    pretrained = str(config_path(config, "model.backbone", "stabilityai/sd-turbo"))
    batch_size = str(config_path(config, "data.batch_size", 1))
    workers = str(config_path(config, "data.num_workers", 0))
    learning_rate = str(config_path(config, "training.learning_rate", 5e-6))
    epochs = str(config_path(config, "training.num_epochs", 50))
    checkpoint_steps = str(config_path(config, "training.save_every", 500))
    gradient_accumulation = str(
        config_path(config, "training.gradient_accumulation_steps", 1)
    )
    scheduler = str(config_path(config, "training.lr_scheduler", "constant"))
    report_to = (
        "wandb" if config_path(config, "logging.use_wandb", False) else "tensorboard"
    )

    if model == "pix2pix":
        command = common + [
            "src/train_pix2pix_turbo.py",
            "--pretrained_model_name_or_path",
            pretrained,
            "--dataset_folder",
            str(dataset_folder),
            "--output_dir",
            str(output_dir),
            "--seed",
            str(seed),
            "--resolution",
            str(config_path(config, "data.image_size", 512)),
            "--train_batch_size",
            batch_size,
            "--dataloader_num_workers",
            workers,
            "--learning_rate",
            learning_rate,
            "--num_training_epochs",
            epochs,
            "--checkpointing_steps",
            checkpoint_steps,
            "--eval_freq",
            str(config_path(config, "training.eval_every", 200)),
            "--gradient_accumulation_steps",
            gradient_accumulation,
            "--lr_scheduler",
            scheduler,
            "--mixed_precision",
            str(config_path(config, "training.mixed_precision", "fp16")),
            "--report_to",
            report_to,
            "--tracker_project_name",
            f"ece1508-pix2pix-{shot}shot",
            "--train_image_prep",
            str(
                config_path(
                    config, "pix2pix_turbo.train_image_prep", "resized_crop_512"
                )
            ),
            "--test_image_prep",
            str(
                config_path(config, "pix2pix_turbo.test_image_prep", "resized_crop_512")
            ),
            "--lambda_l2",
            str(config_path(config, "pix2pix_turbo.lambda_l2", 1.0)),
            "--lambda_lpips",
            str(config_path(config, "pix2pix_turbo.lambda_lpips", 5.0)),
            "--lambda_clipsim",
            str(config_path(config, "pix2pix_turbo.lambda_clipsim", 5.0)),
            "--lambda_gan",
            str(config_path(config, "pix2pix_turbo.lambda_gan", 0.5)),
            "--lora_rank_unet",
            str(config_path(config, "pix2pix_turbo.lora_rank_unet", 8)),
            "--lora_rank_vae",
            str(config_path(config, "pix2pix_turbo.lora_rank_vae", 4)),
        ]
    elif model == "cyclegan":
        command = common + [
            "src/train_cyclegan_turbo.py",
            "--pretrained_model_name_or_path",
            pretrained,
            "--dataset_folder",
            str(dataset_folder),
            "--output_dir",
            str(output_dir),
            "--seed",
            str(seed),
            "--train_batch_size",
            batch_size,
            "--dataloader_num_workers",
            workers,
            "--learning_rate",
            learning_rate,
            "--max_train_epochs",
            epochs,
            "--checkpointing_steps",
            checkpoint_steps,
            "--validation_steps",
            str(config_path(config, "training.eval_every", 200)),
            "--gradient_accumulation_steps",
            gradient_accumulation,
            "--lr_scheduler",
            scheduler,
            "--report_to",
            report_to,
            "--tracker_project_name",
            f"ece1508-cyclegan-{shot}shot",
            "--train_img_prep",
            str(
                config_path(config, "cyclegan_turbo.train_image_prep", "resize_512x512")
            ),
            "--val_img_prep",
            str(config_path(config, "cyclegan_turbo.val_image_prep", "resize_512x512")),
            "--lambda_cycle",
            str(config_path(config, "cyclegan_turbo.lambda_cycle", 1.0)),
            "--lambda_cycle_lpips",
            str(config_path(config, "cyclegan_turbo.lambda_cycle_lpips", 10.0)),
            "--lambda_idt",
            str(config_path(config, "cyclegan_turbo.lambda_identity", 1.0)),
            "--lambda_idt_lpips",
            str(config_path(config, "cyclegan_turbo.lambda_identity_lpips", 1.0)),
            "--lambda_gan",
            str(config_path(config, "cyclegan_turbo.lambda_gan", 0.5)),
            "--lora_rank_unet",
            str(config_path(config, "cyclegan_turbo.lora_rank_unet", 128)),
            "--lora_rank_vae",
            str(config_path(config, "cyclegan_turbo.lora_rank_vae", 4)),
        ]
    else:
        raise ValueError(f"Unknown model: {model}")

    max_train_steps = config_path(config, "training.max_train_steps")
    if max_train_steps is not None:
        command.extend(["--max_train_steps", str(max_train_steps)])
    if config_path(config, "model.enable_xformers", True):
        command.append("--enable_xformers_memory_efficient_attention")
    return command


def run_single_experiment(
    model: str,
    shot: int,
    seed: int,
    config: DictConfig,
    gpu_id: int,
    external_root: Path,
    dry_run: bool = False,
) -> bool:
    processed_root = (
        PROJECT_ROOT / str(config_path(config, "data.root", "data/processed"))
    ).resolve()
    adapter_root = (
        PROJECT_ROOT
        / str(
            config_path(
                config,
                "data.img2img_turbo_root",
                "data/processed/img2img_turbo",
            )
        )
    ).resolve()
    dataset_folder = prepare_dataset_view(
        processed_root=processed_root,
        output_root=adapter_root,
        shot=shot,
        seed=seed,
    )
    output_dir = (
        PROJECT_ROOT / "results" / model / f"{shot}shot" / f"seed{seed}"
    ).resolve()
    command = build_training_command(
        model=model,
        shot=shot,
        seed=seed,
        config=config,
        dataset_folder=dataset_folder,
        output_dir=output_dir,
    )

    print(f"\nRunning {model}: {shot}-shot, seed {seed}")
    print("Command:", shlex.join(command))
    if dry_run:
        return True

    if shutil.which("accelerate") is None:
        raise FileNotFoundError(
            "`accelerate` is not installed in the active environment. "
            "Create/activate the environment before training."
        )

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["ACCELERATE_MIXED_PRECISION"] = str(
        config_path(config, "training.mixed_precision", "fp16")
    )
    result = subprocess.run(command, env=env, cwd=external_root, check=False)
    return result.returncode == 0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", required=True, choices=["pix2pix", "cyclegan", "all"]
    )
    parser.add_argument("--shots", type=int, nargs="+", default=[10, 20, 50])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--external-root", type=Path, default=DEFAULT_EXTERNAL_ROOT)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare dataset views and print commands without launching training",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = OmegaConf.load(args.config.resolve())
    external_root = args.external_root.resolve()
    validate_external_repo(external_root)

    models = ["pix2pix", "cyclegan"] if args.model == "all" else [args.model]
    failed = []
    for model in models:
        for shot in args.shots:
            for seed in args.seeds:
                try:
                    success = run_single_experiment(
                        model=model,
                        shot=shot,
                        seed=seed,
                        config=config,
                        gpu_id=args.gpu,
                        external_root=external_root,
                        dry_run=args.dry_run,
                    )
                except (FileNotFoundError, FileExistsError, ValueError, OSError) as exc:
                    print(
                        f"ERROR: {model} {shot}-shot seed {seed}: {exc}",
                        file=sys.stderr,
                    )
                    success = False
                if not success:
                    failed.append(f"{model}_{shot}shot_seed{seed}")

    if failed:
        print(f"\nFailed runs: {', '.join(failed)}", file=sys.stderr)
        return 1
    print("\nAll requested experiments completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
