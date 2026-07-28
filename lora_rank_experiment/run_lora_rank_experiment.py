#!/usr/bin/env python3
"""Run the config-driven CycleGAN-Turbo U-Net LoRA-rank sweep."""

from __future__ import annotations

import argparse
import copy
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.train import model_unpaired


@dataclass(frozen=True)
class ExperimentSettings:
    output_dir: str
    shots: int
    seed: int
    steps: int
    resolution: int
    precision: str
    unet_ranks: tuple[int, ...]
    vae_rank: int
    learning_rate: float
    batch_size: int
    num_workers: int
    checkpointing_steps: int
    preview_steps: int
    enable_xformers: bool
    gradient_checkpointing: bool

    @classmethod
    def from_config(cls, config: dict) -> "ExperimentSettings":
        section = config.get("lora_rank_experiment")
        if not isinstance(section, dict):
            raise KeyError(
                "Missing required config section: lora_rank_experiment"
            )

        required = (
            "output_dir",
            "shots",
            "seed",
            "steps",
            "resolution",
            "precision",
            "unet_ranks",
            "vae_rank",
            "learning_rate",
            "batch_size",
            "num_workers",
            "checkpointing_steps",
            "preview_steps",
            "enable_xformers",
            "gradient_checkpointing",
        )
        missing = [name for name in required if name not in section]
        if missing:
            raise KeyError(
                "Missing lora_rank_experiment settings: "
                + ", ".join(missing)
            )

        ranks_value = section["unet_ranks"]
        if not isinstance(ranks_value, list):
            raise TypeError(
                "lora_rank_experiment.unet_ranks must be a YAML list"
            )
        for boolean_name in (
            "enable_xformers",
            "gradient_checkpointing",
        ):
            if not isinstance(section[boolean_name], bool):
                raise TypeError(
                    f"lora_rank_experiment.{boolean_name} must be true or false"
                )

        settings = cls(
            output_dir=str(section["output_dir"]),
            shots=int(section["shots"]),
            seed=int(section["seed"]),
            steps=int(section["steps"]),
            resolution=int(section["resolution"]),
            precision=str(section["precision"]),
            unet_ranks=tuple(int(rank) for rank in ranks_value),
            vae_rank=int(section["vae_rank"]),
            learning_rate=float(section["learning_rate"]),
            batch_size=int(section["batch_size"]),
            num_workers=int(section["num_workers"]),
            checkpointing_steps=int(section["checkpointing_steps"]),
            preview_steps=int(section["preview_steps"]),
            enable_xformers=section["enable_xformers"],
            gradient_checkpointing=section["gradient_checkpointing"],
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if not self.output_dir.strip():
            raise ValueError(
                "lora_rank_experiment.output_dir cannot be empty"
            )
        for name in ("shots", "seed", "steps", "vae_rank", "batch_size"):
            if getattr(self, name) <= 0:
                raise ValueError(
                    f"lora_rank_experiment.{name} must be positive"
                )
        if self.num_workers < 0:
            raise ValueError(
                "lora_rank_experiment.num_workers cannot be negative"
            )
        if self.checkpointing_steps <= 0:
            raise ValueError(
                "lora_rank_experiment.checkpointing_steps must be positive"
            )
        if self.preview_steps < 0:
            raise ValueError(
                "lora_rank_experiment.preview_steps cannot be negative"
            )
        if self.learning_rate <= 0:
            raise ValueError(
                "lora_rank_experiment.learning_rate must be positive"
            )
        if self.resolution not in (256, 512):
            raise ValueError(
                "lora_rank_experiment.resolution must be 256 or 512"
            )
        if self.precision not in ("no", "fp16", "bf16"):
            raise ValueError(
                "lora_rank_experiment.precision must be no, fp16, or bf16"
            )
        if not self.unet_ranks:
            raise ValueError(
                "lora_rank_experiment.unet_ranks cannot be empty"
            )
        if any(rank <= 0 for rank in self.unet_ranks):
            raise ValueError(
                "lora_rank_experiment.unet_ranks must all be positive"
            )
        if len(set(self.unet_ranks)) != len(self.unet_ranks):
            raise ValueError(
                "lora_rank_experiment.unet_ranks cannot contain duplicates"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="Physical GPU index exposed to each sequential run (default: 0)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=model_unpaired.DEFAULT_CONFIG,
        help="Config containing the lora_rank_experiment section",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Override lora_rank_experiment.output_dir",
    )
    parser.add_argument(
        "--no-xformers",
        action="store_true",
        help="Disable configured xFormers for every rank",
    )
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help="Skip a rank when its final checkpoint already exists",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print all commands without starting GPU training",
    )
    return parser.parse_args()


def project_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def build_rank_config(
    base_config: dict,
    settings: ExperimentSettings,
    unet_rank: int,
    use_xformers: bool,
) -> dict:
    """Return an isolated config with only rank-specific metadata changed."""
    config = copy.deepcopy(base_config)
    training = config["training"]
    cyclegan = config["cyclegan_turbo"]

    training["resolution"] = settings.resolution
    cyclegan.update(
        {
            "batch_size": settings.batch_size,
            "num_workers": settings.num_workers,
            "learning_rate": settings.learning_rate,
            "max_train_steps": settings.steps,
            "checkpointing_steps": settings.checkpointing_steps,
            "preview_steps": settings.preview_steps,
            "mixed_precision": settings.precision,
            "enable_xformers": use_xformers,
            "gradient_checkpointing": settings.gradient_checkpointing,
            "lora_rank_unet": unet_rank,
            "lora_rank_vae": settings.vae_rank,
            "train_image_prep": (
                f"resize_{settings.resolution}x{settings.resolution}"
            ),
            "val_image_prep": (
                f"resize_{settings.resolution}x{settings.resolution}"
            ),
            "validation_steps": settings.steps + 1,
            "validation_num_images": 1,
            "skip_training_validation": True,
            "allow_tf32": False,
            "tracker_project_name": (
                f"cyclegan_turbo_{settings.shots}shot_rank{unet_rank}"
            ),
        }
    )
    return config


def print_experiment_summary(
    settings: ExperimentSettings,
    dataset: Path,
    output: Path,
    gpu: int | None,
    use_xformers: bool,
) -> None:
    print("CycleGAN-Turbo U-Net LoRA-rank experiment")
    print(f"  ranks:            {', '.join(map(str, settings.unet_ranks))}")
    print(f"  dataset:          {dataset}")
    print(f"  shots / seed:     {settings.shots} / {settings.seed}")
    print(f"  steps:            {settings.steps}")
    print(
        f"  resolution:       "
        f"{settings.resolution}x{settings.resolution}"
    )
    print(f"  precision:        {settings.precision}")
    print(f"  VAE LoRA rank:    {settings.vae_rank}")
    print(f"  learning rate:    {settings.learning_rate}")
    print(f"  batch size:       {settings.batch_size}")
    print(f"  checkpoint every: {settings.checkpointing_steps}")
    print(f"  preview every:    {settings.preview_steps or 'disabled'}")
    print(f"  xFormers:         {use_xformers}")
    selected_gpu = gpu if gpu is not None else "CUDA environment default"
    print(f"  physical GPU:     {selected_gpu}")
    print(f"  output root:      {output}")


def main() -> int:
    args = parse_args()
    if args.gpu is not None and args.gpu < 0:
        raise ValueError("--gpu cannot be negative")

    base_config = model_unpaired.load_config(args.config.resolve())
    settings = ExperimentSettings.from_config(base_config)
    dataset = (
        project_path(base_config["data"]["root"])
        / f"{settings.shots}shot"
        / f"seed{settings.seed}"
    )
    output_root = (
        args.output.resolve()
        if args.output is not None
        else project_path(settings.output_dir)
    )
    use_xformers = settings.enable_xformers and not args.no_xformers

    if not args.dry_run:
        model_unpaired.validate_vendor()
        model_unpaired.validate_dataset(
            dataset,
            expected_shots=settings.shots,
        )

    print_experiment_summary(
        settings=settings,
        dataset=dataset,
        output=output_root,
        gpu=args.gpu,
        use_xformers=use_xformers,
    )

    for unet_rank in settings.unet_ranks:
        config = build_rank_config(
            base_config,
            settings,
            unet_rank,
            use_xformers,
        )
        output_dir = output_root / f"unet_rank_{unet_rank}"
        final_checkpoint = (
            output_dir
            / "checkpoints"
            / f"model_{settings.steps}.pkl"
        )
        existing_checkpoints = list(
            (output_dir / "checkpoints").glob("model_*.pkl")
        )

        if final_checkpoint.is_file() and args.skip_completed:
            print(f"\nSkipping rank {unet_rank}: {final_checkpoint}")
            continue
        if existing_checkpoints:
            raise FileExistsError(
                f"Rank {unet_rank} already has checkpoints in {output_dir}. "
                "Choose a new --output directory, or use --skip-completed when "
                "the final checkpoint exists."
            )

        command = model_unpaired.build_train_command(
            script_path=model_unpaired.TRAIN_SCRIPT,
            dataset_dir=dataset,
            output_dir=output_dir,
            seed=settings.seed,
            config=config,
            max_train_steps=settings.steps,
        )
        print(f"\nRank {unet_rank} command:")
        print("[CMD]", " ".join(command))
        if args.dry_run:
            continue

        output_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            command,
            check=True,
            env=model_unpaired.training_environment(config, args.gpu),
        )
        if not final_checkpoint.is_file():
            raise RuntimeError(
                f"Rank {unet_rank} ended without its final checkpoint: "
                f"{final_checkpoint}"
            )
        print(f"Rank {unet_rank} complete: {final_checkpoint}")

    if args.dry_run:
        print("\nDry run complete; no GPU work was started.")
    else:
        print(f"\nLoRA-rank experiment complete: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
