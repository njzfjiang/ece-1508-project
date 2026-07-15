from pathlib import Path
import sys

import pytest

from src.train.model_paired import (
    DEFAULT_CONFIG,
    build_train_command,
    load_config,
    train_model,
    training_environment,
    validate_vendor_script,
)


def option_value(command: list[str], option: str) -> str:
    return command[command.index(option) + 1]


def test_pix2pix_config_is_forwarded_to_upstream_cli(tmp_path):
    config = load_config(DEFAULT_CONFIG)
    config["training"].update(
        {
            "batch_size": 2,
            "num_workers": 3,
            "max_train_steps": 17,
            "checkpointing_steps": 5,
            "mixed_precision": "no",
        }
    )
    config["pix2pix_turbo"].update(
        {
            "lora_rank_unet": 6,
            "lambda_lpips": 4.5,
            "track_val_fid": True,
        }
    )

    command = build_train_command(
        script_path=Path("train.py"),
        dataset_dir=tmp_path / "data",
        output_dir=tmp_path / "output",
        seed=7,
        config=config,
    )

    assert command[:3] == [
        sys.executable,
        "-m",
        "accelerate.commands.launch",
    ]
    assert option_value(command, "--pretrained_model_name_or_path") == (
        "stabilityai/sd-turbo"
    )
    assert option_value(command, "--train_batch_size") == "2"
    assert option_value(command, "--dataloader_num_workers") == "3"
    assert option_value(command, "--max_train_steps") == "17"
    assert option_value(command, "--checkpointing_steps") == "5"
    assert option_value(command, "--mixed_precision") == "no"
    assert option_value(command, "--seed") == "7"
    assert option_value(command, "--lora_rank_unet") == "6"
    assert option_value(command, "--lambda_lpips") == "4.5"
    assert "--track_val_fid" in command
    assert "--enable_xformers_memory_efficient_attention" in command
    assert "--gradient_checkpointing" not in command


def test_wandb_is_disabled_through_environment():
    config = load_config(DEFAULT_CONFIG)

    environment = training_environment(config)

    assert environment["WANDB_MODE"] == "disabled"


def test_missing_fewshot_dataset_fails_instead_of_skipping(tmp_path):
    config = load_config(DEFAULT_CONFIG)

    with pytest.raises(FileNotFoundError, match="Missing dataset"):
        train_model(
            shots=[10],
            seeds=[1],
            dataset_root=tmp_path / "data",
            output_root=tmp_path / "output",
            script_path=Path("train.py"),
            config=config,
        )


def test_vendor_preflight_requires_all_compatibility_patches(tmp_path):
    script = tmp_path / "train.py"
    script.write_text("# unpatched upstream\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="setup.py --skip-install --skip-prepare"):
        validate_vendor_script(script)

    script.write_text(
        "# max_train_steps is the authoritative stopping condition\n"
        "# Keep trainable parameters in FP32\n"
        "# Optimizer parameters must be unique by identity\n",
        encoding="utf-8",
    )
    validate_vendor_script(script)


def test_existing_checkpoints_are_not_silently_mixed(tmp_path):
    config = load_config(DEFAULT_CONFIG)
    dataset_root = tmp_path / "data"
    (dataset_root / "10shot" / "seed1").mkdir(parents=True)
    output_root = tmp_path / "output"
    checkpoint = output_root / "10shot" / "seed1" / "checkpoints" / "model_99.pkl"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()

    with pytest.raises(FileExistsError, match="Refusing to mix"):
        train_model(
            shots=[10],
            seeds=[1],
            dataset_root=dataset_root,
            output_root=output_root,
            script_path=Path("train.py"),
            config=config,
        )
