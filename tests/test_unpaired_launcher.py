from pathlib import Path

from src.train.model_unpaired import build_train_command, load_config

ROOT = Path(__file__).resolve().parents[1]


def _value_after(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def test_base_cycle_command_uses_unpaired_cli_and_memory_settings():
    config = load_config(ROOT / "configs" / "base.yaml")
    command = build_train_command(
        Path("train_cyclegan_turbo.py"), Path("data"), Path("output"), 3, config
    )

    assert _value_after(command, "--train_batch_size") == "1"
    assert _value_after(command, "--dataloader_num_workers") == "0"
    assert _value_after(command, "--checkpointing_steps") == "2000"
    assert _value_after(command, "--max_train_steps") == "2000"
    assert _value_after(command, "--learning_rate") == "1e-05"
    assert _value_after(command, "--mixed_precision") == "fp16"
    assert _value_after(command, "--preview_steps") == "0"
    assert _value_after(command, "--lora_rank_unet") == "4"
    assert _value_after(command, "--train_img_prep") == "resize_512x512"
    assert "--lambda_cycle_lpips" in command
    assert "--lambda_idt_lpips" in command
    assert "--skip_training_validation" in command
    assert "--gradient_checkpointing" in command
    assert "--resolution" not in command
    assert "--lambda_l2" not in command
    assert command.index("--mixed_precision") < command.index("train_cyclegan_turbo.py")


def test_smoke_cycle_command_uses_256_preprocessing_and_two_steps():
    config = load_config(ROOT / "configs" / "smoke.yaml")
    command = build_train_command(
        Path("train_cyclegan_turbo.py"), Path("data"), Path("output"), 1, config
    )

    assert _value_after(command, "--max_train_steps") == "2"
    assert _value_after(command, "--checkpointing_steps") == "2"
    assert _value_after(command, "--preview_steps") == "0"
    assert _value_after(command, "--train_img_prep") == "resize_256x256"
    assert _value_after(command, "--val_img_prep") == "resize_256x256"
