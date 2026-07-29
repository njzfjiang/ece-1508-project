import sys
from pathlib import Path

import yaml

from src.train.model_paired import build_train_command, training_environment


ROOT = Path(__file__).resolve().parents[1]


def _value_after(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def test_paired_command_uses_current_interpreter_and_safe_batch_size():
    config = yaml.safe_load((ROOT / "configs" / "base.yaml").read_text())
    command = build_train_command(
        Path("train_pix2pix_turbo.py"), Path("data"), Path("output"), 2, config
    )

    assert command[:3] == [sys.executable, "-m", "accelerate.commands.launch"]
    assert _value_after(command, "--num_processes") == "1"
    assert _value_after(command, "--num_machines") == "1"
    assert command.index("--dynamo_backend") < command.index(
        "train_pix2pix_turbo.py"
    )
    assert _value_after(command, "--train_batch_size") == "1"
    assert _value_after(command, "--pretrained_model_name_or_path") == (
        "stabilityai/sd-turbo"
    )
    assert _value_after(command, "--max_train_steps") == "4000"


def test_paired_environment_selects_one_physical_gpu():
    config = yaml.safe_load((ROOT / "configs" / "base.yaml").read_text())
    environment = training_environment(config, gpu=3)

    assert environment["CUDA_VISIBLE_DEVICES"] == "3"
    assert environment["TOKENIZERS_PARALLELISM"] == "false"
    assert environment["WANDB_MODE"] == "disabled"
