from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import train_cyclegan_10shot
from src.train import model_unpaired


def option_value(command: list[str], option: str) -> str:
    return command[command.index(option) + 1]


def make_args(**overrides):
    defaults = {
        "steps": None,
        "resolution": None,
        "precision": None,
        "no_xformers": False,
        "no_gradient_checkpointing": False,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def make_dataset(root: Path, shot: int) -> None:
    root.mkdir(parents=True)
    (root / "fixed_prompt_a.txt").write_text("day\n", encoding="utf-8")
    (root / "fixed_prompt_b.txt").write_text("night\n", encoding="utf-8")
    for folder_name, count in (
        ("train_A", shot),
        ("train_B", shot),
        ("test_A", 1),
        ("test_B", 1),
    ):
        folder = root / folder_name
        folder.mkdir()
        for index in range(count):
            (folder / f"{index}.jpg").touch()


def test_smoke_config_builds_two_step_256_command(tmp_path):
    config = model_unpaired.load_config(
        model_unpaired.PROJECT_ROOT / "configs" / "smoke.yaml"
    )
    settings = model_unpaired.settings_from_config(config, make_args())

    command = model_unpaired.build_train_command(
        Path("train.py"), tmp_path / "data", tmp_path / "output", 7, settings
    )

    assert command[:3] == [
        model_unpaired.sys.executable,
        "-m",
        "accelerate.commands.launch",
    ]
    assert option_value(command, "--max_train_steps") == "2"
    assert option_value(command, "--checkpointing_steps") == "2"
    assert option_value(command, "--mixed_precision") == "fp16"
    assert option_value(command, "--train_img_prep") == "resize_256x256"
    assert option_value(command, "--seed") == "7"
    assert "--skip_training_validation" in command
    assert "--gradient_checkpointing" in command


def test_cli_overrides_change_steps_resolution_and_precision():
    config = model_unpaired.load_config(model_unpaired.DEFAULT_CONFIG)
    settings = model_unpaired.settings_from_config(
        config,
        make_args(steps=3, resolution=256, precision="no"),
    )

    assert settings.max_train_steps == 3
    assert settings.resolution == 256
    assert settings.mixed_precision == "no"
    assert settings.train_image_prep == "resize_256x256"
    assert settings.val_image_prep == "resize_256x256"


def test_base_config_uses_sparse_cyclegan_checkpoints():
    config = model_unpaired.load_config(model_unpaired.DEFAULT_CONFIG)
    settings = model_unpaired.settings_from_config(config, make_args())

    assert settings.checkpointing_steps == 2000


def test_zero_step_override_is_preserved_for_main_validation():
    config = model_unpaired.load_config(model_unpaired.DEFAULT_CONFIG)
    settings = model_unpaired.settings_from_config(config, make_args(steps=0))

    assert settings.max_train_steps == 0


def test_dataset_validation_enforces_shot_count(tmp_path):
    dataset = tmp_path / "dataset"
    make_dataset(dataset, shot=2)

    model_unpaired.validate_dataset(dataset, expected_images=2)
    with pytest.raises(ValueError, match="Expected 3 images"):
        model_unpaired.validate_dataset(dataset, expected_images=3)


def test_dry_run_does_not_create_output(tmp_path):
    dataset = tmp_path / "data" / "10shot" / "seed1"
    make_dataset(dataset, shot=10)
    output_root = tmp_path / "output"
    config = model_unpaired.load_config(
        model_unpaired.PROJECT_ROOT / "configs" / "smoke.yaml"
    )
    settings = model_unpaired.settings_from_config(config, make_args())

    model_unpaired.train_model(
        shots=[10],
        seeds=[1],
        dataset_root=tmp_path / "data",
        output_root=output_root,
        settings=settings,
        config=config,
        gpu=0,
        dry_run=True,
    )

    assert not output_root.exists()


def test_vendor_marker_validation_reports_partial_patch(tmp_path):
    vendor_file = tmp_path / "train.py"
    vendor_file.write_text("# args.skip_training_validation\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Missing markers"):
        model_unpaired._validate_markers(
            vendor_file, model_unpaired.REQUIRED_TRAINER_MARKERS
        )


def test_cycle_directions_backward_sequentially():
    source = model_unpaired.UPSTREAM_TRAINER.read_text(encoding="utf-8")

    first_backward = source.index("accelerator.backward(loss_cycle_a")
    second_forward = source.index("cyc_fake_a = CycleGAN_Turbo.forward_with_networks")
    second_backward = source.index("accelerator.backward(loss_cycle_b")

    assert first_backward < second_forward < second_backward


def test_standalone_wrapper_preserves_10shot_seed1_defaults(monkeypatch):
    captured = {}
    monkeypatch.setattr(train_cyclegan_10shot.sys, "argv", ["launcher", "--dry-run"])
    monkeypatch.setattr(
        train_cyclegan_10shot,
        "main",
        lambda arguments: captured.setdefault("arguments", arguments) or 0,
    )

    train_cyclegan_10shot.compatibility_main()

    assert captured["arguments"][:4] == ["--seed", "1", "--shots", "10"]
