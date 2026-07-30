from pathlib import Path

import yaml

from src.train.model_paired import load_config as load_paired_config
from src.train.model_unpaired import load_config as load_unpaired_config


ROOT = Path(__file__).resolve().parents[1]


def test_5090_profile_is_blackwell_safe_and_isolates_outputs():
    config_path = ROOT / "configs" / "5090.yaml"
    config = yaml.safe_load(config_path.read_text())

    # Exercise both real launcher validators so required top-level sections
    # cannot silently disappear when this profile is updated.
    assert load_paired_config(config_path) == config
    assert load_unpaired_config(config_path) == config

    assert config["training"]["enable_xformers"] is False
    assert config["lora_rank_experiment"]["enable_xformers"] is False
    assert config["training"]["max_train_steps"] == 2000
    assert config["training"]["checkpointing_steps"] == 2000
    assert config["pix2pix_turbo"]["output_dir"].startswith(
        "outputs/full_grid_5090/"
    )
    assert config["cyclegan_turbo"]["output_dir"].startswith(
        "outputs/full_grid_5090/"
    )
    assert config["eval"]["generated_dir"].startswith("results/full_grid_5090/")
    assert config["eval"]["output_dir"].startswith("results/full_grid_5090/")
    assert isinstance(config["logging"]["use_wandb"], bool)
    assert config["logging"]["log_dir"].startswith("results/full_grid_5090/")


def test_5090_requirements_pin_validated_stack_without_xformers():
    requirements = (ROOT / "requirements-5090.txt").read_text().lower()

    for requirement in (
        "torch==2.10.0",
        "torchvision==0.25.0",
        "torchaudio==2.10.0",
        "diffusers==0.39.0",
        "transformers==4.57.6",
        "peft==0.19.1",
        "accelerate==1.13.0",
    ):
        assert requirement in requirements
    assert "\nxformers" not in requirements


def test_5090_smoke_keeps_formal_resolution_with_two_steps():
    config_path = ROOT / "configs" / "5090-smoke.yaml"
    config = load_paired_config(config_path)

    assert load_unpaired_config(config_path) == config
    assert config["training"]["resolution"] == 512
    assert config["training"]["max_train_steps"] == 2
    assert config["training"]["checkpointing_steps"] == 2
    assert config["training"]["enable_xformers"] is False
    assert config["cyclegan_turbo"]["train_image_prep"] == "resize_512x512"
    assert config["cyclegan_turbo"]["checkpointing_steps"] == 2
    assert config["eval"]["test_samples"] == 1
    assert config["logging"]["use_wandb"] is False
