from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_5090_profile_is_blackwell_safe_and_isolates_outputs():
    config = yaml.safe_load((ROOT / "configs" / "5090.yaml").read_text())

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
