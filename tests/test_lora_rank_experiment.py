from pathlib import Path

import yaml

from lora_rank_experiment.run_lora_rank_experiment import (
    ExperimentSettings,
    build_rank_config,
    write_run_metadata,
)

ROOT = Path(__file__).resolve().parents[1]


def test_base_rank_pilot_changes_only_unet_rank_between_runs():
    base = yaml.safe_load((ROOT / "configs" / "base.yaml").read_text())
    settings = ExperimentSettings.from_config(base)

    assert settings.shots == 10
    assert settings.steps == 1000
    assert settings.resolution == 512
    assert settings.precision == "fp16"
    assert settings.unet_ranks == (4, 128)
    assert settings.validation_samples == 100
    assert settings.metrics == ("ssim", "lpips", "clip_similarity", "cmmd")

    rank4 = build_rank_config(base, settings, 4, use_xformers=True)
    rank128 = build_rank_config(base, settings, 128, use_xformers=True)
    cycle4 = rank4["cyclegan_turbo"]
    cycle128 = rank128["cyclegan_turbo"]

    differing = {key for key in cycle4 if cycle4.get(key) != cycle128.get(key)}
    assert differing == {"lora_rank_unet", "tracker_project_name"}


def test_run_metadata_records_config_and_replayable_command(tmp_path):
    output = tmp_path / "rank4"
    config = {"cyclegan_turbo": {"lora_rank_unet": 4}}
    command = ["python", "trainer.py", "--output_dir", "path with spaces"]

    write_run_metadata(output, config, command)

    saved = yaml.safe_load((output / "resolved_config.yaml").read_text())
    assert saved == config
    assert "'path with spaces'" in (output / "command.txt").read_text()
