import json
from pathlib import Path

from omegaconf import OmegaConf
from PIL import Image

from scripts.prepare_img2img_turbo_data import prepare_dataset_view
from src.train.run_experiment import (
    PIX2PIX_FP16_PATCH_MARKER,
    build_training_command,
)


def write_image(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color="navy").save(path)


def make_processed_dataset(root: Path):
    for split, names in {
        "train": ["train_1.jpg", "train_2.jpg", "train_3.jpg"],
        "test": ["test_1.jpg", "test_2.jpg"],
    }.items():
        for domain in ("day", "night"):
            for name in names:
                write_image(root / "day2night" / split / domain / name)

    split_dir = root / "splits" / "fewshot" / "2shot" / "seed1"
    split_dir.mkdir(parents=True)
    split_dir.joinpath("split.json").write_text(
        json.dumps(
            {
                "shot": 2,
                "seed": 1,
                "train_day": ["train_1.jpg", "train_3.jpg"],
                "train_night": ["train_1.jpg", "train_3.jpg"],
            }
        ),
        encoding="utf-8",
    )


def test_adapter_builds_official_layout(tmp_path):
    processed = tmp_path / "processed"
    make_processed_dataset(processed)

    dataset = prepare_dataset_view(
        processed_root=processed,
        output_root=processed / "img2img_turbo",
        shot=2,
        seed=1,
        mode="copy",
    )

    assert sorted(path.name for path in dataset.joinpath("train_A").iterdir()) == [
        "train_1.jpg",
        "train_3.jpg",
    ]
    assert len(list(dataset.joinpath("test_B").iterdir())) == 2
    assert json.loads(dataset.joinpath("train_prompts.json").read_text()) == {
        "train_1.jpg": "a driving scene at night",
        "train_3.jpg": "a driving scene at night",
    }
    assert (
        dataset.joinpath("fixed_prompt_a.txt")
        .read_text()
        .strip()
        .endswith("during the day")
    )
    assert dataset.joinpath("manifest.json").is_file()


def test_commands_use_upstream_argument_names(tmp_path):
    config = OmegaConf.load(Path("configs/base.yaml"))
    dataset = tmp_path / "dataset"
    output = tmp_path / "output"

    pix2pix = build_training_command("pix2pix", 10, 1, config, dataset, output)
    cyclegan = build_training_command("cyclegan", 10, 1, config, dataset, output)

    assert "--dataset_folder" in pix2pix
    assert "--num_training_epochs" in pix2pix
    assert "--lambda_l2" in pix2pix
    assert "--lambda_l1" not in pix2pix
    assert "--max_train_epochs" in cyclegan
    assert "--lambda_idt" in cyclegan
    assert "--lambda_identity" not in cyclegan
    assert "--lora_rank_unet" in pix2pix
    assert "--lora_rank_vae" in cyclegan
    assert pix2pix[pix2pix.index("--report_to") + 1] == "wandb"
    assert cyclegan[cyclegan.index("--report_to") + 1] == "wandb"
    assert pix2pix[pix2pix.index("--mixed_precision") + 1] == "fp16"
    assert "--num_machines" in pix2pix
    assert "--dynamo_backend" in cyclegan


def test_fp16_patch_keeps_trainable_master_weights_in_fp32():
    patch = Path("patches/img2img-turbo-pix2pix-fp16.patch").read_text()
    assert PIX2PIX_FP16_PATCH_MARKER in patch
    assert "-    net_pix2pix.to(dtype=weight_dtype)" in patch
    assert "-    net_disc.to(dtype=weight_dtype)" in patch
