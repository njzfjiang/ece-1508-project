from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_cyclegan_checkpoint_patch_saves_and_compatibly_loads_conv_in():
    patch = (
        ROOT / "patches" / "img2img-turbo-cyclegan-conv-in-checkpoint.patch"
    ).read_text(encoding="utf-8")

    assert 'sd["sd_unet_conv_in"] = base_conv_in.state_dict()' in patch
    assert 'if "sd_unet_conv_in" in sd:' in patch
    assert 'base_conv_in.load_state_dict(sd["sd_unet_conv_in"])' in patch


def test_setup_preserves_memory_fixes_before_checkpoint_and_logging_patches():
    setup = (ROOT / "scripts" / "setup.py").read_text(encoding="utf-8")
    ordered = [
        "img2img-turbo-cyclegan-memory.patch",
        "img2img-turbo-cyclegan-sequential-backward.patch",
        "img2img-turbo-cyclegan-conv-in-checkpoint.patch",
        "img2img-turbo-training-loss-csv.patch",
    ]

    positions = [setup.index(name) for name in ordered]
    assert positions == sorted(positions)


def test_loss_csv_patch_covers_both_trainers():
    patch = (ROOT / "patches" / "img2img-turbo-training-loss-csv.patch").read_text(
        encoding="utf-8"
    )

    assert "a/src/train_pix2pix_turbo.py" in patch
    assert "a/src/train_cyclegan_turbo.py" in patch
    assert patch.count('os.path.join(args.output_dir, "losses.csv")') == 2


def test_preview_patch_preserves_training_rng():
    patch = (ROOT / "patches" / "img2img-turbo-cyclegan-preview.patch").read_text(
        encoding="utf-8"
    )

    assert "torch.random.fork_rng" in patch
    assert "torch.manual_seed(0)" in patch
    assert "finally:" in patch
