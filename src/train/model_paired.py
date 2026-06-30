import subprocess
from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def load_config(config_path: Path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def run(cmd):
    print("\n[CMD]", " ".join(cmd))
    subprocess.run(cmd, check=True)

def train_model(shots, seeds, model_path, dataset_folder, output_dir, script_path):
    cfg = load_config(PROJECT_ROOT / "config" / "base.yaml")["training"]

    for shot in shots:
        for seed in seeds:
            dataset_dir = dataset_folder / f"{shot}shot" / f"seed{seed}"

            if not dataset_dir.exists():
                print(f"Missing dataset {shot}-{seed}, skipping")
                continue

            run_output_dir = output_dir / f"{shot}shot" / f"seed{seed}"
            run_output_dir.mkdir(parents=True, exist_ok=True)

            cmd = [
                "accelerate", "launch", str(script_path),
                "--pretrained_model_name_or_path", str(model_path),
                "--output_dir", str(run_output_dir),
                "--dataset_folder", str(dataset_dir),
                "--resolution", str(cfg["resolution"]),
                "--train_batch_size", str(cfg["batch_size"]),
                "--enable_xformers_memory_efficient_attention",
                "--viz_freq", str(cfg["viz_freq"]),
                "--track_val_fid",
                "--report_to", "wandb",
                "--tracker_project_name", "pix2pix_turbo_darkdriving",
            ]

            print(f"Training {shot}-{seed}")
            run(cmd)

def get_latest_model(model_dir: Path):
    if not model_dir.exists():
        raise FileNotFoundError(model_dir)
    ckpts = list(model_dir.glob("*.pkl"))
    if not ckpts:
        raise FileNotFoundError(model_dir)
    return max(ckpts, key=lambda p: p.stat().st_mtime)

def test_model(shots, seeds, testset_folder, script_path, output_dir, output_base):
    for shot in shots:
        for seed in seeds:
            model_dir = output_base / "train" / f"{shot}shot" / f"seed{seed}" / "checkpoints"
            latest_model_path = get_latest_model(model_dir)

            output_image_dir = output_dir / f"{shot}shot" / f"seed{seed}"
            output_image_dir.mkdir(parents=True, exist_ok=True)

            for test_image in (testset_folder / "test_A").glob("*.jpg"):
                cmd = [
                    "python", str(script_path),
                    "--model_path", str(latest_model_path),
                    "--input_image", str(test_image),
                    "--prompt", "a driving scene during the day",
                    "--output_dir", str(output_image_dir),
                ]
                run(cmd)
            
    
def main():
    model_path = PROJECT_ROOT / "src" / "train" / "night2day.pkl"
    dataset_folder = PROJECT_ROOT / "data" / "processed"
    output_base = PROJECT_ROOT / "outputs" / "pix2pix_turbo"
    pix2pix_turbo_repo = PROJECT_ROOT / "external" / "img2img-turbo"
    shots = [10, 20, 50]
    seeds = [1, 2, 3]
    
    # train
    validate_output_dir = output_base / "train"
    validate_output_dir.mkdir(parents=True, exist_ok=True)
    train_script_path = pix2pix_turbo_repo / "src" / "train_pix2pix_turbo.py"
    train_model(shots, seeds, model_path, dataset_folder, validate_output_dir, train_script_path)
    print("\nModel training completed check the outputs/pix2pix_turbo/train directory for results.")
    
    # test
    testset_folder = dataset_folder / "test"
    testset_folder.mkdir(parents=True, exist_ok=True)
    test_output_dir = output_base / "test"
    test_output_dir.mkdir(parents=True, exist_ok=True)
    test_script_path = pix2pix_turbo_repo / "src" / "inference_paired.py"
    test_model(shots, seeds, testset_folder, test_script_path, test_output_dir, output_base)
    print("\nModel testing completed check the outputs/pix2pix_turbo/test directory for results.")


if __name__ == "__main__":
    main()