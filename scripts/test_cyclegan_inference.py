#!/usr/bin/env python3
"""Compare CycleGAN-Turbo before and after project fine-tuning.

With no ``--checkpoint``, this constructs the generator exactly at step 0:
pretrained SD-Turbo VAE/U-Net weights plus newly initialized CycleGAN-Turbo
LoRA and skip adapters. It does not load the authors' released day-to-night
checkpoint. Later, pass a project checkpoint to produce the comparable
after-training image.
"""

from __future__ import annotations

import argparse
import gc
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_SRC = PROJECT_ROOT / "external" / "img2img-turbo" / "src"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "inference_smoke"
DEFAULT_PROMPT = "a driving scene during the night"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        help="Day image; defaults to the first image in the processed test split",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Project model_*.pkl checkpoint; omit for the step-0 baseline",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--resolution", type=int, choices=[256, 512], default=256)
    parser.add_argument("--lora-rank-unet", type=int, default=4)
    parser.add_argument("--lora-rank-vae", type=int, default=4)
    parser.add_argument(
        "--fp32",
        action="store_true",
        help="Use FP32 instead of the default lower-memory FP16 inference",
    )
    parser.add_argument(
        "--no-xformers",
        action="store_true",
        help="Do not enable xFormers memory-efficient attention",
    )
    return parser.parse_args()


def first_image(directory: Path) -> Path | None:
    if not directory.is_dir():
        return None
    images = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    return images[0] if images else None


def find_default_input(project_root: Path = PROJECT_ROOT) -> Path:
    candidates = (
        project_root / "data" / "processed" / "test" / "test_A",
        project_root
        / "data"
        / "processed"
        / "10shot"
        / "seed1"
        / "test_A",
        project_root
        / "data"
        / "processed"
        / "10shot"
        / "seed1"
        / "train_A",
    )
    for directory in candidates:
        image = first_image(directory)
        if image is not None:
            return image
    raise FileNotFoundError(
        "No default day image was found. Pass one explicitly with --input PATH."
    )


def physical_gpu_description(gpu: int) -> str:
    command = [
        "nvidia-smi",
        f"--id={gpu}",
        "--query-gpu=name,memory.total",
        "--format=csv,noheader",
    ]
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unavailable through nvidia-smi"
    return result.stdout.strip() or "unavailable through nvidia-smi"


def validate_upstream_source() -> None:
    required = (
        UPSTREAM_SRC / "cyclegan_turbo.py",
        UPSTREAM_SRC / "model.py",
        UPSTREAM_SRC / "my_utils" / "training_utils.py",
    )
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing upstream img2img-turbo files: {missing}\n"
            "Run: python scripts/setup.py --skip-install --skip-prepare"
        )


def main() -> int:
    args = parse_args()
    validate_upstream_source()
    input_path = (args.input or find_default_input()).resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input image not found: {input_path}")
    checkpoint = args.checkpoint.resolve() if args.checkpoint else None
    if checkpoint is not None and not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    # Set visibility before importing torch. Inside this process, the selected
    # physical GPU becomes logical cuda:0, matching upstream hard-coded .cuda().
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if str(UPSTREAM_SRC) not in sys.path:
        sys.path.insert(0, str(UPSTREAM_SRC))

    import torch
    from PIL import Image
    from torchvision import transforms

    from cyclegan_turbo import (
        CycleGAN_Turbo,
        VAE_decode,
        VAE_encode,
        initialize_unet,
        initialize_vae,
    )
    from model import make_1step_sched
    from my_utils.training_utils import build_transform

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available to PyTorch. Verify the NVIDIA driver and "
            "CUDA-enabled PyTorch installation."
        )

    print(f"Physical GPU {args.gpu}: {physical_gpu_description(args.gpu)}")
    print(f"PyTorch device: cuda:0 ({torch.cuda.get_device_name(0)})")
    print(f"Input image: {input_path}")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    input_image = Image.open(input_path).convert("RGB")
    transform = build_transform(f"resize_{args.resolution}x{args.resolution}")
    transformed = transform(input_image)
    input_tensor = transforms.ToTensor()(transformed)
    input_tensor = transforms.Normalize([0.5], [0.5])(input_tensor)
    input_tensor = input_tensor.unsqueeze(0).cuda()
    if not args.fp32:
        input_tensor = input_tensor.half()

    if checkpoint is None:
        print("Mode: step-0 baseline (SD-Turbo base; adapters are untrained)")
        print("Loading pretrained SD-Turbo components...")
        from transformers import AutoTokenizer, CLIPTextModel

        tokenizer = AutoTokenizer.from_pretrained(
            "stabilityai/sd-turbo", subfolder="tokenizer", use_fast=False
        )
        text_encoder = CLIPTextModel.from_pretrained(
            "stabilityai/sd-turbo", subfolder="text_encoder"
        ).cuda()
        unet = initialize_unet(args.lora_rank_unet).cuda()
        vae_a2b = initialize_vae(args.lora_rank_vae).cuda()
        vae_encoder = VAE_encode(vae_a2b).cuda()
        vae_decoder = VAE_decode(vae_a2b).cuda()
        scheduler = make_1step_sched()

        unet.eval()
        vae_encoder.eval()
        vae_decoder.eval()
        text_encoder.eval()
        if not args.no_xformers:
            unet.enable_xformers_memory_efficient_attention()
        if not args.fp32:
            unet.half()
            vae_encoder.half()
            vae_decoder.half()
            text_encoder.half()

        prompt_tokens = tokenizer(
            args.prompt,
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids.cuda()
        with torch.inference_mode():
            prompt_embedding = text_encoder(prompt_tokens)[0].detach()
        del tokenizer, text_encoder
        gc.collect()
        torch.cuda.empty_cache()

        timestep = torch.tensor([999], device="cuda").long()
        # VAE encoding samples a latent. Resetting the seed makes before/after
        # comparisons use the same stochastic draw.
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        print("Running one untrained-adapter translation...")
        with torch.inference_mode():
            output = CycleGAN_Turbo.forward_with_networks(
                input_tensor,
                "a2b",
                vae_encoder,
                unet,
                vae_decoder,
                scheduler,
                timestep,
                prompt_embedding,
            )
        result_label = "before_training_step0"
    else:
        print(f"Mode: trained project checkpoint ({checkpoint})")
        model = CycleGAN_Turbo(pretrained_path=str(checkpoint))
        model.eval()
        if not args.no_xformers:
            model.unet.enable_xformers_memory_efficient_attention()
        if not args.fp32:
            model.half()

        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        print("Running one trained day-to-night translation...")
        with torch.inference_mode():
            output = model(
                input_tensor,
                direction="a2b",
                caption=args.prompt,
            )
        result_label = f"after_training_{checkpoint.stem}"

    output_tensor = (output[0].float().cpu() * 0.5 + 0.5).clamp(0, 1)
    output_image = transforms.ToPILImage()(output_tensor)
    output_image = output_image.resize(input_image.size, Image.LANCZOS)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    translated_path = output_dir / f"{result_label}_{input_path.name}"
    output_image.save(translated_path)

    comparison = Image.new("RGB", (input_image.width * 2, input_image.height))
    comparison.paste(input_image, (0, 0))
    comparison.paste(output_image, (input_image.width, 0))
    comparison_path = output_dir / f"{result_label}_comparison.png"
    comparison.save(comparison_path)

    print(f"Translated image: {translated_path}")
    print(f"Day/night comparison: {comparison_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
