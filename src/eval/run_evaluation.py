#!/usr/bin/env python
"""Formal held-out evaluation runner for DarkDriving experiments."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Iterable

import torch
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_SRC = PROJECT_ROOT / "external" / "img2img-turbo" / "src"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "base.yaml"
DEFAULT_PROMPT = "a driving scene at night"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.metrics import MetricsCalculator


def parse_checkpoint_step(path: Path) -> int:
    match = re.fullmatch(r"model_(\d+)\.pkl", path.name)
    if not match:
        raise ValueError(f"Checkpoint does not match model_<step>.pkl: {path}")
    return int(match.group(1))


def resolve_checkpoint(
    results_root: Path,
    model: str,
    shot: int,
    seed: int,
    checkpoint: Path | None = None,
) -> Path:
    if checkpoint is not None:
        checkpoint = checkpoint.resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        return checkpoint

    checkpoint_dir = (
        results_root / model / f"{shot}shot" / f"seed{seed}" / "checkpoints"
    )
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    candidates = []
    for path in checkpoint_dir.glob("model_*.pkl"):
        try:
            candidates.append((parse_checkpoint_step(path), path))
        except ValueError:
            continue
    if not candidates:
        raise FileNotFoundError(f"No model_<step>.pkl checkpoints in {checkpoint_dir}")
    return max(candidates, key=lambda item: item[0])[1].resolve()


def find_test_pairs(processed_root: Path, limit: int | None = None) -> list[tuple[Path, Path]]:
    day_dir = processed_root / "day2night" / "test" / "day"
    night_dir = processed_root / "day2night" / "test" / "night"
    if not day_dir.is_dir() or not night_dir.is_dir():
        raise FileNotFoundError(f"Held-out test directories not found: {day_dir} / {night_dir}")

    day_files = {
        path.name: path
        for path in day_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }
    night_files = {
        path.name: path
        for path in night_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }
    names = sorted(day_files.keys() & night_files.keys())
    if not names:
        raise ValueError(f"No paired test images found in {day_dir} and {night_dir}")
    if limit is not None and limit > 0:
        names = names[:limit]
    return [(day_files[name], night_files[name]) for name in names]


def generate_outputs(
    model: str,
    checkpoint: Path,
    pairs: Iterable[tuple[Path, Path]],
    output_dir: Path,
    prompt: str = DEFAULT_PROMPT,
    use_fp16: bool = False,
    cyclegan_image_prep: str = "resize_512x512",
) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("Checkpoint generation requires CUDA because upstream models call .cuda().")
    if str(EXTERNAL_SRC) not in sys.path:
        sys.path.insert(0, str(EXTERNAL_SRC))

    output_dir.mkdir(parents=True, exist_ok=True)
    if model == "pix2pix":
        from pix2pix_turbo import Pix2Pix_Turbo

        generator = Pix2Pix_Turbo(pretrained_path=str(checkpoint))
        generator.set_eval()
        if use_fp16:
            generator.half()

        with torch.no_grad():
            for day_path, _ in pairs:
                input_image = _load_rgb(day_path)
                resized = _resize_to_multiple_of_eight(input_image)
                control = TF.to_tensor(resized).unsqueeze(0).cuda()
                if use_fp16:
                    control = control.half()
                output = generator(control, prompt)
                output_pil = transforms.ToPILImage()(output[0].cpu() * 0.5 + 0.5)
                output_pil = output_pil.resize(input_image.size, Image.LANCZOS)
                output_pil.save(output_dir / day_path.name)
    elif model == "cyclegan":
        from cyclegan_turbo import CycleGAN_Turbo
        from my_utils.training_utils import build_transform

        generator = CycleGAN_Turbo(pretrained_path=str(checkpoint))
        generator.eval()
        try:
            generator.unet.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
        if use_fp16:
            generator.half()
        transform = build_transform(cyclegan_image_prep)

        with torch.no_grad():
            for day_path, _ in pairs:
                input_image = _load_rgb(day_path)
                input_tensor = transforms.ToTensor()(transform(input_image))
                input_tensor = transforms.Normalize([0.5], [0.5])(input_tensor)
                input_tensor = input_tensor.unsqueeze(0).cuda()
                if use_fp16:
                    input_tensor = input_tensor.half()
                output = generator(input_tensor, direction="a2b", caption=prompt)
                output_pil = transforms.ToPILImage()(output[0].cpu() * 0.5 + 0.5)
                output_pil = output_pil.resize(input_image.size, Image.LANCZOS)
                output_pil.save(output_dir / day_path.name)
    else:
        raise ValueError(f"Unsupported model: {model}")

    return output_dir


def evaluate_generated_pairs(
    pairs: list[tuple[Path, Path]],
    generated_dir: Path,
    output_dir: Path,
    metrics_calculator,
    requested_metrics: list[str],
    metadata: dict,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    generated_features = []
    target_features = []

    for day_path, night_path in pairs:
        generated_path = generated_dir / day_path.name
        if not generated_path.is_file():
            raise FileNotFoundError(f"Generated image missing for {day_path.name}: {generated_path}")

        generated = _image_to_tensor(generated_path)
        target = _image_to_tensor(night_path)
        row = {
            "filename": day_path.name,
            "day_path": str(day_path),
            "night_path": str(night_path),
            "generated_path": str(generated_path),
        }
        if "ssim" in requested_metrics:
            row["ssim"] = metrics_calculator.compute_ssim(generated, target)
        if "lpips" in requested_metrics:
            row["lpips"] = metrics_calculator.compute_lpips(generated, target)
        if "clip_similarity" in requested_metrics:
            row["clip_similarity"] = metrics_calculator.compute_clip_similarity(
                generated, target
            )
        if "cmmd" in requested_metrics:
            generated_features.append(metrics_calculator.extract_clip_features(generated)[0])
            target_features.append(metrics_calculator.extract_clip_features(target)[0])
        rows.append(row)

    if "cmmd" in requested_metrics:
        cmmd = metrics_calculator.compute_cmmd_from_features(
            generated_features, target_features
        )
    else:
        cmmd = None

    per_sample_path = output_dir / "per_sample_metrics.csv"
    fieldnames = ["filename", "day_path", "night_path", "generated_path"] + [
        metric for metric in ("ssim", "lpips", "clip_similarity") if metric in requested_metrics
    ]
    with per_sample_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = _summarize_rows(rows, requested_metrics)
    if cmmd is not None:
        summary["cmmd"] = cmmd
    result = {
        "metadata": metadata,
        "num_samples": len(rows),
        "metrics": summary,
        "per_sample_metrics": str(per_sample_path),
        "generated_dir": str(generated_dir),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                **metadata,
                "filenames": [day_path.name for day_path, _ in pairs],
                "generated_dir": str(generated_dir),
                "per_sample_metrics": str(per_sample_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["pix2pix", "cyclegan"])
    parser.add_argument("--shot", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--generated-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--results-root", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--processed-root", type=Path)
    parser.add_argument("--test-samples", type=int)
    parser.add_argument("--metrics", nargs="+")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--use-fp16", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = OmegaConf.load(args.config.resolve())
    eval_config = config.get("eval", {})
    processed_root = (
        args.processed_root
        or PROJECT_ROOT / str(OmegaConf.select(config, "data.root", default="data/processed"))
    ).resolve()
    results_root = args.results_root.resolve()
    requested_metrics = args.metrics or list(
        OmegaConf.select(config, "eval.metrics", default=["ssim", "lpips", "clip_similarity", "cmmd"])
    )
    test_samples = args.test_samples
    if test_samples is None:
        test_samples = int(OmegaConf.select(config, "eval.test_samples", default=200))

    checkpoint = resolve_checkpoint(
        results_root=results_root,
        model=args.model,
        shot=args.shot,
        seed=args.seed,
        checkpoint=args.checkpoint,
    )
    evaluation_root = PROJECT_ROOT / str(
        OmegaConf.select(config, "eval.output_dir", default="results/evaluation")
    )
    output_dir = (
        args.output_dir
        or evaluation_root / args.model / f"{args.shot}shot" / f"seed{args.seed}"
    ).resolve()
    generated_dir = args.generated_dir or output_dir / "generated"
    pairs = find_test_pairs(processed_root, limit=test_samples)

    if args.generated_dir is None:
        torch.cuda.set_device(args.gpu)
        generate_outputs(
            model=args.model,
            checkpoint=checkpoint,
            pairs=pairs,
            output_dir=generated_dir,
            prompt=args.prompt,
            use_fp16=args.use_fp16,
            cyclegan_image_prep=str(
                OmegaConf.select(config, "cyclegan_turbo.val_image_prep", default="resize_512x512")
            ),
        )

    calculator = MetricsCalculator(
        device="cuda" if torch.cuda.is_available() else "cpu",
        lpips_backbone=str(eval_config.get("lpips_backbone", "alex")),
        clip_model_name=str(eval_config.get("clip_model", "ViT-B/32")),
    )
    result = evaluate_generated_pairs(
        pairs=pairs,
        generated_dir=generated_dir.resolve(),
        output_dir=output_dir,
        metrics_calculator=calculator,
        requested_metrics=requested_metrics,
        metadata={
            "model": args.model,
            "shot": args.shot,
            "seed": args.seed,
            "checkpoint": str(checkpoint),
            "processed_root": str(processed_root),
            "metrics": requested_metrics,
        },
    )
    print(json.dumps(result["metrics"], indent=2))
    return 0


def _load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def _resize_to_multiple_of_eight(image: Image.Image) -> Image.Image:
    width = image.width - image.width % 8
    height = image.height - image.height % 8
    if width <= 0 or height <= 0:
        raise ValueError(f"Image is too small for img2img-turbo inference: {image.size}")
    return image.resize((width, height), Image.LANCZOS)


def _image_to_tensor(path: Path) -> torch.Tensor:
    return transforms.ToTensor()(_load_rgb(path))


def _summarize_rows(rows: list[dict], requested_metrics: list[str]) -> dict:
    summary = {}
    for metric in ("ssim", "lpips", "clip_similarity"):
        if metric not in requested_metrics:
            continue
        values = [float(row[metric]) for row in rows]
        mean = sum(values) / len(values)
        variance = (
            sum((value - mean) ** 2 for value in values) / (len(values) - 1)
            if len(values) > 1
            else 0.0
        )
        summary[metric] = {
            "mean": mean,
            "std": variance**0.5,
            "min": min(values),
            "max": max(values),
        }
    return summary


if __name__ == "__main__":
    raise SystemExit(main())
