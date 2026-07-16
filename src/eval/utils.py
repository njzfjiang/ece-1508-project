from pathlib import Path
import re
import numpy as np
from PIL import Image
from torchvision import transforms


EXT = {".jpg", ".jpeg", ".png"}


def resolve_path(root, p):
    p = Path(p)
    return p if p.is_absolute() else root / p


def get_step(p):
    m = re.match(r"model_(\d+)\.pkl", p.name)
    return int(m.group(1)) if m else -1


def find_checkpoint(root, shot, seed, checkpoint=None):
    if checkpoint:
        return Path(checkpoint).resolve()

    folder = root / f"{shot}shot" / f"seed{seed}" / "checkpoints"
    files = list(folder.glob("model_*.pkl"))
    if not files:
        raise FileNotFoundError(folder)

    return max(files, key=get_step).resolve()


def find_pairs(root, limit=None):
    A = root / "test_A"
    B = root / "test_B"
    names = sorted(p.name for p in A.iterdir() if p.suffix.lower() in EXT)
    pairs = [(A / n, B / n) for n in names if (B / n).exists()]

    return pairs[:limit] if limit else pairs


def load_rgb(p):
    return Image.open(p).convert("RGB")


def tensor(p):
    return transforms.ToTensor()(load_rgb(p))


def resize8(img):
    w = img.width - img.width % 8
    h = img.height - img.height % 8

    return img.resize((w, h), Image.Resampling.LANCZOS)

def summarize(rows, metrics):

    result = {}

    for name in ["ssim", "lpips", "clip_similarity"]:
        if name not in metrics:
            continue

        values = [float(x[name]) for x in rows]

        result[name] = {
            "mean": np.mean(values),
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values),
        }

    return result

def image_to_tensor(img):
    return transforms.ToTensor()(img)