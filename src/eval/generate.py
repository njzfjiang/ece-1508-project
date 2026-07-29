import sys
from pathlib import Path
from typing import Callable, Optional

from PIL import Image
import torch
from torchvision import transforms
import torchvision.transforms.functional as TF
import importlib
from .utils import load_rgb, resize8


EXT_SRC = Path(__file__).resolve().parents[2] / "external/img2img-turbo/src"


def _output_to_pil(output: torch.Tensor) -> Image.Image:
    """Convert a model output to RGB without running FP16 ops on CPU."""
    normalized = (output.detach().float().cpu() * 0.5 + 0.5).clamp(0, 1)
    return transforms.ToPILImage()(normalized)


def generate(
    model,
    checkpoint,
    pairs,
    out,
    prompt,
    fp16=False,
    cyclegan_image_prep="resize_512x512",
):
    if str(EXT_SRC) not in sys.path:
        sys.path.insert(0, str(EXT_SRC))
    out.mkdir(parents=True, exist_ok=True)
    transform: Optional[Callable] = None

    if model == "pix2pix":
        # ensure external src is available and import dynamically to avoid static import errors
        try:
            pix2pix_mod = importlib.import_module("pix2pix_turbo")
            Pix2Pix_Turbo = getattr(pix2pix_mod, "Pix2Pix_Turbo")
        except Exception as e:
            raise RuntimeError(f"Could not import pix2pix_turbo from {EXT_SRC}: {e}")

        net = Pix2Pix_Turbo(pretrained_path=str(checkpoint)).cuda()
        net.set_eval()

    elif model == "cyclegan":
        try:
            cyclegan_mod = importlib.import_module("cyclegan_turbo")
            CycleGAN_Turbo = getattr(cyclegan_mod, "CycleGAN_Turbo")
        except Exception as e:
            raise RuntimeError(f"Could not import cyclegan_turbo from {EXT_SRC}: {e}")
        net = CycleGAN_Turbo(pretrained_path=str(checkpoint)).cuda()
        try:
            net.unet.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
        build_transform = importlib.import_module("my_utils.training_utils").build_transform
        transform = build_transform(cyclegan_image_prep)
    
    else:
        raise ValueError(f"Unknown model: {model}")
        
    net.eval()

    if fp16:
        net.half()

    with torch.inference_mode():
        for src, _ in pairs:
            img = load_rgb(src)

            if model == "pix2pix":
                x = TF.to_tensor(resize8(img)).unsqueeze(0).cuda()
                if fp16:
                    x = x.half()
                y = net(x, prompt)[0]

            else:
                if transform is None:
                    raise RuntimeError("transform is not initialized")
                x = transforms.ToTensor()(transform(img))
                x = transforms.Normalize([0.5] * 3, [0.5] * 3)(x).unsqueeze(0).cuda()
                if fp16:
                    x = x.half()
                y = net(x, direction="a2b", caption=prompt)[0]

            y = _output_to_pil(y)
            y.resize(img.size, Image.Resampling.LANCZOS).save(out / src.name)
