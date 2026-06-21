"""Evaluation metrics for model assessment."""

# src/eval/metrics.py
"""
Evaluation metrics for day-to-night translation.
- SSIM: Structural similarity (structure preservation)
- LPIPS: Perceptual similarity (human-aligned quality)
- CLIP Vision Similarity: Semantic alignment (image-to-image)
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
from skimage.metrics import structural_similarity as ssim_sk
import lpips
import clip
import numpy as np


class MetricsCalculator:
    def __init__(
        self,
        device: str = "cuda",
        lpips_backbone: str = "alex",
        clip_model_name: str = "ViT-B/32",
    ):
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self.device = device

        # LPIPS
        self.lpips_model = lpips.LPIPS(net=lpips_backbone).to(device)
        self.lpips_model.eval()

        # CLIP (image encoder only)
        self.clip_model, self.clip_preprocess = clip.load(
            clip_model_name, device=device
        )
        self.clip_model.eval()

    @staticmethod
    def _as_batch(images: torch.Tensor) -> torch.Tensor:
        if images.dim() == 3:
            images = images.unsqueeze(0)
        if images.dim() != 4 or images.shape[1] != 3:
            raise ValueError("Expected an RGB tensor with shape CHW or BCHW")
        return images

    @staticmethod
    def _to_unit_range(images: torch.Tensor) -> torch.Tensor:
        images = images.detach().float()
        if images.min().item() < 0:
            images = (images + 1) / 2
        return images.clamp(0, 1)

    @staticmethod
    def _validate_pair(img1: torch.Tensor, img2: torch.Tensor):
        if img1.shape != img2.shape:
            raise ValueError(
                f"Image shapes must match, received {img1.shape} and {img2.shape}"
            )

    def compute_ssim(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
        data_range: float = 1.0,
    ) -> float:
        """
        Compute SSIM between two images.
        Args:
            img1, img2: (C, H, W) or (B, C, H, W), values in [0, 1]
        Returns:
            SSIM score (float)
        """
        img1 = self._to_unit_range(self._as_batch(img1))
        img2 = self._to_unit_range(self._as_batch(img2))
        self._validate_pair(img1, img2)

        values = []
        for first, second in zip(img1, img2):
            first_np = first.cpu().numpy().transpose(1, 2, 0)
            second_np = second.cpu().numpy().transpose(1, 2, 0)
            min_side = min(first_np.shape[:2])
            win_size = min(11, min_side if min_side % 2 else min_side - 1)
            if win_size < 3:
                raise ValueError("SSIM requires images at least 3x3 pixels")
            values.append(
                ssim_sk(
                    first_np,
                    second_np,
                    win_size=win_size,
                    data_range=data_range,
                    channel_axis=2,
                )
            )
        return float(np.mean(values))

    def compute_lpips(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
    ) -> float:
        """
        Compute LPIPS between two images.
        Args:
            img1, img2: (B, C, H, W) or (C, H, W), values in [-1, 1] or [0, 1]
        Returns:
            LPIPS distance (float, lower is better)
        """
        img1 = self._as_batch(img1)
        img2 = self._as_batch(img2)
        self._validate_pair(img1, img2)
        img1 = self._to_unit_range(img1) * 2 - 1
        img2 = self._to_unit_range(img2) * 2 - 1

        with torch.no_grad():
            # LPIPS expects NCHW, values in [-1, 1]
            dist = self.lpips_model(img1.to(self.device), img2.to(self.device))

        return float(dist.mean().item())

    def compute_clip_similarity(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
    ) -> float:
        """
        Compute CLIP vision cosine similarity between two images.
        Uses ONLY the image encoder (no text), ensuring fair comparison.
        Args:
            img1, img2: (C, H, W) or (B, C, H, W), values in [0, 1]
        Returns:
            Cosine similarity (float, higher is better)
        """
        img1 = self._to_unit_range(self._as_batch(img1))
        img2 = self._to_unit_range(self._as_batch(img2))
        self._validate_pair(img1, img2)

        # Convert to PIL for CLIP preprocess
        # CLIP expects RGB images normalized with its own mean/std
        batch_size = img1.shape[0]
        features = []

        with torch.no_grad():
            for i in range(batch_size):
                # To PIL
                img1_pil = transforms.ToPILImage()(img1[i].cpu())
                img2_pil = transforms.ToPILImage()(img2[i].cpu())

                # Preprocess
                img1_processed = (
                    self.clip_preprocess(img1_pil).unsqueeze(0).to(self.device)
                )
                img2_processed = (
                    self.clip_preprocess(img2_pil).unsqueeze(0).to(self.device)
                )

                # Encode
                emb1 = self.clip_model.encode_image(img1_processed)
                emb2 = self.clip_model.encode_image(img2_processed)

                # Normalize
                emb1 = F.normalize(emb1, p=2, dim=-1)
                emb2 = F.normalize(emb2, p=2, dim=-1)

                # Similarity
                sim = (emb1 @ emb2.T).item()
                features.append(sim)

        return float(np.mean(features))

    def evaluate_pair(
        self,
        generated: torch.Tensor,
        ground_truth: torch.Tensor,
    ) -> dict:
        """
        Evaluate a single (generated, ground_truth) pair.
        Returns dict with SSIM, LPIPS, CLIP similarity.
        """
        return {
            "ssim": self.compute_ssim(generated, ground_truth),
            "lpips": self.compute_lpips(generated, ground_truth),
            "clip_similarity": self.compute_clip_similarity(generated, ground_truth),
        }
