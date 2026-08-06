from skimage.metrics import structural_similarity as ssim_sk
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

"""Evaluation metrics for day-to-night translation.
- SSIM: structural preservation
- LPIPS: perceptual similarity
- CLIP Vision Similarity: image-encoder cosine similarity
- CMMD: CLIP-feature distributional similarity
"""

class MetricsCalculator:
    def __init__(
        self,
        device: str = "cuda",
        lpips_backbone: str = "alex",
        clip_model_name: str = "ViT-B/32",
        requested_metrics: set[str] | None = None,
    ):
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self.device = device
        self.requested_metrics = requested_metrics or {
            "ssim",
            "lpips",
            "clip_similarity",
            "cmmd",
        }
        self.lpips_model = None
        self.clip_model = None
        self.clip_preprocess = None

        if "lpips" in self.requested_metrics:
            import lpips

            self.lpips_model = lpips.LPIPS(net=lpips_backbone).to(device)
            self.lpips_model.eval()

        if "clip_similarity" in self.requested_metrics:
            import clip

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
            # Odd window size is required for SSIM, so we take the largest odd number <= min_side, but at least 3
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
        if self.lpips_model is None:
            raise RuntimeError("LPIPS was not initialized for this evaluation")
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
        img1_features = self.extract_clip_features(img1)
        img2_features = self.extract_clip_features(img2)
        return self.clip_similarity_from_features(img1_features, img2_features)

    @staticmethod
    def clip_similarity_from_features(
        img1_features: np.ndarray,
        img2_features: np.ndarray,
    ) -> float:
        """Compute mean cosine similarity from normalized CLIP features."""
        img1_features = np.asarray(img1_features, dtype=np.float64)
        img2_features = np.asarray(img2_features, dtype=np.float64)
        if img1_features.shape != img2_features.shape or img1_features.ndim != 2:
            raise ValueError("CLIP feature arrays must have the same 2D shape")
        return float(np.mean(np.sum(img1_features * img2_features, axis=1)))

    def extract_clip_features(self, images: torch.Tensor) -> np.ndarray:
        """Extract normalized CLIP image features for a CHW or BCHW tensor."""
        if self.clip_model is None or self.clip_preprocess is None:
            raise RuntimeError("CLIP was not initialized for this evaluation")
        images = self._to_unit_range(self._as_batch(images))
        features = []

        with torch.no_grad():
            for image in images:
                image_pil = transforms.ToPILImage()(image.cpu())
                processed = self.clip_preprocess(image_pil).unsqueeze(0).to(self.device)
                embedding = self.clip_model.encode_image(processed)
                embedding = F.normalize(embedding, p=2, dim=-1)
                features.append(embedding.cpu().float().numpy()[0])

        return np.stack(features, axis=0)

    def evaluate_pair(
        self,
        generated: torch.Tensor,
        ground_truth: torch.Tensor,
    ) -> dict:
        """
        Evaluate per-image metrics for one generated/ground-truth pair.

        CMMD is intentionally excluded because it is a distribution-level metric.
        """
        return {
            "ssim": self.compute_ssim(generated, ground_truth),
            "lpips": self.compute_lpips(generated, ground_truth),
            "clip_similarity": self.compute_clip_similarity(generated, ground_truth),
        }


class CMMDCalculator:
    """Official CMMD definition implemented with the released OpenAI CLIP weights.

    The reference implementation uses ViT-L/14@336px embeddings, a Gaussian
    kernel with sigma=10, the biased/minimum-variance MMD estimator, and a 1000x
    reporting scale. CMMD is computed once over complete image sets, not per pair.
    """

    def __init__(
        self,
        device: str = "cuda",
        clip_model_name: str = "ViT-L/14@336px",
        batch_size: int = 32,
        sigma: float = 10.0,
        scale: float = 1000.0,
    ):
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        if batch_size <= 0:
            raise ValueError("CMMD batch size must be positive")
        if sigma <= 0:
            raise ValueError("CMMD sigma must be positive")
        if scale <= 0:
            raise ValueError("CMMD scale must be positive")

        import clip

        self.device = device
        self.clip_model_name = clip_model_name
        self.batch_size = batch_size
        self.sigma = sigma
        self.scale = scale
        self.clip_model, self.clip_preprocess = clip.load(
            clip_model_name, device=device
        )
        self.clip_model.eval()

    @property
    def configuration(self) -> dict[str, object]:
        return {
            "clip_model": self.clip_model_name,
            "sigma": self.sigma,
            "scale": self.scale,
            "estimator": "biased_minimum_variance",
        }

    def extract_path_features(self, paths: list[Path]) -> np.ndarray:
        if not paths:
            raise ValueError("CMMD requires at least one image path")

        feature_batches = []
        with torch.no_grad():
            for start in range(0, len(paths), self.batch_size):
                processed = []
                for path in paths[start : start + self.batch_size]:
                    with Image.open(path) as image:
                        processed.append(self.clip_preprocess(image.convert("RGB")))
                batch = torch.stack(processed).to(self.device)
                embeddings = self.clip_model.encode_image(batch)
                embeddings = F.normalize(embeddings, p=2, dim=-1)
                feature_batches.append(embeddings.cpu().float().numpy())

        return np.concatenate(feature_batches, axis=0)

    @staticmethod
    def compute_from_features(
        generated_features: np.ndarray,
        ground_truth_features: np.ndarray,
        sigma: float = 10.0,
        scale: float = 1000.0,
    ) -> float:
        generated_features = np.asarray(generated_features, dtype=np.float64)
        ground_truth_features = np.asarray(ground_truth_features, dtype=np.float64)
        if generated_features.ndim != 2 or ground_truth_features.ndim != 2:
            raise ValueError("CMMD features must be 2D arrays")
        if generated_features.shape[1] != ground_truth_features.shape[1]:
            raise ValueError(
                "Generated and ground-truth CLIP features must have the same width"
            )
        if len(generated_features) == 0 or len(ground_truth_features) == 0:
            raise ValueError("CMMD requires at least one generated and target feature")
        if sigma <= 0 or scale <= 0:
            raise ValueError("CMMD sigma and scale must be positive")

        gamma = 1.0 / (2.0 * sigma**2)
        k_xx = np.exp(
            -gamma * _squared_distances(generated_features, generated_features)
        )
        k_yy = np.exp(
            -gamma * _squared_distances(ground_truth_features, ground_truth_features)
        )
        k_xy = np.exp(
            -gamma * _squared_distances(generated_features, ground_truth_features)
        )
        return float(scale * (k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean()))

    def compute_from_paths(
        self,
        generated_paths: list[Path],
        ground_truth_paths: list[Path],
    ) -> float:
        if len(generated_paths) != len(ground_truth_paths):
            raise ValueError("CMMD image sets must have the same size")
        return self.compute_from_features(
            self.extract_path_features(generated_paths),
            self.extract_path_features(ground_truth_paths),
            sigma=self.sigma,
            scale=self.scale,
        )


def _squared_distances(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first_norm = np.sum(first * first, axis=1, keepdims=True)
    second_norm = np.sum(second * second, axis=1, keepdims=True).T
    distances = first_norm + second_norm - 2.0 * first @ second.T
    return np.maximum(distances, 0.0)
