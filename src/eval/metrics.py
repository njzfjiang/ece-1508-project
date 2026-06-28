from skimage.metrics import structural_similarity as ssim_sk
import numpy as np
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
    ):
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self.device = device

        import clip
        import lpips

        self.lpips_model = lpips.LPIPS(net=lpips_backbone).to(device)
        self.lpips_model.eval()

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

    def extract_clip_features(self, images: torch.Tensor) -> np.ndarray:
        """Extract normalized CLIP image features for a CHW or BCHW tensor."""
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

    @staticmethod
    def compute_cmmd_from_features(
        generated_features: np.ndarray,
        ground_truth_features: np.ndarray,
        sigma: float | None = None,
    ) -> float:
        """Compute squared Gaussian-kernel MMD over CLIP image features."""
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

        combined = np.concatenate([generated_features, ground_truth_features], axis=0)
        if sigma is None:
            distances = _squared_distances(combined, combined)
            positive = distances[distances > 0]
            sigma = float(np.sqrt(np.median(positive))) if positive.size else 1.0
        if sigma <= 0:
            raise ValueError("CMMD sigma must be positive")

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
        return float(k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean())

    def compute_cmmd(
        self,
        generated: torch.Tensor,
        ground_truth: torch.Tensor,
        sigma: float | None = None,
    ) -> float:
        """Compute CLIP Maximum Mean Discrepancy for two image batches."""
        generated_features = self.extract_clip_features(generated)
        ground_truth_features = self.extract_clip_features(ground_truth)
        return self.compute_cmmd_from_features(
            generated_features, ground_truth_features, sigma=sigma
        )

    def evaluate_pair(
        self,
        generated: torch.Tensor,
        ground_truth: torch.Tensor,
    ) -> dict:
        """
        Evaluate a single (generated, ground_truth) pair.
        Returns dict with SSIM, LPIPS, CLIP similarity, and CMMD.
        """
        return {
            "cmmd": self.compute_cmmd(generated, ground_truth),
            "ssim": self.compute_ssim(generated, ground_truth),
            "lpips": self.compute_lpips(generated, ground_truth),
            "clip_similarity": self.compute_clip_similarity(generated, ground_truth),
        }

def _squared_distances(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first_norm = np.sum(first * first, axis=1, keepdims=True)
    second_norm = np.sum(second * second, axis=1, keepdims=True).T
    distances = first_norm + second_norm - 2.0 * first @ second.T
    return np.maximum(distances, 0.0)
