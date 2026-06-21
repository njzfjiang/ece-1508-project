"""Project-native dataset loader for EDA, evaluation, and custom experiments.

Official pix2pix-turbo and CycleGAN-Turbo training does not use this loader.
Those scripts consume the upstream-compatible views created by
scripts/prepare_img2img_turbo_data.py.
"""

import json
from pathlib import Path
from typing import Optional, Tuple, List

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class DarkDrivingFewShotDataset(Dataset):
    """
    Project-native paired DarkDriving dataset for day-to-night translation.

    Supports:
    - Full training set
    - Few-shot splits (loaded from JSON)
    - Paired (day, night) loading

    This class is intentionally paired and is not the unpaired CycleGAN-Turbo
    loader. The official CycleGAN-Turbo loader randomizes samples across
    train_A and train_B in the generated upstream-compatible dataset view.
    """

    def __init__(
        self,
        data_root: Path,
        split: str = "train",  # "train", "val", "test"
        shot: Optional[int] = None,
        seed: Optional[int] = None,
        transform=None,
        target_transform=None,
        image_size: int = 512,
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.shot = shot
        self.seed = seed
        self.image_size = image_size

        if split not in {"train", "val", "test"}:
            raise ValueError(
                f"Unsupported split: {split!r}. Expected train, val, or test."
            )
        if (shot is None) != (seed is None):
            raise ValueError(
                "shot and seed must either both be provided or both be omitted"
            )
        if shot is not None and split != "train":
            raise ValueError("Few-shot split files can only be used with split='train'")
        if shot is not None and shot < 1:
            raise ValueError("shot must be a positive integer")
        if seed is not None and seed < 1:
            raise ValueError(
                "seed must be a positive integer matching directories such as seed1"
            )

        # Default transforms
        if transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ]
            )
        else:
            self.transform = transform

        if target_transform is None:
            self.target_transform = transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ]
            )
        else:
            self.target_transform = target_transform

        # Load pairs based on split and shot
        self.pairs = self._load_pairs()

    def _load_pairs(self) -> List[Tuple[Path, Path]]:
        """Load day-night pairs based on split and shot configuration."""
        if self.shot is not None and self.seed is not None:
            # Few-shot mode: load from split JSON
            split_dir = (
                self.data_root
                / "splits"
                / "fewshot"
                / f"{self.shot}shot"
                / f"seed{self.seed}"
            )
            split_file = split_dir / "split.json"

            if not split_file.exists():
                raise FileNotFoundError(
                    f"Split file not found: {split_file}\n"
                    f"Make sure to run prepare_fewshot_splits.py first with --shot_levels {self.shot}"
                )

            try:
                with open(split_file) as f:
                    split_data = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON format in split file {split_file}: {e}")

            # Get file names from the split
            day_names = split_data["train_day"]
            night_names = split_data["train_night"]
            if len(day_names) != len(night_names):
                raise ValueError(
                    f"Split file has mismatched day/night counts: {split_file}"
                )
            if (
                split_data.get("shot") != self.shot
                or split_data.get("seed") != self.seed
            ):
                raise ValueError(
                    f"Split metadata does not match requested shot={self.shot}, seed={self.seed}: "
                    f"{split_file}"
                )

            # Construct full paths
            day_dir = self.data_root / "day2night" / self.split / "day"
            night_dir = self.data_root / "day2night" / self.split / "night"

            pairs = []
            for day_name, night_name in zip(day_names, night_names):
                day_path = day_dir / day_name
                night_path = night_dir / night_name
                if not day_path.exists() or not night_path.exists():
                    raise FileNotFoundError(
                        f"Pair not found: {day_path} / {night_path}"
                    )
                pairs.append((day_path, night_path))

            return pairs
        else:
            # Full dataset mode: use all files in the split directory
            day_dir = self.data_root / "day2night" / self.split / "day"
            night_dir = self.data_root / "day2night" / self.split / "night"

            if not day_dir.is_dir() or not night_dir.is_dir():
                raise FileNotFoundError(
                    f"Processed split directories not found: {day_dir} / {night_dir}"
                )

            day_files = {
                f.name: f
                for f in day_dir.iterdir()
                if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
            }
            night_files = {
                f.name: f
                for f in night_dir.iterdir()
                if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
            }

            shared_names = sorted(day_files.keys() & night_files.keys())
            if not shared_names:
                raise ValueError(
                    f"No matching day/night image filenames found in {day_dir} and {night_dir}"
                )
            return [(day_files[name], night_files[name]) for name in shared_names]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        day_path, night_path = self.pairs[idx]

        day_image = Image.open(day_path).convert("RGB")
        night_image = Image.open(night_path).convert("RGB")

        day_tensor = self.transform(day_image)
        night_tensor = self.target_transform(night_image)

        return {
            "day": day_tensor,
            "night": night_tensor,
            "day_path": str(day_path),
            "night_path": str(night_path),
        }


def get_dataloaders(
    data_root: Path,
    batch_size: int = 4,
    image_size: int = 512,
    shot: Optional[int] = None,
    seed: Optional[int] = None,
    num_workers: int = 4,
):
    """Create train, val, and test dataloaders."""
    from torch.utils.data import DataLoader

    # For few-shot experiments, train uses the few-shot split, val and test use full sets
    train_dataset = DarkDrivingFewShotDataset(
        data_root, split="train", shot=shot, seed=seed, image_size=image_size
    )
    val_dataset = DarkDrivingFewShotDataset(
        data_root, split="val", shot=None, seed=None, image_size=image_size
    )
    test_dataset = DarkDrivingFewShotDataset(
        data_root, split="test", shot=None, seed=None, image_size=image_size
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader
