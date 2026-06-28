from pathlib import Path
from typing import Optional
import yaml
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class DarkDrivingDataset(Dataset):
    def __init__(
        self,
        root: Path,
        mode: str, # "train" or "test"
        shot: Optional[int] = None,
        seed: Optional[int] = None,
        image_size: int = 512,
    ):
        self.root = Path(root)
        self.mode = mode
        self.shot = shot
        self.seed = seed

        self.tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])

        self.pairs = self._load_pairs()

    def _load_pairs(self):
        if self.mode == "test":
            day_dir = self.root / "test" / "test_A"
            night_dir = self.root / "test" / "test_B"
        else:
            base = self.root / f"{self.shot}shot" / f"seed{self.seed}"
            day_dir = base / "train_A"
            night_dir = base / "train_B"

        return self._collect_pairs(day_dir, night_dir)

    def _collect_pairs(self, day_dir: Path, night_dir: Path):
        day_files = {
            f.name: f
            for f in day_dir.iterdir()
            if f.suffix.lower() in IMAGE_EXTENSIONS
        }
        night_files = {
            f.name: f
            for f in night_dir.iterdir()
            if f.suffix.lower() in IMAGE_EXTENSIONS
        }

        names = sorted(day_files.keys() & night_files.keys())
        return [(day_files[n], night_files[n]) for n in names]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        d, n = self.pairs[idx]

        d = Image.open(d).convert("RGB")
        n = Image.open(n).convert("RGB")

        return {
            "day": self.tf(d),
            "night": self.tf(n),
            "day_path": str(d),
            "night_path": str(n),
        }

def load_config(cfg_path: str | Path):
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)
    
def get_dataloaders(
    shot: int | None = None,
    seed: int | None = None,
):
    cfg = load_config(PROJECT_ROOT / "configs" / "base.yaml")

    data_cfg = cfg["data"]
    root = Path(data_cfg["root"])
    image_size = data_cfg["image_size"]
    batch_size = data_cfg["batch_size"]
    num_workers = data_cfg["num_workers"]

    train_set = DarkDrivingDataset(
        root=root,
        mode="train",
        shot=shot,
        seed=seed,
        image_size=image_size,
    )

    test_set = DarkDrivingDataset(
        root=root,
        mode="test",
        image_size=image_size,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader