"""
spectrogram_dataset.py – Basic image dataset for spectrograms.

Loads spectrograms from folders organized by species: root_dir/<species>/*.png
"""
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def preprocess_image(img: Image.Image, size: tuple = (128, 128)) -> torch.Tensor:
    """Resize and convert a PIL image to a CHW float tensor in [0, 1]."""
    img = img.resize(size, Image.BILINEAR)
    img_array = np.array(img, dtype=np.float32) / 255.0
    if img_array.ndim == 3:
        return torch.from_numpy(img_array.transpose(2, 0, 1))
    return torch.from_numpy(img_array).unsqueeze(0)


class SpectrogramDataset(Dataset):
    """Dataset yielding (image_tensor, label_int) from a species-folder tree."""

    def __init__(self, root_dir: str, transform=None, image_size: tuple = (128, 128)):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_size = image_size
        self.image_paths: list[str] = []
        self.labels: list[int] = []
        self.class_to_idx: dict[str, int] = {}
        self.classes: list[str] = []
        self._load_data()

    def _load_data(self) -> None:
        class_names = sorted(
            d for d in (self.root_dir).iterdir() if d.is_dir()
        )
        self.classes = [c.name for c in class_names]
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}

        for cls_dir in class_names:
            label_idx = self.class_to_idx[cls_dir.name]
            for img_file in sorted(cls_dir.iterdir()):
                if img_file.suffix.lower() in (".png", ".jpg", ".jpeg"):
                    self.image_paths.append(str(img_file))
                    self.labels.append(label_idx)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        else:
            image = preprocess_image(image, self.image_size)

        return image, torch.tensor(label, dtype=torch.long)
