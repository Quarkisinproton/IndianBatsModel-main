"""
spectrogram_with_features_dataset.py – Images + numeric features dataset.

Serves (image, feature_vector, label) tuples for the fused CNN model.
Expects spectrograms in root_dir/<species>/*.png and a features CSV.
"""
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class SpectrogramWithFeaturesDataset(Dataset):
    """Dataset yielding (image, features, label) tuples.

    Image names are expected to match ``{audio_stem}_{segment_index}.png``.
    """

    def __init__(
        self,
        root_dir: str,
        features_csv: Optional[str] = None,
        transform=None,
        numeric_cols: Optional[list] = None,
    ):
        self.root = Path(root_dir)
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        self.image_paths: list[str] = []
        self.labels: list[int] = []
        self.class_to_idx: dict[str, int] = {}
        self.classes: list[str] = []
        self._scan_files()

        self.features_map: dict[str, np.ndarray] = {}
        self.numeric_cols: list[str] = numeric_cols or []

        if features_csv:
            df = pd.read_csv(features_csv)
            if not self.numeric_cols:
                skip = {"json_file", "audio_file", "segment_index", "label", "start", "end"}
                self.numeric_cols = [c for c in df.columns if c not in skip]
            for _, row in df.iterrows():
                audio_path = str(row["audio_file"])
                stem = Path(audio_path).stem
                idx = int(row["segment_index"])
                key = f"{stem}_{idx}.png"
                vec = [
                    float(row.get(c, np.nan)) if not pd.isna(row.get(c, np.nan)) else 0.0
                    for c in self.numeric_cols
                ]
                self.features_map[key] = np.array(vec, dtype=np.float32)

    def _scan_files(self) -> None:
        classes = sorted(d for d in self.root.iterdir() if d.is_dir())
        self.classes = [c.name for c in classes]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        for cls_dir in classes:
            label_idx = self.class_to_idx[cls_dir.name]
            for fname in sorted(cls_dir.iterdir()):
                if fname.suffix.lower() in (".png", ".jpg", ".jpeg"):
                    self.image_paths.append(str(fname))
                    self.labels.append(label_idx)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        key = Path(img_path).name
        feat = self.features_map.get(key)
        if feat is None:
            dim = len(self.numeric_cols) if self.numeric_cols else 1
            feat = np.zeros((dim,), dtype=np.float32)
        feat = torch.tensor(feat, dtype=torch.float32)

        return image, feat, label
