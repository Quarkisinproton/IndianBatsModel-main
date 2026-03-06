"""
spectrogram_with_features_dataset.py - Images + numbers = better predictions

PyTorch Dataset that serves spectrograms alongside numeric features.
The CNN sees the image, the MLP sees the features, everybody wins.
"""
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
import numpy as np
import os


class SpectrogramWithFeaturesDataset(Dataset):
    """Dataset yielding (image, features, label) tuples.
    
    Expects:
    - spectrograms in root_dir/<species>/*.png
    - features CSV with audio_file, segment_index, and numeric columns
    - image names matching "{audio_stem}_{segment_index}.png"
    """
    def __init__(self, root_dir: str, features_csv: str = None, transform=None, numeric_cols: list = None):
        self.root = Path(root_dir)
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}
        self._scan_files()

        self.features_map = {}
        self.numeric_cols = numeric_cols
        if features_csv:
            df = pd.read_csv(features_csv)
            if self.numeric_cols is None:
                self.numeric_cols = [c for c in df.columns if c not in ('json_file','audio_file','segment_index','label','start','end')]
            for _, row in df.iterrows():
                audio_path = str(row['audio_file'])
                stem = Path(audio_path).stem
                idx = int(row['segment_index'])
                key = f"{stem}_{idx}.png"
                vec = []
                for c in self.numeric_cols:
                    val = row.get(c, np.nan)
                    vec.append(float(val) if not pd.isna(val) else 0.0)
                self.features_map[key] = np.array(vec, dtype=np.float32)

        if self.numeric_cols is None:
            # at least one-dim vector
            self.numeric_cols = []

    def _scan_files(self):
        classes = sorted([d for d in os.listdir(self.root) if (self.root / d).is_dir()])
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        for c in classes:
            for fname in sorted(os.listdir(self.root / c)):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(str(self.root / c / fname))
                    self.labels.append(self.class_to_idx[c])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        key = Path(img_path).name
        feat = self.features_map.get(key)
        if feat is None:
            feat = np.zeros((len(self.numeric_cols) if len(self.numeric_cols) else 1,), dtype=np.float32)
        feat = torch.tensor(feat, dtype=torch.float32)

        return image, feat, label
