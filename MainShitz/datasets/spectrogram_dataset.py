"""
spectrogram_dataset.py - Basic image dataset for spectrograms

Simple dataset that loads spectrograms from folders organized by species.
No fancy features, just images and labels. Sometimes simple is good.
"""
from torch.utils.data import Dataset
from PIL import Image
import torch
import numpy as np
import os


def preprocess_image(img, size=(128, 128)):
    """Resize and tensorize. The boring but necessary stuff."""
    img = img.resize(size, Image.BILINEAR)
    img_array = np.array(img, dtype=np.float32) / 255.0
    # HWC to CHW caus PyTorch is picky abt dimensions
    if img_array.ndim == 3:
        img_tensor = torch.from_numpy(img_array.transpose(2, 0, 1))
    else:
        img_tensor = torch.from_numpy(img_array).unsqueeze(0)
    return img_tensor


class SpectrogramDataset(Dataset):
    def __init__(self, root_dir, transform=None, image_size=(128, 128)):
        self.root_dir = root_dir
        self.transform = transform
        self.image_size = image_size
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}
        self.classes = []
        self._load_data()

    def _load_data(self):
        # First, get all class names (folder names) and sort for consistency
        class_names = sorted([d for d in os.listdir(self.root_dir) 
                              if os.path.isdir(os.path.join(self.root_dir, d))])
        self.classes = class_names
        self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        
        for label in class_names:
            species_dir = os.path.join(self.root_dir, label)
            for img_file in os.listdir(species_dir):
                img_path = os.path.join(species_dir, img_file)
                self.image_paths.append(img_path)
                self.labels.append(self.class_to_idx[label])  # Store integer index

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        else:
            image = preprocess_image(image, self.image_size)

        return image, torch.tensor(label, dtype=torch.long)
