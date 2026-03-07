"""
spectrogram_dataset.py - Legacy dataset (see MainShitz version for newer code)

Basic spectrogram loader. Kept around but you probably want the other one.
"""
from torch.utils.data import Dataset
from PIL import Image
import os


class SpectrogramDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self._load_data()

    def _load_data(self):
        for label in os.listdir(self.root_dir):
            species_dir = os.path.join(self.root_dir, label)
            if os.path.isdir(species_dir):
                for img_file in os.listdir(species_dir):
                    img_path = os.path.join(species_dir, img_file)
                    self.image_paths.append(img_path)
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label