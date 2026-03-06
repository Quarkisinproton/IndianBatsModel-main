"""
utils.py - Legacy utilities (mostly superseded by MainShitz/utils.py)

Kept for backwards compatibility. Consider using MainShitz version instead.
"""
import os
import torch
from PIL import Image


def load_model(model_path):
    """Load and set model to eval mode."""
    model = torch.load(model_path)
    model.eval()
    return model


def save_model(model, model_path):
    """Pickle the model to disk."""
    torch.save(model, model_path)


def load_data(data_path):
    """Bulk load images. Hope you have RAM."""
    images = []
    labels = []
    for label in os.listdir(data_path):
        label_path = os.path.join(data_path, label)
        if os.path.isdir(label_path):
            for img_file in os.listdir(label_path):
                img_path = os.path.join(label_path, img_file)
                image = Image.open(img_path)
                images.append(image)
                labels.append(label)
    return images, labels