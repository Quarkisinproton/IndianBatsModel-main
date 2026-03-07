"""
utils.py – Shared helper functions: model I/O, image loading.
"""
import os
from pathlib import Path

import torch
from PIL import Image


def load_model(model_path: str | Path) -> torch.nn.Module:
    """Load a full model (or state-dict) and set to eval mode."""
    model = torch.load(str(model_path), map_location="cpu")
    if isinstance(model, torch.nn.Module):
        model.eval()
    return model


def save_model(model: torch.nn.Module, model_path: str | Path) -> None:
    """Save model state dict to *model_path*."""
    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(path))


def load_images_from_dir(data_path: str | Path):
    """Bulk load PIL images from a species-folder tree.

    Returns:
        (images, labels): Two lists, images as PIL objects, labels as strings.
    """
    images, labels = [], []
    data_path = Path(data_path)
    for label_dir in sorted(data_path.iterdir()):
        if not label_dir.is_dir():
            continue
        for img_file in sorted(label_dir.iterdir()):
            if img_file.suffix.lower() in (".png", ".jpg", ".jpeg"):
                images.append(Image.open(img_file))
                labels.append(label_dir.name)
    return images, labels
