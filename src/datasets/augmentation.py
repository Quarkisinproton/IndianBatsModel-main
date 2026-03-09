"""
augmentation.py – Spectrogram-aware augmentations for **tiny** datasets (50-250 samples).

Strategy:
  - Time-masking  (SpecAugment-style) – zero-out a random vertical stripe.
  - Freq-masking  (SpecAugment-style) – zero-out a random horizontal stripe.
  - Random-erasing (from torchvision) – tiny cutout rectangles.
  - Mixup helper  – blend two spectrograms + their labels (call from training loop).

Usage in CONFIG:
  CONFIG["train"]["augmentation"] = "specaugment"   # or "none"
"""
from __future__ import annotations

import random
import torch
import torchvision.transforms as T


# ────────────────────────────────────────────────────────────────────
#  Callable transforms (composable with torchvision.transforms)
# ────────────────────────────────────────────────────────────────────

class TimeMask:
    """Zero-out a random vertical stripe (time axis) of a spectrogram tensor [C,H,W]."""

    def __init__(self, max_width: int = 30, p: float = 0.5):
        self.max_width = max_width
        self.p = p

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return img
        _, _, w = img.shape
        width = random.randint(1, min(self.max_width, w // 4))
        start = random.randint(0, w - width)
        img = img.clone()
        img[:, :, start : start + width] = 0.0
        return img


class FreqMask:
    """Zero-out a random horizontal stripe (frequency axis) of a spectrogram tensor [C,H,W]."""

    def __init__(self, max_height: int = 20, p: float = 0.5):
        self.max_height = max_height
        self.p = p

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return img
        _, h, _ = img.shape
        height = random.randint(1, min(self.max_height, h // 4))
        start = random.randint(0, h - height)
        img = img.clone()
        img[:, start : start + height, :] = 0.0
        return img


class GaussianNoise:
    """Add faint Gaussian noise – helps the model generalise."""

    def __init__(self, std: float = 0.01, p: float = 0.3):
        self.std = std
        self.p = p

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return img
        return img + torch.randn_like(img) * self.std


# ────────────────────────────────────────────────────────────────────
#  Pre-built transform pipelines
# ────────────────────────────────────────────────────────────────────

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def build_train_transform(augment: str = "specaugment", image_size: int = 224) -> T.Compose:
    """
    Return a composed transform for **training** spectrograms.

    augment options:
      - "none"          – just resize + normalise (baseline).
      - "specaugment"   – SpecAugment-style time/freq masking + noise (recommended).
      - "heavy"         – specaugment + random-erasing + colour jitter.
    """
    base = [
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ]

    if augment == "none":
        return T.Compose(base)

    spec = [
        TimeMask(max_width=30, p=0.5),
        FreqMask(max_height=20, p=0.5),
        GaussianNoise(std=0.01, p=0.3),
    ]

    if augment == "specaugment":
        return T.Compose(base + spec)

    # "heavy"
    heavy = spec + [
        T.RandomErasing(p=0.3, scale=(0.02, 0.15)),
    ]
    return T.Compose(base + heavy)


def build_val_transform(image_size: int = 224) -> T.Compose:
    """Validation / test transform – no augmentation, just resize + normalise."""
    return T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ]
    )


# ────────────────────────────────────────────────────────────────────
#  Mixup (called from the training loop, not from transforms)
# ────────────────────────────────────────────────────────────────────

def mixup_data(
    images: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Apply Mixup to a batch.

    Returns:
        mixed_images, labels_a, labels_b, lam
    Loss should be:  lam * loss(pred, labels_a) + (1 - lam) * loss(pred, labels_b)
    """
    if alpha > 0:
        lam = torch.distributions.Beta(alpha, alpha).sample().item()
    else:
        lam = 1.0

    batch_size = images.size(0)
    index = torch.randperm(batch_size, device=images.device)

    mixed = lam * images + (1 - lam) * images[index]
    return mixed, labels, labels[index], lam
