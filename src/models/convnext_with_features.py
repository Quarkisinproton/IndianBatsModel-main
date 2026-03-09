"""
convnext_with_features.py – ConvNeXt-Tiny backbone + numeric feature fusion.

A **2022-era modernised CNN** that rivals Vision Transformers.  Uses depthwise
convolutions, LayerNorm, and GELU.  28 M params – heavier than ResNet-18 but
often significantly more accurate.

Recommended only when you have >300 spectrograms (or with heavy augmentation).
"""
import torch
import torch.nn as nn
import torchvision.models as models


class ConvNeXtWithFeatures(nn.Module):
    """
    images → ConvNeXt-Tiny (768-d) ─┐
                                      ├──▶ classifier → species
    numeric_features → MLP (32-d)  ──┘
    """

    def __init__(
        self,
        num_classes: int,
        numeric_feat_dim: int = 3,
        pretrained: bool = True,
        dropout: float = 0.4,
    ):
        super().__init__()
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        backbone = models.convnext_tiny(weights=weights)
        # ConvNeXt classifier: Sequential(LayerNorm, Flatten, Linear(768, 1000))
        emb_dim = backbone.classifier[2].in_features  # 768
        backbone.classifier = nn.Identity()
        self.backbone = backbone
        self.emb_dim = emb_dim

        self.has_features = numeric_feat_dim is not None and numeric_feat_dim > 0
        feat_out = 0
        if self.has_features:
            self.feature_mlp = nn.Sequential(
                nn.Linear(numeric_feat_dim, 32),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
            feat_out = 32

        self.classifier = nn.Sequential(
            nn.Linear(emb_dim + feat_out, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(
        self, images: torch.Tensor, numeric_feats: torch.Tensor = None
    ) -> torch.Tensor:
        img_emb = self.backbone(images)
        if self.has_features and numeric_feats is not None and numeric_feats.numel() > 0:
            feat_emb = self.feature_mlp(numeric_feats)
            x = torch.cat([img_emb, feat_emb], dim=1)
        else:
            x = img_emb
        return self.classifier(x)
