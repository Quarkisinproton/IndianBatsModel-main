"""
efficientnet_with_features.py – EfficientNet-B0 backbone + numeric feature fusion.

Best balance of speed and accuracy for **small datasets** (50-500 samples).
Only 5.3M params; produces a 1280-d image embedding before the fusion head.
"""
import torch
import torch.nn as nn
import torchvision.models as models


class EfficientNetWithFeatures(nn.Module):
    """
    images → EfficientNet-B0 (1280-d) ─┐
                                         ├──▶ classifier → species
    numeric_features → small MLP (32-d) ─┘
    """

    def __init__(
        self,
        num_classes: int,
        numeric_feat_dim: int = 3,
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        backbone = models.efficientnet_b0(weights=weights)
        # EfficientNet-B0 classifier: [Dropout, Linear(1280, 1000)]
        emb_dim = backbone.classifier[1].in_features  # 1280
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
