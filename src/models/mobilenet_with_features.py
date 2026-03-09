"""
mobilenet_with_features.py – MobileNetV3-Small backbone + numeric feature fusion.

Smallest pretrained model available: only **2.5 M params**.
Best for: very small datasets (<200 samples) where overfitting is the enemy,
or when you need fast inference (edge / real-time / Raspberry Pi).
"""
import torch
import torch.nn as nn
import torchvision.models as models


class MobileNetWithFeatures(nn.Module):
    """
    images → MobileNetV3-Small (576-d) ─┐
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
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        backbone = models.mobilenet_v3_small(weights=weights)
        # MobileNetV3-Small classifier: Sequential(Linear(576,1024), HS, Dropout, Linear(1024,1000))
        emb_dim = backbone.classifier[0].in_features  # 576
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
