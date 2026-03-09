"""
densenet_with_features.py – DenseNet-121 backbone + numeric feature fusion.

Dense connections mean every layer re-uses features from ALL previous layers.
This provides **strong implicit regularisation** – ideal for tiny datasets
(50-300 samples) where overfitting is the main risk.  8 M params.
"""
import torch
import torch.nn as nn
import torchvision.models as models


class DenseNetWithFeatures(nn.Module):
    """
    images → DenseNet-121 (1024-d) ─┐
                                      ├──▶ classifier → species
    numeric_features → MLP (32-d)  ──┘
    """

    def __init__(
        self,
        num_classes: int,
        numeric_feat_dim: int = 3,
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        backbone = models.densenet121(weights=weights)
        emb_dim = backbone.classifier.in_features  # 1024
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
