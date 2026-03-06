"""
cnn_with_features.py - The main event: ResNet18 + numeric features fusion

This is where the magic happens. Takes spectrogram images through a pretrained
ResNet18, concatenates end-frequency features, and classifies bat species.
Basically a CNN that also knows how to read spreadsheets.
"""
import torch
import torch.nn as nn
import torchvision.models as models


class CNNWithFeatures(nn.Module):
    """
    images -> resnet18 (512-d) -> concat features -> MLP -> species
    """
    def __init__(self, num_classes: int, numeric_feat_dim: int = 1, pretrained: bool = True):
        super().__init__()
        backbone = models.resnet18(pretrained=pretrained)
        # replace final fc with identity to get 512-d embedding
        backbone.fc = nn.Identity()
        self.backbone = backbone
        emb_dim = 512

        if numeric_feat_dim and numeric_feat_dim > 0:
            clf_in = emb_dim + numeric_feat_dim
        else:
            clf_in = emb_dim

        self.classifier = nn.Sequential(
            nn.Linear(clf_in, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, images: torch.Tensor, numeric_feats: torch.Tensor = None) -> torch.Tensor:
        emb = self.backbone(images)
        if numeric_feats is None or numeric_feats.numel() == 0:
            out = self.classifier(emb)
        else:
            x = torch.cat([emb, numeric_feats], dim=1)
            out = self.classifier(x)
        return out
