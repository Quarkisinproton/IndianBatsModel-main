"""src.models – All model architectures for bat species classification.

Available backbones (all accept forward(images, features)):
  - CNN                    – basic 3-layer conv net (no pretrained weights)
  - CNNWithFeatures        – ResNet-18  (512-d)   ← default / baseline
  - EfficientNetWithFeatures – EfficientNet-B0 (1280-d) ← best bang-for-buck
  - MobileNetWithFeatures  – MobileNetV3-Small (576-d)  ← smallest, fastest
  - DenseNetWithFeatures   – DenseNet-121 (1024-d)      ← strong regularisation
  - ConvNeXtWithFeatures   – ConvNeXt-Tiny (768-d)      ← modern CNN
  - SwinWithFeatures       – Swin-Tiny (768-d)          ← attention-based
"""

from src.models.cnn import CNN
from src.models.cnn_with_features import CNNWithFeatures
from src.models.efficientnet_with_features import EfficientNetWithFeatures
from src.models.mobilenet_with_features import MobileNetWithFeatures
from src.models.densenet_with_features import DenseNetWithFeatures
from src.models.convnext_with_features import ConvNeXtWithFeatures
from src.models.swin_with_features import SwinWithFeatures

__all__ = [
    "CNN",
    "CNNWithFeatures",
    "EfficientNetWithFeatures",
    "MobileNetWithFeatures",
    "DenseNetWithFeatures",
    "ConvNeXtWithFeatures",
    "SwinWithFeatures",
]
