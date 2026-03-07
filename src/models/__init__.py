"""src.models – All model architectures for bat species classification."""

from src.models.cnn import CNN
from src.models.cnn_with_features import CNNWithFeatures

__all__ = ["CNN", "CNNWithFeatures"]
