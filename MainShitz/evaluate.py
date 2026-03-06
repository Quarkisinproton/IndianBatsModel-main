"""
evaluate.py - Moment of truth: does our model actually work?

Loads a trained model and runs it on test data to get accuracy.
Hopefully the numbers go up, not down.
"""
import os
import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader
from MainShitz.datasets.spectrogram_dataset import SpectrogramDataset
from MainShitz.datasets.spectrogram_with_features_dataset import SpectrogramWithFeaturesDataset
from MainShitz.models.cnn import CNN
from MainShitz.models.cnn_with_features import CNNWithFeatures
from MainShitz.utils import load_model


def evaluate_model(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    data_cfg = config.get('data', {})
    test_dir = data_cfg.get('test_spectrograms') or data_cfg.get('train_spectrograms') # Fallback to train if test not specified
    features_csv = data_cfg.get('features_csv')
    num_classes = data_cfg.get('num_classes') or 3
    
    train_cfg = config.get('training', {})
    batch_size = train_cfg.get('batch_size') or 16
    model_path = train_cfg.get('model_save_path') or 'models/bat_model.pth'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Determine if we are using features
    use_features = False
    if features_csv and os.path.exists(features_csv):
        dataset = SpectrogramWithFeaturesDataset(test_dir, features_csv)
        use_features = True
    else:
        dataset = SpectrogramDataset(test_dir)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Load model
    if use_features:
        if len(dataset) > 0:
            sample_image, sample_feat, sample_label = dataset[0]
            feat_dim = sample_feat.numel()
        else:
            feat_dim = 1 # Dummy
        
        model = CNNWithFeatures(num_classes=num_classes, numeric_feat_dim=feat_dim, pretrained=False) # Pretrained=False because we load weights
    else:
        model = CNN(num_classes=num_classes)

    # Load state dict
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    else:
        print(f"Model file not found at {model_path} NOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO")
        return

    model.to(device)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            if use_features:
                inputs, feats, labels = batch
                inputs = inputs.to(device)
                feats = feats.to(device)
                labels = labels.to(device)
                outputs = model(inputs, feats)
            else:
                inputs, labels = batch
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    if total > 0:
        accuracy = correct / total
        print(f'Accuracy of the model: {accuracy:.2f}')
        if accuracy < 70:
            print("FYI dis Trash");
        
    else:
        print("No data to evaluate.........WTF is wrong with you")

if __name__ == "__main__":
    evaluate_model('configs/config.yaml')