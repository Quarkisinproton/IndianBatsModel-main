import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from MainShitz.datasets.spectrogram_dataset import SpectrogramDataset
from MainShitz.datasets.spectrogram_with_features_dataset import SpectrogramWithFeaturesDataset
from MainShitz.models.cnn import CNN
from MainShitz.models.cnn_with_features import CNNWithFeatures
from MainShitz.utils import save_model
import yaml
import numpy as np
import json

def train_model(config):
    # Resolve config values with fallbacks
    data_cfg = config.get('data', {})
    train_dir = data_cfg.get('train_spectrograms') or 'data/processed/spectrograms'
    num_classes = data_cfg.get('num_classes') or config.get('model', {}).get('num_classes') or 3
    train_cfg = config.get('training', {})
    batch_size = train_cfg.get('batch_size') or 16
    lr = train_cfg.get('learning_rate') or 1e-3
    num_epochs = train_cfg.get('num_epochs') or 10
    num_workers = train_cfg.get('num_workers') if train_cfg.get('num_workers') is not None else 0

    # If features CSV exists, use fused dataset
    features_csv = data_cfg.get('features_csv')
    if features_csv and os.path.exists(features_csv):
        train_dataset = SpectrogramWithFeaturesDataset(train_dir, features_csv)
        use_features = True
    else:
        # don't override dataset transforms here; allow dataset to apply its defaults
        train_dataset = SpectrogramDataset(train_dir)
        use_features = False

    if len(train_dataset) == 0:
        print(f"CRITICAL ERROR: Training dataset is empty.")
        print(f"Checked directory: {train_dir}")
        print("Possible reasons:")
        print("1. Spectrogram generation failed (check previous steps).")
        print("2. Input audio directories are empty or incorrect.")
        print("3. No images found in the spectrogram directory.")
        raise ValueError("Dataset is empty. Cannot proceed with training.")

    # build a weighted sampler to help with class imbalance when possible
    try:
        if use_features:
            labels = train_dataset.labels
        else:
            # ImageFolder-like dataset exposes .samples as (path, class_idx)
            labels = [y for _, y in train_dataset.samples] if hasattr(train_dataset, 'samples') else getattr(train_dataset, 'targets', [])
        labels = np.array(labels, dtype=int)
        if labels.size > 0:
            class_sample_counts = np.bincount(labels)
            class_weights = 1.0 / class_sample_counts
            sample_weights = class_weights[labels]
            from torch.utils.data import WeightedRandomSampler
            sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        else:
            sampler = None
    except Exception:
        sampler = None

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=(sampler is None), sampler=sampler, num_workers=num_workers)

    # Initialize model, loss function, and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if use_features:
        # infer numeric feature dimension from a sample
        sample_image, sample_feat, sample_label = train_dataset[0]
        feat_dim = sample_feat.numel()
        # use pretrained backbone for better transfer learning when data is small
        model = CNNWithFeatures(num_classes=num_classes, numeric_feat_dim=feat_dim, pretrained=True)
    else:
        model = CNN(num_classes=num_classes)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Prepare for saving model
    model_save_path = train_cfg.get('model_save_path') or 'models/bat_model.pth'
    os.makedirs(os.path.dirname(model_save_path) or '.', exist_ok=True)
    best_loss = float('inf')

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            optimizer.zero_grad()
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

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            num_batches += 1

        if num_batches:
            epoch_loss = running_loss / num_batches
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')
            
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                save_model(model, model_save_path)
                print(f"  New best model saved with loss {best_loss:.4f}")
        else:
            print('No training data found. Check processed spectrograms path:', train_dir)
            break

    # save class mapping for inference if available
    class_map = getattr(train_dataset, 'class_to_idx', None)
    if class_map:
        with open(model_save_path + '.classes.json', 'w') as f:
            json.dump(class_map, f)
    print('Training complete. Best model saved to', model_save_path)

if __name__ == "__main__":
    # Load configuration
    with open('configs/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    train_model(config)