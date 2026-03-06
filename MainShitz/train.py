import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from MainShitz.datasets.spectrogram_dataset import SpectrogramDataset
from MainShitz.datasets.spectrogram_with_features_dataset import SpectrogramWithFeaturesDataset
from MainShitz.models.cnn import CNN
from MainShitz.models.cnn_with_features import CNNWithFeatures
from MainShitz.utils import save_model
import yaml
import numpy as np
import json
import argparse
import matplotlib.pyplot as plt

def train_model(config):
    # Handle config keys: 'train' (from yaml) vs 'training' (legacy)
    train_cfg = config.get('train') or config.get('training', {})
    data_cfg = config.get('data', {})
    
    train_dir = data_cfg.get('train_spectrograms') or 'data/processed/spectrograms'
    num_classes = data_cfg.get('num_classes') or config.get('model', {}).get('num_classes') or 3
    
    batch_size = train_cfg.get('batch_size') or 16
    lr = train_cfg.get('learning_rate') or 1e-3
    weight_decay = train_cfg.get('weight_decay') or 0.0
    num_epochs = train_cfg.get('num_epochs') or 10
    num_workers = train_cfg.get('num_workers') if train_cfg.get('num_workers') is not None else 0
    model_save_path = train_cfg.get('model_save_path') or 'models/bat_model.pth'

    # If features CSV exists, use fused dataset
    features_csv = data_cfg.get('features_csv')
    if features_csv and os.path.exists(features_csv):
        full_dataset = SpectrogramWithFeaturesDataset(train_dir, features_csv)
        use_features = True
    else:
        full_dataset = SpectrogramDataset(train_dir)
        use_features = False

    if len(full_dataset) == 0:
        print(f"CRITICAL ERROR: Dataset is empty.")
        print(f"Checked directory: {train_dir}")
        raise ValueError("Dataset is empty. Cannot proceed with training.")

    # Automatic Split: 80% Train, 20% Validation
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"Dataset split: {train_size} training samples, {val_size} validation samples.")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Model setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if use_features:
        # infer numeric feature dimension from a sample
        # We need to access the underlying dataset from the Subset
        sample_image, sample_feat, sample_label = full_dataset[0]
        feat_dim = sample_feat.numel()
        model = CNNWithFeatures(num_classes=num_classes, numeric_feat_dim=feat_dim, pretrained=True)
    else:
        model = CNN(num_classes=num_classes)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    os.makedirs(os.path.dirname(model_save_path) or '.', exist_ok=True)
    best_loss = float('inf')
    
    # Loss Tracking
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        # --- Training Phase ---
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

        epoch_train_loss = running_loss / num_batches if num_batches > 0 else 0.0
        train_losses.append(epoch_train_loss)

        # --- Validation Phase ---
        model.eval()
        val_running_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
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
                val_running_loss += loss.item()
                val_batches += 1
        
        epoch_val_loss = val_running_loss / val_batches if val_batches > 0 else 0.0
        val_losses.append(epoch_val_loss)

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}')
        
        if epoch_val_loss < best_loss:
            best_loss = epoch_val_loss
            save_model(model, model_save_path)
            print(f"  New best model saved with val loss {best_loss:.4f}")

    # save class mapping
    class_map = getattr(full_dataset, 'class_to_idx', None)
    if class_map:
        with open(model_save_path + '.classes.json', 'w') as f:
            json.dump(class_map, f)
    print('Training complete. Best model saved to', model_save_path)

    # --- Plotting ---
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves')
    plt.legend()
    plt.grid(True)
    
    plot_path = os.path.join(os.path.dirname(model_save_path), 'training_curves.png')
    plt.savefig(plot_path)
    print(f"Training curves saved to {plot_path}")

    # Report final validation loss for hyperparameter tuning
    if val_losses:
        print(f"FINAL_VAL_LOSS: {val_losses[-1]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train IndianBatsModel')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config file')
    args = parser.parse_args()

    # Load configuration
    if not os.path.exists(args.config):
        print(f"Config file not found: {args.config}")
        exit(1)
        
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    train_model(config)