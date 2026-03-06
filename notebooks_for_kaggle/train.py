import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset import BatDataset
from pathlib import Path
import numpy as np
import sys
import importlib.util

# Workaround: Import BatCNN, BatTransformer from models.py (not models/ package)
# This avoids conflict when both models.py and models/ folder exist
spec = importlib.util.spec_from_file_location("models_pytorch", Path(__file__).parent / "models.py")
models_pytorch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(models_pytorch)
BatCNN = models_pytorch.BatCNN
BatTransformer = models_pytorch.BatTransformer

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Dataset
    dataset = BatDataset(
        annotations_file=args.annotations,
        data_dir=args.data_dir,
        target_sample_rate=args.sample_rate,
        n_mels=args.n_mels,
        fixed_duration=args.duration,
        augment=args.augment
    )
    
    # Split dataset
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    
    # Calculate class weights for imbalanced data
    train_species_ids = [dataset.samples[i]['species_id'] for i in train_dataset.indices]
    class_counts = np.bincount(train_species_ids, minlength=2)
    class_weights = len(train_species_ids) / (2 * class_counts)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    print(f"Class counts: {class_counts}, Class weights: {class_weights.cpu().numpy()}")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Model
    if args.model == 'cnn':
        model = BatCNN(n_mels=args.n_mels, n_classes=2).to(device)
    elif args.model == 'transformer':
        model = BatTransformer(n_mels=args.n_mels, n_classes=2).to(device)
    else:
        raise ValueError(f"Unknown model: {args.model}")
        
    # Loss and Optimizer - use weighted loss + label smoothing to prevent overconfidence
    criterion_species = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Learning rate scheduler - reduce LR when stuck
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, verbose=True)
    
    # Training Loop
    best_val_acc = 0.0
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        train_species_acc = 0.0
        
        for batch in train_loader:
            spectrogram = batch['spectrogram'].to(device)
            species_label = batch['species_label'].to(device)
            
            optimizer.zero_grad()
            
            species_logits = model(spectrogram)
            
            loss = criterion_species(species_logits, species_label)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Metrics
            preds = torch.argmax(species_logits, dim=1)
            train_species_acc += (preds == species_label).sum().item()
            
        train_loss /= len(train_loader)
        train_species_acc /= len(train_dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_species_acc = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                spectrogram = batch['spectrogram'].to(device)
                species_label = batch['species_label'].to(device)
                
                species_logits = model(spectrogram)
                
                loss = criterion_species(species_logits, species_label)
                val_loss += loss.item()
                
                preds = torch.argmax(species_logits, dim=1)
                val_species_acc += (preds == species_label).sum().item()
                
        val_loss /= len(val_loader)
        val_species_acc /= len(val_dataset)
        
        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_species_acc:.2f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_species_acc:.2f}")
        
        # Learning rate scheduler step
        scheduler.step(val_species_acc)
        
        # Save model based on validation ACCURACY (more stable than loss for small datasets)
        if val_species_acc > best_val_acc:
            best_val_acc = val_species_acc
            torch.save(model.state_dict(), f"best_model_{args.model}.pth")
            print(f"Saved best model (Val Acc: {val_species_acc:.2f})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Bat Species Model")
    parser.add_argument("--annotations", type=str, default="../data/annotations.json", help="Path to annotations JSON")
    parser.add_argument("--data_dir", type=str, default="../data", help="Path to data directory")
    parser.add_argument("--model", type=str, default="cnn", choices=["cnn", "transformer"], help="Model architecture")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate (reduced to prevent overfitting)")
    parser.add_argument("--sample_rate", type=int, default=44100, help="Target sample rate")
    parser.add_argument("--n_mels", type=int, default=128, help="Number of mel bins")
    parser.add_argument("--duration", type=float, default=3.5, help="Fixed duration in seconds")
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--augment", type=bool, default=False, help="Enable data augmentation")
    
    args = parser.parse_args()
    train(args)
