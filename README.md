# 🦇 IndianBatsModel

## Bat Species Classifier using Deep Learning

A deep learning pipeline that classifies Indian bat species from their echolocation calls. The system converts ultrasonic audio recordings into mel spectrograms and uses a hybrid CNN+MLP architecture that fuses visual features with acoustic parameters (like end-frequency) for improved classification accuracy.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Complete Pipeline Workflow](#complete-pipeline-workflow)
- [Detailed File Documentation](#detailed-file-documentation)
  - [Models](#models)
  - [Datasets](#datasets)
  - [Data Preparation](#data-preparation)
  - [Training & Evaluation](#training--evaluation)
  - [Utilities](#utilities)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

This project provides an end-to-end solution for bat species identification:

1. **Audio → Annotations**: Convert raw audio + annotation files into standardized JSON format
2. **Annotations → Spectrograms**: Generate mel spectrogram images from annotated call segments
3. **Feature Extraction**: Compute acoustic features (end-frequency) for each call
4. **Training**: Train a ResNet18 + MLP fusion model on spectrograms + features
5. **Inference**: Classify new bat calls with the trained model

**Key Features:**
- Supports multiple annotation formats (Whombat, Wombat)
- Pretrained ResNet18 backbone for transfer learning
- Handles class imbalance with weighted sampling
- Multi-GPU training support
- Kaggle-ready inference pipeline

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        CNNWithFeatures                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌──────────────┐     ┌──────────────┐     ┌──────────────┐   │
│   │  Spectrogram │     │   ResNet18   │     │   512-dim    │   │
│   │    Image     │ ──▶ │   Backbone   │ ──▶ │  Embedding   │   │
│   │  (224x224)   │     │ (pretrained) │     │              │   │
│   └──────────────┘     └──────────────┘     └──────┬───────┘   │
│                                                     │           │
│   ┌──────────────┐                                  │           │
│   │   Numeric    │                                  ▼           │
│   │   Features   │ ──────────────────────▶  ┌──────────────┐   │
│   │ (end_freq)   │                          │   Concat     │   │
│   └──────────────┘                          │  (513-dim)   │   │
│                                              └──────┬───────┘   │
│                                                     │           │
│                                              ┌──────▼───────┐   │
│                                              │  MLP Head    │   │
│                                              │ 513→256→N    │   │
│                                              │ (dropout=0.3)│   │
│                                              └──────┬───────┘   │
│                                                     │           │
│                                              ┌──────▼───────┐   │
│                                              │   Species    │   │
│                                              │  Prediction  │   │
│                                              └──────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
IndianBatsModel/
├── MainShitz/                    # 🎯 PRIMARY SOURCE CODE
│   ├── models/                   # Neural network architectures
│   │   ├── cnn.py               # Basic 3-layer CNN (legacy)
│   │   └── cnn_with_features.py # ResNet18 + MLP fusion model (MAIN)
│   ├── datasets/                 # PyTorch Dataset classes
│   │   ├── spectrogram_dataset.py           # Basic image dataset
│   │   └── spectrogram_with_features_dataset.py  # Image + features dataset
│   ├── data_prep/                # Data preprocessing scripts
│   │   ├── wombat_to_spectrograms.py     # JSON + audio → spectrograms
│   │   ├── whombat_project_to_wombat.py  # Whombat → Wombat format converter
│   │   ├── extract_end_frequency.py       # Compute end-frequency features
│   │   ├── generate_annotations.py        # Auto-generate annotations
│   │   ├── audio_to_spectrogram.py        # Simple audio → spectrogram
│   │   └── augment.py                     # Data augmentation (deprecated)
│   ├── train.py                  # Training loop
│   ├── evaluate.py               # Model evaluation
│   └── utils.py                  # Helper functions
├── configs/
│   └── config.yaml               # Hyperparameters and paths
├── notebooks_for_kaggle/         # Kaggle competition notebooks
│   ├── kaggle_inference_pipeline.ipynb
│   ├── kaggle_train.ipynb
│   └── exploratory.ipynb
├── scripts/
│   └── prepare_data.sh           # Data preparation automation
├── tests/
│   └── test_dataset.py           # Unit tests
├── requirements.txt              # Python dependencies
└── setup.py                      # Package installation
```

---

## Installation & Setup

### 🚀 Recommended: Run on Kaggle (Easiest)

The easiest way to use this project is through **Kaggle Notebooks** - no local setup required!

1. Go to [Kaggle](https://www.kaggle.com)
2. Create a new notebook or fork our existing notebooks
3. Upload/link the competition dataset
4. Copy the code from `notebooks_for_kaggle/` folder:
   - `kaggle_train.ipynb` - For training the model
   - `kaggle_inference_pipeline.ipynb` - For making predictions
5. Run the cells - all dependencies are pre-installed on Kaggle!


**Benefits of Kaggle:**
- Free GPU/TPU access (up to 30 hours/week)
- All dependencies pre-installed (torch, librosa, etc.)
- Easy dataset management
- No local setup headaches

### 💻 Local Development (Optional)

If you prefer local development:

```bash
# Clone the repository
git clone https://github.com/Quarkisinproton/IndianBatsModel.git
cd IndianBatsModel

# Install dependencies
pip install -r requirements.txt
```

**Requirements:**
- Python 3.8+
- CUDA-capable GPU (recommended for training)

### Dependencies
```
torch
torchaudio
torchvision
numpy
matplotlib
scikit-learn
pandas
librosa
opencv-python
jupyter
seaborn
PyYAML
tqdm
```

---

## Complete Pipeline Workflow

### Step 1: Convert Whombat Annotations (if needed)
```bash
python -m MainShitz.data_prep.whombat_project_to_wombat \
    --project_json data/annotations.json \
    --out_dir data/wombat_jsons \
    --tag_key Species
```

### Step 2: Generate Spectrograms
```bash
python -m MainShitz.data_prep.wombat_to_spectrograms \
    --raw_audio_dir data/raw/audio \
    --json_dir data/wombat_jsons \
    --out_dir data/processed/spectrograms
```

### Step 3: Extract End-Frequency Features
```bash
python -m MainShitz.data_prep.extract_end_frequency \
    --raw_audio_dir data/raw/audio \
    --json_dir data/wombat_jsons \
    --out_csv data/features.csv
```

### Step 4: Train the Model
```bash
python -m MainShitz.train
```

### Step 5: Evaluate
```bash
python -m MainShitz.evaluate
```

---

## Detailed File Documentation

---

### Models

#### 📄 `MainShitz/models/cnn.py`
**Purpose:** Basic 3-layer CNN for spectrogram classification (legacy/retired)

| Function/Class | Description |
|---------------|-------------|
| `CNN.__init__(num_classes)` | Initializes a simple CNN with 3 conv layers (1→16→32→64 channels), max pooling, and 2 fully connected layers (64*32*32→128→num_classes). Expects single-channel 256x256 input. |
| `CNN.forward(x)` | Forward pass: Conv1→ReLU→Pool → Conv2→ReLU→Pool → Conv3→ReLU→Pool → Flatten → FC1→ReLU → FC2. Returns logits for each class. |

**Architecture Details:**
- Input: `(batch, 1, 256, 256)` grayscale spectrogram
- Conv layers: 3x3 kernels, stride 1, padding 1
- Pooling: 2x2 max pooling
- Output: `(batch, num_classes)` logits

---

#### 📄 `MainShitz/models/cnn_with_features.py`
**Purpose:** Main production model - ResNet18 backbone fused with numeric features via MLP head

| Function/Class | Description |
|---------------|-------------|
| `CNNWithFeatures.__init__(num_classes, numeric_feat_dim=1, pretrained=True)` | Initializes the hybrid model. Loads pretrained ResNet18, replaces final FC with Identity to get 512-d embeddings. Creates MLP classifier that takes (512 + numeric_feat_dim) → 256 → num_classes with ReLU activation and 30% dropout. |
| `CNNWithFeatures.forward(images, numeric_feats=None)` | Forward pass: Images go through ResNet18 backbone → 512-d embedding. If numeric_feats provided, concatenate with embedding. Pass through MLP classifier. Returns logits. |

**Architecture Details:**
- Image input: `(batch, 3, 224, 224)` RGB spectrogram
- Numeric input: `(batch, feat_dim)` e.g., end-frequency
- Backbone: ResNet18 pretrained on ImageNet
- Classifier: Linear(512+feat_dim, 256) → ReLU → Dropout(0.3) → Linear(256, num_classes)

---

### Datasets

#### 📄 `MainShitz/datasets/spectrogram_dataset.py`
**Purpose:** Basic PyTorch Dataset for loading spectrograms organized in folders by species

| Function/Class | Description |
|---------------|-------------|
| `preprocess_image(img, size=(128,128))` | Resizes PIL Image to target size using bilinear interpolation. Converts to float32 numpy array, normalizes to [0,1]. Transposes from HWC to CHW format for PyTorch. Returns tensor. |
| `SpectrogramDataset.__init__(root_dir, transform=None, image_size=(128,128))` | Initializes dataset with root directory path. Scans all subdirectories as class labels. Stores all image paths and corresponding labels. |
| `SpectrogramDataset._load_data()` | Walks through root_dir, treats each subdirectory as a species class. Collects all image file paths and their labels into lists. |
| `SpectrogramDataset.__len__()` | Returns total number of images in dataset. |
| `SpectrogramDataset.__getitem__(idx)` | Loads image at index, converts to RGB. Applies transform if provided, otherwise uses preprocess_image(). Returns (image_tensor, label_string). |

**Expected Directory Structure:**
```
root_dir/
├── Species_A/
│   ├── image1.png
│   └── image2.png
├── Species_B/
│   └── image3.png
```

---

#### 📄 `MainShitz/datasets/spectrogram_with_features_dataset.py`
**Purpose:** Advanced dataset that yields (image, numeric_features, label) tuples for the fusion model

| Function/Class | Description |
|---------------|-------------|
| `SpectrogramWithFeaturesDataset.__init__(root_dir, features_csv=None, transform=None, numeric_cols=None)` | Initializes dataset. Scans root_dir for images organized by class. If features_csv provided, loads CSV and builds a lookup map from image filename to feature vector. Default transform: ToTensor + ImageNet normalization. |
| `SpectrogramWithFeaturesDataset._scan_files()` | Scans root directory, builds class_to_idx mapping (sorted alphabetically), collects all image paths with their numeric labels. |
| `SpectrogramWithFeaturesDataset.__len__()` | Returns total number of images. |
| `SpectrogramWithFeaturesDataset.__getitem__(idx)` | Loads image, applies transform. Looks up features by filename pattern `{audio_stem}_{segment_index}.png`. Returns (image_tensor, feature_tensor, label_tensor). If features not found, returns zeros. |

**Features CSV Format:**
```csv
audio_file,segment_index,end_freq_hz,low_freq_hz,high_freq_hz
recording1.wav,0,45000,35000,55000
recording1.wav,1,42000,33000,52000
```

---

### Data Preparation

#### 📄 `MainShitz/data_prep/wombat_to_spectrograms.py`
**Purpose:** Main spectrogram generator - reads Wombat JSON annotations and audio files, outputs mel spectrogram images organized by species

| Function | Description |
|----------|-------------|
| `ensure_dir(path)` | Creates directory and all parents if they don't exist (equivalent to `mkdir -p`). |
| `load_wombat_json(path)` | Opens and parses a JSON file, returns dict. |
| `find_audio_for_json(json_path, raw_audio_dirs)` | Locates the audio file for a given JSON. First checks if JSON contains `recording`/`audio_file`/`file` field. If path exists, returns it. Otherwise searches raw_audio_dirs by filename or stem matching. Returns Path or None. |
| `extract_segment(y, sr, start_s, end_s)` | Extracts a portion of audio array between start and end times (seconds). Clips to valid bounds. Returns numpy array segment. |
| `make_mel_spectrogram(y, sr, n_mels=128, hop_length=512)` | Computes mel spectrogram using librosa. Converts power to dB scale. Returns 2D numpy array (n_mels x time_frames). |
| `save_spectrogram_image(S_db, out_path, cmap='magma', dpi=100)` | Saves spectrogram as PNG image using matplotlib. Uses 'magma' colormap, removes axes, tight layout. |
| `normalize_annotations(raw_anns)` | Handles various annotation formats (dict, list, None). Always returns a list of annotation dicts. |
| `get_first_present_key(d, keys)` | Searches dict for first key that exists from a list of alternatives. Useful for handling varied JSON schemas (e.g., 'start_time' vs 'start' vs 't0'). |
| `process_audio_file(audio_path, annotations, out_base, species_key='label')` | Loads audio file, iterates through annotations, extracts each segment, generates spectrogram, saves to `out_base/{species}/{audio_stem}_{index}.png`. |
| `process_all(raw_audio_dirs, json_dir, out_dir, species_key='label')` | Main entry point. Scans json_dir for all .json files, finds corresponding audio, processes each file. Shows progress bar if tqdm available. Prints summary of processed files and output directory contents. |

**CLI Usage:**
```bash
python -m MainShitz.data_prep.wombat_to_spectrograms \
    --raw_audio_dir /path/to/audio \
    --json_dir /path/to/annotations \
    --out_dir /path/to/spectrograms \
    --species_key label
```

---

#### 📄 `MainShitz/data_prep/whombat_project_to_wombat.py`
**Purpose:** Converts Whombat project exports (single large JSON) into per-recording Wombat-style JSONs

| Function/Class | Description |
|---------------|-------------|
| `ConvertedProjectSummary` | Dataclass holding conversion statistics: jsons_written, recordings_seen, sound_events_seen, sound_events_written, sound_events_skipped_unlabeled. |
| `_safe_filename_stem(name)` | Sanitizes filename by removing special characters, replacing with underscores. Returns filesystem-safe string. |
| `_basename_from_any_path(path_str)` | Extracts filename from path string, handling both Windows (`C:\...`) and POSIX (`/...`) paths. |
| `convert_whombat_project_to_wombat_jsons(project_json_path, output_dir, tag_key="Species", skip_unlabeled=True)` | Main conversion function. Parses Whombat JSON structure (tags, recordings, sound_events, sound_event_annotations). Maps tag IDs to species labels. Extracts time bounds from geometry coordinates. Groups annotations by recording. Writes one JSON per recording with format: `{audio_file, recording, annotations: [{start_time, end_time, label}]}`. Returns ConvertedProjectSummary. |
| `_main(argv)` | CLI entry point. Parses arguments, calls converter, prints summary. |

**Whombat JSON Structure Handled:**
```json
{
  "data": {
    "tags": [{"id": 1, "key": "Species", "value": "Pipistrellus"}],
    "recordings": [{"uuid": "...", "path": "audio.wav"}],
    "sound_events": [{"uuid": "...", "recording": "...", "geometry": {"coordinates": [start, low_hz, end, high_hz]}}],
    "sound_event_annotations": [{"sound_event": "...", "tags": [1]}]
  }
}
```

---

#### 📄 `MainShitz/data_prep/extract_end_frequency.py`
**Purpose:** Extracts the dominant frequency at the end of each bat call segment for use as a numeric feature

| Function | Description |
|----------|-------------|
| `compute_end_frequency(y, sr, start_s, end_s, n_fft=2048, hop_length=512, tail_frames=3)` | Computes end-frequency estimate for a call segment. Extracts segment, computes STFT, takes last `tail_frames` columns (time frames), averages power across frames for each frequency bin, finds bin with maximum energy, converts bin to Hz. Returns frequency in Hz or NaN if computation fails. |
| `process_all_and_write_csv(raw_audio_dirs, json_dir, out_csv, species_key='label')` | Scans json_dir for annotation JSONs, matches to audio files, loads each audio once, processes all annotations within it. Writes CSV with columns: json_file, audio_file, segment_index, label, start, end, end_freq_hz, low_freq_hz, high_freq_hz. |

**Output CSV Format:**
```csv
json_file,audio_file,segment_index,label,start,end,end_freq_hz,low_freq_hz,high_freq_hz
/path/to/ann.json,/path/to/audio.wav,0,Pipistrellus,0.12,0.15,45000,35000,55000
```

---

#### 📄 `MainShitz/data_prep/generate_annotations.py`
**Purpose:** Auto-generates Wombat-style annotation JSONs when you have unlabeled audio organized in folders

| Function | Description |
|----------|-------------|
| `generate_annotations(raw_audio_dirs, output_dir, label_strategy='folder')` | Creates annotation JSONs for all audio files. Uses librosa to get duration. Label determined by: 'folder' = parent directory name, 'filename' = audio file stem. Generates one JSON per audio file with full-file annotation (start=0, end=duration). |

**Output JSON Format:**
```json
{
  "audio_file": "recording.wav",
  "recording": "/full/path/to/recording.wav",
  "annotations": [
    {"start_time": 0.0, "end_time": 5.2, "label": "Species_A"}
  ]
}
```

---

#### 📄 `MainShitz/data_prep/audio_to_spectrogram.py`
**Purpose:** Simple audio-to-spectrogram converter (legacy, use wombat_to_spectrograms.py instead)

| Function | Description |
|----------|-------------|
| `audio_to_spectrogram(audio_path, output_path, sr=22050, n_fft=2048, hop_length=512, power=2.0)` | Loads audio, computes mel spectrogram, saves as image with colorbar and axis labels using librosa.display. |
| `process_audio_directory(input_dir, output_dir)` | Processes all .wav files in input_dir, maintains species folder structure in output. |

---

#### 📄 `MainShitz/data_prep/augment.py`
**Purpose:** Data augmentation utilities (⚠️ DEPRECATED - uses removed librosa API)

| Function | Description |
|----------|-------------|
| `augment_audio(audio_path, output_path, sample_rate=22050, noise_factor=0.005)` | Adds Gaussian noise to audio. ⚠️ Uses `librosa.output.write_wav` which was removed in librosa 0.8+. |
| `create_spectrogram(audio_path, output_path, sample_rate=22050)` | Creates mel spectrogram image from audio file. |
| `augment_and_save_spectrograms(species_dirs, output_spectrogram_dir)` | Batch processes directories: augments audio then creates spectrograms. |

---

### Training & Evaluation

#### 📄 `MainShitz/train.py`
**Purpose:** Complete training pipeline with config loading, dataset creation, model initialization, training loop, and checkpointing

| Function | Description |
|----------|-------------|
| `train_model(config)` | Main training function. Parses config dict for data paths, hyperparameters. Automatically selects dataset type based on features_csv existence. Creates weighted sampler for class imbalance. Initializes appropriate model (CNN or CNNWithFeatures). Supports multi-GPU via DataParallel. Runs training loop with loss logging. Saves best model checkpoint and class mapping JSON. |

**Config Keys Used:**
- `data.train_spectrograms`: Path to spectrogram images
- `data.features_csv`: Path to features CSV (optional, enables fusion model)
- `data.num_classes`: Number of species classes
- `training.batch_size`: Batch size (default: 16)
- `training.learning_rate`: Learning rate (default: 1e-3)
- `training.num_epochs`: Training epochs (default: 10)
- `training.num_workers`: DataLoader workers (default: 0)
- `training.model_save_path`: Where to save best model

---

#### 📄 `MainShitz/evaluate.py`
**Purpose:** Model evaluation on test set with accuracy computation

| Function | Description |
|----------|-------------|
| `evaluate_model(config_path)` | Loads config YAML, creates test dataset (with or without features), loads trained model weights, runs inference on all test samples, computes and prints accuracy. |

---

### Utilities

#### 📄 `MainShitz/utils.py`
**Purpose:** Helper functions for model I/O and data loading

| Function | Description |
|----------|-------------|
| `load_model(model_path)` | Loads a pickled PyTorch model from disk, sets to eval mode, returns model. |
| `save_model(model, model_path)` | Saves entire model (architecture + weights) to disk using torch.save. |
| `load_data(data_path)` | Loads all images from a directory structure into memory. Returns (list of PIL Images, list of label strings). ⚠️ Memory-intensive for large datasets. |

---

## Configuration

### `configs/config.yaml`

```yaml
train:
  epochs: 50              # Number of training epochs
  batch_size: 16          # Samples per batch
  learning_rate: 0.001    # Adam optimizer LR
  weight_decay: 0.0001    # L2 regularization

data:
  raw_data_path: "data/raw"
  processed_data_path: "data/processed/spectrograms"
  features_csv: "data/features.csv"  # Optional: enables fusion model
  num_classes: 3

model:
  input_size: [1, 128, 128]   # Legacy CNN input shape
  num_classes: 3
  architecture: "cnn"          # "cnn" or "cnn_with_features"

augmentation:
  enabled: true
  techniques:
    - "random_flip"
    - "random_crop"
    - "time_stretch"

logging:
  log_dir: "logs"
  log_level: "info"
```

---

## Usage Examples

### Quick Start: Train on Existing Spectrograms
```python
from MainShitz.train import train_model
import yaml

with open('configs/config.yaml') as f:
    config = yaml.safe_load(f)

train_model(config)
```

### Generate Spectrograms from Scratch
```python
from MainShitz.data_prep.wombat_to_spectrograms import process_all

process_all(
    raw_audio_dirs=['/data/audio'],
    json_dir='/data/annotations',
    out_dir='/data/spectrograms'
)
```

### Extract Features
```python
from MainShitz.data_prep.extract_end_frequency import process_all_and_write_csv

process_all_and_write_csv(
    raw_audio_dirs=['/data/audio'],
    json_dir='/data/annotations',
    out_csv='/data/features.csv'
)
```

### Inference on New Audio
```python
import torch
from MainShitz.models.cnn_with_features import CNNWithFeatures
from MainShitz.data_prep.wombat_to_spectrograms import make_mel_spectrogram
from MainShitz.data_prep.extract_end_frequency import compute_end_frequency
import librosa

# Load model
model = CNNWithFeatures(num_classes=3, numeric_feat_dim=1)
model.load_state_dict(torch.load('models/bat_model.pth'))
model.eval()

# Process new audio
y, sr = librosa.load('new_recording.wav', sr=None)
spectrogram = make_mel_spectrogram(y, sr)
end_freq = compute_end_frequency(y, sr, 0, len(y)/sr)

# Prepare tensors and predict
# ... (see kaggle_inference_pipeline.ipynb for complete example)
```

---

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- ResNet architecture from PyTorch/torchvision
- Audio processing powered by librosa
- Bat annotation tools: Whombat, Wombat