# 🦇 IndianBatsModel

## Bat Species Classification from Echolocation Calls

**A complete deep learning pipeline for classifying Indian bat species from ultrasonic audio recordings.** The system processes annotated bat calls, generates mel-spectrograms, and trains plug-and-play neural networks (ResNet18, Vision Transformer, custom CNN) with acoustic feature fusion for species identification.

---

## 🚀 Quick Start on Kaggle (Recommended)

**This project is intended to run on Kaggle GPUs, not on local laptops.**

### 1) Create a Kaggle Notebook

- Open Kaggle → `Code` → `New Notebook`
- Turn on `GPU` in `Notebook settings`
- Add your dataset(s) via `Add input`

### 2) Pick One Notebook Workflow

- **Training + inference:** `notebooks/Master_Training_n_Testing _notebook.ipynb`
- **Annotation visualization:** `notebooks/05_spectrogram_viz.ipynb`

### 3) Set Paths in Cell 1 and Run

Use Kaggle-style paths (example):

```python
# Master notebook
CONFIG["data"]["spectrogram_dir"] = "/kaggle/input/indianbats/spectrograms"
CONFIG["model"]["architecture"] = "resnet"  # or "transformer" or "cnn"

# Spectrogram visualization notebook
DATASET_ROOT = Path("/kaggle/input/indianbats/audio")
ANNOTATIONS_JSON = Path("/kaggle/input/indianbats/annotations.json")
```

Then run all cells in order.

---

### Quick Kaggle Guide (No Laptop Setup)

1. **Upload data once** to Kaggle Dataset(s)
2. **Attach dataset** to notebook (`Add input`)
3. **Edit only Cell 1 config**
4. **Run all cells**
5. **Save outputs** (model checkpoints, CSV results) to `/kaggle/working/`

---

### Common Kaggle Issues & Fast Fixes

| **Problem** | **Quick Fix** | **Detailed Section** |
|-------------|---------------|----------------------|
| "Resolved audio for 0/120 annotations" | Verify `DATASET_ROOT` points to attached Kaggle input path | [Troubleshooting](#troubleshooting--faq) |
| Runtime disconnected / timeout | Save checkpoints each epoch to `/kaggle/working/` | [Master Training Notebook](#-master-training-notebook-master_training_n_testing-_notebookipynb) |
| CUDA out of memory | Set `CONFIG["train"]["batch_size"] = 8` | [Hyperparameter Tuning](#-hyperparameter-tuning-guide) |
| Low accuracy on small dataset | Use `pretrained=True` and `freeze_backbone_epochs=10` | [Hyperparameter Tuning](#-hyperparameter-tuning-guide) |
| Too many false detections | Increase `confidence_thresh` and `energy_thresh` | [Inference Pipeline](#cell-5-inference-pipeline-production-use) |

---

### Need More Detail?

- **Parameter-by-parameter explanation:** [Master Training Notebook](#-master-training-notebook-master_training_n_testing-_notebookipynb)
- **Whombat UUID parsing and bounding boxes:** [Spectrogram Visualization Notebook](#-spectrogram-visualization-notebook-05_spectrogram_vizipynb)
- **Model choice (ResNet vs ViT vs CNN):** [Available Models & Architectures](#-available-models--architectures)
- **If results look wrong:** [Troubleshooting & FAQ](#-troubleshooting--faq)

---

## 📋 Detailed Documentation Navigation

- [Overview](#overview)
- [🎯 Master Training Notebook](#-master-training-notebook-master_training_n_testing-_notebookipynb) — **Complete training pipeline with swappable architectures**
- [📊 Spectrogram Visualization Notebook](#-spectrogram-visualization-notebook-05_spectrogram_vizipynb) — **Visualize Whombat annotations with bounding boxes**
- [Project Structure](#project-structure)
- [⚙️ Kaggle Setup (Primary)](#️-kaggle-setup-primary)
- [🧪 Usage Examples](#-usage-examples)

---

## Overview

### What This Project Does

1. **Parse Annotations** — Converts Whombat project exports (UUID-linked) to flat annotation lists
2. **Generate Spectrograms** — Creates mel-spectrogram PNGs from annotated audio segments
3. **Extract Features** — Computes acoustic parameters (end-frequency, duration, bandwidth)
4. **Train Models** — Plug-and-play training with ResNet18, Vision Transformer, or custom CNN
5. **Classify New Audio** — Inference pipeline with energy-based event detection for un-annotated recordings

### Key Capabilities

✅ **Plug-and-Play Architectures** — Change one config line to swap between ResNet18+MLP, Vision Transformer, or custom CNN  
✅ **Feature Fusion** — Combines visual (spectrogram) + acoustic (end-frequency) features  
✅ **Class Imbalance Handling** — Weighted sampling for unbalanced datasets  
✅ **Location-Agnostic** — Works in any folder (local, Kaggle, nested directories)  
✅ **Whombat Integration** — Full UUID chain resolution (sound_event_annotations → sound_events → recordings → audio files)  
✅ **Production-Ready Inference** — Energy-based bat call detector + classifier for raw audio

---

## 🎯 Master Training Notebook (`Master_Training_n_Testing _notebook.ipynb`)

**The complete training pipeline in one notebook. Change one config line to swap architectures — everything else adapts automatically.**

### What It Does

This notebook provides **5 modular cells** that take you from raw spectrograms to production inference:

1. **Cell 1 — Configuration Dict** — All hyperparameters in one place
2. **Cell 2 — Model Factory (Strategy Pattern)** — Plug-and-play architecture swapping
3. **Cell 3 — Data Loading + Stratified Splitting** — 70/20/10 train/val/test split with class balancing
4. **Cell 4 — Training + Evaluation** — Full training loop with early stopping, backbone freezing, metrics, plots
5. **Cell 5 — Inference Pipeline** — Event detection + classification for new audio files

---

### Cell 1: Master Configuration (The Control Center)

**Purpose:** Edit ONLY this dict to control the entire pipeline.

```python
CONFIG: Dict[str, Any] = {
    "data": {
        "spectrogram_dir": "data/processed/spectrograms",  # Folder with species subfolders
        "features_csv": "data/processed/features/end_frequencies.csv",  # Optional numeric features
        "num_classes": 3,          # Auto-detected from spectrogram_dir
        "image_size": (128, 128),  # Resize spectrograms to this (H, W)
    },
    
    "split": {
        "train": 0.70,    # Training ratio
        "val": 0.20,      # Validation ratio
        "test": 0.10,     # Test ratio (must sum to 1.0)
        "seed": 42,       # Random seed for reproducibility
        "stratify": True, # Maintain class distribution in splits
    },
    
    "model": {
        "architecture": "resnet",  # ⭐ CHANGE THIS: "resnet" | "transformer" | "cnn"
        "pretrained": True,        # Use ImageNet weights for ResNet/ViT
        "freeze_backbone_epochs": 3,  # Freeze backbone for N epochs (transfer learning)
        "num_features": 3,         # Number of extra numeric features to fuse
    },
    
    "train": {
        "epochs": 50,
        "batch_size": 16,
        "lr": 1e-3,               # Learning rate
        "weight_decay": 1e-4,     # L2 regularization
        "scheduler": "cosine",    # "cosine" | "step" | "none"
        "step_size": 10,          # For step scheduler
        "gamma": 0.5,             # LR decay factor
        "early_stop_patience": 8, # Stop if no improvement for N epochs
        "use_weighted_sampler": True,  # Handle class imbalance
    },
    
    "eval": {
        "show_confusion_matrix": True,
        "show_classification_report": True,
        "show_per_class_f1": True,
        "show_loss_curves": True,
    },
    
    "inference": {
        "model_path": "models/bat_fused_best.pth",
        "audio_dir": "data/raw/test_audio",
        "sample_rate": 250_000,    # Expected sample rate for bat calls
        "min_freq": 15_000,        # High-pass filter (Hz) for bat range
        "energy_thresh": 0.02,     # RMS threshold for event detection
        "min_duration": 0.01,      # Minimum event duration (seconds)
        "pad_duration": 0.05,      # Padding around detected events
        "confidence_thresh": 0.70, # Minimum confidence to report
    },
}
```

#### What Happens When You Change Settings

| **Setting** | **What Happens** | **Example** |
|-------------|------------------|-------------|
| `architecture: "resnet"` | Uses ResNet18 backbone + feature MLP (512-d image embedding + numeric features → 256 → N classes). Fast, accurate. | Good for small-medium datasets |
| `architecture: "transformer"` | Uses Vision Transformer (ViT-B/16) + feature MLP. Attention-based, needs more data. | Best for large datasets (>1000 samples) |
| `architecture: "cnn"` | Uses custom 3-layer CNN. Lightweight, trains from scratch. | Good for edge deployment or baseline |
| `pretrained: False` | Trains backbone from scratch (random init). **Requires more data and epochs.** | Use when your spectrograms are very different from ImageNet |
| `freeze_backbone_epochs: 0` | Fine-tunes backbone from epoch 1. **Higher risk of overfitting** if dataset is small. | Skip freezing if you have >5000 samples |
| `freeze_backbone_epochs: 10` | Freezes backbone for 10 epochs, then unfreezes. Transfer learning strategy. | Recommended for <1000 samples |
| `scheduler: "cosine"` | Smoothly decreases LR from `lr` to near-zero over `epochs`. Good for convergence. | Default choice |
| `scheduler: "step"` | Drops LR by `gamma` every `step_size` epochs (e.g., 0.001 → 0.0005 → 0.00025). | Use for step-wise experimentation |
| `scheduler: "none"` | Fixed LR throughout training. **Faster but may not converge well.** | Debug mode only |
| `use_weighted_sampler: True` | Oversamples minority classes (e.g., if Species A has 100 samples, Species B has 10, B gets sampled 10× more often per epoch). | Essential for imbalanced data |
| `early_stop_patience: 3` | Stops training if validation accuracy doesn't improve for 3 consecutive epochs. Saves time. | Use shorter patience for small datasets |
| `stratify: False` | Random split without preserving class ratios. **Validation set may have different distribution.** | Only if classes are perfectly balanced |

---

### Cell 2: Model Factory (Strategy Pattern)

**Purpose:** Define all available architectures. You don't edit this — just change `CONFIG["model"]["architecture"]` in Cell 1.

#### Current Architectures

1. **`ResNetStrategy`** (`"resnet"`)
   - **Backbone:** ResNet18 pretrained on ImageNet
   - **Image embedding:** 512 dimensions
   - **Feature fusion:** Concatenates image embedding + numeric features (e.g., end-frequency)
   - **Classifier:** `[512+3] → 256 (ReLU, Dropout 0.3) → N classes`
   - **Parameters:** ~11.2M (most are pretrained)
   - **When to use:** Default choice for most datasets

2. **`TransformerStrategy`** (`"transformer"`)
   - **Backbone:** ViT-B/16 (Vision Transformer) pretrained on ImageNet
   - **Image embedding:** 768 dimensions (from final attention layer)
   - **Feature fusion:** MLP for numeric features (3 → 32), concatenates with image embedding
   - **Classifier:** `[768+32] → 128 (ReLU, Dropout 0.4) → N classes`
   - **Parameters:** ~86M
   - **When to use:** Large datasets (>2000 samples), when you want attention maps, cutting-edge performance

3. **Custom CNN** (not yet wired, available in `src/models/cnn.py`)
   - **Architecture:** 3 conv layers (16→32→64 filters) + 2 FC layers
   - **Parameters:** ~0.5M
   - **When to use:** Baseline comparison, edge deployment, training from scratch

#### How to Add a New Architecture

```python
# In Cell 2, add this block:
class MobileNetStrategy(ModelStrategy):
    def name(self) -> str:
        return "MobileNetV3"
    
    def build(self, num_classes: int, **kw) -> nn.Module:
        from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
        weights = MobileNet_V3_Small_Weights.DEFAULT if kw.get("pretrained") else None
        backbone = mobilenet_v3_small(weights=weights)
        backbone.classifier = nn.Identity()  # Remove head, get 576-d embedding
        # ... rest of implementation
        return model

# Register it:
_STRATEGIES["mobilenet"] = MobileNetStrategy()
```

Then set `CONFIG["model"]["architecture"] = "mobilenet"` and re-run.

---

### Cell 3: Data Loading + Stratified Split

**Purpose:** Loads spectrograms from `data/processed/spectrograms/<species>/` folder structure and splits into train/val/test.

#### What Happens

1. **Scans folder structure:**
   ```
   data/processed/spectrograms/
   ├── Pipistrellus/
   │   ├── call_001.png
   │   ├── call_002.png
   │   └── ...
   ├── Myotis/
   │   └── ...
   └── Rhinolophus/
       └── ...
   ```

2. **Auto-detects classes:** Folder names become class labels (e.g., `['Myotis', 'Pipistrellus', 'Rhinolophus']`)

3. **Loads features (optional):** If `features_csv` exists, loads numeric features (e.g., end-frequency) and matches them to spectrograms by filename

4. **Stratified split:** Uses `StratifiedShuffleSplit` to maintain class distribution across train/val/test
   - Example: If Pipistrellus is 60% of dataset, it will be 60% of train, 60% of val, 60% of test

5. **Prints distribution:**
   ```
   Total samples: 450
   Detected 3 classes: ['Myotis', 'Pipistrellus', 'Rhinolophus']
     Train (315): {'Myotis': 105, 'Pipistrellus': 189, 'Rhinolophus': 21}
     Val   ( 90): {'Myotis':  30, 'Pipistrellus':  54, 'Rhinolophus':  6}
     Test  ( 45): {'Myotis':  15, 'Pipistrellus':  27, 'Rhinolophus':  3}
   ```

#### What Changes Affect This

- **`stratify: False`** → Random split (ignores class distribution)
- **`seed: 123`** → Changes which samples go into train/val/test (for reproducibility experiments)
- **`train/val/test ratios`** → Must sum to 1.0; e.g., `0.8/0.1/0.1` gives more training data

---

### Cell 4: Training + Evaluation (The Main Event)

**Purpose:** Trains the model with early stopping, evaluates on test set, shows metrics.

#### Training Loop Steps

1. **Build model** via factory (Cell 2's `build_model()`)
2. **Create data loaders:**
   - If `use_weighted_sampler: True`, oversamples minority classes
   - Otherwise, shuffles training data normally
3. **Freeze backbone** (if `freeze_backbone_epochs > 0`)
   - Only trains classifier head for first N epochs
   - Prevents catastrophic forgetting of ImageNet features
4. **Training epoch:**
   ```
   For each batch:
     1. Forward pass → loss
     2. Backward pass → gradients
     3. Optimizer step
     4. Accumulate loss
   ```
5. **Validation epoch:**
   ```
   For each batch (no gradients):
     1. Forward pass
     2. Compute loss and accuracy
   ```
6. **Unfreeze backbone** at epoch `freeze_backbone_epochs + 1`
7. **Learning rate scheduling:**
   - `cosine`: Smooth decay to 0
   - `step`: Drops LR every `step_size` epochs
8. **Early stopping:**
   - If validation accuracy doesn't improve for `early_stop_patience` epochs → stop
   - Saves best model checkpoint (highest val accuracy)
9. **Load best checkpoint** and evaluate on test set
10. **Show metrics:**
    - Classification report (precision, recall, F1 per class)
    - Confusion matrix heatmap
    - Loss/accuracy curves over epochs

#### Example Output

```
Building model: ResNet18+MLP
Parameters: 11,689,155

Epoch   1/50 │ TrL 1.2340 │ VaL 0.9876 │ VaAcc 0.6111 │ LR 1.00e-03
Epoch   2/50 │ TrL 0.8765 │ VaL 0.7654 │ VaAcc 0.7222 │ LR 9.99e-04
Epoch   3/50 │ TrL 0.6543 │ VaL 0.6543 │ VaAcc 0.7778 │ LR 9.97e-04
Epoch   4/50 │ TrL 0.5432 │ VaL 0.5987 │ VaAcc 0.8111 │ LR 9.94e-04
...
Epoch  12/50 │ TrL 0.2345 │ VaL 0.4321 │ VaAcc 0.8778 │ LR 9.51e-04
Early stopping at epoch 12

Best val accuracy: 0.8778  →  models/bat_fused_best.pth

============================================================
CLASSIFICATION REPORT (Test)
============================================================
              precision    recall  f1-score   support
      Myotis     0.9333    0.9333    0.9333        15
Pipistrellus     0.9259    0.9259    0.9259        27
Rhinolophus     0.6667    0.6667    0.6667         3
   micro avg     0.9111    0.9111    0.9111        45
   macro avg     0.8420    0.8420    0.8420        45
```

#### What Changes Affect Training

| **Change** | **Effect** |
|------------|------------|
| Higher `batch_size` | Faster training, less noisy gradients, needs more GPU memory |
| Lower `batch_size` | Slower, noisier gradients (acts as regularization), works on small GPUs |
| Higher `lr` | Faster convergence BUT risk of overshooting optimal weights |
| Lower `lr` | Slower, more stable convergence, may need more epochs |
| `weight_decay: 0` | No L2 regularization → higher risk of overfitting |
| `weight_decay: 1e-3` | Strong regularization → may underfit on small datasets |

---

### Cell 5: Inference Pipeline (Production Use)

**Purpose:** Classify bat calls in NEW un-annotated audio files.

#### How It Works

1. **Load trained model** from `model_path`
2. **Find audio files** in `audio_dir` (supports `.wav`, `.flac`, `.mp3`)
3. **For each audio file:**
   ```
   a. Load audio → bandpass filter (15-120 kHz for bat range)
   b. Energy-based event detection:
      - Compute RMS energy in sliding windows
      - Detect segments where RMS > energy_thresh
      - Merge overlapping events
   c. For each detected event:
      - Extract segment + padding
      - Generate mel-spectrogram
      - Compute end-frequency feature
      - Classify via model
      - Report if confidence > confidence_thresh
   ```
4. **Output DataFrame:**
   ```
   | file          | start | end   | prediction   | confidence |
   |---------------|-------|-------|--------------|------------|
   | audio_01.wav  | 2.340 | 2.450 | Pipistrellus | 94.23      |
   | audio_01.wav  | 5.120 | 5.280 | Myotis       | 87.65      |
   | audio_02.wav  | -     | -     | No Detection | 0.00       |
   ```

#### What Changes Affect Inference

| **Setting** | **Effect** |
|-------------|------------|
| `min_freq: 20000` | Ignores calls below 20 kHz (filters out noise) |
| `energy_thresh: 0.01` | Lower threshold → detects quieter calls (more false positives) |
| `energy_thresh: 0.05` | Higher threshold → only loud calls (may miss weak ones) |
| `min_duration: 0.005` | Detects very short chirps (<5ms) |
| `confidence_thresh: 0.90` | Only reports highly confident predictions (fewer results) |
| `pad_duration: 0.1` | More context around calls (better for variable-length calls) |

---

### Quick Start (Master Notebook)

```python
# 1. Edit Cell 1 CONFIG dict
CONFIG["model"]["architecture"] = "resnet"  # or "transformer"
CONFIG["data"]["spectrogram_dir"] = "path/to/your/spectrograms"

# 2. Run all cells in order
# Cell 1 → loads config
# Cell 2 → registers models
# Cell 3 → loads data and splits
# Cell 4 → trains model (saves to models/bat_fused_best.pth)
# Cell 5 → (optional) run inference on new audio

# 3. To switch architectures, just change Cell 1 and re-run Cell 4
CONFIG["model"]["architecture"] = "transformer"
# Re-run Cell 4 → trains ViT model instead
```

---

## 📊 Spectrogram Visualization Notebook (`05_spectrogram_viz.ipynb`)

**Parses Whombat annotation exports and renders mel-spectrograms with frequency bounding boxes overlaid. Essential for data quality checks and annotation validation.**

### What It Does

This notebook has **4 cells** that resolve Whombat's UUID chain and visualize annotated bat calls:

1. **Cell 1 — Configuration** — Set paths and spectrogram parameters
2. **Cell 2 — Whombat JSON Parser** — Resolves full UUID chain (annotations → sound_events → recordings → audio files)
3. **Cell 3 — Render Spectrograms** — Dynamic windowing + bounding box overlay
4. **Cell 4 — Grid View by Species** — Shows up to 5 examples per species in subplot grids

---

### Cell 1: Configuration

```python
# Paths
DATASET_ROOT = Path("/kaggle/input/your-dataset")  # Folder containing audio files
ANNOTATIONS_JSON = Path("/kaggle/input/your-dataset/annotations.json")  # Whombat export

# Spectrogram rendering
SR_TARGET: Optional[int] = None  # None = use native sample rate
N_FFT = 2048          # FFT window size
HOP_LENGTH = 512      # Hop length (affects time resolution)
N_MELS = 128          # Number of mel bins (frequency resolution)
F_MIN = 0             # Min frequency (Hz)
F_MAX: Optional[int] = None  # Max frequency (None = Nyquist)
CMAP = "magma"        # Colormap: "magma", "viridis", "plasma", "inferno"

# Visualization
MAX_ANNOTATIONS_TO_SHOW = 30
BBOX_COLOR = "cyan"
BBOX_LINEWIDTH = 1.5
FIGURE_DPI = 120
```

#### What Happens When You Change Settings

| **Setting** | **Effect** | **Example** |
|-------------|------------|-------------|
| `N_FFT: 1024` | Lower frequency resolution, faster rendering | Good for broadband calls |
| `N_FFT: 4096` | Higher frequency resolution, slower | Good for narrow-band calls with harmonics |
| `HOP_LENGTH: 256` | Better time resolution (more frames), larger images | Use for fast modulations |
| `HOP_LENGTH: 1024` | Lower time resolution, smaller images | Use for long calls |
| `N_MELS: 64` | Coarser frequency binning | Faster, less detail |
| `N_MELS: 256` | Finer frequency binning | Slower, more detail |
| `F_MAX: 60000` | Only shows 0-60 kHz (crops above Nyquist for bat calls) | Recommended for 250 kHz sample rate |
| `CMAP: "viridis"` | Green-blue-purple colormap | Better for colorblind users |

---

### Cell 2: Whombat JSON Parser (UUID Chain Resolution)

**Purpose:** Converts Whombat's nested UUID references into a flat list of annotations with resolved audio paths.

#### Whombat Schema (UUID Chain)

Whombat exports have this structure:
```json
{
  "data": {
    "tags": [
      {"id": 1, "key": "Species", "value": "Pipistrellus"}
    ],
    "recordings": [
      {"uuid": "abc-123", "path": "C:\\Data\\audio_01.wav"}
    ],
    "sound_events": [
      {"uuid": "def-456", "recording": "abc-123", "geometry": {"coordinates": [2.3, 15000, 2.5, 60000]}}
    ],
    "sound_event_annotations": [
      {"sound_event": "def-456", "tags": [1]}
    ]
  }
}
```

#### Resolution Steps

1. **`tags` → `tag_id_to_label`**
   - Maps tag ID → species name (e.g., `1 → "Pipistrellus"`)

2. **`recordings` → `rec_uuid_to_info`**
   - Maps recording UUID → `(full_path, basename)`
   - Handles Windows/POSIX paths: `C:\Data\file.wav` → `file.wav`

3. **`sound_events` → `se_uuid_to_geom`**
   - Maps sound event UUID → `(recording_uuid, start_s, end_s, low_hz, high_hz)`
   - Geometry format: `[start_time, low_freq, end_time, high_freq]`

4. **`sound_event_annotations` → `se_uuid_to_tags`**
   - Maps sound event UUID → list of tag IDs

5. **Resolve full chain:**
   ```
   annotation → sound_event UUID → recording UUID → audio basename → search for file
   ```

6. **Audio file search:**
   - Tries exact basename match
   - Tries common extensions (`.wav`, `.WAV`, `.flac`, `.mp3`, `.m4a`)
   - Recursively searches subdirectories

#### Output

```python
ParsedAnnotation(
    recording_uuid="abc-123",
    audio_basename="audio_01.wav",
    audio_path=Path("/data/recordings/audio_01.wav"),  # Resolved!
    start_s=2.3,
    end_s=2.5,
    low_hz=15000.0,
    high_hz=60000.0,
    label="Pipistrellus",
    sound_event_uuid="def-456",
    annotation_index=0
)
```

```
Parsed 120 annotations from 45 recordings
  Tags found: {1: 'Pipistrellus', 2: 'Myotis', 3: 'Rhinolophus'}
  Sound events with geometry: 120

Resolved audio for 118/120 annotations
Species: {'Pipistrellus', 'Myotis', 'Rhinolophus'}
```

---

### Cell 3: Render Spectrograms + Bounding Boxes

**Purpose:** For each annotation, generates a mel-spectrogram with dynamic windowing and overlays a frequency bounding box.

#### Dynamic Windowing Logic

Problem: Bat calls vary from **10 ms** (short chirps) to **5 seconds** (long social calls). Showing the full audio file would make short calls invisible.

Solution: **Compute a display window that centers the annotation:**

```python
if clip_duration < 1.0 s:
    window = max(0.9 s, clip_duration × 3)  # ~3× zoom for short calls
else:
    window = max(5.0 s, clip_duration × 2)  # ~2× zoom for long calls
```

**Example:**
- Call duration: 50 ms → window: 0.9 s (shows call + 0.85 s context)
- Call duration: 2.5 s → window: 5.0 s (shows call + 2.5 s context)

#### Rendering Steps

1. **Load audio** (cached to avoid re-reading same file)
2. **Compute display window** (centered on annotation)
3. **Extract audio segment** `[win_start : win_end]`
4. **Generate mel-spectrogram:**
   ```python
   S = librosa.feature.melspectrogram(
       y=audio_window, sr=sr,
       n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS,
       fmin=F_MIN, fmax=F_MAX
   )
   S_db = librosa.power_to_db(S, ref=np.max)  # Convert to decibels
   ```
5. **Plot spectrogram** (x-axis: time, y-axis: mel/frequency)
6. **Overlay bounding box:**
   ```python
   Rectangle(
       (start_s - win_start, low_hz),  # Bottom-left corner
       width = end_s - start_s,         # Duration
       height = high_hz - low_hz        # Bandwidth
   )
   ```
7. **Add label text** above box (species name)

#### Example Output

```
Rendering 30 annotations ...

[Shows 30 individual spectrograms, each with:]
audio_01.wav | Pipistrellus | 2.300–2.500s | 15.0–60.0 kHz
  ┌─────────────────────────────────┐
  │                                 │  60 kHz
  │     ┌──cyan box──┐              │
  │     │            │              │
  │     │  Annotation│              │
  │     └────────────┘              │  15 kHz
  │                                 │
  └─────────────────────────────────┘
    2.0s     2.5s     3.0s

✓ Rendered 30 spectrograms
```

---

### Cell 4: Grid View by Species

**Purpose:** Shows up to 5 examples per species in a single row for quick comparison.

#### What It Does

1. **Group annotations by species** (label)
2. **For each species:**
   - Take first 5 annotations
   - Create subplot grid: 1 row × N columns
   - Render each annotation in its subplot
3. **Display:**
   ```
   Species: Pipistrellus (89 total)
   ┌──────┬──────┬──────┬──────┬──────┐
   │ Ex 1 │ Ex 2 │ Ex 3 │ Ex 4 │ Ex 5 │
   └──────┴──────┴──────┴──────┴──────┘
   
   Species: Myotis (24 total)
   ┌──────┬──────┬──────┐
   │ Ex 1 │ Ex 2 │ Ex 3 │
   └──────┴──────┴──────┘
   ```

#### Use Cases

- **Data quality check:** Verify all species have consistent call patterns
- **Annotation validation:** Spot mislabeled calls (e.g., Myotis call labeled as Pipistrellus)
- **Frequency range check:** Ensure bounding boxes align with actual call energy
- **Dataset balance:** Quickly see which species have few examples

---

### Quick Start (Spectrogram Viz)

```python
# 1. Edit Cell 1
DATASET_ROOT = Path("path/to/your/audio/folder")
ANNOTATIONS_JSON = Path("path/to/whombat_export.json")

# 2. Run Cell 2 (parses JSON, resolves audio paths)
# Output: "Resolved audio for 118/120 annotations"

# 3. Run Cell 3 (renders first 30 annotated spectrograms)
# Shows individual plots with bounding boxes

# 4. Run Cell 4 (grid view by species)
# Shows up to 5 examples per species side-by-side
```

---

### Troubleshooting

| **Problem** | **Cause** | **Solution** |
|-------------|-----------|--------------|
| "Resolved audio for 0/120 annotations" | Audio files not in `DATASET_ROOT` or subfolders | Add parent directories to `audio_search_dirs` in Cell 2 |
| Bounding boxes don't align with calls | Geometry coordinates are in wrong units (samples vs. seconds) | Check Whombat export: coordinates should be `[start_s, low_hz, end_s, high_hz]` |
| Spectrograms are blank | Audio files are stereo but code expects mono | Librosa auto-converts to mono, but check if files are corrupted |
| "Too short to render" | Call duration < N_FFT samples | Reduce `N_FFT` to 512 or 1024 |
| Out of memory | Rendering too many large spectrograms | Reduce `MAX_ANNOTATIONS_TO_SHOW` to 10 |

---

## 📁 Project Structure

```
IndianBatsModel-main/
├── notebooks/                           # 🎯 PRIMARY NOTEBOOKS
│   ├── Master_Training_n_Testing _notebook.ipynb   # Complete training pipeline with Strategy Pattern
│   └── 05_spectrogram_viz.ipynb                    # Whombat annotation visualizer
│
├── src/                                 # Core source code (location-agnostic)
│   ├── __init__.py                     # Auto-import setup for portability
│   ├── train.py                        # Training entry-point
│   ├── evaluate.py                     # Model evaluation script
│   ├── utils.py                        # Shared helper functions
│   │
│   ├── models/                          # Neural network architectures
│   │   ├── __init__.py
│   │   ├── cnn.py                      # Lightweight 3-layer CNN (~0.5M params)
│   │   ├── cnn_with_features.py        # ResNet18 + MLP fusion (11.2M params)
│   │   └── bat_classifier.py           # Legacy TensorFlow model
│   │
│   ├── datasets/                        # PyTorch Dataset loaders
│   │   ├── __init__.py
│   │   ├── spectrogram_dataset.py      # Basic image-only dataset
│   │   └── spectrogram_with_features_dataset.py  # Image + numeric features
│   │
│   └── data_prep/                       # Preprocessing pipelines
│       ├── __init__.py
│       ├── whombat_project_to_wombat.py          # Whombat export → Wombat JSONs
│       ├── wombat_to_spectrograms.py             # Audio + JSON → spectrograms
│       ├── extract_end_frequency.py              # Compute dominant end-frequency
│       └── generate_annotations.py               # Auto-annotation generator
│
├── data/                                # Dataset storage (managed by user)
│   ├── raw/                            # Original audio files (.wav, .flac)
│   ├── processed/                      # Generated data
│   │   ├── spectrograms/               # PNG spectrograms by species
│   │   └── features/                   # CSVs with numeric features
│   └── annotations/                    # Whombat/Wombat JSON exports
│
├── configs/
│   └── config.yaml                     # Legacy config (use notebook CONFIG dicts instead)
│
├── notebooks_for_kaggle/               # Older Kaggle-specific notebooks
│   ├── kaggle_inference_pipeline.ipynb
│   ├── kaggle_train.ipynb
│   └── kaggle_smart_tuning.ipynb
│
├── scripts/
│   └── prepare_data.sh.deprecated      # Shell automation (replaced by Python modules)
│
├── tests/
│   └── test_dataset.py                 # Unit tests for datasets
│
├── requirements.txt                    # Python dependencies
├── setup.py                            # Package installer
└── README.md                           # This file
```

**Key Design Principles:**

1. **Location-Agnostic:** All code works from any folder via auto-import setup in `src/__init__.py`
2. **Notebook-First:** Primary workflows live in Jupyter notebooks with inline CONFIG dicts
3. **Plug-and-Play Architectures:** Change one line to swap models (ResNet ↔ ViT ↔ CNN)
4. **Modular Data Prep:** Each preprocessing step is an independent script

---

## ⚙️ Kaggle Setup (Primary)

### Recommended Runtime

- **Platform:** Kaggle Notebook
- **Accelerator:** GPU (T4 / P100 or available option)
- **Input data:** Attach via `Add input`
- **Working output path:** `/kaggle/working/`

### Minimal Kaggle Steps

1. Create notebook in Kaggle and enable GPU
2. Attach dataset containing audio/spectrogram/annotation files
3. Open either:
   - `notebooks/Master_Training_n_Testing _notebook.ipynb`, or
   - `notebooks/05_spectrogram_viz.ipynb`
4. Update Cell 1 paths to `/kaggle/input/<dataset-name>/...`
5. Run all cells

### Local Setup (Optional, Not Recommended)

If you still need local execution for development:

```bash
git clone <repo-url>
cd IndianBatsModel-main
pip install -r requirements.txt
```

---

## 📌 Kaggle Run Checklist

Use this short checklist before running either notebook:

1. GPU is enabled in notebook settings
2. Correct Kaggle dataset is attached in `Add input`
3. Cell 1 paths point to `/kaggle/input/...`
4. `architecture` is set (`resnet` for default, `transformer` for larger data)
5. Output paths are in `/kaggle/working/` for persistence within the session

For complete workflow details, continue with the sections below.

---

## 🧪 Usage Examples

### Example 1: Train ResNet Model for 3-Species Classification

```python
# In Master Notebook Cell 1:
CONFIG = {
    "data": {
        "spectrogram_dir": "data/processed/spectrograms",
        "features_csv": "data/processed/features/end_frequencies.csv",
        "num_classes": 3,  # Auto-detected
        "image_size": (128, 128),
    },
    "split": {"train": 0.70, "val": 0.20, "test": 0.10, "seed": 42},
    "model": {
        "architecture": "resnet",
        "pretrained": True,
        "freeze_backbone_epochs": 3,
        "num_features": 3,
    },
    "train": {
        "epochs": 30,
        "batch_size": 16,
        "lr": 1e-3,
        "scheduler": "cosine",
        "early_stop_patience": 8,
        "use_weighted_sampler": True,  # For imbalanced data
    },
}

# Run Cells 2-4 → trains model
# Output: 30 epochs, early stopping at ~epoch 22
# Test accuracy: 0.91
# Saved: models/bat_fused_best.pth
```

---

### Example 2: Switch to Vision Transformer (One Line Change!)

```python
# Change only this line in Cell 1:
CONFIG["model"]["architecture"] = "transformer"  # Was: "resnet"

# Re-run Cell 4 → trains ViT-B/16 model instead
# Everything else (data loading, training loop, metrics) stays the same!
```

---

### Example 3: Visualize Whombat Annotations

```python
# Cell 1 in Spectrogram Viz notebook:
DATASET_ROOT = Path("/data/bat_recordings")
ANNOTATIONS_JSON = Path("/data/whombat_export.json")

# Cell 2 output:
# Parsed 120 annotations from 45 recordings
# Resolved audio for 118/120 annotations (2 files missing)

# Cell 3 renders 30 spectrograms with cyan bounding boxes:
# audio_01.wav | Pipistrellus | 2.300–2.500s | 15.0–60.0 kHz

# Cell 4 shows grid:
# Species: Pipistrellus (89 total) → 5 example spectrograms
# Species: Myotis (24 total) → 3 example spectrograms
# Species: Rhinolophus (7 total) → 5 example spectrograms
```

---

### Example 4: Run Inference on New Audio Files

```python
# In Master Notebook Cell 5:
CONFIG["inference"]["audio_dir"] = "data/raw/new_recordings"
CONFIG["inference"]["model_path"] = "models/bat_fused_best.pth"

# Run Cell 5 → detects and classifies bat calls:
# Output DataFrame:
# | file          | start | end   | prediction   | confidence |
# |---------------|-------|-------|--------------|------------|
# | rec_01.wav    | 2.340 | 2.450 | Pipistrellus | 94.23      |
# | rec_01.wav    | 5.120 | 5.280 | Myotis       | 87.65      |
# | rec_02.wav    | 1.200 | 1.350 | Pipistrellus | 91.04      |
```

---

## 🛠️ Data Preparation Pipeline

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
---

**🎉 README rewrite complete!**

This comprehensive guide focuses on the **Master Training Notebook** and **Spectrogram Visualization Notebook**, explaining:
- ✅ What each CONFIG parameter does
- ✅ What happens when you change architectures (ResNet → Transformer → CNN)
- ✅ Effects of hyperparameter changes (batch_size, learning rate, scheduler, etc.)
- ✅ How Whombat UUID chain resolution works
- ✅ Dynamic windowing for spectrogram visualization
- ✅ Complete data preparation pipeline
- ✅ Troubleshooting guide for common issues

For detailed API documentation of the `src/` modules, see inline docstrings in each Python file.

---