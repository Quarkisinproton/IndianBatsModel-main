"""
IndianBatsModel – canonical source package.

Consolidated from MainShitz/ and Model/ directories.
All core logic lives here now. Import paths:
    from src.models import CNNWithFeatures, CNN
    from src.datasets import SpectrogramDataset, SpectrogramWithFeaturesDataset
    from src.data_prep import generate_spectrograms, convert_whombat, extract_features
"""

import sys
from pathlib import Path

def _setup_imports():
    """
    Auto-detect the repo root (directory containing 'src/') and ensure it's in sys.path.
    This makes the code work regardless of where the repo is copied.
    """
    # Start from this file's directory and go up to repo root
    current = Path(__file__).parent.parent
    
    # Verify we're in the repo root by checking for src/ subdirectory
    if (current / "src").exists() and (current / "src" / "__init__.py").exists():
        repo_root = str(current)
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        return repo_root
    
    # Fallback: assume current working directory is repo root
    cwd = str(Path.cwd())
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
    return cwd

# Auto-setup on import
_setup_imports()
