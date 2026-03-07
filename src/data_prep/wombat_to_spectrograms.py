"""
wombat_to_spectrograms.py – Turn bat squeaks into pretty pictures.

Reads Wombat JSON annotations, finds the corresponding audio files,
chops them into segments, and generates mel spectrograms.
"""
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import librosa
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for servers / notebooks
import matplotlib.pyplot as plt
import numpy as np


def ensure_dir(path: Path) -> None:
    """Create directory tree (like ``mkdir -p``)."""
    path.mkdir(parents=True, exist_ok=True)


def load_wombat_json(path: Path) -> Dict:
    """Read a Wombat-style JSON annotation file."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_audio_for_json(json_path: Path, raw_audio_dirs: List[Path]) -> Optional[Path]:
    """Resolve the audio file for a given JSON annotation."""
    data = load_wombat_json(json_path)
    rec = None
    if isinstance(data, dict):
        rec = data.get("recording") or data.get("audio_file") or data.get("file")

    if rec:
        p = Path(rec)
        if p.is_absolute() and p.exists():
            return p
        for d in raw_audio_dirs:
            candidate = d / p.name
            if candidate.exists():
                return candidate

    # fallback: match by stem name
    stem = json_path.stem
    for d in raw_audio_dirs:
        for ext in (".wav", ".flac", ".mp3", ".m4a"):
            cand = d / (stem + ext)
            if cand.exists():
                return cand
        for f in d.glob("*"):
            if stem in f.stem:
                return f
    return None


def extract_segment(y: np.ndarray, sr: int, start_s: float, end_s: float) -> np.ndarray:
    """Extract an audio segment by time bounds."""
    start = max(0, int(start_s * sr))
    end = min(len(y), int(end_s * sr))
    return y[start:end]


def make_mel_spectrogram(
    y: np.ndarray,
    sr: int,
    n_mels: int = 128,
    hop_length: int = 512,
) -> np.ndarray:
    """Compute a mel-spectrogram in dB scale."""
    if y.size == 0:
        return np.zeros((n_mels, 1), dtype=float)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    return librosa.power_to_db(S, ref=np.max)


def save_spectrogram_image(
    S_db: np.ndarray,
    out_path: Path,
    cmap: str = "magma",
    dpi: int = 100,
) -> None:
    """Save a spectrogram array as a PNG image."""
    ensure_dir(out_path.parent)
    plt.figure(figsize=(3, 3))
    plt.axis("off")
    plt.imshow(S_db, aspect="auto", origin="lower", cmap=cmap)
    plt.tight_layout(pad=0)
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0, dpi=dpi)
    plt.close()


def normalize_annotations(raw_anns) -> List[Dict]:
    """Normalize annotation payloads to a list of dicts."""
    if raw_anns is None:
        return []
    if isinstance(raw_anns, dict):
        return [raw_anns]
    if isinstance(raw_anns, list):
        return raw_anns
    return []


def get_first_present_key(d: Dict, keys: List[str]):
    """Return the value of the first key found in *d*."""
    for k in keys:
        if k in d:
            return d[k]
    return None


def process_audio_file(
    audio_path: Path,
    annotations: Iterable[Dict],
    out_base: Path,
    species_key: str = "label",
) -> None:
    """Generate spectrogram PNGs for all annotations of one audio file."""
    try:
        y, sr = librosa.load(str(audio_path), sr=None)
    except Exception as e:
        print(f"Error loading audio {audio_path}: {e}")
        return

    for i, ann in enumerate(annotations):
        start = get_first_present_key(ann, ["start_time", "start", "t0", "onset"])
        end = get_first_present_key(ann, ["end_time", "end", "t1", "offset"])
        label = get_first_present_key(ann, [species_key, "species", "label", "class"])

        if start is None or end is None or label is None:
            print(f"Skipping annotation {i} in {audio_path.name}: Missing data")
            continue
        try:
            start_f, end_f = float(start), float(end)
        except Exception:
            print(f"Skipping annotation {i}: Invalid time format")
            continue

        seg = extract_segment(y, sr, start_f, end_f)
        if seg.size == 0:
            continue

        S_db = make_mel_spectrogram(seg, sr)
        safe_label = str(label).strip().replace("/", "_").replace("\\", "_").replace(os.sep, "_")
        safe_label = "_".join(safe_label.split())
        out_dir = out_base / safe_label
        ensure_dir(out_dir)
        out_path = out_dir / f"{audio_path.stem}_{i}.png"

        try:
            save_spectrogram_image(S_db, out_path)
        except Exception as e:
            print(f"Error saving spectrogram to {out_path}: {e}")


def process_all(
    raw_audio_dirs: List[str],
    json_dir: str,
    out_dir: str,
    species_key: str = "label",
) -> None:
    """Process all JSON annotations and generate spectrograms."""
    raw_audio_dirs_p = [Path(d) for d in raw_audio_dirs]
    json_dir_p = Path(json_dir)
    out_base = Path(out_dir)
    ensure_dir(out_base)

    print(f"Scanning for JSONs in {json_dir_p} ...")
    json_files = list(json_dir_p.rglob("*.json"))
    print(f"Found {len(json_files)} JSON files.")

    if not json_files:
        print("WARNING: No JSON files found. Spectrogram generation skipped.")
        return

    try:
        from tqdm import tqdm
        iterator = tqdm(json_files, desc="Processing JSONs")
    except ImportError:
        iterator = json_files

    processed = 0
    for jpath in iterator:
        try:
            data = load_wombat_json(jpath)
        except Exception as e:
            print(f"Error loading {jpath}: {e}")
            continue

        audio_path = find_audio_for_json(jpath, raw_audio_dirs_p)
        if audio_path is None:
            print(f"Warning: Could not find audio for {jpath.name}")
            continue

        anns = None
        if isinstance(data, dict):
            for key in ("annotations", "labels", "segments", "events"):
                if key in data:
                    anns = data[key]
                    break
            if anns is None and any(k in data for k in ("start_time", "end_time", species_key)):
                anns = [data]
        else:
            anns = data

        anns = normalize_annotations(anns)
        if not anns:
            continue

        process_audio_file(audio_path, anns, out_base, species_key=species_key)
        processed += 1

    print(f"Processed {processed} files successfully.")
    if out_base.exists():
        subdirs = [d.name for d in out_base.iterdir() if d.is_dir()]
        total = sum(len(list(d.glob("*.png"))) for d in out_base.iterdir() if d.is_dir())
        print(f"Species folders: {subdirs}  |  Total images: {total}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Wombat JSON + audio → spectrogram PNGs")
    parser.add_argument("--raw_audio_dir", required=True, nargs="+")
    parser.add_argument("--json_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--species_key", default="label")
    args = parser.parse_args()
    process_all(args.raw_audio_dir, args.json_dir, args.out_dir, species_key=args.species_key)
