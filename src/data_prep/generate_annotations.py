"""
generate_annotations.py – Auto-generate Wombat-style JSON annotations.

Creates per-audio JSON annotations from raw audio directories.
Labels are derived from folder names or filenames.
"""
import json
from pathlib import Path
from typing import List

import librosa


def generate_annotations(
    raw_audio_dirs: List[str],
    output_dir: str,
    label_strategy: str = "folder",
) -> None:
    """Generate Wombat-style annotation JSONs from audio files.

    Args:
        raw_audio_dirs: Directories containing audio files.
        output_dir: Where to write the JSONs.
        label_strategy: ``'folder'`` (parent dir name) or ``'filename'`` (stem).
    """
    output_dir_p = Path(output_dir)
    output_dir_p.mkdir(parents=True, exist_ok=True)
    raw_dirs = [Path(d) for d in raw_audio_dirs]

    print(f"Generating annotations for {len(raw_dirs)} directories ...")
    print(f"Output: {output_dir_p}  |  Strategy: {label_strategy}")

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None  # type: ignore[assignment]

    for d in raw_dirs:
        if not d.exists():
            print(f"Warning: not found – {d}")
            continue

        audio_files = []
        for ext in ("*.wav", "*.mp3", "*.flac", "*.m4a"):
            audio_files.extend(d.rglob(ext))

        print(f"Found {len(audio_files)} audio files in {d}")
        it = tqdm(audio_files, desc=d.name) if tqdm else audio_files

        for audio_path in it:
            try:
                duration = librosa.get_duration(path=str(audio_path))
                if label_strategy == "folder":
                    label = audio_path.parent.name
                elif label_strategy == "filename":
                    label = audio_path.stem
                else:
                    label = "unknown"

                payload = {
                    "audio_file": audio_path.name,
                    "recording": str(audio_path.absolute()),
                    "annotations": [{
                        "start_time": 0.0,
                        "end_time": duration,
                        "label": label,
                    }],
                }
                json_path = output_dir_p / (audio_path.stem + ".json")
                with json_path.open("w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2)
            except Exception as e:
                print(f"Error processing {audio_path}: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate Wombat JSONs from raw audio")
    parser.add_argument("--raw_audio_dirs", required=True, nargs="+")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--label_strategy", default="folder", choices=["folder", "filename"])
    args = parser.parse_args()
    generate_annotations(args.raw_audio_dirs, args.output_dir, args.label_strategy)
