"""
extract_end_frequency.py – Dominant frequency in the tail of each bat call.

Outputs CSV: json_file, audio_file, segment_index, label, start, end,
             end_freq_hz, low_freq_hz, high_freq_hz
"""
import csv
from pathlib import Path
from typing import List

import librosa
import numpy as np

from src.data_prep import wombat_to_spectrograms as w2s


def compute_end_frequency(
    y: np.ndarray,
    sr: int,
    start_s: float,
    end_s: float,
    n_fft: int = 2048,
    hop_length: int = 512,
    tail_frames: int = 3,
) -> float:
    """Estimate end-frequency (Hz) for an audio segment.

    Takes the last *tail_frames* STFT frames, averages power per frequency
    bin, and returns the frequency of the peak bin.  Returns ``nan`` on
    failure.
    """
    seg = w2s.extract_segment(y, sr, start_s, end_s)
    if seg.size == 0:
        return float("nan")
    S = librosa.stft(seg, n_fft=n_fft, hop_length=hop_length)
    S_power = np.abs(S) ** 2
    if S_power.shape[1] == 0:
        return float("nan")
    tail = S_power[:, max(0, S_power.shape[1] - tail_frames):]
    avg_power = np.mean(tail, axis=1)
    max_bin = int(np.argmax(avg_power))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    return float(freqs[max_bin])


def process_all_and_write_csv(
    raw_audio_dirs: List[str],
    json_dir: str,
    out_csv: str,
    species_key: str = "label",
) -> None:
    """Walk annotation JSONs, compute end-frequency, write CSV."""
    raw_dirs = [Path(d) for d in raw_audio_dirs]
    json_dir_p = Path(json_dir)
    out_csv_p = Path(out_csv)
    out_csv_p.parent.mkdir(parents=True, exist_ok=True)

    header = [
        "json_file", "audio_file", "segment_index", "label",
        "start", "end", "end_freq_hz", "low_freq_hz", "high_freq_hz",
    ]

    with out_csv_p.open("w", newline="") as csvf:
        writer = csv.writer(csvf)
        writer.writerow(header)

        for jpath in json_dir_p.rglob("*.json"):
            try:
                data = w2s.load_wombat_json(jpath)
            except Exception:
                continue
            audio_path = w2s.find_audio_for_json(jpath, raw_dirs)
            if audio_path is None:
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
            anns = w2s.normalize_annotations(anns)

            try:
                y, sr = librosa.load(str(audio_path), sr=None)
            except Exception:
                continue

            for i, ann in enumerate(anns):
                start = w2s.get_first_present_key(ann, ["start_time", "start", "t0", "onset"])
                end = w2s.get_first_present_key(ann, ["end_time", "end", "t1", "offset"])
                label = w2s.get_first_present_key(ann, [species_key, "species", "label", "class"])
                if start is None or end is None:
                    continue
                try:
                    start_f, end_f = float(start), float(end)
                except Exception:
                    continue
                ef = compute_end_frequency(y, sr, start_f, end_f)
                low_f = w2s.get_first_present_key(ann, ["low_freq_hz", "low_f", "min_freq"])
                high_f = w2s.get_first_present_key(ann, ["high_freq_hz", "high_f", "max_freq"])
                writer.writerow([
                    str(jpath), str(audio_path), i, label or "",
                    start_f, end_f, ef, low_f or "", high_f or "",
                ])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract end-frequency per annotation → CSV")
    parser.add_argument("--raw_audio_dir", required=True, nargs="+")
    parser.add_argument("--json_dir", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--species_key", default="label")
    args = parser.parse_args()
    process_all_and_write_csv(args.raw_audio_dir, args.json_dir, args.out_csv, args.species_key)
