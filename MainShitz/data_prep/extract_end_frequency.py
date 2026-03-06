"""
extract_end_frequency.py - Where the call ends, literally

Finds the dominant frequency in the last few frames of each bat call.

Outputs: CSV with json_file, audio_file, segment_index, label, start, end, end_freq_hz
"""
from pathlib import Path
import csv
import math
from typing import Optional

import numpy as np
import librosa

from MainShitz.data_prep import wombat_to_spectrograms as w2s


def compute_end_frequency(y: np.ndarray, sr: int, start_s: float, end_s: float, n_fft: int = 2048, hop_length: int = 512, tail_frames: int = 3) -> float:
    """Compute an estimate of end-frequency (Hz) for a segment.

    Approach:
    - extract the audio segment
    - compute STFT and power spectrum
    - consider the last `tail_frames` columns (time frames)
    - average energy across those frames per frequency bin
    - pick frequency bin with maximum avg energy
    - convert bin index to Hz and return

    Returns nan if computation fails.
    """
    seg = w2s.extract_segment(y, sr, start_s, end_s)
    if seg.size == 0:
        return float('nan')
    S = librosa.stft(seg, n_fft=n_fft, hop_length=hop_length)
    S_power = np.abs(S) ** 2
    # choose last tail_frames (or less if not available)
    if S_power.shape[1] == 0:
        return float('nan')
    tail = S_power[:, max(0, S_power.shape[1] - tail_frames):]
    avg_power = np.mean(tail, axis=1)
    max_bin = int(np.argmax(avg_power))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    return float(freqs[max_bin])


def process_all_and_write_csv(raw_audio_dirs: list, json_dir: str, out_csv: str, species_key: str = 'label') -> None:
    raw_audio_dirs = [Path(d) for d in raw_audio_dirs]
    json_dir = Path(json_dir)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_csv.open('w', newline='') as csvf:
        writer = csv.writer(csvf)
        writer.writerow(['json_file', 'audio_file', 'segment_index', 'label', 'start', 'end', 'end_freq_hz', 'low_freq_hz', 'high_freq_hz'])

        for jpath in json_dir.rglob('*.json'):
            try:
                data = w2s.load_wombat_json(jpath)
            except Exception:
                continue
            audio_path = w2s.find_audio_for_json(jpath, raw_audio_dirs)
            if audio_path is None:
                continue
            # get annotations slike to wombat_to_spectrograms
            anns = None
            if isinstance(data, dict):
                for key in ('annotations', 'labels', 'segments', 'events'):
                    if key in data:
                        anns = data[key]
                        break
                if anns is None and any(k in data for k in ('start_time', 'end_time', species_key)):
                    anns = [data]
            else:
                anns = data
            anns = w2s.normalize_annotations(anns)

            # load audio once
            try:
                y, sr = librosa.load(str(audio_path), sr=None)
            except Exception:
                continue

            for i, ann in enumerate(anns):
                start = w2s.get_first_present_key(ann, ['start_time', 'start', 't0', 'onset'])
                end = w2s.get_first_present_key(ann, ['end_time', 'end', 't1', 'offset'])
                label = w2s.get_first_present_key(ann, [species_key, 'species', 'label', 'class'])
                if start is None or end is None:
                    continue
                try:
                    start_f = float(start)
                    end_f = float(end)
                except Exception:
                    continue
                ef = compute_end_frequency(y, sr, start_f, end_f)
                
                low_f = w2s.get_first_present_key(ann, ['low_freq_hz', 'low_f', 'min_freq'])
                high_f = w2s.get_first_present_key(ann, ['high_freq_hz', 'high_f', 'max_freq'])

                writer.writerow([str(jpath), str(audio_path), i, label or '', start_f, end_f, ef, low_f or '', high_f or ''])


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Extract end-frequency per annotation and write CSV')
    parser.add_argument('--raw_audio_dir', required=True, nargs='+')
    parser.add_argument('--json_dir', required=True)
    parser.add_argument('--out_csv', required=True)
    parser.add_argument('--species_key', default='label')
    args = parser.parse_args()
    process_all_and_write_csv(args.raw_audio_dir, args.json_dir, args.out_csv, species_key=args.species_key)
