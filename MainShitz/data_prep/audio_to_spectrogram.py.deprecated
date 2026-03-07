"""
audio_to_spectrogram.py - Old school spectrogram generator

Does basically the same thing as wombat_to_spectrograms.py but
without the JSON annotation support. Kept around because... reasons.

Honestly you should probably use wombat_to_spectrograms.py instead.
"""
import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import librosa.display


def audio_to_spectrogram(audio_path, output_path, sr=22050, n_fft=2048, hop_length=512, power=2.0):
    """Convert audio file to mel spectrogram image."""
    y, _ = librosa.load(audio_path, sr=sr)
    S = librosa.feature.melspectrogram(y, sr=sr, n_fft=n_fft, hop_length=hop_length, power=power)
    S_dB = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', fmax=8000)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-frequency spectrogram')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def process_audio_directory(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for species in os.listdir(input_dir):
        species_dir = os.path.join(input_dir, species)
        if os.path.isdir(species_dir):
            for audio_file in os.listdir(species_dir):
                audio_path = os.path.join(species_dir, audio_file)
                if audio_file.endswith('.wav'):
                    output_file = f"{os.path.splitext(audio_file)[0]}.png"
                    output_path = os.path.join(output_dir, species, output_file)
                    audio_to_spectrogram(audio_path, output_path)