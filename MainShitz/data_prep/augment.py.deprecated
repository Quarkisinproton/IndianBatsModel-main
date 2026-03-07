"""
augment.py - Making more data out of thin air

Adds noise to audio files to artificially expand the dataset.
It's not cheating, it's "data augmentation". Totally legit.

WARNING: librosa.output.write_wav was removed in librosa 0.8+
This file needs updating if you actually want to use it.
"""
import os
import numpy as np
import librosa
import matplotlib.pyplot as plt


def augment_audio(audio_path, output_path, sample_rate=22050, noise_factor=0.005):
    """Add random noise. Because bats in the wild have noisy environments."""
    audio, sr = librosa.load(audio_path, sr=sample_rate)
    noise = np.random.randn(len(audio))
    augmented_audio = audio + noise_factor * noise
    # NOTE: This will fail on modern librosa - needs soundfile instead
    librosa.output.write_wav(output_path, augmented_audio, sr)

def create_spectrogram(audio_path, output_path, sample_rate=22050):
    audio, sr = librosa.load(audio_path, sr=sample_rate)
    spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, fmax=8000)
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(spectrogram_db, sr=sr, x_axis='time', y_axis='mel', fmax=8000)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-frequency spectrogram')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def augment_and_save_spectrograms(species_dirs, output_spectrogram_dir):
    for species_dir in species_dirs:
        audio_files = [f for f in os.listdir(species_dir) if f.endswith('.wav')]
        for audio_file in audio_files:
            audio_path = os.path.join(species_dir, audio_file)
            augmented_audio_path = os.path.join(species_dir, 'augmented_' + audio_file)
            spectrogram_path = os.path.join(output_spectrogram_dir, audio_file.replace('.wav', '.png'))
            
            augment_audio(audio_path, augmented_audio_path)
            create_spectrogram(augmented_audio_path, spectrogram_path)

# Example usage
if __name__ == "__main__":
    species_dirs = ['data/raw/species_1', 'data/raw/species_2', 'data/raw/species_3']
    output_spectrogram_dir = 'data/processed/spectrograms'
    os.makedirs(output_spectrogram_dir, exist_ok=True)
    augment_and_save_spectrograms(species_dirs, output_spectrogram_dir)