
# Create the processed spectrograms directory if it doesn't exist Becaues i am lazy to check
mkdir -p ../data/processed/spectrograms

# Convert audio files to specto........grams
for species in species_1 species_2 species_3; do
    for audio_file in ../data/raw/$species/*.wav; do
        # Extract the filename without extension(caus not annotations and only 1 species call in 1 wav file so to create annotations based on like filename)
        filename=$(basename "$audio_file" .wav)
        # spectrogram image Gen
        python ../src/data_prep/audio_to_spectrogram.py "$audio_file" "../data/processed/spectrograms/${species}_${filename}.png"
    done
done

echo "Data preparation complete. Spectrograms are saved in ../data/processed/spectrograms."