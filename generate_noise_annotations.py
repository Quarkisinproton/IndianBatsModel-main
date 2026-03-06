import os
import json
import librosa
import glob

def generate_noise_annotations(noise_dir, output_file):
    annotations = []
    
    # Ensure directory exists
    if not os.path.exists(noise_dir):
        print(f"Directory not found: {noise_dir}")
        # We continue even if empty to generate the file structure if needed, 
        # but practically we should probably return or warn.
        # Given the task is to scan, if it's missing, we just find 0 files.
    
    # Scan for .wav files
    wav_files = glob.glob(os.path.join(noise_dir, "*.wav"))
    
    print(f"Found {len(wav_files)} noise files in {noise_dir}.")

    for file_path in wav_files:
        try:
            # Get duration
            duration = librosa.get_duration(path=file_path)
            filename = os.path.basename(file_path)
            
            # Create annotation entry
            entry = {
                "filename": filename,
                "annotations": [
                    {
                        "start_time": 0.0,
                        "end_time": float(duration),
                        "label": "Noise"
                    }
                ]
            }
            annotations.append(entry)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    # Save to JSON
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(annotations, f, indent=2)
    
    print(f"Saved annotations to {output_file}")

if __name__ == "__main__":
    NOISE_DIR = "data/raw/noise"
    OUTPUT_FILE = "data/noise_annotations.json"
    generate_noise_annotations(NOISE_DIR, OUTPUT_FILE)
