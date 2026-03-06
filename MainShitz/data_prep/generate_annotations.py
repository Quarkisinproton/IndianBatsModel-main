"""
generate_annotations.py - Auto-label audio files when you're lazy

Creates Wombat-style JSON annotations from raw audio directories.
Labels are derived from folder names or filenames. Not perfect,
but better than labeling thousands of files by hand.
"""
import argparse
import json
import os
from pathlib import Path
import librosa
from tqdm import tqdm


def generate_annotations(raw_audio_dirs, output_dir, label_strategy='folder'):
    """Generate annotation JSONs. Label strategy: folder or filename."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    raw_audio_dirs = [Path(d) for d in raw_audio_dirs]
    
    print(f"Generating annotations for {len(raw_audio_dirs)} directories...")
    print(f"Output directory: {output_dir}")
    print(f"Label strategy: {label_strategy}")

    for d in raw_audio_dirs:
        if not d.exists():
            print(f"Warning: Directory not found: {d}")
            continue
            
        # Gather all audio files
        audio_files = []
        for ext in ['*.wav', '*.mp3', '*.flac', '*.m4a']:
            audio_files.extend(list(d.rglob(ext)))
            
        print(f"Found {len(audio_files)} audio files in {d}")
        
        for audio_path in tqdm(audio_files, desc=f"Processing {d.name}"):
            try:
                duration = librosa.get_duration(path=audio_path)
                
                
                if label_strategy == 'folder':
                    label = d.name
                elif label_strategy == 'filename':
                    label = audio_path.stem
                else:
                    label = 'unknown'
                
                # Create Wombat-style annotation
                annotation = {
                    "audio_file": str(audio_path.name),
                    "recording": str(audio_path.absolute()),
                    "annotations": [
                        {
                            "start_time": 0.0,
                            "end_time": duration,
                            "label": label
                        }
                    ]
                }
                
                json_name = audio_path.stem + '.json'
                json_path = output_dir / json_name
                
                with open(json_path, 'w') as f:
                    json.dump(annotation, f, indent=2)
                    
            except Exception as e:
                print(f"Error processing {audio_path}: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate dummy Wombat JSON annotations from raw audio files')
    parser.add_argument('--raw_audio_dirs', required=True, nargs='+', help='List of directories containing raw audio')
    parser.add_argument('--output_dir', required=True, help='Directory to save generated JSONs')
    parser.add_argument('--label_strategy', default='folder', choices=['folder', 'filename'], 
                        help='How to determine the species label. "folder" uses the parent directory name. "filename" uses the file stem.')
    
    args = parser.parse_args()
    
    generate_annotations(args.raw_audio_dirs, args.output_dir, args.label_strategy)
