import yaml
import subprocess
import os
import copy

def run_experiments():
    base_config_path = 'configs/config.yaml'
    temp_config_path = 'temp_config.yaml'
    
    # Define experiments
    experiments = [
        {'learning_rate': 0.01, 'batch_size': 16},
        {'learning_rate': 0.001, 'batch_size': 32}
    ]
    
    # Load base config
    if not os.path.exists(base_config_path):
        print(f"Error: Base config not found at {base_config_path}")
        return

    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)
    
    for i, exp in enumerate(experiments):
        print(f"Running experiment {i+1}/{len(experiments)}: {exp}")
        
        # Create a deep copy to avoid modifying the base for subsequent runs
        current_config = copy.deepcopy(base_config)
        
        # Ensure 'train' section exists (matching config.yaml structure)
        if 'train' not in current_config:
            current_config['train'] = {}
            
        # Update parameters
        if 'learning_rate' in exp:
            current_config['train']['learning_rate'] = exp['learning_rate']
        if 'batch_size' in exp:
            current_config['train']['batch_size'] = exp['batch_size']
            
        # Update model save path
        lr = exp.get('learning_rate', current_config['train'].get('learning_rate', 'default'))
        bs = exp.get('batch_size', current_config['train'].get('batch_size', 'default'))
        model_name = f"model_lr{lr}_bs{bs}.pth"
        
        # We inject model_save_path into the 'train' section.
        # The modified train.py will need to look for it there.
        current_config['train']['model_save_path'] = os.path.join('models', model_name)
        
        # Save temp config
        with open(temp_config_path, 'w') as f:
            yaml.dump(current_config, f)
            
        # Run training
        cmd = ["python", "-m", "MainShitz.train", "--config", temp_config_path]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Experiment failed: {e}")
            
    # Cleanup
    if os.path.exists(temp_config_path):
        os.remove(temp_config_path)
        
if __name__ == "__main__":
    run_experiments()
