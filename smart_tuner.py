import optuna
import yaml
import os
import subprocess
import sys

def objective(trial):
    # 1. Suggest Hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    
    print(f"\n--- Trial {trial.number} ---")
    print(f"Params: lr={learning_rate}, bs={batch_size}, wd={weight_decay}")

    # 2. Load Base Config
    base_config_path = 'configs/config.yaml'
    if not os.path.exists(base_config_path):
        raise FileNotFoundError(f"Config file not found: {base_config_path}")
        
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Ensure 'train' section exists
    if 'train' not in config:
        config['train'] = {}
        
    # Update Config
    config['train']['learning_rate'] = learning_rate
    config['train']['batch_size'] = batch_size
    config['train']['weight_decay'] = weight_decay # Note: Ensure train.py uses this if implemented, otherwise it's just passed
    
    # Unique model path to avoid overwriting
    model_save_path = os.path.join('models', f'trial_{trial.number}.pth')
    config['train']['model_save_path'] = model_save_path
    
    # Save Temp Config
    temp_config_path = f'temp_config_{trial.number}.yaml'
    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f)
        
    # 3. Run Training
    cmd = [sys.executable, "-m", "MainShitz.train", "--config", temp_config_path]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        output = result.stdout
        
        # 4. Parse Output for Validation Loss
        final_val_loss = None
        for line in output.splitlines():
            if "FINAL_VAL_LOSS:" in line:
                try:
                    final_val_loss = float(line.split("FINAL_VAL_LOSS:")[1].strip())
                except ValueError:
                    pass
        
        if final_val_loss is None:
            print("Warning: Could not find FINAL_VAL_LOSS in output.")
            print("Output tail:", output[-500:])
            return 999.0 # Return high loss on failure to parse
            
        return final_val_loss

    except subprocess.CalledProcessError as e:
        print(f"Training failed for trial {trial.number}")
        print("Error:", e.stderr)
        return 999.0 # Return high loss on crash
        
    finally:
        # Cleanup temp config
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)

if __name__ == "__main__":
    # Check if optuna is installed
    try:
        import optuna
    except ImportError:
        print("Optuna is not installed. Please run: pip install optuna")
        sys.exit(1)

    # 5. Optimization
    study = optuna.create_study(direction="minimize")
    
    print("Starting Hyperparameter Optimization...")
    study.optimize(objective, n_trials=20)
    
    print("\n" + "="*40)
    print("Optimization Complete")
    print("="*40)
    print("Best Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print(f"Best Validation Loss: {study.best_value}")
    print("="*40)
