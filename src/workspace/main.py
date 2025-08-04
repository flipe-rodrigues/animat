#!/usr/bin/env python3
"""
Main script to test RL training with organized model saving.
"""

import os
import sys
from pathlib import Path

# Add both workspace and parent directory to Python path
workspace_root = Path(__file__).parent
parent_dir = workspace_root.parent
sys.path.insert(0, str(workspace_root))
sys.path.insert(0, str(parent_dir))

def create_model_directory():
    """Create models directory if it doesn't exist."""
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    print(f"✅ Models will be saved to: {models_dir.absolute()}")
    return models_dir

def main():
    """Main training function."""
    print("🚀 Starting RL Training with SB3")
    print("=" * 50)
    
    # Create models directory
    models_dir = create_model_directory()
    
    # Import training function - use absolute imports
    try:
        from workspace.trainers.rl_train import train_arm
        from workspace.wrappers.rl_wrapper import set_seeds
    except ImportError:
        # Fallback to local imports if workspace module doesn't work
        print("Using local imports...")
        sys.path.insert(0, str(workspace_root))
        from workspace.trainers.rl_train import train_arm
        from workspace.wrappers.rl_wrapper import set_seeds
        
    
    # Set seeds for reproducibility
    set_seeds(42)
    
    # Change to models directory for saving
    original_dir = os.getcwd()
    
    try:
        # Change to models directory so all outputs go there
        os.chdir(models_dir)
        
        print(f"🔧 Training from directory: {os.getcwd()}")
        print(f"🎯 All models and logs will be saved here")
        
        # Run training
        model = train_arm()
        
        print(f"\n🎉 Training Complete!")
        print(f"📁 Check the following files in {models_dir.absolute()}:")
        print(f"   - arm_final2.zip (trained model)")
        print(f"   - vec_normalize2.pkl (normalization stats)")
        print(f"   - models2/ (checkpoints)")
        print(f"   - best_model2/ (best model)")
        print(f"   - eval_logs2/ (evaluation logs)")
        print(f"   - tensorboard_logs/ (tensorboard logs)")
        
        return model
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise
    finally:
        # Always return to original directory
        os.chdir(original_dir)

if __name__ == "__main__":
    main()