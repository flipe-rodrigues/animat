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

def create_encoder(encoder_type="identity", **encoder_kwargs):
    """Create encoder based on type specification."""
    from encoders.encoders import IdentityEncoder, ModalitySpecificEncoder, GridEncoder
    
    if encoder_type == "identity":
        encoder = IdentityEncoder(obs_dim=15)
        print(f"🧠 Using Identity Encoder: 15D → 15D")
        
    elif encoder_type == "modality_specific":
        grid_size = encoder_kwargs.get('grid_size', 5)
        encoder = ModalitySpecificEncoder(grid_size=grid_size, raw_obs_dim=15)
        output_dim = 12 + (grid_size * grid_size) + 1  # muscle + grid + target_z
        print(f"🧠 Using Modality Specific Encoder: 15D → {output_dim}D")
        print(f"   - Muscle sensors: 12D (passthrough)")
        print(f"   - Target XY: 2D → {grid_size}×{grid_size} = {grid_size*grid_size}D (grid)")
        print(f"   - Target Z: 1D (passthrough)")
        
    elif encoder_type == "grid_only":
        grid_size = encoder_kwargs.get('grid_size', 5)
        encoder = GridEncoder(
            grid_size=grid_size,
            min_val=[-0.65, -0.90],
            max_val=[0.90, 0.35],
            sigma_scale=0.8
        )
        output_dim = grid_size * grid_size
        print(f"🧠 Using Grid Encoder: 2D → {output_dim}D")
        
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")
    
    return encoder

def main():
    """Main training function."""
    print("🚀 Starting RL Training with SB3")
    print("=" * 50)
    
    # Create models directory
    models_dir = create_model_directory()
    
    # Import training function
    try:
        from workspace.trainers.rl_train import train_arm
        from workspace.wrappers.rl_wrapper import set_seeds
    except ImportError:
        print("Using local imports...")
        sys.path.insert(0, str(workspace_root))
        from workspace.trainers.rl_train import train_arm
        from workspace.wrappers.rl_wrapper import set_seeds
    
    # 🎯 ENCODER CONFIGURATION - Change this to select different encoders!
    encoder_config = {
        "encoder_type": "modality_specific",  # Options: "identity", "modality_specific", "grid_only"
        "grid_size": 5,  # For grid-based encoders
    }
    
    # Create the specified encoder
    encoder = create_encoder(**encoder_config)
    
    # Set seeds for reproducibility
    set_seeds(42)
    
    # Change to models directory for saving
    original_dir = os.getcwd()
    
    try:
        # Change to models directory so all outputs go there
        os.chdir(models_dir)
        
        print(f"🔧 Training from directory: {os.getcwd()}")
        print(f"🎯 All models and logs will be saved here")
        
        # Run training with specified encoder
        model = train_arm(encoder=encoder)
        
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