import numpy as np
import sys
import os
from pathlib import Path

# Add workspace to path for imports
workspace_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(workspace_root))

from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from wrappers.rl_wrapper import create_env

from stable_baselines3.common.vec_env import VecNormalize

# Update SelectiveVecNormalize for the new observation size
class SelectiveVecNormalize(VecNormalize):
    """VecNormalize that only normalizes the first 12 dimensions (muscle sensors)."""
    
    def __init__(self, venv, norm_dims=12, **kwargs):
        self.norm_dims = norm_dims
        
        # Initialize parent class first
        super().__init__(venv, **kwargs)
        
        # Now override the obs_rms to only track norm_dims
        from stable_baselines3.common.running_mean_std import RunningMeanStd
        self.obs_rms = RunningMeanStd(shape=(norm_dims,))
        
        print(f"Selective normalization: normalizing first {norm_dims} dimensions only")
    
    def reset(self):
        """Reset environments and update selective normalization."""
        obs = self.venv.reset()
        if self.training and self.norm_obs:
            # Only update with first norm_dims dimensions
            self.obs_rms.update(obs[:, :self.norm_dims])
        return self.normalize_obs(obs)
    
    def step_wait(self):
        """Override step_wait to handle selective normalization."""
        obs, rewards, dones, infos = self.venv.step_wait()
        
        if self.training and self.norm_obs:
            # Only update with first norm_dims dimensions
            self.obs_rms.update(obs[:, :self.norm_dims])
        
        if self.norm_reward:
            rewards = self.normalize_reward(rewards)
        
        obs = self.normalize_obs(obs)
        
        return obs, rewards, dones, infos
        
    def normalize_obs(self, obs):
        """Override normalize_obs to handle selective normalization."""
        if self.norm_obs:
            obs_to_norm = obs[:, :self.norm_dims]
            obs_rest = obs[:, self.norm_dims:]
            
            # Normalize only first part
            normalized_part = (obs_to_norm - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon)
            normalized_part = np.clip(normalized_part, -self.clip_obs, self.clip_obs)
            
            # Concatenate with unchanged part
            return np.concatenate([normalized_part, obs_rest], axis=1)
        return obs
    
    def _update_obs(self, obs):
        """Update statistics only for first norm_dims dimensions."""
        if self.training and self.norm_obs:
            self.obs_rms.update(obs[:, :self.norm_dims])



def create_selective_normalization(original_vecnorm_path="vec_normalize2.pkl", 
                                  output_path="selective_vec_normalize3.pkl"):
    """Create SelectiveVecNormalize from original VecNormalize."""
    
    print(f"🔄 Creating selective normalization from {original_vecnorm_path}")
    
    # Get the current working directory and workspace root
    current_dir = Path.cwd()
    workspace_root = Path(__file__).parent.parent.parent
    
    # Make paths absolute - look in the models directory relative to current working directory
    if not os.path.isabs(original_vecnorm_path):
        # First try relative to current directory
        original_vecnorm_path = current_dir / "models" / original_vecnorm_path
    if not os.path.isabs(output_path):
        output_path = current_dir / "models" / output_path
    
    print(f"   Reading from: {original_vecnorm_path}")
    print(f"   Writing to: {output_path}")
    
    # Check if input file exists
    if not original_vecnorm_path.exists():
        print(f"❌ Error: {original_vecnorm_path} not found!")
        print(f"   Current directory: {current_dir}")
        print(f"   Looking for models in: {current_dir / 'models'}")
        
        # List what's actually in the models directory
        models_dir = current_dir / "models"
        if models_dir.exists():
            print(f"   Files in models directory:")
            for file in models_dir.iterdir():
                print(f"     - {file.name}")
        else:
            print(f"   Models directory doesn't exist: {models_dir}")
        return None
    
    # Convert back to string for the rest of the function
    original_vecnorm_path = str(original_vecnorm_path)
    output_path = str(output_path)
    
    # Create dummy environment
    dummy_env = create_env(random_seed=42)
    vec_env = DummyVecEnv([lambda: dummy_env])
    
    # Load original normalization
    try:
        original_norm = VecNormalize.load(original_vecnorm_path, vec_env)
        print(f"✅ Loaded original normalization")
        print(f"   Observation shape: {original_norm.obs_rms.mean.shape}")
        print(f"   Mean (first 12): {original_norm.obs_rms.mean[:12]}")
        print(f"   Var (first 12): {original_norm.obs_rms.var[:12]}")
    except Exception as e:
        print(f"❌ Error loading original normalization: {e}")
        vec_env.close()
        return None
    
    # Create selective normalization
    selective_norm = SelectiveVecNormalize(
        vec_env,
        norm_dims=12,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        gamma=0.99
    )
    
    # Transfer statistics for first 12 dimensions
    selective_norm.obs_rms.mean = original_norm.obs_rms.mean[:12].copy()
    selective_norm.obs_rms.var = original_norm.obs_rms.var[:12].copy()
    selective_norm.obs_rms.count = original_norm.obs_rms.count
    
    # Ensure clipping parameters match
    if hasattr(original_norm, 'clip_obs'):
        selective_norm.clip_obs = original_norm.clip_obs
    
    print(f"  ✅ Transferred stats for first 12 dimensions:")
    print(f"     Original mean shape: {original_norm.obs_rms.mean.shape}")
    print(f"     Selective mean shape: {selective_norm.obs_rms.mean.shape}")
    print(f"     Count: {selective_norm.obs_rms.count}")
    print(f"     Clip obs: {selective_norm.clip_obs}")
    
    # Save the selective normalization
    try:
        selective_norm.save(output_path)
        print(f"💾 Saved selective normalization to: {output_path}")
    except Exception as e:
        print(f"❌ Error saving selective normalization: {e}")
        vec_env.close()
        return None
    
    vec_env.close()
    return selective_norm

if __name__ == "__main__":
    # Use the files from your models directory
    create_selective_normalization(
        original_vecnorm_path="vec_normalize2.pkl",
        output_path="selective_vec_normalize3.pkl"
    )