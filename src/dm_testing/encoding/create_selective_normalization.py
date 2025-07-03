import numpy as np
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from shimmy_wrapper import create_env, SelectiveVecNormalize

def create_selective_normalization(original_vecnorm_path="vec_normalize.pkl", 
                                  output_path="selective_vec_normalize.pkl"):
    """Create SelectiveVecNormalize from original VecNormalize."""
    
    print(f"🔄 Creating selective normalization from {original_vecnorm_path}")
    
    # Create dummy environment
    dummy_env = create_env(random_seed=42)
    vec_env = DummyVecEnv([lambda: dummy_env])
    
    # Load original normalization
    original_norm = VecNormalize.load(original_vecnorm_path, vec_env)
    
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
    
    print(f"  Transferred stats for first 12 dimensions:")
    print(f"  Original mean shape: {original_norm.obs_rms.mean.shape}")
    print(f"  Selective mean shape: {selective_norm.obs_rms.mean.shape}")
    print(f"  Count: {selective_norm.obs_rms.count}")
    print(f"  Clip obs: {selective_norm.clip_obs}")
    
    # Save the selective normalization
    selective_norm.save(output_path)
    
    print(f"💾 Saved selective normalization to: {output_path}")
    
    vec_env.close()
    return selective_norm

if __name__ == "__main__":
    create_selective_normalization(
        original_vecnorm_path="vec_normalize2.pkl",
        output_path="selective_vec_normalize3.pkl"
    )