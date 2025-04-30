from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from shimmy_wrapper import create_env, set_seeds
import numpy as np

def create_normalization_stats(model_path="./best_model/best_model.zip", 
                             save_path="vec_normalize.pkl",
                             num_steps=30000):
    """Create and save normalization statistics through exploration"""
    # Set seeds for reproducibility
    set_seeds(42)
    
    # Load model 
    model = SAC.load(model_path)
    
    # Create environment similar to training
    env = create_env()
    vec_env = DummyVecEnv([lambda: env])
    
    # Create a new VecNormalize wrapper
    vec_normalize = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0
    )
    
    # Use the model to collect statistics (better than random actions)
    print(f"Collecting normalization statistics using trained policy...")
    obs = vec_normalize.reset()
    
    for step in range(num_steps):
        # Get action from trained policy
        action, _ = model.predict(obs, deterministic=False)  # Use stochastic actions
        
        # Step the environment and collect stats
        obs, _, _, _ = vec_normalize.step(action)
        
        # Print progress
        if step % 1000 == 0:
            print(f"Progress: {step}/{num_steps} steps")
    
    # Save normalization statistics
    vec_normalize.save(save_path)
    print(f"Normalization statistics saved to {save_path}")

if __name__ == "__main__":
    create_normalization_stats()