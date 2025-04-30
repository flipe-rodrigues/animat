import os
os.environ['MUJOCO_GL'] = 'osmesa'  # Try software rendering

import numpy as np
import imageio
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from environment import make_arm_env
from shimmy_wrapper import set_seeds, create_env

def record_episodes(num_episodes=5, model_path="./best_model/best_model.zip", 
                    output_path="arm_episodes.gif", stats_path="vec_normalize.pkl"):
    """Record episodes with proper normalization"""
    
    # Load the trained model
    model = SAC.load(model_path)
    
    # Container for all frames
    all_frames = []
    
    # Record each episode with a different seed
    for episode in range(1, num_episodes + 1):
        seed = 5 + episode  # Different seed for each episode
        set_seeds(seed)
        
        print(f"\nRecording episode {episode}/{num_episodes} with seed {seed}...")
        
        # Create a SINGLE dm_env instance
        dm_env = make_arm_env(random_seed=seed)
        
        # Create wrapped environment with the SAME dm_env instance
        raw_env = create_env(random_seed=seed, base_env=dm_env) 
        
        # Wrap for vectorization
        eval_vec = DummyVecEnv([lambda env=raw_env: env])
        
        try:
            # Load and apply normalization
            eval_vec = VecNormalize.load(stats_path, eval_vec)
            eval_vec.training = False
            eval_vec.norm_reward = False
        except Exception as e:
            print(f"Could not load normalization: {e}")
            # Continue without normalization
        
        # Episode frames
        episode_frames = []
        obs = eval_vec.reset()
        done = False

        # At the beginning of your recording loop
        print(f"Model expects observation shape: {model.observation_space.shape}")
        print(f"Environment provides observation shape: {obs.shape}")
        
        # Record episode
        step_count = 0
        while not done and step_count < 100:  # Add step limit for safety
            # Get action from model
            action, _ = model.predict(obs, deterministic=True)
            
            # Step environment 
            obs, reward, done, info = eval_vec.step(action)
            step_count += 1
            
            # Render frame - now dm_env is properly in sync
            frame = dm_env.physics.render(height=480, width=640, camera_id=0)
            episode_frames.append(frame)
            
            # Debug info
            if step_count % 10 == 0:
                print(f"  Step {step_count}, reward: {reward}, done: {done}")
            
            # During the recording loop
            if step_count % 20 == 0:
                # Check if the environment is actually changing
                hand_pos = dm_env.physics.bind(dm_env._task._arm.hand).xpos
                target_pos = dm_env.physics.bind(dm_env._task._arm.target).mocap_pos
                print(f"Hand position: {hand_pos}")
                print(f"Target position: {target_pos}")
                print(f"Distance: {np.linalg.norm(hand_pos - target_pos)}")
        
        print(f"Episode completed in {step_count} steps")
        
        # Add episode frames to collection
        all_frames.extend(episode_frames)
    
    # Save video
    imageio.mimsave(output_path, all_frames, duration=50)  # Try a shorter duration for smoother playback
    print(f"Video saved to {output_path}")

if __name__ == "__main__":
    # Test with 5 different seeds to verify consistent performance
    record_episodes(
        num_episodes=10, 
        model_path="./best_model_continued/best_model.zip", 
        output_path="arm_random.gif",
        stats_path="vec_normalize_continued.pkl"
    )