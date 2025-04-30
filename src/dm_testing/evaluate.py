import os
os.environ['MUJOCO_GL'] = 'osmesa'  # Try software rendering

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from environment import make_arm_env
from shimmy_wrapper import set_seeds, create_env

def evaluate_model(num_episodes=20, model_path="./best_model/best_model.zip", 
                  stats_path="vec_normalize.pkl", deterministic=True):
    """Evaluate model performance across multiple episodes"""
    
    # Load the trained model
    model = SAC.load(model_path)
    print(f"Model observation space: {model.observation_space.shape}")
    
    # Track success stats
    successes = 0
    step_counts = []
    distances = []
    target_positions = []
    
    for episode in range(1, num_episodes + 1):
        seed = 45 + episode  # Different seeds than training
        set_seeds(seed)
        
        # Create environment
        dm_env = make_arm_env(random_seed=seed)
        raw_env = create_env(random_seed=seed, base_env=dm_env)
        
        # Wrap for vectorization
        eval_vec = DummyVecEnv([lambda env=raw_env: env])
        
        # Load normalization if available
        try:
            eval_vec = VecNormalize.load(stats_path, eval_vec)
            eval_vec.training = False
            eval_vec.norm_reward = False
            print(f"Using normalization from {stats_path}")
            if 'Using normalization' in locals():
                print(f"Normalization loaded successfully: mean={eval_vec.obs_rms.mean[:5]}, var={eval_vec.obs_rms.var[:5]}")
        except Exception as e:
            print(f"Could not load normalization: {e}")
        
        # Get initial observation
        obs = eval_vec.reset()
        
        # Log target position
        target_pos = dm_env.physics.bind(dm_env._task._arm.target).mocap_pos
        target_positions.append(target_pos)
        print(f"\nEpisode {episode}/{num_episodes}, Target position: {target_pos}")
        
        # Episode variables
        done = False
        step_count = 0
        episode_success = False
        success_reported = False
        min_distance = float('inf')
        
        # Run episode
        while not done and step_count < 150:  # Extended steps
            # Get action from model
            action, _ = model.predict(obs, deterministic=deterministic)
            
            # Step environment
            obs, reward, done, info = eval_vec.step(action)
            step_count += 1

            # Check if success was recorded by the environment
            if 'success' in info[0] and info[0]['success']:
                episode_success = True
                if not success_reported:
                    print(f"  SUCCESS detected at step {step_count}! From environment info.")
                    success_reported = True

            # Sample positions less frequently
            if step_count % 5 == 0:  # Only sample every 5th step
                hand_pos = dm_env.physics.bind(dm_env._task._arm.hand).xpos
                distance = np.linalg.norm(hand_pos - target_pos)
                min_distance = min(min_distance, distance)

            # Check if success threshold is reached at any point
            if distance < 0.08:
                episode_success = True
                if not success_reported:
                    print(f"  SUCCESS detected at step {step_count}! Distance: {distance:.4f}")
                    success_reported = True
                
            # Only log periodically to avoid console spam
            if step_count % 10 == 0 or done:
                print(f"  Step {step_count}: Distance = {distance:.4f}, Reward = {reward[0]:.2f}")
        
        # Record episode results
        step_counts.append(step_count)
        distances.append(min_distance)
        if episode_success:
            successes += 1
            print(f"✓ Episode {episode} SUCCESS in {step_count} steps")
        else:
            print(f"✗ Episode {episode} FAILED, closest distance: {min_distance:.4f}")
    
    # Print summary statistics
    success_rate = successes / num_episodes
    avg_steps = sum(step_counts) / len(step_counts)
    avg_distance = sum(distances) / len(distances)
    
    print("\n===== Evaluation Results =====")
    print(f"Success rate: {success_rate:.2%} ({successes}/{num_episodes})")
    print(f"Average steps: {avg_steps:.1f}")
    print(f"Average closest distance: {avg_distance:.4f}")
    print(f"Target position range: {np.min(target_positions, axis=0)} to {np.max(target_positions, axis=0)}")
    print("=============================")
    
    return {
        'success_rate': success_rate,
        'successes': successes,
        'episodes': num_episodes,
        'avg_steps': avg_steps,
        'avg_distance': avg_distance
    }

if __name__ == "__main__":
    # Test with deterministic policy 
    print("Evaluating with deterministic actions...")
    deterministic_results = evaluate_model(
        num_episodes=20, 
        deterministic=True,
        model_path="./best_model/best_model.zip",
        stats_path="vec_normalize.pkl"
    )
    
    # Test with stochastic policy
    print("\nEvaluating with stochastic actions...")
    stochastic_results = evaluate_model(
        num_episodes=10,
        deterministic=False,
        model_path="./best_model/best_model.zip",
        stats_path="vec_normalize.pkl"
    )