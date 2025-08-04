import torch
import numpy as np
import pickle
import sys
from pathlib import Path

# Add workspace to path
workspace_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(workspace_root))

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from envs.dm_env import make_arm_env
from wrappers.rl_wrapper import set_seeds, create_env
from workspace.wrappers.sl_wrappers.normalization import SelectiveVecNormalize
from tqdm import tqdm

def collect_optimized_demonstrations_50steps_then_convert(model_path="./best_model/best_model.zip", 
                                                         vecnorm_path="vec_normalize.pkl",
                                                         selective_norm_path="selective_vec_normalize.pkl",
                                                         num_episodes=1500,
                                                         target_episode_length=50,
                                                         success_threshold=0.06,
                                                         save_path="sac_demonstrations_50steps_successful_selective.pkl"):
    """Collect demonstrations with original normalization, then convert to selective format."""
    
    print(f"🎯 COLLECTING 50-STEP DEMONSTRATIONS THEN CONVERTING TO SELECTIVE")
    print(f"Episode length: {target_episode_length} steps")
    print(f"Success threshold: {success_threshold}")
    print(f"Step 1: Collect with original normalization: {vecnorm_path}")
    print(f"Step 2: Convert to selective normalization format")
    print("=" * 60)
    
    # Load SAC model
    model = SAC.load(model_path)
    

    # Load both normalizations for conversion
    dummy_env = create_env(random_seed=42)
    dummy_vec = DummyVecEnv([lambda: dummy_env])
    
    # Original normalization (for SAC policy)
    original_norm = VecNormalize.load(vecnorm_path, dummy_vec)
    
    # Selective normalization (for conversion)
    selective_norm = SelectiveVecNormalize.load(selective_norm_path, dummy_vec)
    
    dummy_vec.close()
    
    demonstrations = {
        'observations': [],
        'actions': [],
        'rewards': [],
        'dones': [],
        'episode_starts': [],
        'episode_info': []
    }
    
    success_count = 0
    collected_count = 0
    processed_count = 0
    
    for episode in tqdm(range(num_episodes), desc="Collecting with original normalization"):
        seed = 42 + episode
        set_seeds(seed)
        processed_count += 1
        
        # Create environment
        dm_env = make_arm_env(random_seed=seed)
        raw_env = create_env(random_seed=seed, base_env=dm_env)
        eval_vec = DummyVecEnv([lambda env=raw_env: env])
        
        # Use ORIGINAL normalization for SAC policy
        eval_vec = VecNormalize.load(vecnorm_path, eval_vec)
        eval_vec.training = False
        
        # Track episode data
        temp_obs = []
        temp_actions = []
        temp_rewards = []
        temp_distances = []
        
        obs = eval_vec.reset()
        done = False
        step_count = 0
        
        # Run for exactly 50 steps
        while not done and step_count < target_episode_length:
            # Get action from SAC policy (using original normalization)
            action, _ = model.predict(obs, deterministic=True)
            
            # Store step data
            temp_obs.append(obs[0].copy())
            
            # Normalize actions to [0, 1] range
            # Assuming actions are in [-1, 1], convert to [0, 1]
            normalized_action = (action[0] + 1.0) / 2.0
            normalized_action = np.clip(normalized_action, 0.0, 1.0)  # Add this line
            temp_actions.append(normalized_action.copy())
            
            # Step environment with original action
            obs, reward, done, info = eval_vec.step(action)
            
            # Calculate distance to target
            hand_pos = dm_env.physics.bind(dm_env._task._arm.hand).xpos
            target_pos = dm_env.physics.bind(dm_env._task._arm.target).mocap_pos
            distance = np.linalg.norm(hand_pos - target_pos)
            temp_distances.append(distance)
            temp_rewards.append(reward[0])
            
            step_count += 1
            done = done[0] if isinstance(done, np.ndarray) else done
        
        # Check success criterion
        final_distance = temp_distances[-1] if temp_distances else float('inf')
        min_distance = min(temp_distances) if temp_distances else float('inf')
        total_reward = sum(temp_rewards)
        
        is_successful_at_50 = final_distance < success_threshold
        
        if is_successful_at_50:
            success_count += 1
            
            # Convert observations to selective normalization format
            converted_obs = []
            for obs in temp_obs:
                # Step 1: Denormalize using original normalization
                denormalized = obs * np.sqrt(original_norm.obs_rms.var + original_norm.epsilon) + original_norm.obs_rms.mean
                
                # Step 2: Apply selective normalization
                muscle_part = denormalized[:12]
                rest_part = denormalized[12:]
                
                # Normalize only muscle sensors
                normalized_muscle = (muscle_part - selective_norm.obs_rms.mean) / np.sqrt(selective_norm.obs_rms.var + selective_norm.epsilon)
                # Use the same clipping as selective_norm
                normalized_muscle = np.clip(normalized_muscle, -selective_norm.clip_obs, selective_norm.clip_obs)
                
                # Combine normalized muscle + raw rest
                converted_obs_step = np.concatenate([normalized_muscle, rest_part])
                converted_obs.append(converted_obs_step)
            
            # Store this successful episode with converted observations
            episode_start_idx = len(demonstrations['observations'])
            
            demonstrations['observations'].extend(converted_obs)
            demonstrations['actions'].extend(temp_actions)
            demonstrations['rewards'].extend(temp_rewards)
            demonstrations['dones'].extend([False] * (len(temp_rewards) - 1) + [True])
            demonstrations['episode_starts'].append(episode_start_idx)
            
            # Store episode info
            episode_info = {
                'episode_idx': collected_count,
                'source_episode': episode,
                'seed': seed,
                'total_reward': total_reward,
                'final_distance': final_distance,
                'min_distance': min_distance,
                'success': True,
                'steps': target_episode_length
            }
            demonstrations['episode_info'].append(episode_info)
            collected_count += 1
        
        # Close environment for this episode
        eval_vec.close()
        
        # Progress update every 100 episodes
        if processed_count % 100 == 0:
            success_rate = 100 * success_count / processed_count
            print(f"  Episodes {processed_count-99:4d}-{processed_count:4d}: "
                  f"Success rate: {success_count:3d}/{processed_count:4d} ({success_rate:5.1f}%), "
                  f"Collected: {collected_count:4d}")
            
        # Early stop if we have enough successful episodes
        if collected_count >= 3000:
            print(f"\n✅ Collected target of 1000 successful episodes!")
            print(f"   Processed {processed_count} episodes total")
            break
    
    # Convert to numpy arrays
    demonstrations['observations'] = np.array(demonstrations['observations'])
    demonstrations['actions'] = np.array(demonstrations['actions'])
    demonstrations['rewards'] = np.array(demonstrations['rewards'])
    demonstrations['dones'] = np.array(demonstrations['dones'])
    demonstrations['episode_starts'] = np.array(demonstrations['episode_starts'])
    
    # Add metadata
    demonstrations['metadata'] = {
        'num_episodes_processed': processed_count,
        'num_episodes_collected': collected_count,
        'success_count': success_count,
        'success_rate': success_count / processed_count,
        'success_threshold': success_threshold,
        'episode_length': target_episode_length,
        'total_steps': len(demonstrations['observations']),
        'collection_type': 'successful_with_converted_selective_normalization',
        'normalization_type': 'Converted_to_SelectiveVecNormalize_first_12_dims',
        'model_path': model_path,
        'original_vecnorm_path': vecnorm_path,
        'selective_norm_path': selective_norm_path
    }
    
    # Save demonstrations
    with open(save_path, 'wb') as f:
        pickle.dump(demonstrations, f)
    
    print(f"\n🎉 SELECTIVE NORMALIZATION COLLECTION SUMMARY")
    print("=" * 60)
    print(f"Episodes processed: {processed_count}")
    print(f"Episodes collected: {collected_count} (100% successful)")
    print(f"Normalization: Original → Converted to SelectiveVecNormalize")
    print(f"Data saved to: {save_path}")
    
    return demonstrations

def quick_data_check(save_path="sac_demonstrations_50steps_successful_selective.pkl"):
    """Quick check of the collected data."""
    with open(save_path, 'rb') as f:
        data = pickle.load(f)
    
    # Fix the path to include models/ directory
    sac_model = SAC.load("models/arm_final2.zip")
    print(f"SAC was trained on action space: {sac_model.action_space}")

    print(f"\n🔍 DATA CHECK: {save_path}")
    print("=" * 40)
    print(f"Total observations: {len(data['observations'])}")
    print(f"Total actions: {len(data['actions'])}")
    print(f"Observation shape: {data['observations'][0].shape}")
    print(f"Action shape: {data['actions'][0].shape}")
    print(f"Episodes: {len(data['episode_info'])}")
    print(f"All successful: {all(ep['success'] for ep in data['episode_info'])}")
    print(f"All 50 steps: {all(ep['steps'] == 50 for ep in data['episode_info'])}")
    
    # Check action range
    actions = np.array(data['actions'])
    print(f"Action range: [{np.min(actions):.3f}, {np.max(actions):.3f}]")
    print(f"Action mean: {np.mean(actions):.3f}")
    print(f"Action std: {np.std(actions):.3f}")
    
    # Check observation ranges - adapt to actual observation dimensions
    observations = np.array(data['observations'])
    obs_dim = observations.shape[1]
    
    if obs_dim >= 12:
        muscle_obs = observations[:, :12]  # First 12 dims (muscle sensors)
        print(f"\n📊 OBSERVATION RANGES (obs_dim={obs_dim}):")
        print(f"Muscle sensors (0-11): mean=[{muscle_obs.mean(axis=0).min():.3f}, {muscle_obs.mean(axis=0).max():.3f}], "
              f"std=[{muscle_obs.std(axis=0).min():.3f}, {muscle_obs.std(axis=0).max():.3f}]")
        
        if obs_dim > 12:
            remaining_obs = observations[:, 12:]
            print(f"Remaining dims (12-{obs_dim-1}): range=[{remaining_obs.min():.3f}, {remaining_obs.max():.3f}]")
    else:
        print(f"\n📊 OBSERVATION RANGES (obs_dim={obs_dim}):")
        print(f"All observations: range=[{observations.min():.3f}, {observations.max():.3f}]")

if __name__ == "__main__":
    # Collect demonstrations with original normalization, then convert
    demonstrations = collect_optimized_demonstrations_50steps_then_convert(
        model_path="models/arm_final2.zip",
        vecnorm_path="models/vec_normalize2.pkl",  # Original normalization for SAC
        selective_norm_path="models/selective_vec_normalize3.pkl",  # Target normalization
        num_episodes=10000,  # Increase to collect 3000 successful episodes
        target_episode_length=50,
        success_threshold=0.04,
        save_path="sac_demonstrations_50steps_successful_selective5.pkl"
    )
    
    # Quick verification
    quick_data_check("sac_demonstrations_50steps_successful_selective5.pkl")