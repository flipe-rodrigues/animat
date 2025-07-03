import torch
import numpy as np
import pickle
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from environment import make_arm_env
from shimmy_wrapper import set_seeds, create_env, SelectiveVecNormalize
from tqdm import tqdm
import os


def load_reachable_positions():
    """Load the actual reachable positions from the pickle file."""
    reachable_path = os.path.join('..', '..', '..', 'mujoco/candidate_targets.pkl')
    
    try:
        with open(reachable_path, 'rb') as f:
            reachable_data = pickle.load(f)
        
        if isinstance(reachable_data, dict):
            if 'positions' in reachable_data:
                positions = np.array(reachable_data['positions'])
            elif 'reachable_positions' in reachable_data:
                positions = np.array(reachable_data['reachable_positions'])
            else:
                positions = np.array(list(reachable_data.values())[0])
        else:
            positions = np.array(reachable_data)
        
        if positions.shape[1] > 2:
            positions = positions[:, :2]
            
        print(f"✅ Loaded {len(positions)} reachable positions")
        return positions
        
    except Exception as e:
        print(f"❌ Error loading reachable positions: {e}")
        return None


def check_target_coverage(target_positions, reachable_positions, grid_size=15):
    """Check how well targets cover the workspace."""
    if len(target_positions) == 0:
        return 0.0
    
    # Create grid
    x_min, x_max = reachable_positions[:, 0].min(), reachable_positions[:, 0].max()
    y_min, y_max = reachable_positions[:, 1].min(), reachable_positions[:, 1].max()
    
    x_edges = np.linspace(x_min, x_max, grid_size + 1)
    y_edges = np.linspace(y_min, y_max, grid_size + 1)
    
    # Find reachable grid cells
    reach_x = np.digitize(reachable_positions[:, 0], x_edges) - 1
    reach_y = np.digitize(reachable_positions[:, 1], y_edges) - 1
    reach_x = np.clip(reach_x, 0, grid_size - 1)
    reach_y = np.clip(reach_y, 0, grid_size - 1)
    reachable_cells = set(zip(reach_x, reach_y))
    
    # Find target grid cells
    target_x = np.digitize(target_positions[:, 0], x_edges) - 1
    target_y = np.digitize(target_positions[:, 1], y_edges) - 1
    target_x = np.clip(target_x, 0, grid_size - 1)
    target_y = np.clip(target_y, 0, grid_size - 1)
    target_cells = set(zip(target_x, target_y))
    
    # Calculate coverage
    covered_cells = target_cells.intersection(reachable_cells)
    coverage = len(covered_cells) / len(reachable_cells)
    
    return coverage


def collect_diverse_target_demonstrations(model_path="arm_final2.zip",
                                        vecnorm_path="vec_normalize2.pkl", 
                                        selective_norm_path="selective_vec_normalize3.pkl",
                                        num_episodes=3500,  # Increased as requested
                                        episode_length=50,
                                        success_threshold=0.06,
                                        target_coverage_threshold=0.8,  # 80% workspace coverage
                                        max_episodes_to_collect=3000,  # Target 3000 episodes
                                        save_path="sac_diverse_targets.pkl"):
    """
    Simple collection: successful episodes with manually set diverse targets.
    """
    
    print(f"🎯 COLLECTING DIVERSE TARGET DEMONSTRATIONS")
    print(f"Target: {max_episodes_to_collect} successful episodes from {num_episodes} total")
    print(f"Workspace coverage goal: {target_coverage_threshold:.0%}")
    print("=" * 50)
    
    # Load reachable positions
    reachable_positions = load_reachable_positions()
    if reachable_positions is None:
        print("❌ Cannot load workspace positions. Aborting.")
        return None
    
    # Create target pool from 80% of reachable positions
    np.random.seed(42)  # For reproducible target selection
    n_targets = int(len(reachable_positions) * 0.8)  # 80% of workspace
    target_indices = np.random.choice(len(reachable_positions), n_targets, replace=False)
    target_pool = reachable_positions[target_indices]
    
    print(f"📍 Created target pool: {len(target_pool)} targets from {len(reachable_positions)} reachable positions")
    print(f"   Target range: X=[{target_pool[:, 0].min():.3f}, {target_pool[:, 0].max():.3f}], "
          f"Y=[{target_pool[:, 1].min():.3f}, {target_pool[:, 1].max():.3f}]")
    
    # Load model and normalizations
    model = SAC.load(model_path)
    dummy_env = create_env(random_seed=42)
    dummy_vec = DummyVecEnv([lambda: dummy_env])
    original_norm = VecNormalize.load(vecnorm_path, dummy_vec)
    selective_norm = SelectiveVecNormalize.load(selective_norm_path, dummy_vec)
    dummy_vec.close()
    
    # Data storage
    demonstrations = {
        'observations': [],
        'actions': [],
        'rewards': [],
        'dones': [],
        'episode_starts': [],
        'target_positions': []
    }
    
    collected_count = 0
    success_count = 0
    processed_count = 0
    all_targets = []
    
    # Debug tracking
    debug_distances = []
    debug_velocities = []
    
    print("🔄 Collecting episodes with manual targets...")
    
    for episode in tqdm(range(num_episodes), desc="Episodes"):
        if collected_count >= max_episodes_to_collect:
            print(f"\n✅ Collected {max_episodes_to_collect} episodes!")
            break
            
        seed = 42 + episode
        set_seeds(seed)
        processed_count += 1
        
        # Select target from pool (cycle through all targets)
        target_idx = episode % len(target_pool)
        manual_target = target_pool[target_idx]
        
        # Create environment
        dm_env = make_arm_env(random_seed=seed)
        
        # MANUALLY SET THE TARGET POSITION
        dm_env.physics.bind(dm_env._task._arm.target).mocap_pos[:2] = manual_target
        
        raw_env = create_env(random_seed=seed, base_env=dm_env)
        eval_vec = DummyVecEnv([lambda env=raw_env: env])
        eval_vec = VecNormalize.load(vecnorm_path, eval_vec)
        eval_vec.training = False
        
        # Verify target was set correctly
        actual_target = dm_env.physics.bind(dm_env._task._arm.target).mocap_pos[:2].copy()
        
        # Run episode
        obs = eval_vec.reset()
        episode_obs = []
        episode_actions = []
        episode_rewards = []
        distances = []
        
        done = False
        step_count = 0
        
        while not done and step_count < episode_length:
            action, _ = model.predict(obs, deterministic=True)
            episode_obs.append(obs[0].copy())
            
            # Normalize action to [0,1]
            normalized_action = (action[0] + 1.0) / 2.0
            normalized_action = np.clip(normalized_action, 0.0, 1.0)
            episode_actions.append(normalized_action.copy())
            
            obs, reward, done, info = eval_vec.step(action)
            
            # Track distance to target
            hand_pos = dm_env.physics.bind(dm_env._task._arm.hand).xpos
            distance = np.linalg.norm(hand_pos[:2] - actual_target)
            distances.append(distance)
            episode_rewards.append(reward[0])
            
            step_count += 1
            done = done[0] if isinstance(done, np.ndarray) else done
        
        # Check if successful
        final_distance = distances[-1] if distances else float('inf')
        hand_vel = dm_env.physics.bind(dm_env._task._arm.hand).cvel[:3]
        final_velocity = np.linalg.norm(hand_vel)
        
        debug_distances.append(final_distance)
        debug_velocities.append(final_velocity)
        
        is_successful = (final_distance < success_threshold) and (final_velocity < 0.2)
        
        # Debug output for first few episodes
        if processed_count <= 5:
            print(f"\n🔍 DEBUG Episode {processed_count}:")
            print(f"   Manual target: {manual_target}")
            print(f"   Actual target: {actual_target}")
            print(f"   Final distance: {final_distance:.4f} (threshold: {success_threshold})")
            print(f"   Final velocity: {final_velocity:.4f} (threshold: 0.2)")
            print(f"   Success: {is_successful}")
        
        if is_successful:
            success_count += 1
            
            # Convert observations (denormalize then apply selective norm)
            converted_obs = []
            for obs_step in episode_obs:
                # Denormalize
                denormalized = obs_step * np.sqrt(original_norm.obs_rms.var + original_norm.epsilon) + original_norm.obs_rms.mean
                
                # Apply selective normalization
                muscle_part = denormalized[:12]
                rest_part = denormalized[12:]
                normalized_muscle = (muscle_part - selective_norm.obs_rms.mean) / np.sqrt(selective_norm.obs_rms.var + selective_norm.epsilon)
                normalized_muscle = np.clip(normalized_muscle, -selective_norm.clip_obs, selective_norm.clip_obs)
                
                converted_obs_step = np.concatenate([normalized_muscle, rest_part])
                converted_obs.append(converted_obs_step)
            
            # Store episode
            episode_start_idx = len(demonstrations['observations'])
            demonstrations['observations'].extend(converted_obs)
            demonstrations['actions'].extend(episode_actions)
            demonstrations['rewards'].extend(episode_rewards)
            demonstrations['dones'].extend([False] * (len(episode_rewards) - 1) + [True])
            demonstrations['episode_starts'].append(episode_start_idx)
            demonstrations['target_positions'].append(actual_target)
            
            collected_count += 1
            all_targets.append(actual_target)
        
        eval_vec.close()
        
        # Progress update
        if processed_count % 200 == 0:
            current_coverage = check_target_coverage(all_targets, reachable_positions) if all_targets else 0
            success_rate = 100 * success_count / processed_count
            avg_distance = np.mean(debug_distances[-200:]) if debug_distances else 0
            avg_velocity = np.mean(debug_velocities[-200:]) if debug_velocities else 0
            
            print(f"  Episodes {processed_count-199:4d}-{processed_count:4d}: "
                  f"Success: {success_count:4d}/{processed_count:4d} ({success_rate:5.1f}%), "
                  f"Collected: {collected_count:4d}, Coverage: {current_coverage:.1%}")
            print(f"    Avg distance: {avg_distance:.4f}, Avg velocity: {avg_velocity:.4f}")
    
    # Final statistics
    final_coverage = check_target_coverage(all_targets, reachable_positions)
    
    # Convert to numpy and save
    demonstrations['observations'] = np.array(demonstrations['observations'])
    demonstrations['actions'] = np.array(demonstrations['actions'])
    demonstrations['rewards'] = np.array(demonstrations['rewards'])
    demonstrations['dones'] = np.array(demonstrations['dones'])
    demonstrations['episode_starts'] = np.array(demonstrations['episode_starts'])
    demonstrations['target_positions'] = np.array(demonstrations['target_positions'])
    
    # Add metadata
    demonstrations['metadata'] = {
        'num_episodes_processed': processed_count,
        'num_episodes_collected': collected_count,
        'success_count': success_count,
        'success_rate': success_count / processed_count if processed_count > 0 else 0,
        'final_target_coverage': final_coverage,
        'target_coverage_threshold': target_coverage_threshold,
        'success_threshold': success_threshold,
        'episode_length': episode_length,
        'total_steps': len(demonstrations['observations']),
        'reachable_positions': reachable_positions,
        'target_pool': target_pool,
        'model_path': model_path,
        'collection_method': 'manual_target_setting'
    }
    
    # Save
    with open(save_path, 'wb') as f:
        pickle.dump(demonstrations, f)
    
    print(f"\n🎉 COLLECTION COMPLETE")
    print("=" * 50)
    print(f"Episodes processed: {processed_count}")
    print(f"Successful episodes: {success_count} ({100*success_count/processed_count:.1f}%)")
    print(f"Episodes collected: {collected_count}")
    print(f"Target workspace coverage: {final_coverage:.1%}")
    print(f"Total timesteps: {len(demonstrations['observations'])}")
    print(f"Unique targets used: {len(np.unique(all_targets, axis=0))}")
    print(f"Data saved to: {save_path}")
    
    return demonstrations


def visualize_targets(demonstrations, save_path="target_distribution.png"):
    """Simple visualization of target distribution."""
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    reachable_pos = demonstrations['metadata']['reachable_positions']
    target_pos = demonstrations['target_positions']
    
    # Target distribution
    ax1.scatter(reachable_pos[:, 0], reachable_pos[:, 1], c='lightgray', s=1, alpha=0.3, label='Reachable')
    ax1.scatter(target_pos[:, 0], target_pos[:, 1], c='red', s=15, alpha=0.7, label='Targets')
    ax1.set_title('Target Distribution')
    ax1.set_xlabel('X Position (m)')
    ax1.set_ylabel('Y Position (m)')
    ax1.set_aspect('equal')
    ax1.legend()
    
    # Target density
    ax2.hexbin(target_pos[:, 0], target_pos[:, 1], gridsize=15, cmap='Reds')
    ax2.set_title('Target Density')
    ax2.set_xlabel('X Position (m)')
    ax2.set_ylabel('Y Position (m)')
    ax2.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Visualization saved to: {save_path}")


if __name__ == "__main__":
    # Collection with manual target setting for 80% workspace coverage
    demonstrations = collect_diverse_target_demonstrations(
        model_path="arm_final2.zip",
        vecnorm_path="vec_normalize2.pkl",
        selective_norm_path="selective_vec_normalize3.pkl",
        num_episodes=3500,  # Process 3500 episodes
        episode_length=50,
        success_threshold=0.06,
        target_coverage_threshold=0.8,  # 80% workspace coverage
        max_episodes_to_collect=3000,  # Target 3000 successful episodes
        save_path="sac_diverse_targets.pkl"
    )
    
    if demonstrations is not None:
        visualize_targets(demonstrations)
