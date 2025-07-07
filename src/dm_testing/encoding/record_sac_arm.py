import os
os.environ['MUJOCO_GL'] = 'osmesa'  # Use software rendering for headless systems

import numpy as np
import imageio
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from environment import make_arm_env
from shimmy_wrapper import set_seeds, create_env

def record_sac_episodes_dm_control(num_episodes=5, 
                                  model_path="arm_final2", 
                                  vecnorm_path="vec_normalize2.pkl",
                                  output_path="arm_sac_performance.gif", 
                                  fps=20,
                                  episode_duration_limit=100):
    """Record episodes with trained SAC model using VecNormalize."""
    
    print(f"🎬 RECORDING SAC EPISODES")
    print("=" * 40)
    
    # Load the trained SAC model
    print(f"Loading SAC model from {model_path}...")
    model = SAC.load(model_path)
    print(f"✅ SAC model loaded successfully")
    
    # Load normalization stats
    print(f"Loading normalization stats from {vecnorm_path}...")
    dummy_env = create_env(random_seed=42)
    dummy_vec = DummyVecEnv([lambda: dummy_env])
    
    # Load VecNormalize
    vec_normalize = VecNormalize.load(vecnorm_path, dummy_vec)
    
    obs_mean = vec_normalize.obs_rms.mean  # Shape: (93,)
    obs_std = np.sqrt(vec_normalize.obs_rms.var + vec_normalize.epsilon)  # Shape: (93,)
    
    dummy_vec.close()
    
    print(f"✅ Loaded normalization stats")
    print(f"   Obs mean shape: {obs_mean.shape}")
    print(f"   Obs std shape: {obs_std.shape}")
    print(f"   Epsilon: {vec_normalize.epsilon}")
    print(f"   Clip obs: {vec_normalize.clip_obs}")
    
    # Initialize trackers
    successful_episodes = []
    all_frames = []
    episode_stats = []

    # Record episodes until we get the desired number of successes
    episode = 0
    while len(successful_episodes) < num_episodes:
        episode += 1
        seed = 82 + episode
        set_seeds(seed)
        
        print(f"\nRecording episode {episode} (need {num_episodes - len(successful_episodes)} more successes) with seed {seed}...")
        
        # Create CLEAN environment (no VecNormalize, just encoding)
        dm_env = make_arm_env(random_seed=seed)
        raw_env = create_env(random_seed=seed, base_env=dm_env) 
        eval_vec = DummyVecEnv([lambda env=raw_env: env])
        
        # Initialize episode variables
        obs = eval_vec.reset()  # This gives encoded obs [1, 93]
        done = False
        episode_reward = 0.0
        distances = []
        episode_frames = []
        muscle_activations_history = []
        action_history = []
        
        # NORMALIZE ALL OBSERVATIONS using VecNormalize stats
        obs = (obs - obs_mean[None, :]) / obs_std[None, :]
        obs = np.clip(obs, -vec_normalize.clip_obs, vec_normalize.clip_obs)
        
        print(f"   First obs stats: mean={obs.mean():.3f}, std={obs.std():.3f}")

        # Record episode
        step_count = 0
        while not done and step_count < episode_duration_limit:
            # SAC policy prediction (deterministic for evaluation)
            action, _ = model.predict(obs, deterministic=True)
            
            # Track action outputs
            action_history.append(action[0].copy())  # Remove batch dimension
            
            # Step environment
            obs, reward, done, info = eval_vec.step(action)
            
            # NORMALIZE the new observation
            obs = (obs - obs_mean[None, :]) / obs_std[None, :]
            obs = np.clip(obs, -vec_normalize.clip_obs, vec_normalize.clip_obs)
            
            episode_reward += reward[0]
            step_count += 1
            
            # Get target and hand positions for distance tracking
            hand_pos = dm_env.physics.bind(dm_env._task._arm.hand).xpos
            target_pos = dm_env.physics.bind(dm_env._task._arm.target).mocap_pos
            distance = np.linalg.norm(hand_pos - target_pos)
            distances.append(distance)
            
            # Get muscle activations for visualization and analysis
            muscle_activations = dm_env.physics.data.ctrl  # This should be [0-1]
            muscle_activations_history.append(muscle_activations.copy())
            
            # Update tendon colors based on activations
            if step_count == 1:
                inactive_color = np.array([0.1, 0.1, 0.4])  # Dark blue
                active_color = np.array([1.0, 0.2, 0.2])    # Bright red
                color_range = (inactive_color, active_color)

            for i, activation in enumerate(muscle_activations):
                if i < len(dm_env.physics.model.tendon_rgba):
                    inactive_color, active_color = color_range
                    interpolated_color = inactive_color + activation * (active_color - inactive_color)
                    dm_env.physics.model.tendon_rgba[i][:3] = interpolated_color
                    dm_env.physics.model.tendon_rgba[i][3] = 1.0
            
            # Render frame
            frame = dm_env.physics.render(height=1080, width=1920, camera_id=0)
            episode_frames.append(frame)
            
            # Debug info
            if step_count % 20 == 0:
                print(f"    Step {step_count:2d}: reward={reward[0]:6.3f}, "
                      f"distance={distance:.4f}, "
                      f"action_max={np.abs(action).max():.3f}, activations_max={muscle_activations.max():.3f}")
            
            done = done[0] if isinstance(done, np.ndarray) else done
        
        # Episode results
        final_distance = distances[-1] if distances else float('inf')
        min_distance = min(distances) if distances else float('inf')
        is_success = final_distance < 0.06
        
        # Analyze SAC action outputs
        if action_history:
            action_array = np.array(action_history)  # Shape: (steps, action_dim)
            action_usage = {
                'mean_actions': action_array.mean(axis=0),
                'max_actions': action_array.max(axis=0),
                'min_actions': action_array.min(axis=0),
                'std_actions': action_array.std(axis=0),
                'action_range': action_array.max(axis=0) - action_array.min(axis=0)
            }
        else:
            action_usage = None
        
        # Analyze muscle activations
        if muscle_activations_history:
            muscle_array = np.array(muscle_activations_history)  # Shape: (steps, num_muscles)
            num_muscles = muscle_array.shape[1]
            muscle_usage = {
                'mean_activations': muscle_array.mean(axis=0),
                'max_activations': muscle_array.max(axis=0),
                'activity_ratio': (muscle_array > 0.1).mean(axis=0),  # Fraction of time each muscle was active
                'total_activation': muscle_array.sum(axis=0),
                'num_muscles': num_muscles
            }
        else:
            muscle_usage = None
        
        episode_info = {
            'episode': episode,
            'seed': seed,
            'steps': step_count,
            'total_reward': episode_reward,
            'final_distance': final_distance,
            'min_distance': min_distance,
            'success': is_success,
            'action_usage': action_usage,
            'muscle_usage': muscle_usage
        }
        
        print(f"  📊 Episode {episode} results:")
        print(f"    Steps: {step_count}")
        print(f"    Total reward: {episode_reward:.2f}")
        print(f"    Final distance: {final_distance:.4f}")
        print(f"    Min distance: {min_distance:.4f}")
        print(f"    Success: {'✅' if is_success else '❌'}")
        
        if action_usage:
            print(f"    SAC action range: [{action_usage['min_actions'].min():.3f}, {action_usage['max_actions'].max():.3f}]")
            print(f"    SAC action std: {action_usage['std_actions'].mean():.3f}")
        
        if muscle_usage:
            active_muscles = np.sum(muscle_usage['activity_ratio'] > 0.2)  # Muscles active >20% of time
            print(f"    Active muscles: {active_muscles}/{muscle_usage['num_muscles']}")
            print(f"    Max muscle usage: {muscle_usage['max_activations'].max():.2f}")
        
        episode_stats.append(episode_info)
        
        if is_success:
            print(f"    ✅ Adding successful episode to video")
            all_frames.extend(episode_frames)
            successful_episodes.append(episode_info)
        else:
            print(f"    ❌ Skipping failed episode")
        
        eval_vec.close()
        
        if len(successful_episodes) >= num_episodes:
            print(f"\n🎯 Reached target of {num_episodes} successful episodes after {episode} attempts")
            break
    
    # Comprehensive analysis
    total_attempts = episode
    success_rate = len(successful_episodes) / total_attempts if total_attempts > 0 else 0
    
    print(f"\n📊 COMPREHENSIVE SAC ANALYSIS:")
    print("=" * 40)
    print(f"   Success rate: {len(successful_episodes)}/{total_attempts} ({success_rate:.1%})")
    print(f"   Total frames: {len(all_frames)}")
    
    # Analyze successful episodes
    if successful_episodes:
        avg_steps = np.mean([ep['steps'] for ep in successful_episodes])
        avg_reward = np.mean([ep['total_reward'] for ep in successful_episodes])
        avg_final_dist = np.mean([ep['final_distance'] for ep in successful_episodes])
        
        print(f"\n🎯 SUCCESSFUL EPISODES ANALYSIS:")
        print(f"   Average steps: {avg_steps:.1f}")
        print(f"   Average reward: {avg_reward:.2f}")
        print(f"   Average final distance: {avg_final_dist:.4f}")
        
        # SAC Action analysis for successful episodes
        successful_action_data = []
        for ep in successful_episodes:
            if ep['action_usage']:
                successful_action_data.append(ep['action_usage'])
        
        if successful_action_data:
            print(f"\n🎮 SAC ACTION OUTPUT ANALYSIS (successful episodes):")
            avg_mean_actions = np.mean([au['mean_actions'] for au in successful_action_data], axis=0)
            avg_action_ranges = np.mean([au['action_range'] for au in successful_action_data], axis=0)
            avg_action_stds = np.mean([au['std_actions'] for au in successful_action_data], axis=0)
            
            for i in range(len(avg_mean_actions)):
                print(f"   Action {i:2d}: mean={avg_mean_actions[i]:.3f}, "
                      f"range={avg_action_ranges[i]:.3f}, std={avg_action_stds[i]:.3f}")
        
        # Muscle usage analysis for successful episodes
        successful_muscle_data = []
        for ep in successful_episodes:
            if ep['muscle_usage']:
                successful_muscle_data.append(ep['muscle_usage'])
        
        if successful_muscle_data:
            print(f"\n💪 MUSCLE USAGE ANALYSIS (successful episodes):")
            first_muscle_data = successful_muscle_data[0]
            num_muscles = first_muscle_data['num_muscles']
            
            avg_mean_activations = np.mean([mu['mean_activations'] for mu in successful_muscle_data], axis=0)
            avg_activity_ratios = np.mean([mu['activity_ratio'] for mu in successful_muscle_data], axis=0)
            
            for i in range(num_muscles):
                print(f"   Muscle {i:2d}: avg_activation={avg_mean_activations[i]:.3f}, "
                      f"active_ratio={avg_activity_ratios[i]:.3f}")
            
            # Identify most/least used muscles
            most_used = np.argmax(avg_activity_ratios)
            least_used = np.argmin(avg_activity_ratios)
            print(f"   Most used muscle: {most_used} (active {avg_activity_ratios[most_used]:.1%} of time)")
            print(f"   Least used muscle: {least_used} (active {avg_activity_ratios[least_used]:.1%} of time)")
            
            # Count truly active muscles
            truly_active = np.sum(avg_activity_ratios > 0.1)
            print(f"   Truly active muscles (>10% usage): {truly_active}/{num_muscles}")
    
    # Save results
    all_frames = False
    if all_frames:
        print(f"\n🎬 Saving video to {output_path}...")
        frame_duration = 1000 // fps
        imageio.mimsave(output_path, all_frames, duration=frame_duration, loop=0)
        print(f"✅ Video saved successfully!")
        
        # Summary for the SAC model
        print(f"\n🤖 SAC MODEL SUMMARY:")
        print(f"   Model: {model_path}")
        print(f"   VecNormalize: {vecnorm_path}")
        print(f"   Success rate: {success_rate:.1%}")
        print(f"   Video: {output_path}")
        
    else:
        print(f"⚠️ No successful episodes to save!")
    
    return successful_episodes, episode_stats

if __name__ == "__main__":
    # Record your trained SAC model
    successful_eps, all_eps = record_sac_episodes_dm_control(
        num_episodes=30, 
        model_path="arm_final2",  # Your SAC model
        vecnorm_path="vec_normalize2.pkl",  # Your normalization stats
        output_path="arm_sac_performance.gif",
        fps=20,
        episode_duration_limit=50  # Match training episode length
    )
    
    if successful_eps:
        print(f"\n🎉 SUCCESS! SAC model results:")
        print(f"   Trained policy performance")
        print(f"   Task completion demonstrated!")