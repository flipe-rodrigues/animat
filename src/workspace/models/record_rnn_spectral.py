import os
os.environ['MUJOCO_GL'] = 'osmesa'  # Use software rendering for headless systems

import torch
import numpy as np
import imageio
import sys
from pathlib import Path

# Fix the path setup - go to workspace directory
current_file = Path(__file__).resolve()
workspace_root = current_file.parent.parent  # Go up from models/ to workspace/
sys.path.insert(0, str(workspace_root))

print(f"🔧 Current file: {current_file}")
print(f"🔧 Workspace root: {workspace_root}")
print(f"🔧 Python path: {sys.path[:3]}")  # Show first 3 entries

# Now the imports should work
try:
    from stable_baselines3.common.vec_env import DummyVecEnv
    from envs.dm_env import make_arm_env
    from wrappers.rl_wrapper import set_seeds, create_env
    from wrappers.sl_wrappers.normalization import SelectiveVecNormalize
    from networks.rnn import RNNPolicy
    print("✅ All imports successful!")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print(f"   Current working directory: {os.getcwd()}")
    print(f"   Files in workspace: {list(workspace_root.iterdir())}")
    sys.exit(1)

import torch.nn as nn

def calculate_sparsity(tensor, threshold=1e-4):
    """Calculate sparsity of a tensor."""
    return (torch.abs(tensor) < threshold).float().mean().item()

def load_spectral_clamped_rnn(model_path="./rnn_test_25hidden_spectral_clamped2/best_rnn_full.pth"):
    """Load trained spectral radius clamped RNN model."""
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    config = checkpoint['config']
    
    # Create model with saved config - fix parameter names
    policy = RNNPolicy(
        input_dim=config.get('obs_dim', config.get('input_dim', 15)),  # Handle both old and new names
        action_dim=config.get('action_dim', 4),
        hidden_size=config['hidden_size'],
        activation=nn.Sigmoid(),
        alpha=1.0/config.get('tau_mem', 10.0)
    )
    
    # Load weights
    policy.load_state_dict(checkpoint['model_state_dict'])
    policy.eval()
    
    print(f"✅ Loaded SPECTRAL RADIUS CLAMPED RNN model:")
    print(f"   Training loss: {checkpoint.get('loss', 'N/A'):.6f}")
    print(f"   Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"   Hidden size: {config['hidden_size']}")
    print(f"   Action dim: {config.get('action_dim', 4)}")
    print(f"   Input dim: {config.get('obs_dim', config.get('input_dim', 15))}")
    print(f"   Tau mem: {config.get('tau_mem', 'N/A')}")
    print(f"   Spectral radius clamping: {config.get('spectral_radius_clamping', False)}")
    print(f"   Target ρ: {config.get('rho_target', 'N/A')}")
    
    # Analyze spectral radius
    print(f"\n🔍 MODEL SPECTRAL RADIUS ANALYSIS:")
    print("=" * 40)
    with torch.no_grad():
        W_h = policy.W_h.detach().cpu().numpy()
        eigenvalues = np.linalg.eigvals(W_h)
        current_rho = np.max(np.abs(eigenvalues))
        print(f"   Current spectral radius: {current_rho:.4f}")
        
        # Weight statistics
        total_params = 0
        total_sparse_params = 0
        
        print(f"\n📊 WEIGHT STATISTICS & SPARSITY:")
        for name, param in policy.named_parameters():
            param_mean = torch.abs(param).mean()
            param_max = torch.abs(param).max()
            param_std = torch.std(param)
            sparsity = calculate_sparsity(param)
            num_params = param.numel()
            
            total_params += num_params
            total_sparse_params += int(sparsity * num_params)
            
            print(f"   {name:15s}: |Mean|={param_mean:.6f}, |Max|={param_max:.6f}, "
                  f"Std={param_std:.6f}, Sparsity={sparsity:.3f}")
        
        overall_sparsity = total_sparse_params / total_params if total_params > 0 else 0
        print(f"\n🎯 OVERALL SPARSITY: {overall_sparsity:.3f} ({total_sparse_params}/{total_params} params)")
    
    return policy, config

def record_spectral_clamped_episodes(num_episodes=5, 
                                    model_path="./rnn_test_25hidden_spectral_clamped2/best_rnn_full.pth", 
                                    output_path="arm_rnn_spectral_clamped.gif", 
                                    normalization_path="selective_vec_normalize3.pkl",
                                    fps=20,
                                    episode_duration_limit=50):
    """Record episodes with spectral radius clamped RNN model."""
    
    print(f"🎬 RECORDING SPECTRAL RADIUS CLAMPED RNN EPISODES")
    print("=" * 55)
    
    # Load the trained RNN model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy, config = load_spectral_clamped_rnn(model_path)
    policy = policy.to(device)

    # Load normalization stats
    print(f"Loading normalization stats from {normalization_path}...")
    dummy_env = create_env(random_seed=42)
    dummy_vec = DummyVecEnv([lambda: dummy_env])
    
    # Load normalization
    vec_normalize = SelectiveVecNormalize.load(normalization_path, dummy_vec)
    
    # Check if selective (12 dims) or full normalization
    obs_mean = vec_normalize.obs_rms.mean
    obs_std = np.sqrt(vec_normalize.obs_rms.var + vec_normalize.epsilon)
    is_selective = len(obs_mean) == 12
    
    dummy_vec.close()
    
    print(f"✅ Loaded normalization stats:")
    print(f"   Type: {'Selective (first 12 dims)' if is_selective else 'Full observation space'}")
    print(f"   Mean shape: {obs_mean.shape}")
    print(f"   Std shape: {obs_std.shape}")
    print(f"   Epsilon: {vec_normalize.epsilon}")
    print(f"   Clip obs: {vec_normalize.clip_obs}")
    
    if config.get('spectral_radius_clamping'):
        print(f"🔧 Model trained with SPECTRAL RADIUS CLAMPING!")
        print(f"   Target ρ: {config.get('rho_target', 'N/A')}")

    # Initialize trackers
    successful_episodes = []
    all_frames = []
    episode_stats = []

    # Record episodes
    episode = 0
    while len(successful_episodes) < num_episodes:
        episode += 1
        seed = 82 + episode
        set_seeds(seed)
        
        print(f"\nRecording episode {episode} (need {num_episodes - len(successful_episodes)} more) with seed {seed}...")
        
        # Create environment
        dm_env = make_arm_env(random_seed=seed)
        raw_env = create_env(random_seed=seed, base_env=dm_env) 
        eval_vec = DummyVecEnv([lambda env=raw_env: env])
        
        # Initialize episode variables
        obs = eval_vec.reset()
        done = False
        episode_reward = 0.0
        distances = []
        episode_frames = []
        muscle_activations_history = []
        action_history = []
        hidden_states_norms = []
        
        # Initialize hidden state
        hidden = policy.init_hidden(1, device)
        
        # Normalize observations
        if is_selective:
            obs[:, :12] = (obs[:, :12] - obs_mean[None, :]) / obs_std[None, :]
            obs[:, :12] = np.clip(obs[:, :12], -vec_normalize.clip_obs, vec_normalize.clip_obs)
        else:
            obs = (obs - obs_mean[None, :]) / obs_std[None, :]
            obs = np.clip(obs, -vec_normalize.clip_obs, vec_normalize.clip_obs)

        # Record episode
        step_count = 0
        while not done and step_count < episode_duration_limit:
            # RNN forward pass
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
            with torch.no_grad():
                action_tensor, hidden = policy(obs_tensor, hidden)
                action = action_tensor.cpu().numpy()
                
                # Track outputs
                action_history.append(action[0].copy())
                
                # Track hidden state norm
                if isinstance(hidden, tuple):
                    hidden_norm = torch.norm(hidden[0]).item()
                else:
                    hidden_norm = torch.norm(hidden).item()
                hidden_states_norms.append(hidden_norm)
 
            # Step environment
            obs, reward, done, info = eval_vec.step(action)
            
            # Normalize new observation
            if is_selective:
                obs[:, :12] = (obs[:, :12] - obs_mean[None, :]) / obs_std[None, :]
                obs[:, :12] = np.clip(obs[:, :12], -vec_normalize.clip_obs, vec_normalize.clip_obs)
            else:
                obs = (obs - obs_mean[None, :]) / obs_std[None, :]
                obs = np.clip(obs, -vec_normalize.clip_obs, vec_normalize.clip_obs)
            
            episode_reward += reward[0]
            step_count += 1
            
            # Get distance
            hand_pos = dm_env.physics.bind(dm_env._task._arm.hand).xpos
            target_pos = dm_env.physics.bind(dm_env._task._arm.target).mocap_pos
            distance = np.linalg.norm(hand_pos - target_pos)
            distances.append(distance)
            
            # Get muscle activations
            muscle_activations = dm_env.physics.data.ctrl
            muscle_activations_history.append(muscle_activations.copy())
            
            # Update tendon colors
            if step_count == 1:
                inactive_color = np.array([0.1, 0.1, 0.4])
                active_color = np.array([1.0, 0.2, 0.2])

            for i, activation in enumerate(muscle_activations):
                if i < len(dm_env.physics.model.tendon_rgba):
                    interpolated_color = inactive_color + activation * (active_color - inactive_color)
                    dm_env.physics.model.tendon_rgba[i][:3] = interpolated_color
                    dm_env.physics.model.tendon_rgba[i][3] = 1.0
            
            # Render frame
            frame = dm_env.physics.render(height=1080, width=1920, camera_id=0)
            episode_frames.append(frame)
            
            # Debug info
            if step_count % 20 == 0:
                print(f"    Step {step_count:2d}: reward={reward[0]:6.3f}, "
                      f"distance={distance:.4f}, hidden_norm={hidden_norm:.3f}")
            
            done = done[0] if isinstance(done, np.ndarray) else done
        
        # Episode results
        final_distance = distances[-1] if distances else float('inf')
        min_distance = min(distances) if distances else float('inf')
        is_success = final_distance < 0.06
        
        # Hidden state stability
        if hidden_states_norms:
            max_hidden_norm = max(hidden_states_norms)
            avg_hidden_norm = np.mean(hidden_states_norms)
            hidden_stability = max_hidden_norm / avg_hidden_norm if avg_hidden_norm > 0 else float('inf')
        else:
            max_hidden_norm = avg_hidden_norm = hidden_stability = 0.0
        
        # Action analysis
        if action_history:
            action_array = np.array(action_history)
            action_usage = {
                'mean_actions': action_array.mean(axis=0),
                'max_actions': action_array.max(axis=0),
                'min_actions': action_array.min(axis=0),
                'std_actions': action_array.std(axis=0),
                'action_range': action_array.max(axis=0) - action_array.min(axis=0)
            }
        else:
            action_usage = None
        
        # Muscle analysis
        if muscle_activations_history:
            muscle_array = np.array(muscle_activations_history)
            num_muscles = muscle_array.shape[1]
            muscle_usage = {
                'mean_activations': muscle_array.mean(axis=0),
                'max_activations': muscle_array.max(axis=0),
                'activity_ratio': (muscle_array > 0.1).mean(axis=0),
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
            'max_hidden_norm': max_hidden_norm,
            'avg_hidden_norm': avg_hidden_norm,
            'hidden_stability': hidden_stability,
            'action_usage': action_usage,
            'muscle_usage': muscle_usage
        }
        
        print(f"  📊 Episode {episode} results:")
        print(f"    Steps: {step_count}")
        print(f"    Total reward: {episode_reward:.2f}")
        print(f"    Final distance: {final_distance:.4f}")
        print(f"    Min distance: {min_distance:.4f}")
        print(f"    Hidden stability: {hidden_stability:.2f}")
        print(f"    Success: {'✅' if is_success else '❌'}")
        
        if action_usage:
            print(f"    Action range: [{action_usage['min_actions'].min():.3f}, {action_usage['max_actions'].max():.3f}]")
        
        if muscle_usage:
            active_muscles = np.sum(muscle_usage['activity_ratio'] > 0.2)
            print(f"    Active muscles: {active_muscles}/{muscle_usage['num_muscles']}")
        
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
    
    # Analysis
    total_attempts = episode
    success_rate = len(successful_episodes) / total_attempts if total_attempts > 0 else 0
    
    print(f"\n📊 SPECTRAL RADIUS CLAMPED RNN ANALYSIS:")
    print("=" * 50)
    print(f"   Success rate: {len(successful_episodes)}/{total_attempts} ({success_rate:.1%})")
    print(f"   Total frames: {len(all_frames)}")
    
    if successful_episodes:
        avg_steps = np.mean([ep['steps'] for ep in successful_episodes])
        avg_reward = np.mean([ep['total_reward'] for ep in successful_episodes])
        avg_final_dist = np.mean([ep['final_distance'] for ep in successful_episodes])
        avg_hidden_stability = np.mean([ep['hidden_stability'] for ep in successful_episodes])
        
        print(f"\n🎯 SUCCESSFUL EPISODES:")
        print(f"   Average steps: {avg_steps:.1f}")
        print(f"   Average reward: {avg_reward:.2f}")
        print(f"   Average final distance: {avg_final_dist:.4f}")
        print(f"   Average hidden stability: {avg_hidden_stability:.2f}")
        
        # Hidden state analysis
        max_stability = max([ep['hidden_stability'] for ep in successful_episodes])
        min_stability = min([ep['hidden_stability'] for ep in successful_episodes])
        print(f"\n🧠 HIDDEN STATE STABILITY:")
        print(f"   Stability range: {min_stability:.2f} - {max_stability:.2f}")
        print(f"   Well-controlled: {'✅' if max_stability < 5.0 else '⚠️'}")
        
        # Action analysis
        successful_action_data = [ep['action_usage'] for ep in successful_episodes if ep['action_usage']]
        if successful_action_data:
            print(f"\n🎮 ACTION ANALYSIS:")
            avg_mean_actions = np.mean([au['mean_actions'] for au in successful_action_data], axis=0)
            avg_action_ranges = np.mean([au['action_range'] for au in successful_action_data], axis=0)
            
            for i in range(len(avg_mean_actions)):
                print(f"   Action {i:2d}: mean={avg_mean_actions[i]:.3f}, range={avg_action_ranges[i]:.3f}")
        
        # Muscle analysis
        successful_muscle_data = [ep['muscle_usage'] for ep in successful_episodes if ep['muscle_usage']]
        if successful_muscle_data:
            print(f"\n💪 MUSCLE ANALYSIS:")
            num_muscles = successful_muscle_data[0]['num_muscles']
            avg_activity_ratios = np.mean([mu['activity_ratio'] for mu in successful_muscle_data], axis=0)
            
            most_used = np.argmax(avg_activity_ratios)
            least_used = np.argmin(avg_activity_ratios)
            truly_active = np.sum(avg_activity_ratios > 0.1)
            
            print(f"   Most used muscle: {most_used} ({avg_activity_ratios[most_used]:.1%})")
            print(f"   Least used muscle: {least_used} ({avg_activity_ratios[least_used]:.1%})")
            print(f"   Truly active muscles: {truly_active}/{num_muscles}")
    
    # Save video
    all_frames = False
    if all_frames:
        print(f"\n🎬 Saving video to {output_path}...")
        frame_duration = 1000 // fps
        imageio.mimsave(output_path, all_frames, duration=frame_duration, loop=0)
        print(f"✅ Video saved successfully!")
        
        print(f"\n🧠 SPECTRAL RADIUS CLAMPED RNN SUMMARY:")
        print(f"   Model: {model_path}")
        print(f"   Normalization: {normalization_path}")
        print(f"   Success rate: {success_rate:.1%}")
        print(f"   Video: {output_path}")
        print(f"   Spectral clamping: {config.get('spectral_radius_clamping', False)}")
        print(f"   Target ρ: {config.get('rho_target', 'N/A')}")
        print(f"   Hidden size: {config['hidden_size']}")
        
    else:
        print(f"⚠️ No successful episodes to save!")
    
    return successful_episodes, episode_stats

if __name__ == "__main__":
    successful_eps, all_eps = record_spectral_clamped_episodes(
        num_episodes=30, 
        model_path="models/rnn_test_64hidden_spectral_clamped4/best_rnn_full.pth",
        output_path="models/arm_rnn_spectral_clamped_performance.gif",
        normalization_path="models/selective_vec_normalize3.pkl",
        fps=20,
        episode_duration_limit=50
    )
    
    if successful_eps:
        print(f"\n🎉 SUCCESS! Spectral radius clamped RNN results:")
        print(f"   Controlled recurrent dynamics")
        print(f"   Stable hidden state evolution")
        print(f"   Task completion demonstrated!")