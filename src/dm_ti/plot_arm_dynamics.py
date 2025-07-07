#!/usr/bin/env python3
"""
Plot muscle activations and hand kinematics for a trained arm policy.
This script runs the policy on the environment and generates plots of:
- Muscle activations over time
- Hand velocity over time 
- Hand position and trajectory
- Distance to target over time
- Reward over time
"""

import os
os.environ['MUJOCO_GL'] = 'osmesa'  # Use software rendering

import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from tianshou.data import Batch
from tianshou.policy import PPOPolicy
from torch.distributions import Independent, Normal

# Import environment creation functions - ADAPTED
from environment import make_arm_env
from shimmy_wrapper import create_env, set_seeds
from policy_networks import RecurrentActorNetwork, CriticNetwork, ModalitySpecificEncoder


def get_args():
    parser = argparse.ArgumentParser(description='Plot muscle activations and hand kinematics')
    parser.add_argument('--checkpoint', type=str, default='logs/ppo_rnn/policy_best.pth', 
                       help='Path to the checkpoint file')
    parser.add_argument('--output-dir', type=str, default='plots', 
                       help='Directory to save plots')
    parser.add_argument('--num-episodes', type=int, default=3, 
                       help='Number of episodes to plot')
    parser.add_argument('--hidden-size', type=int, default=64,  # ADAPTED
                       help='Hidden layer size (must match training)')
    parser.add_argument('--num-layers', type=int, default=1,    # ADAPTED
                       help='Number of RNN layers (must match training)')
    parser.add_argument('--seed', type=int, default=42, 
                       help='Random seed')
    parser.add_argument('--device', type=str, default='cpu', 
                       help='Device to run on')
    parser.add_argument('--style', type=str, default='seaborn-v0_8-whitegrid', 
                       help='Plot style')
    return parser.parse_args()


def make_policy(actor, critic, action_space, device):
    """Create policy with EXACT same configuration as training/evaluation."""
    # ADAPTED: Use same optimizer and hyperparameters as evaluate_policy.py
    optim = torch.optim.Adam([
        {'params': actor.parameters()},
        {'params': critic.parameters()},
    ], lr=1e-3)
    
    def dist_fn(logits):
        mu, sigma = logits
        return Independent(Normal(mu, sigma), 1)
    
    policy = PPOPolicy(
        actor=actor, 
        critic=critic, 
        optim=optim, 
        dist_fn=dist_fn,
        discount_factor=0.99, 
        gae_lambda=0.95, 
        max_grad_norm=1.0,
        vf_coef=0.5, 
        ent_coef=0.01, 
        action_scaling=True,
        action_space=action_space,
        eps_clip=0.2,
        value_clip=1,
        dual_clip=None,
        advantage_normalization=1,
        recompute_advantage=1,
        reward_normalization=0
    )
    return policy.to(device)


def run_episodes_and_collect_data(policy, dm_env, env, args):
    """Run episodes and collect detailed data for plotting."""
    all_episodes_data = []
    success_count = 0
    
    for episode in range(args.num_episodes):
        print(f"Episode {episode + 1}/{args.num_episodes}")
        
        # Reset environment and get initial observation
        obs, _ = env.reset(seed=args.seed + episode)
        terminated = False
        truncated = False
        state = None  # ADAPTED: Reset RNN state for each episode
        episode_reward = 0
        timesteps = 0
        
        # Data structures for this episode
        episode_data = {
            'timestep': [],
            'time': [],
            'hand_pos': [],
            'hand_vel': [],
            'target_pos': [],
            'distance': [],
            'reward': [],
            'muscle_activations': [],
            'cumulative_reward': [],
            'action': [],
            'rnn_state_norm': []  # ADAPTED: Track RNN state evolution
        }
        
        while not (terminated or truncated) and timesteps < 150:  # ADAPTED: Add max steps
            # Format observation for policy - ADAPTED
            batch = Batch(obs=np.expand_dims(obs, axis=0), info={})
            
            # Get action from policy - ADAPTED
            with torch.no_grad():
                result = policy.forward(batch, state=state)
            action = result.act[0]
            state = result.state
            
            # NaN protection - ADAPTED
            if np.isnan(action).any():
                print("WARNING: NaN detected in action, using zero action")
                action = np.zeros_like(action)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            timesteps += 1
            
            # Extract data
            hand_pos = dm_env.physics.bind(dm_env._task._arm.hand).xpos.copy()
            hand_vel = dm_env.physics.bind(dm_env._task._arm.hand).cvel[3:6].copy()  # linear velocity
            target_pos = dm_env.physics.bind(dm_env._task._arm.target).mocap_pos.copy()
            distance = np.linalg.norm(hand_pos - target_pos)
            
            # Get muscle activations 
            muscle_acts = dm_env.physics.data.act.copy()
            
            # ADAPTED: Track RNN state evolution
            if state is not None:
                # Calculate norm of RNN hidden state for visualization
                if hasattr(state, 'h'):
                    rnn_state_norm = torch.norm(state.h).item()
                else:
                    rnn_state_norm = torch.norm(state).item() if torch.is_tensor(state) else 0.0
            else:
                rnn_state_norm = 0.0
            
            # Store data
            episode_data['timestep'].append(timesteps)
            episode_data['time'].append(dm_env.physics.data.time)
            episode_data['hand_pos'].append(hand_pos)
            episode_data['hand_vel'].append(hand_vel)
            episode_data['target_pos'].append(target_pos)
            episode_data['distance'].append(distance)
            episode_data['reward'].append(reward)
            episode_data['muscle_activations'].append(muscle_acts)
            episode_data['cumulative_reward'].append(episode_reward)
            episode_data['action'].append(action.detach().cpu().numpy())
            episode_data['rnn_state_norm'].append(rnn_state_norm)
        
        # Convert lists to numpy arrays for easier plotting
        for key in episode_data:
            if key in ['timestep', 'time', 'distance', 'reward', 'cumulative_reward', 'rnn_state_norm']:
                episode_data[key] = np.array(episode_data[key])
            else:
                episode_data[key] = np.array(episode_data[key])
        
        # Track success
        final_distance = episode_data['distance'][-1]
        success = final_distance < 0.1
        episode_data['success'] = success
        
        if success:
            success_count += 1
            
        print(f"  Episode finished: Reward={episode_reward:.4f}, Steps={timesteps}, " 
              f"Final Distance={final_distance:.4f}, Success={success}")
        
        all_episodes_data.append(episode_data)
    
    print(f"\nSuccess rate: {success_count}/{args.num_episodes} ({100*success_count/args.num_episodes:.1f}%)")
    return all_episodes_data


def plot_episode_dynamics(episode_data, episode_idx, output_dir):
    """Create detailed plots for a single episode."""
    # Set up the figure with subplots
    plt.figure(figsize=(20, 16))
    gs = GridSpec(5, 2, height_ratios=[1.5, 1, 1, 1, 1])
    
    # 1. Muscle activations plot (larger)
    ax1 = plt.subplot(gs[0, :])
    muscle_acts = episode_data['muscle_activations']
    n_muscles = muscle_acts.shape[1]
    
    for i in range(n_muscles):
        ax1.plot(episode_data['time'], muscle_acts[:, i], label=f"Muscle {i+1}")
    
    ax1.set_title(f"Episode {episode_idx+1}: Muscle Activations", fontsize=16)
    ax1.set_ylabel("Activation", fontsize=14)
    ax1.set_ylim(-0.1, 1.1)
    ax1.grid(True)
    if n_muscles <= 12:  # Only show legend if not too many muscles
        ax1.legend(loc='upper right', ncol=min(6, n_muscles))
    
    # 2. Hand velocity plot
    ax2 = plt.subplot(gs[1, 0])
    hand_vel = np.array(episode_data['hand_vel'])
    
    ax2.plot(episode_data['time'], hand_vel[:, 0], 'r-', label="X Velocity")
    ax2.plot(episode_data['time'], hand_vel[:, 1], 'g-', label="Y Velocity")
    ax2.plot(episode_data['time'], hand_vel[:, 2], 'b-', label="Z Velocity")
    
    # Calculate speed (magnitude of velocity)
    speed = np.linalg.norm(hand_vel, axis=1)
    ax2.plot(episode_data['time'], speed, 'k--', label="Speed", linewidth=2)
    
    ax2.set_title("Hand Velocity Components", fontsize=14)
    ax2.set_ylabel("Velocity (m/s)", fontsize=12)
    ax2.grid(True)
    ax2.legend(loc='upper right')
    
    # 3. Distance to target plot
    ax3 = plt.subplot(gs[1, 1])
    ax3.plot(episode_data['time'], episode_data['distance'], 'b-', linewidth=2)
    ax3.axhline(y=0.1, color='r', linestyle='--', label="Success Threshold")
    
    ax3.set_title("Distance to Target", fontsize=14)
    ax3.set_ylabel("Distance (m)", fontsize=12)
    ax3.set_ylim(bottom=0)
    ax3.grid(True)
    ax3.legend(loc='upper right')
    
    # 4. Reward plot
    ax4 = plt.subplot(gs[2, 0])
    ax4.plot(episode_data['time'], episode_data['reward'], 'g-')
    
    ax4.set_title("Per-Step Reward", fontsize=14)
    ax4.set_ylabel("Reward", fontsize=12)
    ax4.grid(True)
    
    # 5. Cumulative reward plot
    ax5 = plt.subplot(gs[2, 1])
    ax5.plot(episode_data['time'], episode_data['cumulative_reward'], 'c-', linewidth=2)
    
    ax5.set_title("Cumulative Reward", fontsize=14)
    ax5.set_ylabel("Reward", fontsize=12)
    ax5.grid(True)
    
    # 6. Hand trajectory 3D plot
    ax6 = plt.subplot(gs[3, 0], projection='3d')
    hand_pos = np.array(episode_data['hand_pos'])
    target_pos = np.array(episode_data['target_pos'])
    
    # Plot trajectory with color gradient to show progression
    points = ax6.scatter(hand_pos[:, 0], hand_pos[:, 1], hand_pos[:, 2], 
                         c=range(len(hand_pos)), cmap='viridis', s=15)
    
    # Plot start and end points
    ax6.scatter(hand_pos[0, 0], hand_pos[0, 1], hand_pos[0, 2], 
                color='blue', s=100, label='Start', marker='o')
    ax6.scatter(hand_pos[-1, 0], hand_pos[-1, 1], hand_pos[-1, 2], 
                color='green', s=100, label='End', marker='o')
    
    # Plot target
    ax6.scatter(target_pos[0, 0], target_pos[0, 1], target_pos[0, 2], 
                color='red', s=150, label='Target', marker='*')
    
    ax6.set_title("Hand Trajectory", fontsize=14)
    ax6.set_xlabel("X", fontsize=12)
    ax6.set_ylabel("Y", fontsize=12)
    ax6.set_zlabel("Z", fontsize=12)
    ax6.legend()
    
    # 7. Input actions plot
    ax7 = plt.subplot(gs[3, 1])
    actions = np.array(episode_data['action'])
    n_actions = actions.shape[1]
    
    for i in range(n_actions):
        ax7.plot(episode_data['time'], actions[:, i], label=f"Action {i+1}")
    
    ax7.set_title("Network Output Actions", fontsize=14)
    ax7.set_xlabel("Time (s)", fontsize=12)
    ax7.set_ylabel("Action Value", fontsize=12)
    ax7.grid(True)
    if n_actions <= 8:  # Only show legend if not too many actions
        ax7.legend(loc='upper right')
    
    # 8. ADAPTED: RNN State Evolution plot
    ax8 = plt.subplot(gs[4, 0])
    ax8.plot(episode_data['time'], episode_data['rnn_state_norm'], 'purple', linewidth=2)
    ax8.set_title("LeakyRNN Hidden State Norm", fontsize=14)
    ax8.set_xlabel("Time (s)", fontsize=12)
    ax8.set_ylabel("State Norm", fontsize=12)
    ax8.grid(True)
    
    # 9. Action magnitude plot
    ax9 = plt.subplot(gs[4, 1])
    actions = np.array(episode_data['action'])
    action_magnitude = np.linalg.norm(actions, axis=1)
    ax9.plot(episode_data['time'], action_magnitude, 'orange', linewidth=2)
    ax9.set_title("Action Magnitude", fontsize=14)
    ax9.set_xlabel("Time (s)", fontsize=12)
    ax9.set_ylabel("||Action||", fontsize=12)
    ax9.grid(True)
    
    # Add success/failure indicator
    success_text = "SUCCESS" if episode_data['success'] else "FAILURE"
    color = "green" if episode_data['success'] else "red"
    plt.figtext(0.5, 0.01, f"Episode Result: {success_text}", 
                ha="center", fontsize=16, bbox={"facecolor":color, "alpha":0.3, "pad":5})
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95, bottom=0.05)
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, f'episode_{episode_idx+1}_dynamics.png'), dpi=150)
    plt.close()


# Keep the existing plot_all_episodes_comparison function unchanged...
def plot_all_episodes_comparison(all_episodes_data, output_dir):
    """Create comparison plots across all episodes."""
    num_episodes = len(all_episodes_data)
    colors = plt.cm.viridis(np.linspace(0, 1, num_episodes))
    
    # 1. Distance to target comparison
    plt.figure(figsize=(12, 7))
    for i, episode_data in enumerate(all_episodes_data):
        label = f"Episode {i+1}" + (" (Success)" if episode_data['success'] else " (Failure)")
        plt.plot(episode_data['time'], episode_data['distance'], 
                 color=colors[i], linewidth=2, label=label)
    
    plt.axhline(y=0.1, color='r', linestyle='--', label="Success Threshold")
    plt.title("Distance to Target Across Episodes", fontsize=16)
    plt.xlabel("Time (s)", fontsize=14)
    plt.ylabel("Distance (m)", fontsize=14)
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_episodes_distance.png'), dpi=150)
    plt.close()
    
    # 2. Hand speed comparison 
    plt.figure(figsize=(12, 7))
    for i, episode_data in enumerate(all_episodes_data):
        hand_vel = np.array(episode_data['hand_vel'])
        speed = np.linalg.norm(hand_vel, axis=1)
        label = f"Episode {i+1}" + (" (Success)" if episode_data['success'] else " (Failure)")
        plt.plot(episode_data['time'], speed, color=colors[i], linewidth=2, label=label)
    
    plt.title("Hand Speed Across Episodes", fontsize=16)
    plt.xlabel("Time (s)", fontsize=14)
    plt.ylabel("Speed (m/s)", fontsize=14)
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_episodes_speed.png'), dpi=150)
    plt.close()
    
    # 3. 3D trajectory comparison
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    for i, episode_data in enumerate(all_episodes_data):
        hand_pos = np.array(episode_data['hand_pos'])
        target_pos = np.array(episode_data['target_pos'])
        
        # Plot trajectory with color gradient
        points = ax.plot(hand_pos[:, 0], hand_pos[:, 1], hand_pos[:, 2], 
                         color=colors[i], alpha=0.7,
                         label=f"Episode {i+1}")
        
        # Plot end points
        ax.scatter(hand_pos[-1, 0], hand_pos[-1, 1], hand_pos[-1, 2], 
                   color=colors[i], s=100, marker='o')
        
        # Plot target (only once)
        if i == 0:
            ax.scatter(target_pos[0, 0], target_pos[0, 1], target_pos[0, 2], 
                       color='red', s=150, label='Target', marker='*')
    
    ax.set_title("Hand Trajectories Across Episodes", fontsize=16)
    ax.set_xlabel("X", fontsize=14)
    ax.set_ylabel("Y", fontsize=14)
    ax.set_zlabel("Z", fontsize=14)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_episodes_trajectories.png'), dpi=150)
    plt.close()
    
    # 4. ADAPTED: RNN state comparison
    plt.figure(figsize=(12, 7))
    for i, episode_data in enumerate(all_episodes_data):
        label = f"Episode {i+1}" + (" (Success)" if episode_data['success'] else " (Failure)")
        plt.plot(episode_data['time'], episode_data['rnn_state_norm'], 
                 color=colors[i], linewidth=2, label=label)
    
    plt.title("LeakyRNN Hidden State Norm Across Episodes", fontsize=16)
    plt.xlabel("Time (s)", fontsize=14)
    plt.ylabel("State Norm", fontsize=14)
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_episodes_rnn_state.png'), dpi=150)
    plt.close()


def main():
    # Get arguments and set style
    args = get_args()
    try:
        plt.style.use(args.style)
    except:
        print(f"Warning: Style {args.style} not found. Using default style.")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set random seeds
    set_seeds(args.seed)
    
    print("Creating DM Control environment...")
    # Create the DM Control environment - ADAPTED
    dm_env = make_arm_env(random_seed=args.seed)
    env = create_env(random_seed=args.seed, base_env=dm_env)
    
    # Get environment shapes
    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape
    
    print(f"Creating policy with observation shape {obs_shape} and action shape {act_shape}")
    
    # ADAPTED: Create policy using same approach as evaluate_policy.py
    # Create shared encoder
    shared_encoder = ModalitySpecificEncoder(target_size=40).to(args.device)
    
    # Create actor & critic with shared encoder
    actor = RecurrentActorNetwork(
        obs_shape=obs_shape, 
        action_shape=act_shape,
        encoder=shared_encoder,
        hidden_size=args.hidden_size, 
        num_layers=args.num_layers,
        device=args.device
    ).to(args.device)
    
    critic = CriticNetwork(
        obs_shape=obs_shape, 
        encoder=shared_encoder,
        hidden_size=args.hidden_size, 
        device=args.device
    ).to(args.device)
    
    # Create policy
    policy = make_policy(actor, critic, env.action_space, args.device)
    
    # Load checkpoint - ADAPTED from evaluate_policy.py
    print(f"Loading checkpoint from {args.checkpoint}")
    try:
        checkpoint = torch.load(args.checkpoint, map_location=args.device)
        
        if isinstance(checkpoint, dict) and 'policy' in checkpoint:
            # New format with individual components
            policy.load_state_dict(checkpoint['policy'], strict=True)
            if 'encoder' in checkpoint:
                shared_encoder.load_state_dict(checkpoint['encoder'])
            print("Model loaded successfully (new format)")
        else:
            # Old format - just policy state dict
            policy.load_state_dict(checkpoint, strict=True)
            print("Model loaded successfully (old format)")
            
    except Exception as e:
        print(f"Strict loading failed: {e}")
        print("Attempting non-strict loading...")
        if isinstance(checkpoint, dict) and 'policy' in checkpoint:
            policy.load_state_dict(checkpoint['policy'], strict=False)
        else:
            policy.load_state_dict(checkpoint, strict=False)
        print("Model loaded with non-strict parameter matching")
    
    # Switch policy to evaluation mode
    policy.eval()
    
    # Run episodes and collect data
    all_episodes_data = run_episodes_and_collect_data(policy, dm_env, env, args)
    
    # Create plots
    print("Creating individual episode plots...")
    for i, episode_data in enumerate(all_episodes_data):
        plot_episode_dynamics(episode_data, i, args.output_dir)
    
    print("Creating comparison plots...")
    plot_all_episodes_comparison(all_episodes_data, args.output_dir)
    
    print(f"Plots saved to {args.output_dir}")
    
    # Close environment
    env.close()


if __name__ == "__main__":
    main()