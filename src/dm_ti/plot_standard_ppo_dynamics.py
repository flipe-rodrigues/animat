#!/usr/bin/env python3
"""
Plot muscle activations and hand kinematics for a trained standard PPO policy.
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
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic

# Import environment creation functions - ADAPTED for standard PPO
from environment import make_arm_env
from shimmy_wrapper import create_env, set_seeds
from bio_networks import ModalitySpecificEncoder
from train_standard_ppo import NormalizedEncodingWrapper


def get_args():
    parser = argparse.ArgumentParser(description='Plot muscle activations and hand kinematics for standard PPO')
    parser.add_argument('--checkpoint', type=str, default='models/encoded_ppo/encoded_ppo_best.pth', 
                       help='Path to the checkpoint file')
    parser.add_argument('--output-dir', type=str, default='plots_standard_ppo', 
                       help='Directory to save plots')
    parser.add_argument('--num-episodes', type=int, default=3, 
                       help='Number of episodes to plot')
    parser.add_argument('--hidden-sizes', type=int, nargs='+', default=[64, 64],  # ADAPTED
                       help='Hidden layer sizes (must match training)')
    parser.add_argument('--seed', type=int, default=42, 
                       help='Random seed')
    parser.add_argument('--device', type=str, default='cpu', 
                       help='Device to run on')
    parser.add_argument('--style', type=str, default='seaborn-v0_8-whitegrid', 
                       help='Plot style')
    return parser.parse_args()


def create_standard_policy(state_shape, action_shape, max_action, hidden_sizes, device, action_space):
    """Create a standard MLP-based PPO policy with encoding - ADAPTED from evaluate_standard_ppo.py"""
    # Create encoder - EXACTLY as in train_standard_ppo.py
    encoder = ModalitySpecificEncoder(
        target_size=39,  # FIXED: match the encoder used in training
        device=device
    )
    
    encoded_state_shape = encoder.output_size
    input_norm = torch.nn.LayerNorm(encoded_state_shape)  # FIXED: Add normalizer as in training
    
    # Actor network
    net_a = Net(
        encoded_state_shape, 
        hidden_sizes=hidden_sizes,
        activation=torch.nn.Tanh,
        device=device
    )
    actor_base = ActorProb(
        net_a, 
        action_shape,
        max_action=max_action,
        unbounded=True,
        device=device
    )
    # Wrap with normalized encoder to match training
    actor = NormalizedEncodingWrapper(actor_base, encoder, input_norm)
   
    # Critic network
    net_c = Net(
        encoded_state_shape, 
        hidden_sizes=hidden_sizes,
        activation=torch.nn.Tanh,
        device=device
    )
    critic_base = Critic(net_c, device=device)
    # Wrap with normalized encoder to match training
    critic = NormalizedEncodingWrapper(critic_base, encoder, input_norm)
    
    # Create optimizer (not needed for evaluation but PPOPolicy requires it)
    optim = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()), 
        lr=3e-4
    )
    
    # Create PPO policy with EXACTLY the same parameters as training
    policy = PPOPolicy(
        actor=actor,
        critic=critic,
        optim=optim,
        # FIXED: Use the same dist_fn as in training
        dist_fn=lambda x: torch.distributions.Independent(
            torch.distributions.Normal(*x), 1
        ),
        discount_factor=0.99,
        eps_clip=0.2,
        vf_coef=0.5,
        ent_coef=0.04,  # FIXED: match training value
        action_space=action_space,
        action_scaling=True,
        reward_normalization=False,
        advantage_normalization=True,
        recompute_advantage=True,
        value_clip=True,
    )
    
    return policy


def run_episodes_and_collect_data(policy, dm_env, env, args):
    """Run episodes and collect detailed data for plotting - ADAPTED for standard PPO."""
    all_episodes_data = []
    success_count = 0
    
    for episode in range(args.num_episodes):
        print(f"Episode {episode + 1}/{args.num_episodes}")
        
        # Reset environment and get initial observation
        obs, _ = env.reset(seed=args.seed + episode)
        terminated = False
        truncated = False
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
            'action_magnitude': []  # ADAPTED: No RNN state for standard PPO
        }
        
        while not (terminated or truncated) and timesteps < 150:  # ADAPTED: Add max steps
            # Format observation for policy - ADAPTED for standard PPO
            batch_obs = np.expand_dims(obs, axis=0)
            batch = Batch(obs=batch_obs, info={})
            
            # Get action from policy - ADAPTED (no state tracking)
            with torch.no_grad():
                result = policy.forward(batch)
            action = result.act[0]
            
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
            
            # Calculate action magnitude
            action_magnitude = np.linalg.norm(action.detach().cpu().numpy())
            
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
            episode_data['action_magnitude'].append(action_magnitude)
        
        # Convert lists to numpy arrays for easier plotting
        for key in episode_data:
            if key in ['timestep', 'time', 'distance', 'reward', 'cumulative_reward', 'action_magnitude']:
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
    """Create detailed plots for a single episode - ADAPTED for standard PPO."""
    # Set up the figure with subplots
    plt.figure(figsize=(20, 16))
    gs = GridSpec(5, 2, height_ratios=[1.5, 1, 1, 1, 1])
    
    # 1. Muscle activations plot (larger)
    ax1 = plt.subplot(gs[0, :])
    muscle_acts = episode_data['muscle_activations']
    n_muscles = muscle_acts.shape[1]
    
    for i in range(n_muscles):
        ax1.plot(episode_data['time'], muscle_acts[:, i], label=f"Muscle {i+1}")
    
    ax1.set_title(f"Episode {episode_idx+1}: Muscle Activations (Standard PPO)", fontsize=16)
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
    
    # 8. ADAPTED: Action magnitude plot (instead of RNN state)
    ax8 = plt.subplot(gs[4, 0])
    ax8.plot(episode_data['time'], episode_data['action_magnitude'], 'orange', linewidth=2)
    ax8.set_title("Action Magnitude (MLP Policy)", fontsize=14)
    ax8.set_xlabel("Time (s)", fontsize=12)
    ax8.set_ylabel("||Action||", fontsize=12)
    ax8.grid(True)
    
    # 9. Hand position components plot
    ax9 = plt.subplot(gs[4, 1])
    hand_pos = np.array(episode_data['hand_pos'])
    target_pos = np.array(episode_data['target_pos'])
    
    ax9.plot(episode_data['time'], hand_pos[:, 0], 'r-', label="Hand X")
    ax9.plot(episode_data['time'], hand_pos[:, 1], 'g-', label="Hand Y")
    ax9.plot(episode_data['time'], hand_pos[:, 2], 'b-', label="Hand Z")
    
    # Plot target position (should be constant)
    ax9.axhline(y=target_pos[0, 0], color='r', linestyle='--', alpha=0.5, label="Target X")
    ax9.axhline(y=target_pos[0, 1], color='g', linestyle='--', alpha=0.5, label="Target Y") 
    ax9.axhline(y=target_pos[0, 2], color='b', linestyle='--', alpha=0.5, label="Target Z")
    
    ax9.set_title("Hand Position Components", fontsize=14)
    ax9.set_xlabel("Time (s)", fontsize=12)
    ax9.set_ylabel("Position (m)", fontsize=12)
    ax9.grid(True)
    ax9.legend(loc='upper right')
    
    # Add success/failure indicator
    success_text = "SUCCESS" if episode_data['success'] else "FAILURE"
    color = "green" if episode_data['success'] else "red"
    plt.figtext(0.5, 0.01, f"Episode Result: {success_text} (Standard PPO)", 
                ha="center", fontsize=16, bbox={"facecolor":color, "alpha":0.3, "pad":5})
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95, bottom=0.05)
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, f'standard_ppo_episode_{episode_idx+1}_dynamics.png'), dpi=150)
    plt.close()


def plot_all_episodes_comparison(all_episodes_data, output_dir):
    """Create comparison plots across all episodes - ADAPTED for standard PPO."""
    num_episodes = len(all_episodes_data)
    colors = plt.cm.viridis(np.linspace(0, 1, num_episodes))
    
    # 1. Distance to target comparison
    plt.figure(figsize=(12, 7))
    for i, episode_data in enumerate(all_episodes_data):
        label = f"Episode {i+1}" + (" (Success)" if episode_data['success'] else " (Failure)")
        plt.plot(episode_data['time'], episode_data['distance'], 
                 color=colors[i], linewidth=2, label=label)
    
    plt.axhline(y=0.1, color='r', linestyle='--', label="Success Threshold")
    plt.title("Distance to Target Across Episodes (Standard PPO)", fontsize=16)
    plt.xlabel("Time (s)", fontsize=14)
    plt.ylabel("Distance (m)", fontsize=14)
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'standard_ppo_all_episodes_distance.png'), dpi=150)
    plt.close()
    
    # 2. Hand speed comparison 
    plt.figure(figsize=(12, 7))
    for i, episode_data in enumerate(all_episodes_data):
        hand_vel = np.array(episode_data['hand_vel'])
        speed = np.linalg.norm(hand_vel, axis=1)
        label = f"Episode {i+1}" + (" (Success)" if episode_data['success'] else " (Failure)")
        plt.plot(episode_data['time'], speed, color=colors[i], linewidth=2, label=label)
    
    plt.title("Hand Speed Across Episodes (Standard PPO)", fontsize=16)
    plt.xlabel("Time (s)", fontsize=14)
    plt.ylabel("Speed (m/s)", fontsize=14)
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'standard_ppo_all_episodes_speed.png'), dpi=150)
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
    
    ax.set_title("Hand Trajectories Across Episodes (Standard PPO)", fontsize=16)
    ax.set_xlabel("X", fontsize=14)
    ax.set_ylabel("Y", fontsize=14)
    ax.set_zlabel("Z", fontsize=14)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'standard_ppo_all_episodes_trajectories.png'), dpi=150)
    plt.close()
    
    # 4. ADAPTED: Action magnitude comparison (instead of RNN state)
    plt.figure(figsize=(12, 7))
    for i, episode_data in enumerate(all_episodes_data):
        label = f"Episode {i+1}" + (" (Success)" if episode_data['success'] else " (Failure)")
        plt.plot(episode_data['time'], episode_data['action_magnitude'], 
                 color=colors[i], linewidth=2, label=label)
    
    plt.title("Action Magnitude Across Episodes (Standard PPO)", fontsize=16)
    plt.xlabel("Time (s)", fontsize=14)
    plt.ylabel("||Action||", fontsize=14)
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'standard_ppo_all_episodes_action_magnitude.png'), dpi=150)
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
    state_shape = env.observation_space.shape
    action_shape = env.action_space.shape
    max_action = env.action_space.high[0]
    action_space = env.action_space
    
    print(f"Creating standard PPO policy with observation dim {state_shape[0]} and action dim {action_shape[0]}")
    
    # ADAPTED: Create standard PPO policy using same approach as evaluate_standard_ppo.py
    policy = create_standard_policy(
        state_shape=state_shape[0],
        action_shape=action_shape[0],
        max_action=max_action,
        hidden_sizes=args.hidden_sizes,
        device=args.device,
        action_space=action_space
    )
    
    # Load checkpoint - ADAPTED from evaluate_standard_ppo.py
    print(f"Loading checkpoint from {args.checkpoint}")
    try:
        checkpoint = torch.load(args.checkpoint, map_location=args.device)
        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                policy.load_state_dict(checkpoint['model'])
                print("Loaded checkpoint with 'model' key")
            elif 'state_dict' in checkpoint:
                policy.load_state_dict(checkpoint['state_dict'])
                print("Loaded checkpoint with 'state_dict' key")
            else:
                # Try direct load as a last resort
                policy.load_state_dict(checkpoint)
                print("Loaded checkpoint dictionary directly")
        else:
            # Assume the checkpoint is just the model state dict
            policy.load_state_dict(checkpoint)
            print("Loaded model weights directly")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("WARNING: Running with randomly initialized model!")
    
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