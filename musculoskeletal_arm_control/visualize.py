"""
Visualization script for trained policies.
"""
import torch
import numpy as np
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from envs.reaching_env import ReachingEnv
from agents.mlp_policy import MLPPolicy
from agents.rnn_policy import RNNPolicy


def visualize_policy(
    policy,
    env,
    num_episodes: int = 5,
    is_rnn: bool = False,
    policy_name: str = "Policy"
):
    """
    Visualize a trained policy in the environment.
    
    Args:
        policy: Trained policy
        env: Environment with rendering enabled
        num_episodes: Number of episodes to visualize
        is_rnn: Whether policy is RNN-based
        policy_name: Name for display
    """
    policy.eval()
    
    for episode in range(num_episodes):
        print(f"\n{policy_name} - Episode {episode + 1}/{num_episodes}")
        
        obs, info = env.reset()
        if is_rnn:
            hidden = None
        
        done = False
        episode_reward = 0
        step = 0
        
        print(f"Target: ({info['target_position'][0]:.3f}, {info['target_position'][1]:.3f})")
        print(f"Required hold time: {info['required_hold_time']:.2f}s")
        
        while not done:
            # Get action
            with torch.no_grad():
                if is_rnn:
                    action, hidden = policy.predict(obs, hidden=hidden)
                else:
                    action = policy.predict(obs, deterministic=True)
            
            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            step += 1
            
            # Render
            env.render()
            time.sleep(0.01)  # Slow down for visualization
            
            # Print status every 50 steps
            if step % 50 == 0:
                hand_pos = info['hand_position']
                dist = info['distance_to_target']
                print(f"  Step {step}: Hand=({hand_pos[0]:.3f}, {hand_pos[1]:.3f}), "
                      f"Dist={dist:.3f}, Reward={episode_reward:.1f}")
        
        # Final status
        success = info.get('success', False)
        print(f"  Episode finished in {step} steps")
        print(f"  Total reward: {episode_reward:.2f}")
        print(f"  Success: {success}")
        print(f"  Time at target: {info['time_at_target']:.2f}s")
        
        # Wait before next episode
        if episode < num_episodes - 1:
            time.sleep(1.0)


def main():
    """Main visualization script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize trained policies")
    parser.add_argument(
        "--policy", type=str, choices=["teacher", "student", "both"],
        default="both", help="Which policy to visualize"
    )
    parser.add_argument(
        "--teacher-path", type=str, default="checkpoints/teacher_mlp.pt",
        help="Path to teacher MLP checkpoint"
    )
    parser.add_argument(
        "--student-path", type=str, default="checkpoints/student_rnn.pt",
        help="Path to student RNN checkpoint"
    )
    parser.add_argument(
        "--episodes", type=int, default=3,
        help="Number of episodes to visualize"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Musculoskeletal Arm Control - Policy Visualization")
    print("=" * 60)
    
    # Create environment with rendering
    env = ReachingEnv(
        render_mode="human",
        hold_time_range=(0.2, 0.8),
        reach_threshold=0.03,
        hold_threshold=0.04,
        max_episode_steps=1000
    )
    
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # Load and visualize teacher
    if args.policy in ["teacher", "both"]:
        if os.path.exists(args.teacher_path):
            print(f"\nLoading teacher MLP from {args.teacher_path}")
            teacher_policy = MLPPolicy(
                obs_dim=obs_dim,
                action_dim=action_dim,
                hidden_dims=(256, 256),
                activation="relu"
            )
            teacher_policy.load_state_dict(torch.load(args.teacher_path))
            
            visualize_policy(
                policy=teacher_policy,
                env=env,
                num_episodes=args.episodes,
                is_rnn=False,
                policy_name="Teacher MLP"
            )
        else:
            print(f"Teacher checkpoint not found: {args.teacher_path}")
    
    # Load and visualize student
    if args.policy in ["student", "both"]:
        if os.path.exists(args.student_path):
            print(f"\nLoading student RNN from {args.student_path}")
            student_policy = RNNPolicy(
                obs_dim=obs_dim,
                action_dim=action_dim,
                hidden_size=128,
                num_layers=1
            )
            student_policy.load_state_dict(torch.load(args.student_path))
            
            visualize_policy(
                policy=student_policy,
                env=env,
                num_episodes=args.episodes,
                is_rnn=True,
                policy_name="Student RNN"
            )
        else:
            print(f"Student checkpoint not found: {args.student_path}")
    
    env.close()
    print("\nVisualization complete!")


if __name__ == "__main__":
    main()
