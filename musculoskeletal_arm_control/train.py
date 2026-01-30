# %%
"""
Main training script for musculoskeletal arm control.
Trains MLP with RL, then distills to RNN via behavioral cloning.
"""
import torch
import numpy as np
import sys
import os

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from envs.reaching_env import ReachingEnv
from agents.mlp_policy import MLPPolicy
from agents.rnn_policy import RNNPolicy
from training.ppo_trainer import PPOTrainer
from training.behavioral_cloning import BehavioralCloningTrainer


def train_teacher_mlp(
    env, num_updates: int = 500, save_path: str = "checkpoints/teacher_mlp.pt"
):
    """
    Train teacher MLP policy with PPO.

    Args:
        env: Environment
        num_updates: Number of PPO updates
        save_path: Path to save trained model

    Returns:
        teacher_policy: Trained MLP policy
    """
    print("=" * 60)
    print("Phase 1: Training Teacher MLP with PPO")
    print("=" * 60)

    # Create MLP policy
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    teacher_policy = MLPPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dims=(256, 256),
        activation="tanh",
    )

    # Create PPO trainer
    trainer = PPOTrainer(
        policy=teacher_policy,
        env=env,
        learning_rate=3e-4,
        num_steps=2048,
        batch_size=64,
        num_epochs=10,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Train
    trainer.train(num_updates=num_updates)

    # Get final stats
    stats = trainer.get_stats()
    print(f"\nFinal Teacher MLP Stats:")
    print(f"  Mean Reward: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
    print(f"  Mean Episode Length: {stats['mean_length']:.1f}")

    # Save model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(teacher_policy.state_dict(), save_path)
    print(f"\nSaved teacher MLP to {save_path}")

    return teacher_policy


def train_student_rnn(
    env,
    teacher_policy,
    num_demo_episodes: int = 200,
    num_epochs: int = 50,
    save_path: str = "checkpoints/student_rnn.pt",
):
    """
    Train student RNN policy via behavioral cloning.

    Args:
        env: Environment
        teacher_policy: Pre-trained teacher MLP
        num_demo_episodes: Number of demonstration episodes to collect
        num_epochs: Number of training epochs
        save_path: Path to save trained model

    Returns:
        student_policy: Trained RNN policy
    """
    print("\n" + "=" * 60)
    print("Phase 2: Training Student RNN via Behavioral Cloning")
    print("=" * 60)

    # Create RNN policy
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    student_policy = RNNPolicy(
        obs_dim=obs_dim, action_dim=action_dim, hidden_size=128, num_layers=1
    )

    # Create behavioral cloning trainer
    trainer = BehavioralCloningTrainer(
        student_policy=student_policy,
        teacher_policy=teacher_policy,
        env=env,
        learning_rate=1e-3,
        batch_size=32,
        sequence_length=50,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Collect demonstrations from teacher
    trainer.collect_demonstrations(num_episodes=num_demo_episodes)

    # Train student
    trainer.train(num_epochs=num_epochs, train_ratio=0.8)

    # Evaluate student
    print("\nEvaluating Student RNN:")
    eval_stats = trainer.evaluate(num_episodes=20)
    print(
        f"  Mean Reward: {eval_stats['mean_reward']:.2f} ± {eval_stats['std_reward']:.2f}"
    )
    print(f"  Mean Episode Length: {eval_stats['mean_length']:.1f}")
    print(f"  Success Rate: {eval_stats['success_rate']:.2%}")

    # Save model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(student_policy.state_dict(), save_path)
    print(f"\nSaved student RNN to {save_path}")

    return student_policy


def compare_policies(env, teacher_policy, student_policy, num_episodes: int = 10):
    """
    Compare teacher and student policies side by side.

    Args:
        env: Environment
        teacher_policy: Teacher MLP policy
        student_policy: Student RNN policy
        num_episodes: Number of episodes to compare
    """
    print("\n" + "=" * 60)
    print("Phase 3: Comparing Teacher and Student Policies")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    teacher_policy = teacher_policy.to(device)
    student_policy = student_policy.to(device)
    teacher_policy.eval()
    student_policy.eval()

    teacher_rewards = []
    student_rewards = []
    teacher_successes = 0
    student_successes = 0

    for episode in range(num_episodes):
        # Test teacher
        obs, _ = env.reset()
        done = False
        episode_reward = 0

        while not done:
            with torch.no_grad():
                action = teacher_policy.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward

            if info.get("success", False):
                teacher_successes += 1

        teacher_rewards.append(episode_reward)

        # Test student
        obs, _ = env.reset()
        hidden = None
        done = False
        episode_reward = 0

        while not done:
            with torch.no_grad():
                action, hidden = student_policy.predict(obs, hidden=hidden)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward

            if info.get("success", False):
                student_successes += 1

        student_rewards.append(episode_reward)

    print(f"\nTeacher MLP:")
    print(
        f"  Mean Reward: {np.mean(teacher_rewards):.2f} ± {np.std(teacher_rewards):.2f}"
    )
    print(f"  Success Rate: {teacher_successes / num_episodes:.2%}")

    print(f"\nStudent RNN:")
    print(
        f"  Mean Reward: {np.mean(student_rewards):.2f} ± {np.std(student_rewards):.2f}"
    )
    print(f"  Success Rate: {student_successes / num_episodes:.2%}")

    print(
        f"\nPerformance Ratio (Student/Teacher): {np.mean(student_rewards) / np.mean(teacher_rewards):.2%}"
    )


def main():
    """Main training pipeline."""
    print("\n" + "=" * 60)
    print("Musculoskeletal Arm Control with RNN Distillation")
    print("=" * 60)

    # Set random seeds
    np.random.seed(42)
    torch.manual_seed(42)

    # Create environment
    print("\nCreating environment...")
    env = ReachingEnv(
        render_mode="human",
        hold_time_range=(0.2, 0.8),
        reach_threshold=0.03,
        hold_threshold=0.04,
        max_episode_steps=1000,
    )

    print(f"Observation space: {env.observation_space.shape}")
    print(f"Action space: {env.action_space.shape}")

    # Phase 1: Train teacher MLP with RL
    teacher_policy = train_teacher_mlp(
        env=env, num_updates=500, save_path="checkpoints/teacher_mlp.pt"
    )

    # Phase 2: Train student RNN via distillation
    student_policy = train_student_rnn(
        env=env,
        teacher_policy=teacher_policy,
        num_demo_episodes=200,
        num_epochs=50,
        save_path="checkpoints/student_rnn.pt",
    )

    # Phase 3: Compare policies
    compare_policies(
        env=env,
        teacher_policy=teacher_policy,
        student_policy=student_policy,
        num_episodes=20,
    )

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)

    env.close()


if __name__ == "__main__":
    main()
