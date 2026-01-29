"""
Main training script for PPO + Distillation pipeline.

This script:
1. Trains an MLP policy using PPO
2. Distills the MLP into an RNN using behavioral cloning
3. Evaluates both policies
"""

import numpy as np
import torch
import argparse
import logging
from pathlib import Path
import matplotlib.pyplot as plt

# Import your existing modules
import sys
sys.path.insert(0, '/mnt/user-data/uploads')
from plants import SequentialReacher
from encoders import GridTargetEncoder
from networks import AlphaOnlyRNN, FullRNN
from utils import relu, tanh

# Import new modules
from ppo_trainer import PPOTrainer, RolloutBuffer
from step_env import StepBasedReachingEnv
from distillation import DistillationTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_env_config():
    """Create default environment configuration"""
    return {
        'plant': {
            'plant_xml_file': 'arm.xml',
        },
        'encoder': {
            'grid_size': 10,
            'sigma': 0.1,
        },
        'env': {
            'target_duration_distro': {
                'mean': 0.5,
                'min': 0.3,
                'max': 0.8,
            },
            'iti_distro': {
                'mean': 0.2,
                'min': 0.1,
                'max': 0.4,
            },
            'num_targets': 5,
            'randomize_gravity': False,
            'loss_weights': {
                'distance': 1.0,
                'energy': 0.1,
                'ridge': 0.0,
                'lasso': 0.0,
            },
            'profile': False,
        }
    }


def create_rnn_config(rnn_type='full', hidden_size=64):
    """Create RNN configuration"""
    # These will be filled in after creating the environment
    config = {
        'rnn_type': rnn_type,
        'hidden_size': hidden_size,
        'smoothing_factor': 0.1,
        'use_bias': True,
    }
    return config


def train_ppo(
    env,
    trainer,
    n_steps: int = 1000000,
    rollout_length: int = 2048,
    save_dir: Path = Path('./checkpoints'),
    save_freq: int = 50,
):
    """
    Train MLP policy with PPO.
    
    Args:
        env: Step-based environment
        trainer: PPOTrainer instance
        n_steps: Total training steps
        rollout_length: Steps per rollout
        save_dir: Directory to save checkpoints
        save_freq: Save checkpoint every N updates
    """
    logger.info("Starting PPO training...")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    rollout_buffer = RolloutBuffer()
    obs = env.reset(seed=0)
    episode_reward = 0
    episode_length = 0
    episode_count = 0
    
    training_stats = {
        'steps': [],
        'mean_reward': [],
        'policy_loss': [],
        'value_loss': [],
    }
    
    for step in range(n_steps):
        # Collect rollout
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(trainer.device)
            action, log_prob, value = trainer.policy.get_action(obs_tensor)
            action = action.cpu().numpy().squeeze()
            log_prob = log_prob.cpu().item()
            value = value.cpu().item()
        
        # Step environment
        next_obs, reward, done, info = env.step(action)
        
        # Store transition
        rollout_buffer.add(obs, action, reward, value, log_prob, done)
        
        episode_reward += reward
        episode_length += 1
        obs = next_obs
        
        # Handle episode end
        if done:
            trainer.episode_rewards.append(episode_reward)
            trainer.episode_lengths.append(episode_length)
            episode_count += 1
            
            obs = env.reset(seed=episode_count)
            episode_reward = 0
            episode_length = 0
        
        # Update policy
        if (step + 1) % rollout_length == 0:
            update_stats = trainer.update(rollout_buffer, obs)
            rollout_buffer.clear()
            
            # Log statistics
            mean_reward = np.mean(trainer.episode_rewards) if trainer.episode_rewards else 0
            
            logger.info(
                f"Step {step + 1}/{n_steps} | "
                f"Episodes: {episode_count} | "
                f"Mean Reward: {mean_reward:.2f} | "
                f"Policy Loss: {update_stats['policy_loss']:.4f} | "
                f"Value Loss: {update_stats['value_loss']:.4f}"
            )
            
            # Record stats
            training_stats['steps'].append(step + 1)
            training_stats['mean_reward'].append(mean_reward)
            training_stats['policy_loss'].append(update_stats['policy_loss'])
            training_stats['value_loss'].append(update_stats['value_loss'])
            
            # Save checkpoint
            if trainer.update_count % save_freq == 0:
                save_path = save_dir / f'ppo_checkpoint_{trainer.update_count}.pt'
                trainer.save(str(save_path))
    
    # Save final checkpoint
    final_path = save_dir / 'ppo_final.pt'
    trainer.save(str(final_path))
    logger.info(f"PPO training complete. Saved to {final_path}")
    
    return training_stats


def distill_to_rnn(
    teacher_policy,
    student_rnn_config,
    env,
    n_demo_episodes: int = 200,
    n_epochs: int = 100,
    save_dir: Path = Path('./checkpoints'),
):
    """
    Distill MLP teacher into RNN student.
    
    Args:
        teacher_policy: Trained MLPPolicy
        student_rnn_config: Configuration for student RNN
        env: Step-based environment
        n_demo_episodes: Number of demonstration episodes to collect
        n_epochs: Training epochs for distillation
        save_dir: Directory to save checkpoints
    """
    logger.info("Starting distillation...")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Create student RNN
    rnn_type = student_rnn_config['rnn_type']
    if rnn_type == 'alpha':
        student_rnn = AlphaOnlyRNN(
            target_size=env.encoder.size,
            length_size=env.plant.num_sensors_len,
            velocity_size=env.plant.num_sensors_vel,
            force_size=env.plant.num_sensors_frc,
            hidden_size=0,  # Not used for AlphaOnlyRNN
            output_size=env.action_dim,
            activation=None,  # Not used for AlphaOnlyRNN
            smoothing_factor=student_rnn_config['smoothing_factor'],
            use_bias=student_rnn_config['use_bias'],
        )
    else:  # 'full'
        # Choose activation
        activation = tanh  # or relu
        
        student_rnn = FullRNN(
            target_size=env.encoder.size,
            length_size=env.plant.num_sensors_len,
            velocity_size=env.plant.num_sensors_vel,
            force_size=env.plant.num_sensors_frc,
            hidden_size=student_rnn_config['hidden_size'],
            output_size=env.action_dim,
            activation=activation,
            smoothing_factor=student_rnn_config['smoothing_factor'],
            use_bias=student_rnn_config['use_bias'],
        )
    
    # Create distillation trainer
    distiller = DistillationTrainer(
        teacher_policy=teacher_policy,
        student_rnn=student_rnn,
        learning_rate=1e-3,
    )
    
    # Collect demonstrations
    dataset = distiller.collect_teacher_demonstrations(
        env=env,
        n_episodes=n_demo_episodes,
        deterministic=True,
    )
    
    # Train student
    history = distiller.train_on_demonstrations(
        dataset=dataset,
        n_epochs=n_epochs,
        batch_size=64,
    )
    
    # Save checkpoint
    distill_path = save_dir / 'distillation_final.pt'
    distiller.save(str(distill_path))
    
    # Export to numpy RNN
    trained_rnn = distiller.export_to_numpy_rnn()
    
    logger.info(f"Distillation complete. Saved to {distill_path}")
    
    return trained_rnn, history


def evaluate_policy(policy, env, n_episodes: int = 10, render: bool = False):
    """
    Evaluate a policy.
    
    Args:
        policy: Policy to evaluate (MLPPolicy or RNN)
        env: Step-based environment
        n_episodes: Number of episodes
        render: Whether to render
    
    Returns:
        Mean episode reward
    """
    is_mlp = hasattr(policy, 'get_action')
    device = next(policy.parameters()).device if is_mlp else None
    
    rewards = []
    
    for episode in range(n_episodes):
        obs = env.reset(seed=episode)
        if not is_mlp:
            policy.reset_state()
        
        episode_reward = 0
        done = False
        
        while not done:
            if is_mlp:
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
                    action, _, _ = policy.get_action(obs_tensor, deterministic=True)
                    action = action.cpu().numpy().squeeze()
            else:
                # RNN (numpy)
                tgt_obs = obs[:env.encoder.size]
                len_obs = obs[env.encoder.size:env.encoder.size + env.plant.num_sensors_len]
                vel_obs = obs[env.encoder.size + env.plant.num_sensors_len:
                             env.encoder.size + env.plant.num_sensors_len + env.plant.num_sensors_vel]
                frc_obs = obs[env.encoder.size + env.plant.num_sensors_len + env.plant.num_sensors_vel:]
                
                action = policy.step(tgt_obs, len_obs, vel_obs, frc_obs)
            
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            
            if render:
                env.render(render_speed=1.0)
        
        rewards.append(episode_reward)
    
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    
    logger.info(f"Evaluation: {mean_reward:.2f} ± {std_reward:.2f} (over {n_episodes} episodes)")
    
    return mean_reward


def plot_training_results(ppo_stats, distill_history, save_path: Path):
    """Plot training results"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # PPO reward
    axes[0, 0].plot(ppo_stats['steps'], ppo_stats['mean_reward'])
    axes[0, 0].set_xlabel('Steps')
    axes[0, 0].set_ylabel('Mean Episode Reward')
    axes[0, 0].set_title('PPO Training - Reward')
    axes[0, 0].grid(True)
    
    # PPO losses
    axes[0, 1].plot(ppo_stats['steps'], ppo_stats['policy_loss'], label='Policy Loss')
    axes[0, 1].plot(ppo_stats['steps'], ppo_stats['value_loss'], label='Value Loss')
    axes[0, 1].set_xlabel('Steps')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('PPO Training - Losses')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Distillation training loss
    axes[1, 0].plot(distill_history['train_loss'], label='Train')
    axes[1, 0].plot(distill_history['val_loss'], label='Validation')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('MSE Loss')
    axes[1, 0].set_title('Distillation - Behavioral Cloning Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Remove empty subplot
    fig.delaxes(axes[1, 1])
    
    plt.tight_layout()
    plt.savefig(save_path)
    logger.info(f"Saved training plots to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train MLP with PPO and distill to RNN')
    parser.add_argument('--ppo-steps', type=int, default=500000, help='PPO training steps')
    parser.add_argument('--rollout-length', type=int, default=2048, help='PPO rollout length')
    parser.add_argument('--demo-episodes', type=int, default=200, help='Demonstration episodes')
    parser.add_argument('--distill-epochs', type=int, default=100, help='Distillation epochs')
    parser.add_argument('--rnn-type', type=str, default='full', choices=['alpha', 'full'])
    parser.add_argument('--rnn-hidden-size', type=int, default=64, help='RNN hidden size')
    parser.add_argument('--eval-episodes', type=int, default=10, help='Evaluation episodes')
    parser.add_argument('--save-dir', type=str, default='./checkpoints', help='Save directory')
    parser.add_argument('--skip-ppo', action='store_true', help='Skip PPO, load checkpoint')
    parser.add_argument('--ppo-checkpoint', type=str, help='PPO checkpoint to load')
    parser.add_argument('--render-eval', action='store_true', help='Render during evaluation')
    
    args = parser.parse_args()
    
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Create environment
    logger.info("Creating environment...")
    env_config = create_env_config()
    
    plant = SequentialReacher(**env_config['plant'])
    encoder = GridTargetEncoder(**env_config['encoder'])
    
    # Update encoder bounds based on plant workspace
    x_bounds, y_bounds = plant.get_workspace_bounds()
    encoder.x_bounds = x_bounds
    encoder.y_bounds = y_bounds
    encoder._initialize_grid()
    
    env = StepBasedReachingEnv(
        plant=plant,
        target_encoder=encoder,
        env_config=env_config,
        max_steps_per_episode=1000,
    )
    
    logger.info(f"Environment: obs_dim={env.obs_dim}, action_dim={env.action_dim}")
    
    # Stage 1: Train MLP with PPO
    if not args.skip_ppo:
        policy_config = {
            'obs_dim': env.obs_dim,
            'action_dim': env.action_dim,
            'hidden_sizes': [256, 256],
            'activation': 'tanh',
        }
        
        ppo_trainer = PPOTrainer(
            env_config=env_config,
            policy_config=policy_config,
            learning_rate=3e-4,
        )
        
        ppo_stats = train_ppo(
            env=env,
            trainer=ppo_trainer,
            n_steps=args.ppo_steps,
            rollout_length=args.rollout_length,
            save_dir=save_dir,
        )
        
        teacher_policy = ppo_trainer.policy
    else:
        # Load checkpoint
        logger.info(f"Loading PPO checkpoint from {args.ppo_checkpoint}")
        checkpoint = torch.load(args.ppo_checkpoint)
        
        from ppo_trainer import MLPPolicy
        policy_config = checkpoint['policy_config']
        teacher_policy = MLPPolicy(**policy_config)
        teacher_policy.load_state_dict(checkpoint['policy_state_dict'])
        
        ppo_stats = None
    
    # Evaluate teacher
    logger.info("Evaluating teacher MLP policy...")
    teacher_reward = evaluate_policy(
        teacher_policy,
        env,
        n_episodes=args.eval_episodes,
        render=args.render_eval,
    )
    
    # Stage 2: Distill to RNN
    rnn_config = create_rnn_config(
        rnn_type=args.rnn_type,
        hidden_size=args.rnn_hidden_size,
    )
    
    trained_rnn, distill_history = distill_to_rnn(
        teacher_policy=teacher_policy,
        student_rnn_config=rnn_config,
        env=env,
        n_demo_episodes=args.demo_episodes,
        n_epochs=args.distill_epochs,
        save_dir=save_dir,
    )
    
    # Evaluate student
    logger.info("Evaluating student RNN policy...")
    student_reward = evaluate_policy(
        trained_rnn,
        env,
        n_episodes=args.eval_episodes,
        render=args.render_eval,
    )
    
    # Summary
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Teacher MLP:  {teacher_reward:.2f}")
    logger.info(f"Student RNN:  {student_reward:.2f}")
    logger.info(f"Performance:  {100 * student_reward / teacher_reward:.1f}% of teacher")
    logger.info("=" * 60)
    
    # Plot results
    if ppo_stats is not None:
        plot_training_results(
            ppo_stats,
            distill_history,
            save_dir / 'training_results.png'
        )
    
    # Cleanup
    env.close()


if __name__ == '__main__':
    main()
