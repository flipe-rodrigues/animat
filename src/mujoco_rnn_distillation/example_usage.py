"""
Example: Train MLP with PPO and distill to RNN

This script demonstrates the complete pipeline with minimal configuration.
"""

import sys
sys.path.insert(0, '/mnt/user-data/uploads')

import numpy as np
import torch
from pathlib import Path

# Import your modules
from plants import SequentialReacher
from encoders import GridTargetEncoder
from networks import FullRNN
from utils import tanh

# Import training modules
from ppo_trainer import PPOTrainer
from step_env import StepBasedReachingEnv
from distillation import DistillationTrainer


def quick_example():
    """
    Quick example: Train for a short time to verify everything works.
    """
    print("=" * 60)
    print("PPO + Distillation Quick Example")
    print("=" * 60)
    
    # 1. Setup environment
    print("\n[1/6] Setting up environment...")
    
    env_config = {
        'plant': {'plant_xml_file': 'arm.xml'},
        'encoder': {'grid_size': 10, 'sigma': 0.1},
        'env': {
            'target_duration_distro': {'mean': 0.5, 'min': 0.3, 'max': 0.8},
            'iti_distro': {'mean': 0.2, 'min': 0.1, 'max': 0.4},
            'num_targets': 3,  # Fewer targets for quick training
            'randomize_gravity': False,
            'loss_weights': {
                'distance': 1.0,
                'energy': 0.1,
                'ridge': 0.0,
                'lasso': 0.0,
            },
        }
    }
    
    plant = SequentialReacher(**env_config['plant'])
    encoder = GridTargetEncoder(**env_config['encoder'])
    
    # Update encoder bounds
    x_bounds, y_bounds = plant.get_workspace_bounds()
    encoder.x_bounds = x_bounds
    encoder.y_bounds = y_bounds
    encoder._initialize_grid()
    
    env = StepBasedReachingEnv(
        plant=plant,
        target_encoder=encoder,
        env_config=env_config,
        max_steps_per_episode=500,
    )
    
    print(f"  Observation dim: {env.obs_dim}")
    print(f"  Action dim: {env.action_dim}")
    
    # 2. Create PPO trainer
    print("\n[2/6] Creating PPO trainer...")
    
    policy_config = {
        'obs_dim': env.obs_dim,
        'action_dim': env.action_dim,
        'hidden_sizes': [128, 128],  # Smaller network for quick training
        'activation': 'tanh',
    }
    
    ppo_trainer = PPOTrainer(
        env_config=env_config,
        policy_config=policy_config,
        learning_rate=3e-4,
    )
    
    print(f"  MLP parameters: {sum(p.numel() for p in ppo_trainer.policy.parameters())}")
    
    # 3. Train with PPO (short training for demo)
    print("\n[3/6] Training MLP with PPO (10 updates)...")
    
    from ppo_trainer import RolloutBuffer
    
    rollout_buffer = RolloutBuffer()
    obs = env.reset(seed=0)
    
    n_updates = 10
    rollout_length = 512
    
    for update in range(n_updates):
        # Collect rollout
        for _ in range(rollout_length):
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(ppo_trainer.device)
                action, log_prob, value = ppo_trainer.policy.get_action(obs_tensor)
                action = action.cpu().numpy().squeeze()
                log_prob = log_prob.cpu().item()
                value = value.cpu().item()
            
            next_obs, reward, done, info = env.step(action)
            rollout_buffer.add(obs, action, reward, value, log_prob, done)
            
            obs = next_obs
            if done:
                obs = env.reset(seed=update)
        
        # Update
        stats = ppo_trainer.update(rollout_buffer, obs)
        rollout_buffer.clear()
        
        mean_reward = np.mean(ppo_trainer.episode_rewards) if ppo_trainer.episode_rewards else 0
        print(f"  Update {update + 1}/{n_updates} - Reward: {mean_reward:.2f}")
    
    # 4. Evaluate teacher
    print("\n[4/6] Evaluating teacher MLP...")
    
    teacher_rewards = []
    for episode in range(5):
        obs = env.reset(seed=episode)
        episode_reward = 0
        done = False
        
        while not done:
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(ppo_trainer.device)
                action, _, _ = ppo_trainer.policy.get_action(obs_tensor, deterministic=True)
                action = action.cpu().numpy().squeeze()
            
            obs, reward, done, info = env.step(action)
            episode_reward += reward
        
        teacher_rewards.append(episode_reward)
    
    teacher_mean = np.mean(teacher_rewards)
    print(f"  Teacher reward: {teacher_mean:.2f} ± {np.std(teacher_rewards):.2f}")
    
    # 5. Distill to RNN
    print("\n[5/6] Distilling to RNN...")
    
    # Create student RNN
    student_rnn = FullRNN(
        target_size=env.encoder.size,
        length_size=env.plant.num_sensors_len,
        velocity_size=env.plant.num_sensors_vel,
        force_size=env.plant.num_sensors_frc,
        hidden_size=32,  # Small RNN for quick training
        output_size=env.action_dim,
        activation=tanh,
        smoothing_factor=0.1,
        use_bias=True,
    )
    
    print(f"  RNN parameters: {student_rnn.num_params}")
    
    # Create distillation trainer
    distiller = DistillationTrainer(
        teacher_policy=ppo_trainer.policy,
        student_rnn=student_rnn,
        learning_rate=1e-3,
    )
    
    # Collect demonstrations
    print("  Collecting 20 demonstration episodes...")
    dataset = distiller.collect_teacher_demonstrations(
        env=env,
        n_episodes=20,
        deterministic=True,
    )
    
    # Train
    print("  Training for 20 epochs...")
    history = distiller.train_on_demonstrations(
        dataset=dataset,
        n_epochs=20,
        batch_size=64,
    )
    
    print(f"  Final loss: {history['val_loss'][-1]:.6f}")
    
    # Export to numpy
    trained_rnn = distiller.export_to_numpy_rnn()
    
    # 6. Evaluate student
    print("\n[6/6] Evaluating student RNN...")
    
    student_rewards = []
    for episode in range(5):
        obs = env.reset(seed=episode)
        trained_rnn.reset_state()
        episode_reward = 0
        done = False
        
        while not done:
            # Split observation
            tgt_obs = obs[:env.encoder.size]
            len_obs = obs[env.encoder.size:env.encoder.size + env.plant.num_sensors_len]
            vel_obs = obs[env.encoder.size + env.plant.num_sensors_len:
                         env.encoder.size + env.plant.num_sensors_len + env.plant.num_sensors_vel]
            frc_obs = obs[env.encoder.size + env.plant.num_sensors_len + env.plant.num_sensors_vel:]
            
            action = trained_rnn.step(tgt_obs, len_obs, vel_obs, frc_obs)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
        
        student_rewards.append(episode_reward)
    
    student_mean = np.mean(student_rewards)
    print(f"  Student reward: {student_mean:.2f} ± {np.std(student_rewards):.2f}")
    
    # Summary
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Teacher MLP:  {teacher_mean:.2f}")
    print(f"Student RNN:  {student_mean:.2f}")
    print(f"Performance:  {100 * student_mean / teacher_mean:.1f}% of teacher")
    print("=" * 60)
    
    env.close()
    
    return {
        'teacher_reward': teacher_mean,
        'student_reward': student_mean,
        'trained_rnn': trained_rnn,
    }


if __name__ == '__main__':
    # Run the example
    results = quick_example()
    
    print("\nExample complete!")
    print("\nFor full training, run:")
    print("  python train_ppo_distill.py --ppo-steps 500000")
