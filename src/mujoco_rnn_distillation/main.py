"""
Main training script for PPO + Distillation pipeline.

This script runs the complete training process with periodic visualization,
rendering, and checkpointing - similar to the CMA-ES training approach.
"""

import sys
sys.path.insert(0, '/mnt/user-data/uploads')

import numpy as np
import torch
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime

# Import your existing modules
from plants import SequentialReacher
from encoders import GridTargetEncoder
from networks import FullRNN, AlphaOnlyRNN
from environments import SequentialReachingEnv
from utils import tanh, relu, alpha_from_tau

# Import training modules
from ppo_trainer import PPOTrainer, RolloutBuffer, MLPPolicy
from step_env import StepBasedReachingEnv
from distillation import DistillationTrainer


class TrainingManager:
    """Manages the complete training pipeline with visualization and checkpointing"""
    
    def __init__(
        self,
        reacher,
        target_encoder,
        env_config,
        rnn_config,
        save_dir='./models',
    ):
        self.reacher = reacher
        self.target_encoder = target_encoder
        self.env_config = env_config
        self.rnn_config = rnn_config
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamp for this run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.save_dir / f"run_{self.timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Create environments
        self.episodic_env = SequentialReachingEnv(**env_config['env'])
        self.step_env = StepBasedReachingEnv(
            plant=reacher,
            target_encoder=target_encoder,
            env_config=env_config,
            max_steps_per_episode=2000,
        )
        
        # Training history
        self.history = {
            'ppo': {'updates': [], 'rewards': [], 'policy_loss': [], 'value_loss': []},
            'distill': {'epochs': [], 'train_loss': [], 'val_loss': []},
            'eval': {'checkpoints': [], 'teacher_rewards': [], 'student_rewards': []},
        }
        
        print("=" * 70)
        print(f"Training Manager Initialized")
        print("=" * 70)
        print(f"Run directory: {self.run_dir}")
        print(f"Observation dim: {self.step_env.obs_dim}")
        print(f"Action dim: {self.step_env.action_dim}")
        print("=" * 70)
    
    def train_ppo(
        self,
        n_updates=1000,
        rollout_length=2048,
        eval_freq=10,
        render_freq=50,
        save_freq=100,
    ):
        """
        Train MLP policy with PPO.
        
        Args:
            n_updates: Number of PPO updates
            rollout_length: Steps per rollout
            eval_freq: Evaluate every N updates
            render_freq: Render every N updates
            save_freq: Save checkpoint every N updates
        """
        print("\n" + "=" * 70)
        print("STAGE 1: PPO TRAINING")
        print("=" * 70)
        
        # Create policy
        policy_config = {
            'obs_dim': self.step_env.obs_dim,
            'action_dim': self.step_env.action_dim,
            'hidden_sizes': [256, 256],
            'activation': 'tanh',
        }
        
        self.ppo_trainer = PPOTrainer(
            env_config=self.env_config,
            policy_config=policy_config,
            learning_rate=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_epsilon=0.2,
            entropy_coef=0.01,
        )
        
        print(f"MLP parameters: {sum(p.numel() for p in self.ppo_trainer.policy.parameters())}")
        print(f"Training for {n_updates} updates...\n")
        
        # Training loop
        rollout_buffer = RolloutBuffer()
        obs = self.step_env.reset(seed=0)
        episode_reward = 0
        episode_count = 0
        
        for update in range(n_updates):
            # Collect rollout
            for _ in range(rollout_length):
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.ppo_trainer.device)
                    action, log_prob, value = self.ppo_trainer.policy.get_action(obs_tensor)
                    action = action.cpu().numpy().squeeze()
                    log_prob = log_prob.cpu().item()
                    value = value.cpu().item()
                
                next_obs, reward, done, info = self.step_env.step(action)
                rollout_buffer.add(obs, action, reward, value, log_prob, done)
                
                episode_reward += reward
                obs = next_obs
                
                if done:
                    self.ppo_trainer.episode_rewards.append(episode_reward)
                    obs = self.step_env.reset(seed=episode_count)
                    episode_reward = 0
                    episode_count += 1
            
            # Update policy
            stats = self.ppo_trainer.update(rollout_buffer, obs)
            rollout_buffer.clear()
            
            # Record history
            mean_reward = np.mean(self.ppo_trainer.episode_rewards) if self.ppo_trainer.episode_rewards else 0
            self.history['ppo']['updates'].append(update)
            self.history['ppo']['rewards'].append(mean_reward)
            self.history['ppo']['policy_loss'].append(stats['policy_loss'])
            self.history['ppo']['value_loss'].append(stats['value_loss'])
            
            # Logging
            print(f"Update {update + 1}/{n_updates} | "
                  f"Episodes: {episode_count} | "
                  f"Reward: {mean_reward:.2f} | "
                  f"Policy Loss: {stats['policy_loss']:.4f} | "
                  f"Value Loss: {stats['value_loss']:.4f}")
            
            # Evaluation
            if (update + 1) % eval_freq == 0:
                print(f"\n>>> Evaluating at update {update + 1}...")
                eval_reward = self._evaluate_mlp(n_episodes=5)
                self.history['eval']['checkpoints'].append(update)
                self.history['eval']['teacher_rewards'].append(eval_reward)
                print(f">>> Teacher MLP Reward: {eval_reward:.2f}\n")
            
            # Render
            if (update + 1) % render_freq == 0:
                print(f"\n>>> Rendering at update {update + 1}...")
                self._render_and_plot_mlp()
            
            # Save checkpoint
            if (update + 1) % save_freq == 0:
                checkpoint_path = self.run_dir / f"ppo_update_{update + 1}.pt"
                self.ppo_trainer.save(str(checkpoint_path))
                print(f">>> Saved checkpoint: {checkpoint_path}\n")
        
        # Save final model
        final_path = self.run_dir / "ppo_final.pt"
        self.ppo_trainer.save(str(final_path))
        print(f"\n>>> Saved final PPO model: {final_path}")
        
        # Final evaluation
        print("\n>>> Final PPO evaluation...")
        final_reward = self._evaluate_mlp(n_episodes=20)
        print(f">>> Final Teacher Reward: {final_reward:.2f}")
        
        # Final render and plot
        print("\n>>> Final render and plot...")
        self._render_and_plot_mlp()
        
        return final_reward
    
    def distill_to_rnn(
        self,
        n_demo_episodes=200,
        n_epochs=100,
        batch_size=64,
        eval_freq=10,
    ):
        """
        Distill MLP teacher into RNN student.
        
        Args:
            n_demo_episodes: Number of demonstration episodes
            n_epochs: Training epochs
            batch_size: Batch size
            eval_freq: Evaluate every N epochs
        """
        print("\n" + "=" * 70)
        print("STAGE 2: DISTILLATION TO RNN")
        print("=" * 70)
        
        # Create student RNN
        rnn_type = self.rnn_config['rnn_type']
        
        if rnn_type == 'alpha':
            self.student_rnn = AlphaOnlyRNN(
                target_size=self.target_encoder.size,
                length_size=self.reacher.num_sensors_len,
                velocity_size=self.reacher.num_sensors_vel,
                force_size=self.reacher.num_sensors_frc,
                hidden_size=0,
                output_size=self.reacher.num_actuators,
                activation=None,
                smoothing_factor=self.rnn_config['smoothing_factor'],
                use_bias=self.rnn_config['use_bias'],
            )
        else:  # 'full'
            activation = self.rnn_config.get('activation', tanh)
            
            self.student_rnn = FullRNN(
                target_size=self.target_encoder.size,
                length_size=self.reacher.num_sensors_len,
                velocity_size=self.reacher.num_sensors_vel,
                force_size=self.reacher.num_sensors_frc,
                hidden_size=self.rnn_config['hidden_size'],
                output_size=self.reacher.num_actuators,
                activation=activation,
                smoothing_factor=self.rnn_config['smoothing_factor'],
                use_bias=self.rnn_config['use_bias'],
            )
        
        print(f"RNN type: {rnn_type}")
        print(f"RNN parameters: {self.student_rnn.num_params}")
        
        # Create distillation trainer
        self.distiller = DistillationTrainer(
            teacher_policy=self.ppo_trainer.policy,
            student_rnn=self.student_rnn,
            learning_rate=1e-3,
            weight_decay=1e-4,
        )
        
        # Collect demonstrations
        print(f"\nCollecting {n_demo_episodes} demonstrations from teacher...")
        dataset = self.distiller.collect_teacher_demonstrations(
            env=self.step_env,
            n_episodes=n_demo_episodes,
            deterministic=True,
        )
        print(f"Dataset size: {len(dataset['observations'])} transitions")
        
        # Training loop with periodic evaluation
        print(f"\nTraining student RNN for {n_epochs} epochs...\n")
        
        # Prepare data
        n_samples = len(dataset['observations'])
        n_val = int(n_samples * 0.1)
        indices = np.random.permutation(n_samples)
        
        train_obs = dataset['observations'][indices[n_val:]]
        train_actions = dataset['actions'][indices[n_val:]]
        val_obs = dataset['observations'][indices[:n_val]]
        val_actions = dataset['actions'][indices[:n_val]]
        
        train_obs_tensor = torch.FloatTensor(train_obs).to(self.distiller.device)
        train_actions_tensor = torch.FloatTensor(train_actions).to(self.distiller.device)
        val_obs_tensor = torch.FloatTensor(val_obs).to(self.distiller.device)
        val_actions_tensor = torch.FloatTensor(val_actions).to(self.distiller.device)
        
        best_val_loss = float('inf')
        
        for epoch in range(n_epochs):
            # Training
            self.distiller.student.train()
            train_losses = []
            
            batch_indices = np.random.permutation(len(train_obs_tensor))
            for start in range(0, len(train_obs_tensor), batch_size):
                end = start + batch_size
                batch_idx = batch_indices[start:end]
                
                batch_obs = train_obs_tensor[batch_idx]
                batch_actions = train_actions_tensor[batch_idx]
                
                self.distiller.optimizer.zero_grad()
                
                # Forward pass
                pred_actions = []
                self.distiller.student.reset_state()
                for obs in batch_obs:
                    pred_action = self.distiller.student(obs, reset_state=False)
                    pred_actions.append(pred_action)
                pred_actions = torch.stack(pred_actions)
                
                # Loss
                loss = torch.nn.functional.mse_loss(pred_actions, batch_actions)
                loss.backward()
                self.distiller.optimizer.step()
                
                train_losses.append(loss.item())
            
            # Validation
            self.distiller.student.eval()
            with torch.no_grad():
                self.distiller.student.reset_state()
                val_pred = []
                for obs in val_obs_tensor:
                    pred = self.distiller.student(obs, reset_state=False)
                    val_pred.append(pred)
                val_pred = torch.stack(val_pred)
                val_loss = torch.nn.functional.mse_loss(val_pred, val_actions_tensor).item()
            
            train_loss = np.mean(train_losses)
            
            # Record history
            self.history['distill']['epochs'].append(epoch)
            self.history['distill']['train_loss'].append(train_loss)
            self.history['distill']['val_loss'].append(val_loss)
            
            # Logging
            print(f"Epoch {epoch + 1}/{n_epochs} | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f}")
            
            # Track best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.distiller.best_params = self.distiller.student.params.data.clone()
            
            # Periodic evaluation
            if (epoch + 1) % eval_freq == 0:
                print(f"\n>>> Evaluating student at epoch {epoch + 1}...")
                current_rnn = self.distiller.export_to_numpy_rnn()
                eval_reward = self._evaluate_rnn(current_rnn, n_episodes=5)
                self.history['eval']['student_rewards'].append(eval_reward)
                print(f">>> Student RNN Reward: {eval_reward:.2f}\n")
        
        # Load best parameters
        if hasattr(self.distiller, 'best_params'):
            self.distiller.student.params.data = self.distiller.best_params
        
        # Save final model
        distill_path = self.run_dir / "distillation_final.pt"
        self.distiller.save(str(distill_path))
        print(f"\n>>> Saved distillation checkpoint: {distill_path}")
        
        # Export to numpy RNN
        self.trained_rnn = self.distiller.export_to_numpy_rnn()
        
        # Save numpy RNN
        rnn_path = self.run_dir / "trained_rnn.pkl"
        with open(rnn_path, 'wb') as f:
            pickle.dump(self.trained_rnn, f)
        print(f">>> Saved numpy RNN: {rnn_path}")
        
        # Final evaluation
        print("\n>>> Final student evaluation...")
        final_reward = self._evaluate_rnn(self.trained_rnn, n_episodes=20)
        print(f">>> Final Student Reward: {final_reward:.2f}")
        
        # Final render and plot
        print("\n>>> Final student render and plot...")
        self._render_and_plot_rnn(self.trained_rnn)
        
        return final_reward
    
    def _evaluate_mlp(self, n_episodes=10):
        """Evaluate MLP policy"""
        rewards = []
        for episode in range(n_episodes):
            obs = self.step_env.reset(seed=episode)
            episode_reward = 0
            done = False
            
            while not done:
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.ppo_trainer.device)
                    action, _, _ = self.ppo_trainer.policy.get_action(obs_tensor, deterministic=True)
                    action = action.cpu().numpy().squeeze()
                
                obs, reward, done, info = self.step_env.step(action)
                episode_reward += reward
            
            rewards.append(episode_reward)
        
        return np.mean(rewards)
    
    def _evaluate_rnn(self, rnn, n_episodes=10):
        """Evaluate RNN policy"""
        rewards = []
        for episode in range(n_episodes):
            obs = self.step_env.reset(seed=episode)
            rnn.reset_state()
            episode_reward = 0
            done = False
            
            while not done:
                # Split observation
                tgt_obs = obs[:self.target_encoder.size]
                len_obs = obs[self.target_encoder.size:
                             self.target_encoder.size + self.reacher.num_sensors_len]
                vel_obs = obs[self.target_encoder.size + self.reacher.num_sensors_len:
                             self.target_encoder.size + self.reacher.num_sensors_len + 
                             self.reacher.num_sensors_vel]
                frc_obs = obs[self.target_encoder.size + self.reacher.num_sensors_len + 
                             self.reacher.num_sensors_vel:]
                
                action = rnn.step(tgt_obs, len_obs, vel_obs, frc_obs)
                obs, reward, done, info = self.step_env.step(action)
                episode_reward += reward
            
            rewards.append(episode_reward)
        
        return np.mean(rewards)
    
    def _render_and_plot_mlp(self):
        """Render and plot MLP policy using episodic environment"""
        # Create a temporary MLP wrapper that works with episodic env
        class MLPWrapper:
            def __init__(self, mlp_policy, device, obs_dim):
                self.policy = mlp_policy
                self.device = device
                self.obs_dim = obs_dim
            
            def reset_state(self):
                pass  # MLP is stateless
            
            def step(self, tgt_obs, len_obs, vel_obs, frc_obs):
                obs = np.concatenate([tgt_obs, len_obs, vel_obs, frc_obs])
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                    action, _, _ = self.policy.get_action(obs_tensor, deterministic=True)
                    return action.cpu().numpy().squeeze()
        
        wrapper = MLPWrapper(self.ppo_trainer.policy, self.ppo_trainer.device, self.step_env.obs_dim)
        
        # Evaluate with logging
        self.episodic_env.evaluate(
            wrapper,
            self.reacher,
            self.target_encoder,
            seed=0,
            render=True,
            render_speed=1.0,
            log=True,
        )
        
        # Plot
        self.episodic_env.plot()
    
    def _render_and_plot_rnn(self, rnn):
        """Render and plot RNN policy using episodic environment"""
        self.episodic_env.evaluate(
            rnn,
            self.reacher,
            self.target_encoder,
            seed=0,
            render=True,
            render_speed=1.0,
            log=True,
        )
        self.episodic_env.plot()
    
    def plot_training_history(self):
        """Plot complete training history"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # PPO Reward
        if self.history['ppo']['rewards']:
            axes[0, 0].plot(self.history['ppo']['updates'], self.history['ppo']['rewards'])
            axes[0, 0].set_xlabel('Update')
            axes[0, 0].set_ylabel('Mean Episode Reward')
            axes[0, 0].set_title('PPO Training - Reward')
            axes[0, 0].grid(True)
        
        # PPO Losses
        if self.history['ppo']['policy_loss']:
            axes[0, 1].plot(self.history['ppo']['updates'], self.history['ppo']['policy_loss'], label='Policy')
            axes[0, 1].plot(self.history['ppo']['updates'], self.history['ppo']['value_loss'], label='Value')
            axes[0, 1].set_xlabel('Update')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].set_title('PPO Training - Losses')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        # Distillation Loss
        if self.history['distill']['train_loss']:
            axes[0, 2].plot(self.history['distill']['epochs'], self.history['distill']['train_loss'], label='Train')
            axes[0, 2].plot(self.history['distill']['epochs'], self.history['distill']['val_loss'], label='Val')
            axes[0, 2].set_xlabel('Epoch')
            axes[0, 2].set_ylabel('MSE Loss')
            axes[0, 2].set_title('Distillation - Loss')
            axes[0, 2].legend()
            axes[0, 2].grid(True)
        
        # Comparative Evaluation
        if self.history['eval']['teacher_rewards']:
            axes[1, 0].plot(self.history['eval']['checkpoints'], 
                          self.history['eval']['teacher_rewards'], 
                          'o-', label='Teacher MLP')
            if self.history['eval']['student_rewards']:
                # Match student evaluations to checkpoints
                student_checkpoints = self.history['eval']['checkpoints'][:len(self.history['eval']['student_rewards'])]
                axes[1, 0].plot(student_checkpoints,
                              self.history['eval']['student_rewards'],
                              's-', label='Student RNN')
            axes[1, 0].set_xlabel('Training Checkpoint')
            axes[1, 0].set_ylabel('Mean Reward')
            axes[1, 0].set_title('Policy Comparison')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Performance Ratio
        if self.history['eval']['teacher_rewards'] and self.history['eval']['student_rewards']:
            min_len = min(len(self.history['eval']['teacher_rewards']), 
                         len(self.history['eval']['student_rewards']))
            if min_len > 0:
                teacher = np.array(self.history['eval']['teacher_rewards'][:min_len])
                student = np.array(self.history['eval']['student_rewards'][:min_len])
                ratio = 100 * student / (teacher + 1e-8)
                axes[1, 1].plot(self.history['eval']['checkpoints'][:min_len], ratio, 'o-')
                axes[1, 1].axhline(y=100, color='r', linestyle='--', label='100%')
                axes[1, 1].set_xlabel('Training Checkpoint')
                axes[1, 1].set_ylabel('Performance (%)')
                axes[1, 1].set_title('Student/Teacher Performance Ratio')
                axes[1, 1].legend()
                axes[1, 1].grid(True)
        
        # Remove empty subplot
        fig.delaxes(axes[1, 2])
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.run_dir / 'training_history.png'
        plt.savefig(plot_path, dpi=150)
        print(f"Saved training history plot: {plot_path}")
        plt.show()
    
    def save_history(self):
        """Save training history"""
        history_path = self.run_dir / 'training_history.pkl'
        with open(history_path, 'wb') as f:
            pickle.dump(self.history, f)
        print(f"Saved training history: {history_path}")


if __name__ == "__main__":
    
    # Initialize the plant
    reacher = SequentialReacher(plant_xml_file="arm.xml")
    
    # Initialize the target encoder
    target_encoder = GridTargetEncoder(
        grid_size=8,
        x_bounds=reacher.get_workspace_bounds()[0],
        y_bounds=reacher.get_workspace_bounds()[1],
        sigma=0.25,
    )
    
    # Environment configuration
    env_config = {
        'plant': {'plant_xml_file': 'arm.xml'},
        'encoder': {
            'grid_size': 8,
            'sigma': 0.25,
        },
        'env': {
            'target_duration_distro': {'mean': 3, 'min': 1, 'max': 6},
            'iti_distro': {'mean': 1, 'min': 0, 'max': 3},
            'num_targets': 10,
            'randomize_gravity': True,
            'loss_weights': {
                'distance': 1,
                'energy': 0.1,
                'ridge': 0,
                'lasso': 0,
            },
        }
    }
    
    # RNN configuration
    rnn_config = {
        'rnn_type': 'full',  # 'full' or 'alpha'
        'hidden_size': 25,
        'smoothing_factor': alpha_from_tau(tau=10e-3, dt=reacher.model.opt.timestep),
        'use_bias': True,
        'activation': tanh,
    }
    
    # Create training manager
    manager = TrainingManager(
        reacher=reacher,
        target_encoder=target_encoder,
        env_config=env_config,
        rnn_config=rnn_config,
        save_dir='./models',
    )
    
    # Stage 1: Train with PPO
    print("\nStarting PPO training...")
    teacher_reward = manager.train_ppo(
        n_updates=1000,        # Total PPO updates
        rollout_length=2048,   # Steps per rollout
        eval_freq=10,          # Evaluate every 10 updates
        render_freq=50,        # Render every 50 updates
        save_freq=100,         # Save every 100 updates
    )
    
    # Stage 2: Distill to RNN
    print("\nStarting distillation...")
    student_reward = manager.distill_to_rnn(
        n_demo_episodes=200,   # Demonstration episodes
        n_epochs=100,          # Training epochs
        batch_size=64,         # Batch size
        eval_freq=10,          # Evaluate every 10 epochs
    )
    
    # Final summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Teacher MLP Reward:  {teacher_reward:.2f}")
    print(f"Student RNN Reward:  {student_reward:.2f}")
    print(f"Performance Ratio:   {100 * student_reward / teacher_reward:.1f}%")
    print(f"Run directory:       {manager.run_dir}")
    print("=" * 70)
    
    # Plot training history
    print("\nPlotting training history...")
    manager.plot_training_history()
    
    # Save history
    manager.save_history()
    
    print("\nAll done! Check the output directory for saved models and plots.")
