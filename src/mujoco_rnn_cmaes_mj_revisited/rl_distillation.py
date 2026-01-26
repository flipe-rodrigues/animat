"""
Alternative training approach: RL (PPO) for MLP teacher + Distillation to RNN student

This module provides an alternative to CMA-ES optimization by:
1. Training an MLP policy using PPO (Proximal Policy Optimization)
2. Using behavioral cloning / distillation to transfer knowledge to the RNN
3. Fine-tuning the RNN with RL if needed

Advantages over CMA-ES:
- Faster initial learning with gradient-based methods
- Can leverage large-scale parallelization
- Teacher-student approach allows for architecture flexibility
- Can incorporate online learning and adaptation
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.optim import Adam
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import deque
import pickle


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class RLConfig:
    """Configuration for RL training"""
    # PPO hyperparameters
    lr_actor: float = 3e-4
    lr_critic: float = 1e-3
    gamma: float = 0.99
    gae_lambda: float = 0.95
    epsilon_clip: float = 0.2
    entropy_coef: float = 0.01
    value_loss_coef: float = 0.5
    max_grad_norm: float = 0.5
    
    # Training
    num_epochs: int = 10
    batch_size: int = 64
    buffer_size: int = 2048
    num_iterations: int = 1000
    
    # Logging
    log_interval: int = 10
    save_interval: int = 100


@dataclass
class DistillationConfig:
    """Configuration for distillation learning"""
    lr: float = 1e-3
    num_epochs: int = 100
    batch_size: int = 128
    temperature: float = 1.0
    alpha: float = 0.5  # Weight between hard targets (actions) and soft targets (logits)
    
    # Data collection
    num_trajectories: int = 1000
    max_trajectory_length: int = 1000
    
    # Logging
    log_interval: int = 10
    save_interval: int = 50


# ============================================================================
# MLP TEACHER NETWORK (for RL training)
# ============================================================================

class MLPActor(nn.Module):
    """MLP policy network for PPO"""
    
    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        output_size: int,
        activation: str = 'tanh'
    ):
        super().__init__()
        
        self.activation = getattr(torch, activation)
        
        # Build layers
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            prev_size = hidden_size
        
        self.hidden_layers = nn.ModuleList(layers)
        
        # Output layers for mean and log_std
        self.mean_layer = nn.Linear(prev_size, output_size)
        self.log_std_layer = nn.Linear(prev_size, output_size)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using orthogonal initialization"""
        for layer in self.hidden_layers:
            nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
            nn.init.constant_(layer.bias, 0)
        
        nn.init.orthogonal_(self.mean_layer.weight, gain=0.01)
        nn.init.constant_(self.mean_layer.bias, 0)
        nn.init.constant_(self.log_std_layer.weight, 0)
        nn.init.constant_(self.log_std_layer.bias, -0.5)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input observations
            
        Returns:
            mean: Mean of action distribution
            std: Standard deviation of action distribution
        """
        h = x
        for layer in self.hidden_layers:
            h = self.activation(layer(h))
        
        mean = self.mean_layer(h)
        log_std = self.log_std_layer(h)
        std = torch.exp(torch.clamp(log_std, -20, 2))
        
        return mean, std
    
    def get_action(
        self, 
        x: torch.Tensor, 
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action from policy
        
        Args:
            x: Input observations
            deterministic: If True, return mean action
            
        Returns:
            action: Sampled action
            log_prob: Log probability of action
            mean: Mean of distribution (for distillation)
        """
        mean, std = self.forward(x)
        
        if deterministic:
            action = mean
            log_prob = None
        else:
            dist = Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
        
        # Clip action to valid range [0, 1]
        action = torch.clamp(action, 0, 1)
        
        return action, log_prob, mean


class MLPCritic(nn.Module):
    """Value network for PPO"""
    
    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        activation: str = 'tanh'
    ):
        super().__init__()
        
        self.activation = getattr(torch, activation)
        
        # Build layers
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            prev_size = hidden_size
        
        self.hidden_layers = nn.ModuleList(layers)
        self.value_layer = nn.Linear(prev_size, 1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using orthogonal initialization"""
        for layer in self.hidden_layers:
            nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
            nn.init.constant_(layer.bias, 0)
        
        nn.init.orthogonal_(self.value_layer.weight, gain=1.0)
        nn.init.constant_(self.value_layer.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input observations
            
        Returns:
            value: State value estimate
        """
        h = x
        for layer in self.hidden_layers:
            h = self.activation(layer(h))
        
        value = self.value_layer(h)
        return value.squeeze(-1)


# ============================================================================
# PPO TRAINER
# ============================================================================

class RolloutBuffer:
    """Buffer for storing rollout data"""
    
    def __init__(self, buffer_size: int, obs_dim: int, action_dim: int):
        self.buffer_size = buffer_size
        self.ptr = 0
        self.size = 0
        
        # Buffers
        self.observations = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)
        
        # GAE
        self.advantages = np.zeros(buffer_size, dtype=np.float32)
        self.returns = np.zeros(buffer_size, dtype=np.float32)
    
    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        value: float,
        log_prob: float,
        done: bool
    ):
        """Add a transition to the buffer"""
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)
    
    def compute_gae(self, last_value: float, gamma: float, gae_lambda: float):
        """Compute Generalized Advantage Estimation"""
        gae = 0
        for t in reversed(range(self.size)):
            if t == self.size - 1:
                next_value = last_value
                next_non_terminal = 1.0 - self.dones[t]
            else:
                next_value = self.values[t + 1]
                next_non_terminal = 1.0 - self.dones[t]
            
            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            gae = delta + gamma * gae_lambda * next_non_terminal * gae
            self.advantages[t] = gae
            self.returns[t] = gae + self.values[t]
    
    def get(self) -> Dict[str, torch.Tensor]:
        """Get all data as tensors"""
        indices = np.arange(self.size)
        
        return {
            'observations': torch.FloatTensor(self.observations[indices]),
            'actions': torch.FloatTensor(self.actions[indices]),
            'old_log_probs': torch.FloatTensor(self.log_probs[indices]),
            'advantages': torch.FloatTensor(self.advantages[indices]),
            'returns': torch.FloatTensor(self.returns[indices]),
        }
    
    def clear(self):
        """Clear the buffer"""
        self.ptr = 0
        self.size = 0


class PPOTrainer:
    """PPO trainer for the MLP teacher policy"""
    
    def __init__(
        self,
        env,
        actor: MLPActor,
        critic: MLPCritic,
        config: RLConfig
    ):
        self.env = env
        self.actor = actor
        self.critic = critic
        self.config = config
        
        # Optimizers
        self.actor_optimizer = Adam(actor.parameters(), lr=config.lr_actor)
        self.critic_optimizer = Adam(critic.parameters(), lr=config.lr_critic)
        
        # Get dimensions
        sample_obs = self._get_sample_observation()
        obs_dim = sample_obs.shape[0]
        action_dim = env.plant.num_actuators
        
        # Rollout buffer
        self.buffer = RolloutBuffer(config.buffer_size, obs_dim, action_dim)
        
        # Tracking
        self.iteration = 0
        self.episode_rewards = deque(maxlen=100)
    
    def _get_sample_observation(self) -> np.ndarray:
        """Get a sample observation to determine dimensionality"""
        self.env.plant.reset()
        target_pos = self.env.plant.sample_targets(1)[0]
        self.env.plant.update_target(target_pos)
        
        tgt_obs = self.env.target_encoder.encode(
            target_pos[0], target_pos[1]
        ).flatten()
        len_obs = self.env.plant.get_len_obs()
        vel_obs = self.env.plant.get_vel_obs()
        frc_obs = self.env.plant.get_frc_obs()
        
        return np.concatenate([tgt_obs, len_obs, vel_obs, frc_obs])
    
    def collect_rollouts(self) -> float:
        """Collect rollouts using current policy"""
        total_reward = 0
        num_episodes = 0
        
        while self.buffer.size < self.config.buffer_size:
            episode_reward = self._collect_episode()
            total_reward += episode_reward
            num_episodes += 1
            self.episode_rewards.append(episode_reward)
        
        return total_reward / num_episodes if num_episodes > 0 else 0
    
    def _collect_episode(self) -> float:
        """Collect a single episode"""
        # Reset environment
        self.env.plant.reset()
        
        # Sample targets
        target_positions = self.env.plant.sample_targets(self.env.num_targets)
        target_idx = 0
        target_position = target_positions[target_idx]
        self.env.plant.update_target(target_position)
        self.env.plant.enable_target()
        
        episode_reward = 0
        steps = 0
        max_steps = 1000
        
        while target_idx < self.env.num_targets and steps < max_steps:
            # Get observation
            tgt_obs = self.env.target_encoder.encode(
                target_position[0], target_position[1]
            ).flatten()
            tgt_obs *= 1 if self.env.plant.target_is_active else 0
            
            len_obs = self.env.plant.get_len_obs()
            vel_obs = self.env.plant.get_vel_obs()
            frc_obs = self.env.plant.get_frc_obs()
            
            obs = np.concatenate([tgt_obs, len_obs, vel_obs, frc_obs])
            
            # Get action from policy
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                action_tensor, log_prob, _ = self.actor.get_action(obs_tensor)
                value = self.critic(obs_tensor)
                
                action = action_tensor.squeeze(0).numpy()
                log_prob = log_prob.item()
                value = value.item()
            
            # Step environment
            self.env.plant.step(action)
            
            # Compute reward
            hand_pos = self.env.plant.get_hand_pos()
            distance = (
                np.linalg.norm(target_position - hand_pos)
                if self.env.plant.target_is_active
                else 0
            )
            energy = np.sum(action**2)
            
            reward = -(
                distance * self.env.loss_weights['distance'] +
                energy * self.env.loss_weights['energy']
            )
            
            episode_reward += reward
            
            # Check for target change
            done = False
            if target_idx < self.env.num_targets - 1:
                # Simplified target timing - could be improved
                if steps % 100 == 0 and steps > 0:
                    target_idx += 1
                    if target_idx < self.env.num_targets:
                        target_position = target_positions[target_idx]
                        self.env.plant.randomize_gravity_direction()
                        self.env.plant.update_target(target_position)
                        self.env.plant.enable_target()
            else:
                done = True
            
            # Add to buffer
            if self.buffer.size < self.config.buffer_size:
                self.buffer.add(obs, action, reward, value, log_prob, done)
            
            steps += 1
            
            if done:
                break
        
        return episode_reward
    
    def update(self):
        """Update policy using PPO"""
        # Compute advantages
        with torch.no_grad():
            last_obs = torch.FloatTensor(self.buffer.observations[self.buffer.size - 1]).unsqueeze(0)
            last_value = self.critic(last_obs).item()
        
        self.buffer.compute_gae(last_value, self.config.gamma, self.config.gae_lambda)
        
        # Get data
        data = self.buffer.get()
        
        # Normalize advantages
        advantages = data['advantages']
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Update for multiple epochs
        for epoch in range(self.config.num_epochs):
            # Create mini-batches
            indices = np.random.permutation(self.buffer.size)
            
            for start in range(0, self.buffer.size, self.config.batch_size):
                end = start + self.config.batch_size
                if end > self.buffer.size:
                    continue
                
                batch_indices = indices[start:end]
                
                # Get batch data
                obs_batch = data['observations'][batch_indices]
                actions_batch = data['actions'][batch_indices]
                old_log_probs_batch = data['old_log_probs'][batch_indices]
                advantages_batch = advantages[batch_indices]
                returns_batch = data['returns'][batch_indices]
                
                # Compute current log probs and values
                mean, std = self.actor.forward(obs_batch)
                dist = Normal(mean, std)
                log_probs = dist.log_prob(actions_batch).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()
                
                values = self.critic(obs_batch)
                
                # PPO actor loss
                ratio = torch.exp(log_probs - old_log_probs_batch)
                surr1 = ratio * advantages_batch
                surr2 = torch.clamp(
                    ratio,
                    1 - self.config.epsilon_clip,
                    1 + self.config.epsilon_clip
                ) * advantages_batch
                actor_loss = -torch.min(surr1, surr2).mean()
                actor_loss -= self.config.entropy_coef * entropy
                
                # Critic loss
                critic_loss = F.mse_loss(values, returns_batch)
                
                # Update actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
                self.actor_optimizer.step()
                
                # Update critic
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm)
                self.critic_optimizer.step()
        
        # Clear buffer
        self.buffer.clear()
    
    def train(self) -> Dict[str, List[float]]:
        """Main training loop"""
        training_stats = {
            'iteration': [],
            'episode_reward_mean': [],
            'episode_reward_std': [],
        }
        
        for iteration in range(self.config.num_iterations):
            self.iteration = iteration
            
            # Collect rollouts
            avg_reward = self.collect_rollouts()
            
            # Update policy
            self.update()
            
            # Logging
            if iteration % self.config.log_interval == 0:
                reward_mean = np.mean(self.episode_rewards)
                reward_std = np.std(self.episode_rewards)
                
                training_stats['iteration'].append(iteration)
                training_stats['episode_reward_mean'].append(reward_mean)
                training_stats['episode_reward_std'].append(reward_std)
                
                print(f"Iteration {iteration}: "
                      f"Reward = {reward_mean:.2f} ± {reward_std:.2f}")
            
            # Save checkpoint
            if iteration % self.config.save_interval == 0:
                self.save_checkpoint(f"ppo_checkpoint_iter_{iteration}.pt")
        
        return training_stats
    
    def save_checkpoint(self, filepath: str):
        """Save model checkpoint"""
        checkpoint = {
            'iteration': self.iteration,
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.iteration = checkpoint['iteration']
        print(f"Checkpoint loaded from {filepath}")


# ============================================================================
# DISTILLATION TO RNN STUDENT
# ============================================================================

class DistillationDataset:
    """Dataset of teacher demonstrations for distillation"""
    
    def __init__(self):
        self.observations = []
        self.actions = []
        self.teacher_logits = []  # Mean and std from teacher
    
    def add_trajectory(
        self,
        observations: List[np.ndarray],
        actions: List[np.ndarray],
        teacher_logits: List[Tuple[np.ndarray, np.ndarray]]
    ):
        """Add a trajectory to the dataset"""
        self.observations.extend(observations)
        self.actions.extend(actions)
        self.teacher_logits.extend(teacher_logits)
    
    def get_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample a random batch"""
        indices = np.random.choice(len(self.observations), batch_size, replace=False)
        
        obs = torch.FloatTensor([self.observations[i] for i in indices])
        actions = torch.FloatTensor([self.actions[i] for i in indices])
        
        # Extract means and stds
        means = torch.FloatTensor([self.teacher_logits[i][0] for i in indices])
        stds = torch.FloatTensor([self.teacher_logits[i][1] for i in indices])
        
        return {
            'observations': obs,
            'actions': actions,
            'teacher_means': means,
            'teacher_stds': stds,
        }
    
    def __len__(self):
        return len(self.observations)


class RNNStudentWrapper:
    """
    Wrapper around the original NeuroMuscularRNN to make it compatible with PyTorch training
    """
    
    def __init__(self, rnn):
        self.rnn = rnn
        
        # Convert to PyTorch parameters
        params = rnn.get_params()
        self.params = nn.Parameter(torch.FloatTensor(params))
        
        # Get RNN architecture info
        self.input_size_tgt = rnn.input_size_tgt
        self.input_size_len = rnn.input_size_len
        self.input_size_vel = rnn.input_size_vel
        self.input_size_frc = rnn.input_size_frc
        self.hidden_size = rnn.hidden_size
        self.output_size = rnn.output_size
    
    def forward(
        self, 
        obs: torch.Tensor,
        reset_state: bool = True
    ) -> torch.Tensor:
        """
        Forward pass through RNN
        
        Args:
            obs: Concatenated observation [tgt, len, vel, frc]
            reset_state: Whether to reset hidden state
            
        Returns:
            action: Output action
        """
        # Convert to numpy
        obs_np = obs.detach().cpu().numpy()
        
        # Split observation
        tgt_end = self.input_size_tgt
        len_end = tgt_end + self.input_size_len
        vel_end = len_end + self.input_size_vel
        
        tgt_obs = obs_np[:, :tgt_end]
        len_obs = obs_np[:, tgt_end:len_end]
        vel_obs = obs_np[:, len_end:vel_end]
        frc_obs = obs_np[:, vel_end:]
        
        # Update RNN parameters
        self.rnn.set_params(self.params.detach().cpu().numpy())
        
        # Reset state if needed
        if reset_state:
            self.rnn.reset_state()
        
        # Forward pass for each item in batch
        actions = []
        for i in range(obs.shape[0]):
            action = self.rnn.step(
                tgt_obs[i],
                len_obs[i],
                vel_obs[i],
                frc_obs[i]
            )
            actions.append(action)
        
        return torch.FloatTensor(np.array(actions))
    
    def parameters(self):
        """Return parameters for optimizer"""
        return [self.params]


class DistillationTrainer:
    """Trainer for distilling teacher MLP to student RNN"""
    
    def __init__(
        self,
        teacher: MLPActor,
        student_rnn,
        env,
        config: DistillationConfig
    ):
        self.teacher = teacher
        self.student = RNNStudentWrapper(student_rnn)
        self.env = env
        self.config = config
        
        # Optimizer
        self.optimizer = Adam(self.student.parameters(), lr=config.lr)
        
        # Dataset
        self.dataset = DistillationDataset()
    
    def collect_teacher_data(self):
        """Collect demonstrations from teacher policy"""
        print("Collecting teacher demonstrations...")
        
        for traj_idx in range(self.config.num_trajectories):
            observations = []
            actions = []
            teacher_logits = []
            
            # Reset environment
            self.env.plant.reset()
            target_positions = self.env.plant.sample_targets(self.env.num_targets)
            target_idx = 0
            target_position = target_positions[target_idx]
            self.env.plant.update_target(target_position)
            self.env.plant.enable_target()
            
            steps = 0
            
            while target_idx < self.env.num_targets and steps < self.config.max_trajectory_length:
                # Get observation
                tgt_obs = self.env.target_encoder.encode(
                    target_position[0], target_position[1]
                ).flatten()
                tgt_obs *= 1 if self.env.plant.target_is_active else 0
                
                len_obs = self.env.plant.get_len_obs()
                vel_obs = self.env.plant.get_vel_obs()
                frc_obs = self.env.plant.get_frc_obs()
                
                obs = np.concatenate([tgt_obs, len_obs, vel_obs, frc_obs])
                
                # Get teacher action and logits
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                    action_tensor, _, mean = self.teacher.get_action(obs_tensor, deterministic=True)
                    _, std = self.teacher.forward(obs_tensor)
                    
                    action = action_tensor.squeeze(0).numpy()
                    mean = mean.squeeze(0).numpy()
                    std = std.squeeze(0).numpy()
                
                # Store
                observations.append(obs)
                actions.append(action)
                teacher_logits.append((mean, std))
                
                # Step environment
                self.env.plant.step(action)
                
                # Update target
                if steps % 100 == 0 and steps > 0:
                    target_idx += 1
                    if target_idx < self.env.num_targets:
                        target_position = target_positions[target_idx]
                        self.env.plant.randomize_gravity_direction()
                        self.env.plant.update_target(target_position)
                        self.env.plant.enable_target()
                
                steps += 1
            
            # Add trajectory to dataset
            self.dataset.add_trajectory(observations, actions, teacher_logits)
            
            if (traj_idx + 1) % 100 == 0:
                print(f"Collected {traj_idx + 1}/{self.config.num_trajectories} trajectories")
        
        print(f"Total samples collected: {len(self.dataset)}")
    
    def train(self) -> Dict[str, List[float]]:
        """Train student RNN to imitate teacher"""
        training_stats = {
            'epoch': [],
            'action_loss': [],
            'kl_loss': [],
            'total_loss': [],
        }
        
        print("Training student RNN...")
        
        for epoch in range(self.config.num_epochs):
            epoch_action_losses = []
            epoch_kl_losses = []
            epoch_total_losses = []
            
            # Number of batches
            num_batches = len(self.dataset) // self.config.batch_size
            
            for _ in range(num_batches):
                # Get batch
                batch = self.dataset.get_batch(self.config.batch_size)
                
                # Forward pass
                student_actions = self.student.forward(
                    batch['observations'],
                    reset_state=True
                )
                
                # Action loss (MSE with teacher actions)
                action_loss = F.mse_loss(student_actions, batch['actions'])
                
                # KL divergence loss (match teacher distribution)
                # Approximate: just match the means
                kl_loss = F.mse_loss(student_actions, batch['teacher_means'])
                
                # Combined loss
                total_loss = (
                    (1 - self.config.alpha) * action_loss +
                    self.config.alpha * kl_loss / (self.config.temperature ** 2)
                )
                
                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                
                # Track losses
                epoch_action_losses.append(action_loss.item())
                epoch_kl_losses.append(kl_loss.item())
                epoch_total_losses.append(total_loss.item())
            
            # Logging
            if epoch % self.config.log_interval == 0:
                avg_action_loss = np.mean(epoch_action_losses)
                avg_kl_loss = np.mean(epoch_kl_losses)
                avg_total_loss = np.mean(epoch_total_losses)
                
                training_stats['epoch'].append(epoch)
                training_stats['action_loss'].append(avg_action_loss)
                training_stats['kl_loss'].append(avg_kl_loss)
                training_stats['total_loss'].append(avg_total_loss)
                
                print(f"Epoch {epoch}: "
                      f"Action Loss = {avg_action_loss:.4f}, "
                      f"KL Loss = {avg_kl_loss:.4f}, "
                      f"Total Loss = {avg_total_loss:.4f}")
            
            # Save checkpoint
            if epoch % self.config.save_interval == 0:
                self.save_checkpoint(f"distillation_checkpoint_epoch_{epoch}.pt")
        
        # Update original RNN with learned parameters
        self.student.rnn.set_params(self.student.params.detach().cpu().numpy())
        
        return training_stats
    
    def save_checkpoint(self, filepath: str):
        """Save student checkpoint"""
        checkpoint = {
            'student_params': self.student.params.detach().cpu().numpy(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load student checkpoint"""
        checkpoint = torch.load(filepath)
        self.student.params.data = torch.FloatTensor(checkpoint['student_params'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.student.rnn.set_params(checkpoint['student_params'])
        print(f"Checkpoint loaded from {filepath}")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def train_with_rl_distillation(
    env,
    rnn,
    rl_config: Optional[RLConfig] = None,
    distillation_config: Optional[DistillationConfig] = None,
    teacher_checkpoint: Optional[str] = None
):
    """
    Complete pipeline: Train teacher with RL, then distill to student RNN
    
    Args:
        env: Environment for training
        rnn: Student RNN to train
        rl_config: Configuration for RL training
        distillation_config: Configuration for distillation
        teacher_checkpoint: Optional path to pre-trained teacher
        
    Returns:
        Tuple of (trained_rnn, teacher, training_stats)
    """
    if rl_config is None:
        rl_config = RLConfig()
    
    if distillation_config is None:
        distillation_config = DistillationConfig()
    
    # Determine input/output dimensions
    sample_obs = env.plant.get_len_obs()
    obs_dim = (
        env.target_encoder.size +
        len(sample_obs) * 3  # len, vel, frc
    )
    action_dim = env.plant.num_actuators
    
    # Create teacher network
    teacher = MLPActor(
        input_size=obs_dim,
        hidden_sizes=[128, 128],
        output_size=action_dim,
        activation='tanh'
    )
    
    critic = MLPCritic(
        input_size=obs_dim,
        hidden_sizes=[128, 128],
        activation='tanh'
    )
    
    # Train teacher with PPO (or load checkpoint)
    if teacher_checkpoint is None:
        print("=" * 80)
        print("PHASE 1: Training Teacher Policy with PPO")
        print("=" * 80)
        
        ppo_trainer = PPOTrainer(env, teacher, critic, rl_config)
        rl_stats = ppo_trainer.train()
        ppo_trainer.save_checkpoint("teacher_final.pt")
    else:
        print(f"Loading pre-trained teacher from {teacher_checkpoint}")
        checkpoint = torch.load(teacher_checkpoint)
        teacher.load_state_dict(checkpoint['actor_state_dict'])
        rl_stats = None
    
    # Distill to student RNN
    print("\n" + "=" * 80)
    print("PHASE 2: Distilling to Student RNN")
    print("=" * 80)
    
    distillation_trainer = DistillationTrainer(
        teacher, rnn, env, distillation_config
    )
    
    distillation_trainer.collect_teacher_data()
    distillation_stats = distillation_trainer.train()
    
    # Save final student
    with open('student_rnn_final.pkl', 'wb') as f:
        pickle.dump(rnn, f)
    
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    
    return rnn, teacher, {
        'rl_stats': rl_stats,
        'distillation_stats': distillation_stats
    }
