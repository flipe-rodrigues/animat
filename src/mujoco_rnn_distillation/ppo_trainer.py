"""
PPO Trainer for Sequential Reaching Task

This module implements Proximal Policy Optimization (PPO) to train
an MLP policy that can later be distilled into an RNN.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Beta
from collections import deque
import logging
from typing import Dict, List, Tuple, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLPPolicy(nn.Module):
    """
    MLP policy network for reaching task.
    
    Uses Beta distribution for bounded action space [0, 1].
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: List[int] = [256, 256],
        activation: str = 'tanh',
    ):
        """
        Initialize MLP policy.
        
        Args:
            obs_dim: Observation dimension (target + length + velocity + force)
            action_dim: Action dimension (number of muscles)
            hidden_sizes: Hidden layer sizes
            activation: Activation function ('tanh', 'relu', 'elu')
        """
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # Build network
        layers = []
        input_dim = obs_dim
        
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_dim))
            
            if activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'elu':
                layers.append(nn.ELU())
            else:
                raise ValueError(f"Unknown activation: {activation}")
            
            input_dim = hidden_dim
        
        self.shared_net = nn.Sequential(*layers)
        
        # Beta distribution parameters (alpha, beta > 1 for smooth distributions)
        # Output log(alpha-1) and log(beta-1) for numerical stability
        self.alpha_head = nn.Linear(input_dim, action_dim)
        self.beta_head = nn.Linear(input_dim, action_dim)
        
        # Value head for critic
        self.value_head = nn.Linear(input_dim, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        
        # Small initialization for policy heads
        nn.init.orthogonal_(self.alpha_head.weight, gain=0.01)
        nn.init.orthogonal_(self.beta_head.weight, gain=0.01)
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            obs: Observation tensor [batch_size, obs_dim]
        
        Returns:
            alpha: Beta distribution alpha parameter
            beta: Beta distribution beta parameter
            value: State value estimate
        """
        features = self.shared_net(obs)
        
        # Beta parameters (ensure > 1 for smooth distributions)
        alpha = torch.exp(self.alpha_head(features)) + 1.0
        beta = torch.exp(self.beta_head(features)) + 1.0
        
        # Value estimate
        value = self.value_head(features)
        
        return alpha, beta, value
    
    def get_action(
        self, 
        obs: torch.Tensor, 
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.
        
        Args:
            obs: Observation tensor
            deterministic: If True, return mode of distribution
        
        Returns:
            action: Sampled action
            log_prob: Log probability of action
            value: Value estimate
        """
        alpha, beta, value = self.forward(obs)
        dist = Beta(alpha, beta)
        
        if deterministic:
            # Mode of beta distribution: (alpha-1)/(alpha+beta-2)
            action = (alpha - 1) / (alpha + beta - 2)
        else:
            action = dist.sample()
        
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        
        return action, log_prob, value
    
    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions under current policy.
        
        Args:
            obs: Observation tensor
            actions: Action tensor
        
        Returns:
            log_probs: Log probabilities
            values: Value estimates
            entropy: Policy entropy
        """
        alpha, beta, value = self.forward(obs)
        dist = Beta(alpha, beta)
        
        log_probs = dist.log_prob(actions).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        
        return log_probs, value, entropy


class RolloutBuffer:
    """Experience buffer for PPO"""
    
    def __init__(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
    ):
        """Add transition to buffer"""
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def get(self) -> Dict[str, np.ndarray]:
        """Get all data as arrays"""
        return {
            'observations': np.array(self.observations),
            'actions': np.array(self.actions),
            'rewards': np.array(self.rewards),
            'values': np.array(self.values),
            'log_probs': np.array(self.log_probs),
            'dones': np.array(self.dones),
        }
    
    def clear(self):
        """Clear buffer"""
        self.__init__()


class PPOTrainer:
    """
    PPO trainer for sequential reaching task.
    """
    
    def __init__(
        self,
        env_config: Dict,
        policy_config: Dict,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        n_epochs: int = 10,
        batch_size: int = 64,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        """
        Initialize PPO trainer.
        
        Args:
            env_config: Environment configuration dict
            policy_config: Policy network configuration
            learning_rate: Learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_epsilon: PPO clipping parameter
            value_loss_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Maximum gradient norm for clipping
            n_epochs: Number of optimization epochs per update
            batch_size: Minibatch size
            device: Device to train on
        """
        self.device = torch.device(device)
        
        # Hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        
        # Store configs
        self.env_config = env_config
        self.policy_config = policy_config
        
        # Create policy
        self.policy = MLPPolicy(**policy_config).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        
        # Training stats
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.update_count = 0
        
        logger.info(f"Initialized PPO trainer on {self.device}")
        logger.info(f"Policy: {sum(p.numel() for p in self.policy.parameters())} parameters")
    
    def compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        last_value: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Generalized Advantage Estimation.
        
        Args:
            rewards: Array of rewards
            values: Array of value estimates
            dones: Array of done flags
            last_value: Value estimate for final state
        
        Returns:
            advantages: Computed advantages
            returns: Computed returns (targets for value function)
        """
        advantages = np.zeros_like(rewards)
        last_gae = 0
        
        # Append last value for easier computation
        values_with_last = np.append(values, last_value)
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                next_value = 0
                last_gae = 0
            else:
                next_value = values_with_last[t + 1]
            
            delta = rewards[t] + self.gamma * next_value - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * last_gae
            advantages[t] = last_gae
        
        returns = advantages + values
        
        return advantages, returns
    
    def update(self, rollout_buffer: RolloutBuffer, last_obs: np.ndarray) -> Dict[str, float]:
        """
        Update policy using collected rollout data.
        
        Args:
            rollout_buffer: Buffer containing rollout data
            last_obs: Final observation for bootstrapping
        
        Returns:
            Dictionary of training statistics
        """
        # Get data from buffer
        data = rollout_buffer.get()
        
        # Compute last value for GAE
        with torch.no_grad():
            last_obs_tensor = torch.FloatTensor(last_obs).unsqueeze(0).to(self.device)
            _, _, last_value = self.policy.get_action(last_obs_tensor)
            last_value = last_value.cpu().item()
        
        # Compute advantages and returns
        advantages, returns = self.compute_gae(
            data['rewards'],
            data['values'],
            data['dones'],
            last_value,
        )
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convert to tensors
        obs_tensor = torch.FloatTensor(data['observations']).to(self.device)
        actions_tensor = torch.FloatTensor(data['actions']).to(self.device)
        old_log_probs = torch.FloatTensor(data['log_probs']).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).unsqueeze(1).to(self.device)
        returns_tensor = torch.FloatTensor(returns).unsqueeze(1).to(self.device)
        
        # Training statistics
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_clipfrac = 0
        n_updates = 0
        
        # Multiple epochs of optimization
        for epoch in range(self.n_epochs):
            # Generate random minibatch indices
            indices = np.random.permutation(len(obs_tensor))
            
            for start in range(0, len(obs_tensor), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                # Get batch
                batch_obs = obs_tensor[batch_indices]
                batch_actions = actions_tensor[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]
                batch_returns = returns_tensor[batch_indices]
                
                # Evaluate actions under current policy
                log_probs, values, entropy = self.policy.evaluate_actions(
                    batch_obs, batch_actions
                )
                
                # Policy loss (PPO clipped objective)
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(
                    ratio,
                    1 - self.clip_epsilon,
                    1 + self.clip_epsilon
                ) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = nn.functional.mse_loss(values, batch_returns)
                
                # Entropy bonus
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = (
                    policy_loss
                    + self.value_loss_coef * value_loss
                    + self.entropy_coef * entropy_loss
                )
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Track statistics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                
                # Compute clip fraction
                with torch.no_grad():
                    clipfrac = ((ratio - 1.0).abs() > self.clip_epsilon).float().mean()
                    total_clipfrac += clipfrac.item()
                
                n_updates += 1
        
        self.update_count += 1
        
        # Return statistics
        return {
            'policy_loss': total_policy_loss / n_updates,
            'value_loss': total_value_loss / n_updates,
            'entropy': total_entropy / n_updates,
            'clipfrac': total_clipfrac / n_updates,
            'n_updates': n_updates,
        }
    
    def save(self, path: str):
        """Save policy checkpoint"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'update_count': self.update_count,
            'policy_config': self.policy_config,
            'env_config': self.env_config,
        }, path)
        logger.info(f"Saved checkpoint to {path}")
    
    def load(self, path: str):
        """Load policy checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.update_count = checkpoint['update_count']
        logger.info(f"Loaded checkpoint from {path}")
