"""
PPO (Proximal Policy Optimization) trainer for MLP policy.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple
import gymnasium as gym
from collections import deque


class PPOTrainer:
    """PPO trainer for continuous control."""

    def __init__(
        self,
        policy: nn.Module,
        env: gym.Env,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        num_epochs: int = 10,
        batch_size: int = 64,
        num_steps: int = 2048,
        device: str = "cpu",
    ):
        """
        Initialize PPO trainer.

        Args:
            policy: Policy network
            env: Gymnasium environment
            learning_rate: Learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_epsilon: PPO clipping parameter
            value_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Maximum gradient norm for clipping
            num_epochs: Number of PPO epochs per update
            batch_size: Mini-batch size
            num_steps: Number of steps to collect before update
            device: Device to use
        """
        self.policy = policy.to(device)
        self.env = env
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.device = device

        self.optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

        # Storage for trajectories
        self.reset_storage()

        # Tracking
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.update_count = 0

    def reset_storage(self):
        """Reset trajectory storage."""
        self.observations = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.log_probs = []

    def compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        next_value: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Generalized Advantage Estimation.

        Args:
            rewards: Rewards (num_steps,)
            values: Value estimates (num_steps,)
            dones: Done flags (num_steps,)
            next_value: Value of next state

        Returns:
            advantages: GAE advantages (num_steps,)
            returns: Discounted returns (num_steps,)
        """
        advantages = np.zeros_like(rewards)
        last_gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value_t = next_value
            else:
                next_value_t = values[t + 1]

            delta = rewards[t] + self.gamma * next_value_t * (1 - dones[t]) - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
            advantages[t] = last_gae

        returns = advantages + values
        return advantages, returns

    def collect_trajectories(self) -> Dict[str, np.ndarray]:
        """
        Collect trajectories using current policy.

        Returns:
            batch: Dictionary of trajectory data
        """
        self.reset_storage()

        obs, _ = self.env.reset()
        episode_reward = 0
        episode_length = 0

        for step in range(self.num_steps):
            # Get action from policy
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

            with torch.no_grad():
                action, value = self.policy.get_action_and_value(obs_tensor)
                action = action.squeeze(0).cpu().numpy()
                value = value.squeeze(0).cpu().item()

            # Take step in environment
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            # Store transition
            self.observations.append(obs)
            self.actions.append(action)
            self.rewards.append(reward)
            self.values.append(value)
            self.dones.append(done)

            episode_reward += reward
            episode_length += 1

            obs = next_obs

            if done:
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                obs, _ = self.env.reset()
                episode_reward = 0
                episode_length = 0

        # Get value of final state for GAE
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            next_value = self.policy.get_value(obs_tensor).cpu().item()

        # Convert to numpy arrays
        observations = np.array(self.observations)
        actions = np.array(self.actions)
        rewards = np.array(self.rewards)
        values = np.array(self.values)
        dones = np.array(self.dones)

        # Compute advantages and returns
        advantages, returns = self.compute_gae(rewards, values, dones, next_value)

        return {
            "observations": observations,
            "actions": actions,
            "advantages": advantages,
            "returns": returns,
            "values": values,
        }

    def update_policy(self, batch: Dict[str, np.ndarray]):
        """
        Update policy using PPO.

        Args:
            batch: Batch of trajectory data
        """
        observations = torch.FloatTensor(batch["observations"]).to(self.device)
        actions = torch.FloatTensor(batch["actions"]).to(self.device)
        advantages = torch.FloatTensor(batch["advantages"]).to(self.device)
        returns = torch.FloatTensor(batch["returns"]).to(self.device)
        old_values = torch.FloatTensor(batch["values"]).to(self.device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update
        for epoch in range(self.num_epochs):
            # Generate random indices for mini-batches
            indices = np.random.permutation(len(observations))

            for start in range(0, len(observations), self.batch_size):
                end = start + self.batch_size
                idx = indices[start:end]

                # Get mini-batch
                mb_obs = observations[idx]
                mb_actions = actions[idx]
                mb_advantages = advantages[idx]
                mb_returns = returns[idx]

                # Forward pass
                pred_actions, pred_values = self.policy.get_action_and_value(mb_obs)

                # Policy loss (assuming deterministic policy for simplicity)
                action_loss = nn.functional.mse_loss(pred_actions, mb_actions)

                # Value loss
                value_loss = nn.functional.mse_loss(pred_values.squeeze(), mb_returns)

                # Total loss
                loss = action_loss + self.value_coef * value_loss

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

        self.update_count += 1

    def train(self, num_updates: int):
        """
        Train policy for specified number of updates.

        Args:
            num_updates: Number of PPO updates
        """
        for update in range(num_updates):
            # Collect trajectories
            batch = self.collect_trajectories()

            # Update policy
            self.update_policy(batch)

            # Log progress
            if len(self.episode_rewards) > 0:
                mean_reward = np.mean(self.episode_rewards)
                mean_length = np.mean(self.episode_lengths)
                print(
                    f"Update {update + 1}/{num_updates} | "
                    f"Mean Reward: {mean_reward:.2f} | "
                    f"Mean Length: {mean_length:.1f}"
                )

    def get_stats(self) -> Dict[str, float]:
        """Get training statistics."""
        if len(self.episode_rewards) > 0:
            return {
                "mean_reward": np.mean(self.episode_rewards),
                "std_reward": np.std(self.episode_rewards),
                "mean_length": np.mean(self.episode_lengths),
                "num_updates": self.update_count,
            }
        return {}
