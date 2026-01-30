"""
Behavioral cloning (imitation learning) trainer for RNN distillation.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
import gymnasium as gym
from collections import deque


class BehavioralCloningTrainer:
    """Trainer for RNN distillation via behavioral cloning."""
    
    def __init__(
        self,
        student_policy: nn.Module,
        teacher_policy: nn.Module,
        env: gym.Env,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        sequence_length: int = 50,
        device: str = "cpu"
    ):
        """
        Initialize behavioral cloning trainer.
        
        Args:
            student_policy: RNN policy to train
            teacher_policy: Pre-trained MLP policy to imitate
            env: Gymnasium environment
            learning_rate: Learning rate
            batch_size: Number of sequences per batch
            sequence_length: Length of each sequence
            device: Device to use
        """
        self.student = student_policy.to(device)
        self.teacher = teacher_policy.to(device)
        self.env = env
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.device = device
        
        self.optimizer = optim.Adam(student_policy.parameters(), lr=learning_rate)
        
        # Put teacher in eval mode
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
        
        # Dataset storage
        self.dataset = []
        
        # Tracking
        self.train_losses = deque(maxlen=100)
        self.val_losses = deque(maxlen=20)
        self.episode_rewards = deque(maxlen=100)
        
    def collect_demonstrations(self, num_episodes: int):
        """
        Collect demonstration data from teacher policy.
        
        Args:
            num_episodes: Number of episodes to collect
        """
        print(f"Collecting {num_episodes} demonstration episodes...")
        
        for episode in range(num_episodes):
            obs, _ = self.env.reset()
            done = False
            episode_data = {
                "observations": [],
                "actions": []
            }
            episode_reward = 0
            
            while not done:
                # Get action from teacher
                with torch.no_grad():
                    action = self.teacher.predict(obs, deterministic=True)
                
                # Store transition
                episode_data["observations"].append(obs)
                episode_data["actions"].append(action)
                
                # Take step
                obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                episode_reward += reward
            
            # Add episode to dataset
            self.dataset.append(episode_data)
            self.episode_rewards.append(episode_reward)
            
            if (episode + 1) % 10 == 0:
                mean_reward = np.mean(list(self.episode_rewards)[-10:])
                print(f"Episode {episode + 1}/{num_episodes} | "
                      f"Recent mean reward: {mean_reward:.2f}")
        
        print(f"Collected {len(self.dataset)} episodes, "
              f"mean reward: {np.mean(self.episode_rewards):.2f}")
    
    def create_sequences(
        self, split: str = "train", train_ratio: float = 0.8
    ) -> List[Dict[str, np.ndarray]]:
        """
        Create training sequences from dataset.
        
        Args:
            split: 'train' or 'val'
            train_ratio: Ratio of data to use for training
            
        Returns:
            sequences: List of sequence dictionaries
        """
        # Split dataset
        split_idx = int(len(self.dataset) * train_ratio)
        if split == "train":
            episodes = self.dataset[:split_idx]
        else:
            episodes = self.dataset[split_idx:]
        
        sequences = []
        
        for episode in episodes:
            obs_array = np.array(episode["observations"])
            action_array = np.array(episode["actions"])
            episode_length = len(obs_array)
            
            # Create overlapping sequences
            for start_idx in range(0, episode_length - self.sequence_length, 
                                   self.sequence_length // 2):
                end_idx = start_idx + self.sequence_length
                if end_idx > episode_length:
                    break
                
                sequences.append({
                    "observations": obs_array[start_idx:end_idx],
                    "actions": action_array[start_idx:end_idx]
                })
        
        return sequences
    
    def train_epoch(self, sequences: List[Dict[str, np.ndarray]]) -> float:
        """
        Train for one epoch on sequences.
        
        Args:
            sequences: List of training sequences
            
        Returns:
            mean_loss: Mean loss for epoch
        """
        self.student.train()
        
        # Shuffle sequences
        indices = np.random.permutation(len(sequences))
        epoch_losses = []
        
        for start in range(0, len(sequences), self.batch_size):
            end = min(start + self.batch_size, len(sequences))
            batch_indices = indices[start:end]
            
            # Prepare batch
            batch_obs = []
            batch_actions = []
            
            for idx in batch_indices:
                batch_obs.append(sequences[idx]["observations"])
                batch_actions.append(sequences[idx]["actions"])
            
            # Convert to tensors (seq_len, batch_size, dim)
            obs_tensor = torch.FloatTensor(np.array(batch_obs)).transpose(0, 1).to(self.device)
            action_tensor = torch.FloatTensor(np.array(batch_actions)).transpose(0, 1).to(self.device)
            
            # Forward pass through student RNN
            pred_actions, _ = self.student(obs_tensor, hidden=None)
            
            # Compute loss (MSE between student and teacher actions)
            loss = nn.functional.mse_loss(pred_actions, action_tensor)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            epoch_losses.append(loss.item())
        
        return np.mean(epoch_losses)
    
    def validate(self, sequences: List[Dict[str, np.ndarray]]) -> float:
        """
        Validate on sequences.
        
        Args:
            sequences: List of validation sequences
            
        Returns:
            mean_loss: Mean validation loss
        """
        self.student.eval()
        
        val_losses = []
        
        with torch.no_grad():
            for start in range(0, len(sequences), self.batch_size):
                end = min(start + self.batch_size, len(sequences))
                batch_sequences = sequences[start:end]
                
                # Prepare batch
                batch_obs = []
                batch_actions = []
                
                for seq in batch_sequences:
                    batch_obs.append(seq["observations"])
                    batch_actions.append(seq["actions"])
                
                # Convert to tensors
                obs_tensor = torch.FloatTensor(np.array(batch_obs)).transpose(0, 1).to(self.device)
                action_tensor = torch.FloatTensor(np.array(batch_actions)).transpose(0, 1).to(self.device)
                
                # Forward pass
                pred_actions, _ = self.student(obs_tensor, hidden=None)
                
                # Compute loss
                loss = nn.functional.mse_loss(pred_actions, action_tensor)
                val_losses.append(loss.item())
        
        return np.mean(val_losses)
    
    def train(self, num_epochs: int, train_ratio: float = 0.8):
        """
        Train student policy via behavioral cloning.
        
        Args:
            num_epochs: Number of training epochs
            train_ratio: Ratio of data for training vs validation
        """
        if len(self.dataset) == 0:
            raise ValueError("No demonstration data collected. Call collect_demonstrations() first.")
        
        # Create sequences
        print("Creating training sequences...")
        train_sequences = self.create_sequences("train", train_ratio)
        val_sequences = self.create_sequences("val", train_ratio)
        print(f"Created {len(train_sequences)} train and {len(val_sequences)} val sequences")
        
        # Training loop
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            # Train
            train_loss = self.train_epoch(train_sequences)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate(val_sequences)
            self.val_losses.append(val_loss)
            
            # Log
            print(f"Epoch {epoch + 1}/{num_epochs} | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"  New best validation loss: {best_val_loss:.6f}")
    
    def evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
        """
        Evaluate student policy in environment.
        
        Args:
            num_episodes: Number of episodes to evaluate
            
        Returns:
            stats: Dictionary of evaluation statistics
        """
        self.student.eval()
        
        episode_rewards = []
        episode_lengths = []
        success_count = 0
        
        for episode in range(num_episodes):
            obs, _ = self.env.reset()
            hidden = None
            done = False
            episode_reward = 0
            episode_length = 0
            
            while not done:
                # Get action from student
                with torch.no_grad():
                    action, hidden = self.student.predict(obs, hidden=hidden)
                
                # Take step
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                episode_reward += reward
                episode_length += 1
                
                if info.get("success", False):
                    success_count += 1
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
        
        return {
            "mean_reward": np.mean(episode_rewards),
            "std_reward": np.std(episode_rewards),
            "mean_length": np.mean(episode_lengths),
            "success_rate": success_count / num_episodes
        }
