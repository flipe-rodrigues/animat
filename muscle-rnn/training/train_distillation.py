"""Distillation Learning for Muscle-Driven Arm Controller.

Trains an RNN student by distilling knowledge from a pre-trained controller
using Stable Baselines 3's behavioral cloning approach.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional, Dict, Any

try:
    from stable_baselines3.common.policies import BasePolicy
    from stable_baselines3.common.evaluation import evaluate_policy
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False


class TrajectoryDataset(Dataset):
    """Dataset of trajectory sequences for RNN training."""

    def __init__(self, observations: List[np.ndarray], actions: List[np.ndarray], seq_len: int = 50):
        self.obs_sequences = []
        self.action_sequences = []
        for obs_traj, action_traj in zip(observations, actions):
            for start in range(0, len(obs_traj) - seq_len + 1, seq_len // 2):
                self.obs_sequences.append(obs_traj[start : start + seq_len])
                self.action_sequences.append(action_traj[start : start + seq_len])
        self.obs_sequences = np.array(self.obs_sequences) if self.obs_sequences else np.array([]).reshape(0, seq_len, 0)
        self.action_sequences = np.array(self.action_sequences) if self.action_sequences else np.array([]).reshape(0, seq_len, 0)

    def __len__(self):
        return len(self.obs_sequences)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.obs_sequences[idx], dtype=torch.float32),
            torch.tensor(self.action_sequences[idx], dtype=torch.float32),
        )


class BehaviorCloningDataset(Dataset):
    """Dataset for behavioral cloning (single timesteps)."""

    def __init__(self, observations: np.ndarray, actions: np.ndarray):
        self.observations = torch.tensor(observations, dtype=torch.float32)
        self.actions = torch.tensor(actions, dtype=torch.float32)

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        return self.observations[idx], self.actions[idx]


def collect_random_data(env, num_episodes: int, max_steps: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Collect data using random actions."""
    all_obs, all_actions = [], []
    
    for _ in range(num_episodes):
        obs, _ = env.reset()
        ep_obs, ep_actions = [obs.copy()], []
        
        for _ in range(max_steps):
            action = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(action)
            ep_obs.append(obs.copy())
            ep_actions.append(action.copy())
            if terminated or truncated:
                break
        
        all_obs.append(np.array(ep_obs[:-1]))
        all_actions.append(np.array(ep_actions))
    
    return all_obs, all_actions


def collect_expert_data(
    controller,
    env,
    num_episodes: int,
    max_steps: int,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Collect data using a trained controller (SB3-compatible predict interface)."""
    controller.eval()
    all_obs, all_actions = [], []

    for _ in range(num_episodes):
        obs, _ = env.reset()
        controller.reset_state()
        ep_obs, ep_actions = [], []
        
        for _ in range(max_steps):
            ep_obs.append(obs.copy())
            action, _ = controller.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            ep_actions.append(action.copy())
            if terminated or truncated:
                break
        
        all_obs.append(np.array(ep_obs))
        all_actions.append(np.array(ep_actions))

    return all_obs, all_actions


def evaluate_controller(
    controller,
    env,
    num_episodes: int,
    max_steps: int,
) -> float:
    """Evaluate controller using SB3-compatible predict interface, returns success rate."""
    controller.eval()
    successes = 0

    for _ in range(num_episodes):
        obs, _ = env.reset()
        controller.reset_state()
        
        for _ in range(max_steps):
            action, _ = controller.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                if info.get("phase") == "done":
                    successes += 1
                break

    return successes / num_episodes


def train_teacher_bc(
    teacher: nn.Module,
    env,
    num_epochs: int = 100,
    num_data_episodes: int = 500,
    max_steps: int = 300,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: torch.device = None,
    verbose: bool = True,
) -> List[float]:
    """Train MLP teacher using behavioral cloning with DAgger-style data aggregation.
    
    Returns list of training losses.
    """
    if device is None:
        device = torch.device("cpu")
    
    teacher = teacher.to(device)
    optimizer = optim.Adam(teacher.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # Initial random data
    obs_list, action_list = collect_random_data(env, num_data_episodes // 2, max_steps)
    all_obs = np.vstack(obs_list)
    all_actions = np.vstack(action_list)
    
    losses = []
    best_success = 0
    
    for epoch in range(num_epochs):
        dataset = BehaviorCloningDataset(all_obs, all_actions)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        teacher.train()
        epoch_loss = 0
        for obs, act in loader:
            obs, act = obs.to(device), act.to(device)
            optimizer.zero_grad()
            action, _ = teacher.forward(obs)
            loss = criterion(action, act)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        losses.append(epoch_loss / len(loader))
        
        if (epoch + 1) % 10 == 0:
            teacher.eval()
            success = evaluate_controller(teacher, env, 10, max_steps)
            teacher.train()
            if verbose:
                print(f"Teacher epoch {epoch+1}: loss={losses[-1]:.4f}, success={success:.1%}")
            
            # Aggregate expert data if improving
            if success > best_success:
                best_success = success
                new_obs, new_actions = collect_expert_data(
                    teacher, env, num_data_episodes // 10, max_steps
                )
                all_obs = np.vstack([all_obs, np.vstack(new_obs)])
                all_actions = np.vstack([all_actions, np.vstack(new_actions)])
    
    return losses


def train_student_distillation(
    student: nn.Module,
    teacher: nn.Module,
    env,
    num_epochs: int = 200,
    num_data_episodes: int = 500,
    max_steps: int = 300,
    seq_len: int = 50,
    batch_size: int = 32,
    lr: float = 1e-4,
    device: torch.device = None,
    verbose: bool = True,
) -> Tuple[List[float], List[Tuple[int, float]]]:
    """Train RNN student by distilling from teacher using SB3-compatible interface.
    
    Both teacher and student use the predict() method for inference.
    
    Returns (losses, [(epoch, success_rate), ...])
    """
    if device is None:
        device = torch.device("cpu")
    
    student = student.to(device)
    teacher = teacher.to(device).eval()
    
    optimizer = optim.Adam(student.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # Collect expert data from teacher using predict interface
    obs_list, action_list = collect_expert_data(teacher, env, num_data_episodes, max_steps)
    dataset = TrajectoryDataset(obs_list, action_list, seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    if verbose:
        print(f"Training on {len(dataset)} sequences")
    
    losses = []
    success_history = []
    best_success, best_state = 0, None
    
    for epoch in range(num_epochs):
        student.train()
        epoch_loss = 0
        
        for obs_seq, target_actions in loader:
            obs_seq = obs_seq.to(device)
            target_actions = target_actions.to(device)
            
            # Reset student state and run forward pass for each timestep
            batch_size_actual = obs_seq.shape[0]
            student.reset_state(batch_size_actual, device)
            
            pred_actions = []
            for t in range(obs_seq.shape[1]):
                action, _ = student.forward(obs_seq[:, t, :])
                pred_actions.append(action)
            pred_actions = torch.stack(pred_actions, dim=1)
            
            # Get teacher actions for comparison
            with torch.no_grad():
                teacher_actions = []
                for t in range(obs_seq.shape[1]):
                    action, _ = teacher.forward(obs_seq[:, t, :])
                    teacher_actions.append(action)
                teacher_actions = torch.stack(teacher_actions, dim=1)
            
            loss = criterion(pred_actions, teacher_actions)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        
        losses.append(epoch_loss / max(len(loader), 1))
        
        if (epoch + 1) % 10 == 0:
            student.eval()
            success = evaluate_controller(student, env, 10, max_steps)
            success_history.append((epoch + 1, success))
            
            if verbose:
                print(f"Student epoch {epoch+1}: loss={losses[-1]:.4f}, success={success:.1%}")
            
            if success > best_success:
                best_success = success
                best_state = {k: v.clone() for k, v in student.state_dict().items()}
    
    if best_state:
        student.load_state_dict(best_state)
    
    return losses, success_history


# -----------------------------------------------------------------------------
# Stable Baselines 3 Integration
# -----------------------------------------------------------------------------

def train_with_sb3_bc(
    student: nn.Module,
    teacher: nn.Module,
    env,
    num_demonstrations: int = 1000,
    max_steps: int = 300,
    batch_size: int = 32,
    num_epochs: int = 100,
    device: torch.device = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Train student using Stable Baselines 3 behavioral cloning approach.
    
    This uses the SB3-compatible predict() interface on both teacher and student.
    
    Args:
        student: Student controller with predict() method
        teacher: Teacher controller with predict() method
        env: Gymnasium environment
        num_demonstrations: Number of episodes to collect from teacher
        max_steps: Max steps per episode
        batch_size: Training batch size
        num_epochs: Number of training epochs
        device: Torch device
        verbose: Print progress
        
    Returns:
        Dictionary with training history
    """
    if not HAS_SB3:
        raise ImportError("stable_baselines3 is required. Install with: pip install stable-baselines3")
    
    if device is None:
        device = torch.device("cpu")
    
    student = student.to(device)
    teacher = teacher.to(device).eval()
    
    # Collect demonstrations from teacher
    if verbose:
        print(f"Collecting {num_demonstrations} demonstrations from teacher...")
    
    obs_list, action_list = collect_expert_data(teacher, env, num_demonstrations, max_steps)
    all_obs = np.vstack(obs_list)
    all_actions = np.vstack(action_list)
    
    if verbose:
        print(f"Collected {len(all_obs)} observation-action pairs")
    
    # Create dataset and train
    dataset = BehaviorCloningDataset(all_obs, all_actions)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = optim.Adam(student.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    
    history = {'losses': [], 'success_rates': []}
    best_state = None
    best_success = 0
    
    for epoch in range(num_epochs):
        student.train()
        epoch_loss = 0
        
        for obs_batch, action_batch in loader:
            obs_batch = obs_batch.to(device)
            action_batch = action_batch.to(device)
            
            optimizer.zero_grad()
            pred_action, _ = student.forward(obs_batch)
            loss = criterion(pred_action, action_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(loader)
        history['losses'].append(avg_loss)
        
        # Evaluate periodically
        if (epoch + 1) % 10 == 0:
            student.eval()
            success = evaluate_controller(student, env, 20, max_steps)
            history['success_rates'].append((epoch + 1, success))
            
            if verbose:
                print(f"Epoch {epoch+1}: loss={avg_loss:.4f}, success={success:.1%}")
            
            if success > best_success:
                best_success = success
                best_state = {k: v.clone() for k, v in student.state_dict().items()}
    
    if best_state:
        student.load_state_dict(best_state)
    
    return history
