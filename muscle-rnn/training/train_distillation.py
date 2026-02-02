"""
Distillation Learning Training for Muscle-Driven Arm Controller

Trains an RNN by distilling knowledge from a pre-trained MLP teacher.
The MLP is first trained using standard RL or behavioral cloning,
then the RNN learns to imitate the MLP's behavior.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import pickle
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, asdict
import time
import matplotlib.pyplot as plt
from collections import deque

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from envs.reaching import ReachingEnv
from models.controllers import RNNController, MLPController, create_controller
from core.config import ModelConfig
from envs.plant import parse_mujoco_xml, calibrate_sensors
from core.constants import (
    DEFAULT_MLP_HIDDEN_SIZES,
    DEFAULT_TEACHER_LR,
    DEFAULT_TEACHER_EPOCHS,
    DEFAULT_TEACHER_BATCH_SIZE,
    DEFAULT_TEACHER_DATA_EPISODES,
    DEFAULT_STUDENT_LR,
    DEFAULT_STUDENT_EPOCHS,
    DEFAULT_STUDENT_BATCH_SIZE,
    DEFAULT_STUDENT_SEQ_LEN,
    DEFAULT_ACTION_LOSS_WEIGHT,
    DEFAULT_HIDDEN_LOSS_WEIGHT,
    DEFAULT_MAX_EPISODE_STEPS,
    DEFAULT_NUM_EVAL_EPISODES,
    DEFAULT_CHECKPOINT_EVERY,
    DEFAULT_CALIBRATION_EPISODES,
)


@dataclass
class DistillationConfig:
    """Configuration for distillation training."""
    # Teacher training (MLP)
    teacher_hidden_sizes: List[int] = None
    teacher_lr: float = DEFAULT_TEACHER_LR
    teacher_epochs: int = DEFAULT_TEACHER_EPOCHS
    teacher_batch_size: int = DEFAULT_TEACHER_BATCH_SIZE
    teacher_data_episodes: int = DEFAULT_TEACHER_DATA_EPISODES
    
    # Student training (RNN)
    student_lr: float = DEFAULT_STUDENT_LR
    student_epochs: int = DEFAULT_STUDENT_EPOCHS
    student_batch_size: int = DEFAULT_STUDENT_BATCH_SIZE
    student_seq_len: int = DEFAULT_STUDENT_SEQ_LEN
    
    # Loss weights
    action_loss_weight: float = DEFAULT_ACTION_LOSS_WEIGHT
    hidden_loss_weight: float = DEFAULT_HIDDEN_LOSS_WEIGHT
    
    # Training parameters
    max_steps_per_episode: int = DEFAULT_MAX_EPISODE_STEPS
    n_eval_episodes: int = DEFAULT_NUM_EVAL_EPISODES
    
    # Checkpointing
    checkpoint_freq: int = DEFAULT_CHECKPOINT_EVERY
    
    # Paths
    xml_path: str = ""
    output_dir: str = "outputs/distillation"
    
    def __post_init__(self):
        if self.teacher_hidden_sizes is None:
            self.teacher_hidden_sizes = list(DEFAULT_MLP_HIDDEN_SIZES)


class TrajectoryDataset(Dataset):
    """Dataset of trajectory sequences for training."""
    
    def __init__(
        self,
        observations: List[np.ndarray],
        actions: List[np.ndarray],
        seq_len: int = DEFAULT_STUDENT_SEQ_LEN
    ):
        self.seq_len = seq_len
        
        # Stack all trajectories
        self.obs_sequences = []
        self.action_sequences = []
        
        for obs_traj, action_traj in zip(observations, actions):
            traj_len = len(obs_traj)
            
            # Create overlapping sequences
            for start in range(0, traj_len - seq_len + 1, seq_len // 2):
                end = start + seq_len
                self.obs_sequences.append(obs_traj[start:end])
                self.action_sequences.append(action_traj[start:end])
        
        self.obs_sequences = np.array(self.obs_sequences)
        self.action_sequences = np.array(self.action_sequences)
    
    def __len__(self):
        return len(self.obs_sequences)
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.obs_sequences[idx], dtype=torch.float32),
            torch.tensor(self.action_sequences[idx], dtype=torch.float32)
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


def collect_random_data(
    xml_path: str,
    sensor_stats: Dict,
    n_episodes: int,
    max_steps: int
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Collect data using random actions."""
    env = ReachingEnv(xml_path, sensor_stats=sensor_stats)
    
    all_obs = []
    all_actions = []
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        
        ep_obs = [obs.copy()]
        ep_actions = []
        
        for step in range(max_steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            ep_obs.append(obs.copy())
            ep_actions.append(action.copy())
            
            if terminated or truncated:
                break
        
        all_obs.append(np.array(ep_obs[:-1]))  # Exclude last obs
        all_actions.append(np.array(ep_actions))
    
    env.close()
    return all_obs, all_actions


def collect_expert_data(
    xml_path: str,
    sensor_stats: Dict,
    controller: nn.Module,
    n_episodes: int,
    max_steps: int,
    device: torch.device
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Collect data using a trained controller."""
    env = ReachingEnv(xml_path, sensor_stats=sensor_stats)
    controller.eval()
    
    all_obs = []
    all_actions = []
    
    with torch.no_grad():
        for ep in range(n_episodes):
            obs, _ = env.reset()
            controller.init_hidden(1, device)
            
            ep_obs = []
            ep_actions = []
            
            for step in range(max_steps):
                ep_obs.append(obs.copy())
                
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                action, _, _ = controller.forward(obs_tensor)
                action = action.squeeze(0).cpu().numpy()
                
                obs, reward, terminated, truncated, info = env.step(action)
                ep_actions.append(action.copy())
                
                if terminated or truncated:
                    break
            
            all_obs.append(np.array(ep_obs))
            all_actions.append(np.array(ep_actions))
    
    env.close()
    return all_obs, all_actions


def train_teacher_mlp(
    xml_path: str,
    model_config: ModelConfig,
    sensor_stats: Dict,
    config: DistillationConfig,
    device: torch.device
) -> MLPController:
    """
    Train the MLP teacher using behavioral cloning with data augmentation.
    
    The teacher learns from a combination of:
    1. Random exploration data
    2. Iterative self-improvement (DAgger-like)
    """
    print("Training MLP teacher...")
    
    # Create teacher
    teacher = MLPController(model_config, config.teacher_hidden_sizes).to(device)
    optimizer = optim.Adam(teacher.parameters(), lr=config.teacher_lr)
    criterion = nn.MSELoss()
    
    # Initial random data collection
    print("Collecting initial random data...")
    obs_list, action_list = collect_random_data(
        xml_path, sensor_stats,
        n_episodes=config.teacher_data_episodes // 2,
        max_steps=config.max_steps_per_episode
    )
    
    # Flatten for behavioral cloning
    all_obs = np.vstack(obs_list)
    all_actions = np.vstack(action_list)
    
    print(f"Collected {len(all_obs)} samples")
    
    # Training loop with iterative data collection
    best_success_rate = 0
    
    for epoch in range(config.teacher_epochs):
        # Create dataset and loader
        dataset = BehaviorCloningDataset(all_obs, all_actions)
        loader = DataLoader(dataset, batch_size=config.teacher_batch_size, shuffle=True)
        
        # Train epoch
        teacher.train()
        epoch_loss = 0
        n_batches = 0
        
        for obs_batch, action_batch in loader:
            obs_batch = obs_batch.to(device)
            action_batch = action_batch.to(device)
            
            # Forward pass
            pred_action, _, _ = teacher.forward(obs_batch)
            
            # Loss
            loss = criterion(pred_action, action_batch)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_loss = epoch_loss / n_batches
        
        # Evaluate
        if (epoch + 1) % 10 == 0:
            success_rate = evaluate_controller_simple(
                teacher, xml_path, model_config, sensor_stats,
                n_episodes=config.n_eval_episodes,
                max_steps=config.max_steps_per_episode,
                device=device
            )
            
            print(f"Teacher Epoch {epoch+1}: loss={avg_loss:.4f}, success_rate={success_rate:.2%}")
            
            # Collect more data with current policy (DAgger-style)
            if success_rate > best_success_rate:
                best_success_rate = success_rate
                
                # Collect expert data
                new_obs, new_actions = collect_expert_data(
                    xml_path, sensor_stats, teacher,
                    n_episodes=config.teacher_data_episodes // 10,
                    max_steps=config.max_steps_per_episode,
                    device=device
                )
                
                # Add to dataset
                new_obs_flat = np.vstack(new_obs)
                new_actions_flat = np.vstack(new_actions)
                
                all_obs = np.vstack([all_obs, new_obs_flat])
                all_actions = np.vstack([all_actions, new_actions_flat])
                
                print(f"  Added {len(new_obs_flat)} expert samples, total: {len(all_obs)}")
    
    print(f"Teacher training complete, best success rate: {best_success_rate:.2%}")
    return teacher


def evaluate_controller_simple(
    controller: nn.Module,
    xml_path: str,
    model_config: ModelConfig,
    sensor_stats: Dict,
    n_episodes: int,
    max_steps: int,
    device: torch.device
) -> float:
    """Simple evaluation returning success rate."""
    env = ReachingEnv(xml_path, sensor_stats=sensor_stats)
    controller.eval()
    
    successes = 0
    
    with torch.no_grad():
        for ep in range(n_episodes):
            obs, _ = env.reset()
            controller.init_hidden(1, device)
            
            for step in range(max_steps):
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                action, _, _ = controller.forward(obs_tensor)
                action = action.squeeze(0).cpu().numpy()
                
                obs, reward, terminated, truncated, info = env.step(action)
                
                if terminated or truncated:
                    if info.get('phase') == 'done':
                        successes += 1
                    break
    
    env.close()
    return successes / n_episodes


def train_student_rnn(
    teacher: MLPController,
    xml_path: str,
    model_config: ModelConfig,
    sensor_stats: Dict,
    config: DistillationConfig,
    device: torch.device
) -> RNNController:
    """
    Train the RNN student to imitate the MLP teacher.
    
    The student learns to:
    1. Match the teacher's actions
    2. Develop temporal representations through recurrence
    """
    print("\nTraining RNN student via distillation...")
    
    # Create student
    student = RNNController(model_config).to(device)
    optimizer = optim.Adam(student.parameters(), lr=config.student_lr)
    action_criterion = nn.MSELoss()
    
    # Collect teacher demonstrations
    print("Collecting teacher demonstrations...")
    obs_list, action_list = collect_expert_data(
        xml_path, sensor_stats, teacher,
        n_episodes=config.teacher_data_episodes,
        max_steps=config.max_steps_per_episode,
        device=device
    )
    
    # Create sequence dataset
    dataset = TrajectoryDataset(obs_list, action_list, seq_len=config.student_seq_len)
    loader = DataLoader(dataset, batch_size=config.student_batch_size, shuffle=True)
    
    print(f"Created {len(dataset)} training sequences")
    
    # Training history
    loss_history = []
    success_history = []
    
    best_success_rate = 0
    best_student_state = None
    
    teacher.eval()
    
    for epoch in range(config.student_epochs):
        student.train()
        epoch_loss = 0
        n_batches = 0
        
        for obs_seq, action_seq in loader:
            obs_seq = obs_seq.to(device)
            action_seq = action_seq.to(device)
            
            batch_size = obs_seq.shape[0]
            seq_len = obs_seq.shape[1]
            
            # Initialize hidden state
            student.init_hidden(batch_size, device)
            
            # Forward through sequence
            pred_actions, _, _ = student.forward_sequence(obs_seq)
            
            # Get teacher actions for comparison
            with torch.no_grad():
                teacher_actions = []
                for t in range(seq_len):
                    t_action, _, _ = teacher.forward(obs_seq[:, t, :])
                    teacher_actions.append(t_action)
                teacher_actions = torch.stack(teacher_actions, dim=1)
            
            # Action matching loss
            action_loss = action_criterion(pred_actions, teacher_actions)
            
            # Total loss
            loss = config.action_loss_weight * action_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_loss = epoch_loss / n_batches
        loss_history.append(avg_loss)
        
        # Evaluate
        if (epoch + 1) % 10 == 0:
            success_rate = evaluate_controller_simple(
                student, xml_path, model_config, sensor_stats,
                n_episodes=config.n_eval_episodes,
                max_steps=config.max_steps_per_episode,
                device=device
            )
            success_history.append((epoch + 1, success_rate))
            
            print(f"Student Epoch {epoch+1}: loss={avg_loss:.4f}, success_rate={success_rate:.2%}")
            
            if success_rate > best_success_rate:
                best_success_rate = success_rate
                best_student_state = student.state_dict().copy()
    
    # Restore best student
    if best_student_state is not None:
        student.load_state_dict(best_student_state)
    
    print(f"\nStudent training complete, best success rate: {best_success_rate:.2%}")
    
    return student, loss_history, success_history


class DistillationTrainer:
    """Full distillation training pipeline."""
    
    def __init__(self, config: DistillationConfig, model_config: ModelConfig, sensor_stats: Dict):
        self.config = config
        self.model_config = model_config
        self.sensor_stats = sensor_stats
        
        # Create output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
    
    def train(self) -> Dict[str, Any]:
        """Run full distillation training."""
        start_time = time.time()
        
        # Stage 1: Train MLP teacher
        teacher = train_teacher_mlp(
            self.config.xml_path,
            self.model_config,
            self.sensor_stats,
            self.config,
            self.device
        )
        
        # Save teacher
        torch.save({
            'model_state_dict': teacher.state_dict(),
            'model_config': asdict(self.model_config),
        }, self.output_dir / 'teacher_mlp.pt')
        
        # Stage 2: Train RNN student
        student, loss_history, success_history = train_student_rnn(
            teacher,
            self.config.xml_path,
            self.model_config,
            self.sensor_stats,
            self.config,
            self.device
        )
        
        # Save student
        torch.save({
            'model_state_dict': student.state_dict(),
            'model_config': asdict(self.model_config),
        }, self.output_dir / 'student_rnn.pt')
        
        # Final evaluation
        teacher_success = evaluate_controller_simple(
            teacher, self.config.xml_path, self.model_config, self.sensor_stats,
            n_episodes=50, max_steps=self.config.max_steps_per_episode, device=self.device
        )
        
        student_success = evaluate_controller_simple(
            student, self.config.xml_path, self.model_config, self.sensor_stats,
            n_episodes=50, max_steps=self.config.max_steps_per_episode, device=self.device
        )
        
        total_time = time.time() - start_time
        
        # Plot results
        self._plot_results(loss_history, success_history, teacher_success, student_success)
        
        results = {
            'teacher_success_rate': teacher_success,
            'student_success_rate': student_success,
            'loss_history': loss_history,
            'success_history': success_history,
            'total_time': total_time,
        }
        
        # Save results
        with open(self.output_dir / 'results.json', 'w') as f:
            json.dump({
                'teacher_success_rate': teacher_success,
                'student_success_rate': student_success,
                'total_time': total_time,
            }, f, indent=2)
        
        return results
    
    def _plot_results(
        self,
        loss_history: List[float],
        success_history: List[Tuple[int, float]],
        teacher_success: float,
        student_success: float
    ):
        """Plot training results."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss curve
        axes[0].plot(loss_history)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Distillation Loss')
        axes[0].set_title('Student Training Loss')
        axes[0].grid(True)
        
        # Success rate
        if success_history:
            epochs, rates = zip(*success_history)
            axes[1].plot(epochs, rates, 'o-', label='Student')
            axes[1].axhline(y=teacher_success, color='r', linestyle='--', label=f'Teacher ({teacher_success:.2%})')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Success Rate')
            axes[1].set_title('Success Rate During Training')
            axes[1].legend()
            axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_results.png', dpi=150)
        plt.close()


def run_distillation_training(
    xml_path: str,
    output_dir: str = "outputs/distillation",
    teacher_epochs: int = DEFAULT_TEACHER_EPOCHS,
    student_epochs: int = DEFAULT_STUDENT_EPOCHS,
    calibration_episodes: int = DEFAULT_CALIBRATION_EPISODES // 2
) -> Dict[str, Any]:
    """
    Main entry point for distillation training.
    
    Args:
        xml_path: Path to MuJoCo XML file
        output_dir: Output directory for checkpoints
        teacher_epochs: Epochs for teacher training
        student_epochs: Epochs for student training
        calibration_episodes: Episodes for sensor calibration
        
    Returns:
        Training results
    """
    # Parse model
    parsed_model = parse_mujoco_xml(xml_path)
    print(f"Model: {parsed_model.model_name}")
    print(f"Joints: {parsed_model.n_joints}")
    print(f"Muscles: {parsed_model.n_muscles}")
    
    # Calibrate sensors
    print("Calibrating sensors...")
    sensor_stats = calibrate_sensors(xml_path, n_episodes=calibration_episodes)
    
    # Create model config
    model_config = ModelConfig(
        n_muscles=parsed_model.n_muscles,
        n_joints=parsed_model.n_joints,
        n_target_units=25,
        rnn_hidden_size=128,
        proprioceptive_hidden_size=32,
        target_hidden_size=32,
        output_hidden_size=64,
    )
    
    # Create training config
    train_config = DistillationConfig(
        xml_path=xml_path,
        output_dir=output_dir,
        teacher_epochs=teacher_epochs,
        student_epochs=student_epochs,
    )
    
    # Save configs
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(output_path / 'sensor_stats.pkl', 'wb') as f:
        pickle.dump(sensor_stats, f)
    
    with open(output_path / 'model_config.json', 'w') as f:
        json.dump(asdict(model_config), f, indent=2)
    
    with open(output_path / 'train_config.json', 'w') as f:
        json.dump(asdict(train_config), f, indent=2)
    
    # Train
    trainer = DistillationTrainer(train_config, model_config, sensor_stats)
    results = trainer.train()
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Distillation Training for Muscle Arm')
    parser.add_argument('xml_path', help='Path to MuJoCo XML file')
    parser.add_argument('--output-dir', default='outputs/distillation', help='Output directory')
    parser.add_argument('--teacher-epochs', type=int, default=DEFAULT_TEACHER_EPOCHS, help='Teacher training epochs')
    parser.add_argument('--student-epochs', type=int, default=DEFAULT_STUDENT_EPOCHS, help='Student training epochs')
    
    args = parser.parse_args()
    
    results = run_distillation_training(
        xml_path=args.xml_path,
        output_dir=args.output_dir,
        teacher_epochs=args.teacher_epochs,
        student_epochs=args.student_epochs
    )
    
    print(f"\nTraining complete!")
    print(f"Teacher success rate: {results['teacher_success_rate']:.2%}")
    print(f"Student success rate: {results['student_success_rate']:.2%}")
