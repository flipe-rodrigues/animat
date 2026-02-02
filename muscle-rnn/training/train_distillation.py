"""Distillation Learning for Muscle-Driven Arm Controller.

Trains an RNN by distilling knowledge from a pre-trained MLP teacher.
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

from envs.reaching import ReachingEnv
from envs.plant import parse_mujoco_xml, calibrate_sensors
from models.controllers import RNNController, MLPController
from core.config import ModelConfig
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
    DEFAULT_MAX_EPISODE_STEPS,
    DEFAULT_NUM_EVAL_EPISODES,
    DEFAULT_CALIBRATION_EPISODES,
    DEFAULT_NUM_TARGET_UNITS,
    DEFAULT_RNN_HIDDEN_SIZE,
    DEFAULT_PLOT_DPI,
)


@dataclass
class DistillationConfig:
    """Configuration for distillation training."""
    teacher_hidden_sizes: List[int] = None
    teacher_lr: float = DEFAULT_TEACHER_LR
    teacher_epochs: int = DEFAULT_TEACHER_EPOCHS
    teacher_batch_size: int = DEFAULT_TEACHER_BATCH_SIZE
    teacher_data_episodes: int = DEFAULT_TEACHER_DATA_EPISODES
    student_lr: float = DEFAULT_STUDENT_LR
    student_epochs: int = DEFAULT_STUDENT_EPOCHS
    student_batch_size: int = DEFAULT_STUDENT_BATCH_SIZE
    student_seq_len: int = DEFAULT_STUDENT_SEQ_LEN
    action_loss_weight: float = DEFAULT_ACTION_LOSS_WEIGHT
    max_steps_per_episode: int = DEFAULT_MAX_EPISODE_STEPS
    num_eval_episodes: int = DEFAULT_NUM_EVAL_EPISODES
    xml_path: str = ""
    output_dir: str = "outputs/distillation"

    def __post_init__(self):
        if self.teacher_hidden_sizes is None:
            self.teacher_hidden_sizes = list(DEFAULT_MLP_HIDDEN_SIZES)


class TrajectoryDataset(Dataset):
    """Dataset of trajectory sequences for training."""

    def __init__(self, observations: List[np.ndarray], actions: List[np.ndarray], seq_len: int = DEFAULT_STUDENT_SEQ_LEN):
        self.obs_sequences, self.action_sequences = [], []
        for obs_traj, action_traj in zip(observations, actions):
            for start in range(0, len(obs_traj) - seq_len + 1, seq_len // 2):
                self.obs_sequences.append(obs_traj[start : start + seq_len])
                self.action_sequences.append(action_traj[start : start + seq_len])
        self.obs_sequences = np.array(self.obs_sequences)
        self.action_sequences = np.array(self.action_sequences)

    def __len__(self):
        return len(self.obs_sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.obs_sequences[idx], dtype=torch.float32), torch.tensor(self.action_sequences[idx], dtype=torch.float32)


class BehaviorCloningDataset(Dataset):
    """Dataset for behavioral cloning (single timesteps)."""

    def __init__(self, observations: np.ndarray, actions: np.ndarray):
        self.observations = torch.tensor(observations, dtype=torch.float32)
        self.actions = torch.tensor(actions, dtype=torch.float32)

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        return self.observations[idx], self.actions[idx]


def collect_random_data(xml_path: str, sensor_stats: Dict, num_episodes: int, max_steps: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Collect data using random actions."""
    env = ReachingEnv(xml_path, sensor_stats=sensor_stats)
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

    env.close()
    return all_obs, all_actions


def collect_expert_data(xml_path: str, sensor_stats: Dict, controller: nn.Module, num_episodes: int, max_steps: int, device: torch.device) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Collect data using a trained controller."""
    env = ReachingEnv(xml_path, sensor_stats=sensor_stats)
    controller.eval()
    all_obs, all_actions = [], []

    with torch.no_grad():
        for _ in range(num_episodes):
            obs, _ = env.reset()
            controller.init_hidden(1, device)
            ep_obs, ep_actions = [], []
            for _ in range(max_steps):
                ep_obs.append(obs.copy())
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                action, _, _ = controller.forward(obs_tensor)
                action = action.squeeze(0).cpu().numpy()
                obs, _, terminated, truncated, _ = env.step(action)
                ep_actions.append(action.copy())
                if terminated or truncated:
                    break
            all_obs.append(np.array(ep_obs))
            all_actions.append(np.array(ep_actions))

    env.close()
    return all_obs, all_actions


def evaluate_simple(controller: nn.Module, xml_path: str, sensor_stats: Dict, num_episodes: int, max_steps: int, device: torch.device) -> float:
    """Simple evaluation returning success rate."""
    env = ReachingEnv(xml_path, sensor_stats=sensor_stats)
    controller.eval()
    successes = 0

    with torch.no_grad():
        for _ in range(num_episodes):
            obs, _ = env.reset()
            controller.init_hidden(1, device)
            for _ in range(max_steps):
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                action, _, _ = controller.forward(obs_tensor)
                obs, _, terminated, truncated, info = env.step(action.squeeze(0).cpu().numpy())
                if terminated or truncated:
                    if info.get("phase") == "done":
                        successes += 1
                    break

    env.close()
    return successes / num_episodes


def train_teacher(xml_path: str, model_config: ModelConfig, sensor_stats: Dict, config: DistillationConfig, device: torch.device) -> MLPController:
    """Train MLP teacher using behavioral cloning."""
    print("Training MLP teacher...")
    teacher = MLPController(model_config, config.teacher_hidden_sizes).to(device)
    optimizer = optim.Adam(teacher.parameters(), lr=config.teacher_lr)
    criterion = nn.MSELoss()

    obs_list, action_list = collect_random_data(xml_path, sensor_stats, config.teacher_data_episodes // 2, config.max_steps_per_episode)
    all_obs, all_actions = np.vstack(obs_list), np.vstack(action_list)
    print(f"Collected {len(all_obs)} samples")

    best_success_rate = 0
    for epoch in range(config.teacher_epochs):
        dataset = BehaviorCloningDataset(all_obs, all_actions)
        loader = DataLoader(dataset, batch_size=config.teacher_batch_size, shuffle=True)
        teacher.train()
        epoch_loss = sum(
            (optimizer.zero_grad(), criterion((action := teacher.forward(obs.to(device))[0]), act.to(device)).backward(), optimizer.step(), criterion(action, act.to(device)).item())[-1]
            for obs, act in loader
        ) / len(loader)

        if (epoch + 1) % 10 == 0:
            success_rate = evaluate_simple(teacher, xml_path, sensor_stats, config.num_eval_episodes, config.max_steps_per_episode, device)
            print(f"Teacher Epoch {epoch + 1}: loss={epoch_loss:.4f}, success={success_rate:.2%}")
            if success_rate > best_success_rate:
                best_success_rate = success_rate
                new_obs, new_actions = collect_expert_data(xml_path, sensor_stats, teacher, config.teacher_data_episodes // 10, config.max_steps_per_episode, device)
                all_obs = np.vstack([all_obs, np.vstack(new_obs)])
                all_actions = np.vstack([all_actions, np.vstack(new_actions)])

    return teacher


def train_student(teacher: MLPController, xml_path: str, model_config: ModelConfig, sensor_stats: Dict, config: DistillationConfig, device: torch.device) -> Tuple[RNNController, List, List]:
    """Train RNN student via distillation."""
    print("\nTraining RNN student...")
    student = RNNController(model_config).to(device)
    optimizer = optim.Adam(student.parameters(), lr=config.student_lr)
    criterion = nn.MSELoss()

    obs_list, action_list = collect_expert_data(xml_path, sensor_stats, teacher, config.teacher_data_episodes, config.max_steps_per_episode, device)
    dataset = TrajectoryDataset(obs_list, action_list, config.student_seq_len)
    loader = DataLoader(dataset, batch_size=config.student_batch_size, shuffle=True)
    print(f"Created {len(dataset)} training sequences")

    loss_history, success_history = [], []
    best_success_rate, best_state = 0, None
    teacher.eval()

    for epoch in range(config.student_epochs):
        student.train()
        epoch_loss = 0
        for obs_seq, _ in loader:
            obs_seq = obs_seq.to(device)
            student.init_hidden(obs_seq.shape[0], device)
            pred_actions, _, _ = student.forward_sequence(obs_seq)
            with torch.no_grad():
                teacher_actions = torch.stack([teacher.forward(obs_seq[:, t, :])[0] for t in range(obs_seq.shape[1])], dim=1)
            loss = config.action_loss_weight * criterion(pred_actions, teacher_actions)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        loss_history.append(epoch_loss / len(loader))
        if (epoch + 1) % 10 == 0:
            success_rate = evaluate_simple(student, xml_path, sensor_stats, config.num_eval_episodes, config.max_steps_per_episode, device)
            success_history.append((epoch + 1, success_rate))
            print(f"Student Epoch {epoch + 1}: loss={loss_history[-1]:.4f}, success={success_rate:.2%}")
            if success_rate > best_success_rate:
                best_success_rate, best_state = success_rate, student.state_dict().copy()

    if best_state:
        student.load_state_dict(best_state)
    return student, loss_history, success_history


class DistillationTrainer:
    """Full distillation training pipeline."""

    def __init__(self, config: DistillationConfig, model_config: ModelConfig, sensor_stats: Dict):
        self.config, self.model_config, self.sensor_stats = config, model_config, sensor_stats
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self) -> Dict[str, Any]:
        start = time.time()
        teacher = train_teacher(self.config.xml_path, self.model_config, self.sensor_stats, self.config, self.device)
        torch.save({"model_state_dict": teacher.state_dict(), "model_config": asdict(self.model_config)}, self.output_dir / "teacher_mlp.pt")

        student, loss_history, success_history = train_student(teacher, self.config.xml_path, self.model_config, self.sensor_stats, self.config, self.device)
        torch.save({"model_state_dict": student.state_dict(), "model_config": asdict(self.model_config)}, self.output_dir / "student_rnn.pt")

        teacher_success = evaluate_simple(teacher, self.config.xml_path, self.sensor_stats, 50, self.config.max_steps_per_episode, self.device)
        student_success = evaluate_simple(student, self.config.xml_path, self.sensor_stats, 50, self.config.max_steps_per_episode, self.device)

        self._plot_results(loss_history, success_history, teacher_success, student_success)
        return {"teacher_success_rate": teacher_success, "student_success_rate": student_success, "total_time": time.time() - start}

    def _plot_results(self, loss_history, success_history, teacher_success, student_success):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(loss_history)
        axes[0].set(xlabel="Epoch", ylabel="Loss", title="Student Training Loss")
        axes[0].grid(True)
        if success_history:
            epochs, rates = zip(*success_history)
            axes[1].plot(epochs, rates, "o-", label="Student")
            axes[1].axhline(teacher_success, color="r", linestyle="--", label=f"Teacher ({teacher_success:.2%})")
            axes[1].set(xlabel="Epoch", ylabel="Success Rate", title="Success Rate")
            axes[1].legend()
            axes[1].grid(True)
        plt.tight_layout()
        plt.savefig(self.output_dir / "training_results.png", dpi=DEFAULT_PLOT_DPI)
        plt.close()


def run_distillation_training(xml_path: str, output_dir: str = "outputs/distillation", teacher_epochs: int = DEFAULT_TEACHER_EPOCHS, student_epochs: int = DEFAULT_STUDENT_EPOCHS, calibration_episodes: int = DEFAULT_CALIBRATION_EPISODES // 2) -> Dict[str, Any]:
    """Main entry point for distillation training."""
    parsed_model = parse_mujoco_xml(xml_path)
    print(f"Model: {parsed_model.model_name} (Muscles: {parsed_model.num_muscles})")

    sensor_stats = calibrate_sensors(xml_path, num_episodes=calibration_episodes)
    model_config = ModelConfig(num_muscles=parsed_model.num_muscles, num_sensors=parsed_model.num_sensors, num_target_units=DEFAULT_NUM_TARGET_UNITS, rnn_hidden_size=DEFAULT_RNN_HIDDEN_SIZE)
    train_config = DistillationConfig(xml_path=xml_path, output_dir=output_dir, teacher_epochs=teacher_epochs, student_epochs=student_epochs)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / "sensor_stats.pkl", "wb") as f:
        pickle.dump(sensor_stats, f)
    with open(output_path / "model_config.json", "w") as f:
        json.dump(asdict(model_config), f, indent=2)

    return DistillationTrainer(train_config, model_config, sensor_stats).train()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("xml_path")
    parser.add_argument("--output-dir", default="outputs/distillation")
    parser.add_argument("--teacher-epochs", type=int, default=DEFAULT_TEACHER_EPOCHS)
    parser.add_argument("--student-epochs", type=int, default=DEFAULT_STUDENT_EPOCHS)
    args = parser.parse_args()

    results = run_distillation_training(args.xml_path, args.output_dir, args.teacher_epochs, args.student_epochs)
    print(f"\nTeacher: {results['teacher_success_rate']:.2%}, Student: {results['student_success_rate']:.2%}")
