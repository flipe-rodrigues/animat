# %% [markdown]
# # Distillation Training Example
#
# Train an RNN controller by distilling from a pre-trained MLP teacher.
# All parameters are specified explicitly - no config objects.

# %% Imports
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from pathlib import Path

# Project root - works from both scripts and interactive notebooks
project_root = Path(__file__).parent.parent if "__file__" in dir() else Path.cwd()
if not (project_root / "mujoco").exists():
    project_root = Path(r"f:\\Documents\\GitHub\\python\\animat\\muscle-rnn")

import sys

sys.path.insert(0, str(project_root))

from plants.mujoco import MuJoCoPlant
from envs.reaching import ReachingEnv
from controllers import Controller, ControllerConfig, create_rnn_controller, create_mlp_controller
from training.train_distillation import (
    train_teacher_bc,
    train_student_distillation,
    evaluate_controller,
    collect_random_data,
    BehaviorCloningDataset,
)

# %% Setup: create plant and calibrate
xml_path = str(project_root / "mujoco" / "two-joint-planar-arm.xml")

plant = MuJoCoPlant(xml_path)
print(f"Model: {plant.model_name}")
print(
    f"  Joints: {plant.num_joints}, Muscles: {plant.num_muscles}, Sensors: {plant.num_sensors}"
)

print("\nCalibrating sensors...")
sensor_stats = plant.calibrate(num_episodes=30)
print("Done.")

# %% Create controllers
teacher_config = ControllerConfig(
    num_muscles=plant.num_muscles,
    core_units=[128, 128],  # list for MLP
    target_grid_size=4,
    target_sigma=0.5,
)
teacher = Controller(teacher_config)

student_config = ControllerConfig(
    num_muscles=plant.num_muscles,
    core_units=32,  # int for RNN
    target_grid_size=4,
    target_sigma=0.5,
)
student = Controller(student_config)

print(f"\nTeacher (MLP) parameters: {teacher.num_params}")
print(f"Student (RNN) parameters: {student.num_params}")

# %%
print("\nTeacher (MLP) weight shapes:")
for name, param in teacher.named_parameters():
    print(f"  {name}: {param.shape}")

print("\nStudent (RNN) weight shapes:")
for name, param in student.named_parameters():
    print(f"  {name}: {param.shape}")


# %% Create environment
env = ReachingEnv(xml_path, sensor_stats=sensor_stats)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("Device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# %% Train MLP teacher
print("\n--- Training MLP Teacher ---")
teacher_losses = train_teacher_bc(
    teacher=teacher,
    env=env,
    num_epochs=30,  # Use 100+ for real training
    num_data_episodes=200,
    max_steps=300,
    batch_size=32,
    lr=1e-3,
    device=device,
)

# %% Evaluate teacher
teacher_success = evaluate_controller(teacher, env, 20, 300)
print(f"\nTeacher success rate: {teacher_success:.1%}")

# %% Train RNN student via distillation
print("\n--- Training RNN Student ---")
student_losses, success_history = train_student_distillation(
    student=student,
    teacher=teacher,
    env=env,
    num_epochs=50,  # Use 200+ for real training
    num_data_episodes=200,
    max_steps=300,
    seq_len=50,
    batch_size=32,
    lr=1e-4,
    device=device,
)

# %% Final evaluation
student_success = evaluate_controller(student, env, 30, 300)
print(f"\n--- Final Results ---")
print(f"Teacher success rate: {teacher_success:.1%}")
print(f"Student success rate: {student_success:.1%}")

# %% Plot training curves
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Teacher loss
axes[0].plot(teacher_losses)
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].set_title("Teacher (MLP) Training")
axes[0].grid(True)

# Student loss
axes[1].plot(student_losses)
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].set_title("Student (RNN) Training")
axes[1].grid(True)

# Success comparison
if success_history:
    epochs, rates = zip(*success_history)
    axes[2].plot(epochs, rates, "b-o", label="Student")
    axes[2].axhline(
        teacher_success,
        color="r",
        linestyle="--",
        label=f"Teacher ({teacher_success:.0%})",
    )
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Success Rate")
    axes[2].set_title("Success Rate")
    axes[2].legend()
    axes[2].grid(True)

plt.tight_layout()
plt.show()

# %% Run a test episode with the student
print("\n--- Test Episode ---")
env_render = ReachingEnv(xml_path, render_mode="rgb_array", sensor_stats=sensor_stats)
student.eval()

obs, info = env_render.reset()
student.reset_state()

positions = [info["hand_position"].copy()]
target_pos = info["target_position"].copy()
episode_reward = 0

for step in range(300):
    action, _ = student.predict(obs, deterministic=True)

    obs, reward, done, trunc, info = env_render.step(action)
    positions.append(info["hand_position"].copy())
    episode_reward += reward

    if done or trunc:
        break

env_render.close()
print(
    f"Episode ended at step {step+1}, reward={episode_reward:.1f}, phase={info.get('phase')}"
)

# %% Plot trajectory
positions = np.array(positions)

plt.figure(figsize=(8, 8))
plt.plot(positions[:, 0], positions[:, 1], "b-", linewidth=2, label="Trajectory")
plt.plot(positions[0, 0], positions[0, 1], "go", markersize=12, label="Start")
plt.plot(positions[-1, 0], positions[-1, 1], "bs", markersize=10, label="End")
plt.plot(target_pos[0], target_pos[1], "r*", markersize=15, label="Target")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Student Controller Reaching Trajectory")
plt.legend()
plt.axis("equal")
plt.grid(True)
plt.show()

# %% Save models
output_dir = project_root / "outputs" / "distillation_example"
output_dir.mkdir(parents=True, exist_ok=True)

torch.save(
    {
        "state_dict": teacher.state_dict(),
        "num_muscles": plant.num_muscles,
        "num_sensors": plant.num_sensors,
        "success_rate": teacher_success,
    },
    output_dir / "teacher.pt",
)

torch.save(
    {
        "state_dict": student.state_dict(),
        "num_muscles": plant.num_muscles,
        "num_sensors": plant.num_sensors,
        "success_rate": student_success,
    },
    output_dir / "student.pt",
)

print(f"\nModels saved to {output_dir}")

# %% Cleanup
env.close()
plant.close()
