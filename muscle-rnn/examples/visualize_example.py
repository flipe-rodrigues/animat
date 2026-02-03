#%% [markdown]
# # Visualization Example
# 
# Load a trained controller and visualize its behavior.
# Shows trajectory plots and network activity.

#%% Imports
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# Project root - works from both scripts and interactive notebooks
project_root = Path(__file__).parent.parent if '__file__' in dir() else Path.cwd()
if not (project_root / "mujoco").exists():
    project_root = Path(r"f:\\Documents\\GitHub\\python\\animat\\muscle-rnn")

import sys
sys.path.insert(0, str(project_root))

from plants.mujoco import MuJoCoPlant
from envs.reaching import ReachingEnv
from models.controllers import RNNController, ControllerConfig

#%% Setup
xml_path = str(project_root / "mujoco" / "two-joint-planar-arm.xml")
checkpoint_path = str(project_root / "outputs" / "cmaes_example" / "controller.pt")

plant = MuJoCoPlant(xml_path)
print(f"Model: {plant.model_name}")
print(f"  Muscles: {plant.num_muscles}, Sensors: {plant.num_sensors}")

print("\nCalibrating sensors...")
sensor_stats = plant.calibrate(num_episodes=30)

#%% Load or create controller
def load_checkpoint(path: str, plant):
    """Load controller from checkpoint."""
    ckpt = torch.load(path, map_location='cpu')
    
    # Handle both old (with ModelConfig) and new formats
    if 'model_config' in ckpt:
        cfg = ckpt['model_config']
        config = ControllerConfig(
            num_muscles=cfg.get('num_muscles', plant.num_muscles),
            num_core_units=cfg.get('num_core_units', cfg.get('rnn_hidden_size', 32)),
            target_grid_size=cfg.get('target_grid_size', 4),
        )
    else:
        config = ControllerConfig(
            num_muscles=ckpt.get('num_muscles', plant.num_muscles),
            num_core_units=ckpt.get('num_core_units', ckpt.get('rnn_hidden_size', 32)),
            target_grid_size=ckpt.get('target_grid_size', 4),
        )
    
    controller = RNNController(config)
    
    # Load state dict
    if 'model_state_dict' in ckpt:
        controller.load_state_dict(ckpt['model_state_dict'])
    elif 'state_dict' in ckpt:
        controller.load_state_dict(ckpt['state_dict'])
    
    controller.eval()
    return controller, ckpt

try:
    controller, ckpt = load_checkpoint(checkpoint_path, plant)
    print(f"\nLoaded from {checkpoint_path}")
    print(f"  Fitness: {ckpt.get('fitness', 'N/A')}")
except FileNotFoundError:
    print(f"\nCheckpoint not found: {checkpoint_path}")
    print("Creating random controller for demo...")
    config = ControllerConfig(
        num_muscles=plant.num_muscles,
        num_core_units=32,
        target_grid_size=4,
    )
    controller = RNNController(config)
    controller.eval()

#%% Record an episode with trajectory data
def record_episode(controller, env, num_muscles, max_steps=300, seed=None):
    """Record episode with full trajectory data."""
    device = torch.device("cpu")
    obs, info = env.reset(seed=seed)
    controller._reset_state()
    
    data = {
        "observations": [obs.copy()],
        "actions": [],
        "rewards": [],
        "hand_positions": [info["hand_position"].copy()],
        "target_position": info["target_position"].copy(),
        "alpha": [],
        "gamma_static": [],
        "gamma_dynamic": [],
        "muscle_lengths": [],
    }
    
    with torch.no_grad():
        for step in range(max_steps):
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            action, step_info = controller.forward(obs_t)
            action_np = action.squeeze(0).numpy()
            
            data["actions"].append(action_np.copy())
            data["alpha"].append(step_info["alpha"].squeeze(0).numpy().copy())
            data["gamma_static"].append(step_info["gamma_static"].squeeze(0).numpy().copy())
            data["gamma_dynamic"].append(step_info["gamma_dynamic"].squeeze(0).numpy().copy())
            data["muscle_lengths"].append(obs[:num_muscles].copy())
            
            obs, reward, done, trunc, info = env.step(action_np)
            
            data["observations"].append(obs.copy())
            data["rewards"].append(reward)
            data["hand_positions"].append(info["hand_position"].copy())
            
            if done or trunc:
                break
    
    # Convert to arrays
    for k in ["observations", "actions", "rewards", "hand_positions", "alpha", "gamma_static", "gamma_dynamic", "muscle_lengths"]:
        data[k] = np.array(data[k])
    
    data["success"] = info.get("phase") == "done"
    data["final_phase"] = info.get("phase", "running")
    return data

#%% Record episodes
env = ReachingEnv(xml_path, render_mode="rgb_array", sensor_stats=sensor_stats)

num_episodes = 5
episodes = []
for i in range(num_episodes):
    data = record_episode(controller, env, plant.num_muscles, max_steps=300, seed=i*42)
    episodes.append(data)
    print(f"Episode {i+1}: reward={sum(data['rewards']):.1f}, steps={len(data['actions'])}, success={data['success']}")

env.close()

#%% Plot trajectories
fig, axes = plt.subplots(2, 3, figsize=(14, 9))

# Trajectory plot
ax = axes[0, 0]
colors = plt.cm.viridis(np.linspace(0, 1, num_episodes))
for i, ep in enumerate(episodes):
    pos = ep["hand_positions"]
    ax.plot(pos[:, 0], pos[:, 1], color=colors[i], alpha=0.8, label=f'Ep {i+1}')
    ax.plot(pos[0, 0], pos[0, 1], 'o', color=colors[i], markersize=8)
    ax.plot(pos[-1, 0], pos[-1, 1], 's', color=colors[i], markersize=6)
ax.plot(episodes[0]["target_position"][0], episodes[0]["target_position"][1], 
        'r*', markersize=15, label='Target')
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_title("Hand Trajectories")
ax.legend(fontsize=8)
ax.axis('equal')
ax.grid(True)

# Rewards over time
ax = axes[0, 1]
for i, ep in enumerate(episodes):
    ax.plot(np.cumsum(ep["rewards"]), color=colors[i], alpha=0.8)
ax.set_xlabel("Step")
ax.set_ylabel("Cumulative Reward")
ax.set_title("Reward Accumulation")
ax.grid(True)

# Distance to target
ax = axes[0, 2]
for i, ep in enumerate(episodes):
    dist = np.linalg.norm(ep["hand_positions"] - ep["target_position"], axis=1)
    ax.plot(dist, color=colors[i], alpha=0.8)
ax.set_xlabel("Step")
ax.set_ylabel("Distance")
ax.set_title("Distance to Target")
ax.grid(True)

# Muscle activations (first episode)
ax = axes[1, 0]
alpha = episodes[0]["alpha"]
im = ax.imshow(alpha.T, aspect='auto', cmap='hot', vmin=0, vmax=1)
ax.set_xlabel("Step")
ax.set_ylabel("Muscle")
ax.set_title("Muscle Activations (Ep 1)")
plt.colorbar(im, ax=ax)

# Muscle lengths (first episode)
ax = axes[1, 1]
lengths = episodes[0]["muscle_lengths"]
im = ax.imshow(lengths.T, aspect='auto', cmap='viridis')
ax.set_xlabel("Step")
ax.set_ylabel("Muscle")
ax.set_title("Muscle Lengths (Ep 1)")
plt.colorbar(im, ax=ax)

# Gamma static (first episode)
ax = axes[1, 2]
gamma_static = episodes[0]["gamma_static"]
im = ax.imshow(gamma_static.T, aspect='auto', cmap='RdBu', vmin=0, vmax=1)
ax.set_xlabel("Step")
ax.set_ylabel("Muscle")
ax.set_title("Gamma Static (Ep 1)")
plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.show()

#%% Plot single episode in detail
ep = episodes[0]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# XY trajectory with time coloring
ax = axes[0, 0]
pos = ep["hand_positions"]
target = ep["target_position"]
scatter = ax.scatter(pos[:, 0], pos[:, 1], c=range(len(pos)), cmap='viridis', s=20)
ax.plot(pos[0, 0], pos[0, 1], 'go', markersize=12, label='Start', zorder=5)
ax.plot(target[0], target[1], 'r*', markersize=20, label='Target', zorder=5)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_title(f"Trajectory (success={ep['success']})")
ax.legend()
ax.axis('equal')
ax.grid(True)
plt.colorbar(scatter, ax=ax, label='Step')

# Individual muscle activations
ax = axes[0, 1]
for m in range(min(6, plant.num_muscles)):
    ax.plot(ep["alpha"][:, m], label=f'M{m}')
ax.set_xlabel("Step")
ax.set_ylabel("Activation")
ax.set_title("Muscle Activations")
ax.legend(fontsize=8, ncol=2)
ax.grid(True)

# Position components over time
ax = axes[1, 0]
ax.plot(pos[:, 0], 'b-', label='X')
ax.plot(pos[:, 1], 'r-', label='Y')
ax.axhline(target[0], color='b', linestyle='--', alpha=0.5)
ax.axhline(target[1], color='r', linestyle='--', alpha=0.5)
ax.set_xlabel("Step")
ax.set_ylabel("Position")
ax.set_title("Position Components")
ax.legend()
ax.grid(True)

# Rewards per step
ax = axes[1, 1]
ax.bar(range(len(ep["rewards"])), ep["rewards"], width=1, alpha=0.7)
ax.set_xlabel("Step")
ax.set_ylabel("Reward")
ax.set_title("Reward per Step")
ax.grid(True)

plt.tight_layout()
plt.show()

#%% Success statistics
successes = sum(1 for ep in episodes if ep["success"])
print(f"\n--- Summary ---")
print(f"Success rate: {successes}/{num_episodes} = {successes/num_episodes:.0%}")
print(f"Avg total reward: {np.mean([sum(ep['rewards']) for ep in episodes]):.1f}")
print(f"Avg episode length: {np.mean([len(ep['actions']) for ep in episodes]):.0f} steps")

#%% Cleanup
plant.close()
