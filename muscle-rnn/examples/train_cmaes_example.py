# %% [markdown]
# # CMA-ES Training Example
#
# Train an RNN controller for a muscle-driven arm using CMA-ES.
# Uses the `cmaes` PyPI package (https://pypi.org/project/cmaes/).

# %% Imports
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# Project root - works from both scripts and interactive notebooks
project_root = Path(__file__).parent.parent if "__file__" in dir() else Path.cwd()
if not (project_root / "mujoco").exists():
    project_root = Path(r"f:\\Documents\\GitHub\\python\\animat\\muscle-rnn")

import sys

sys.path.insert(0, str(project_root))

from cmaes import CMA
from plants.mujoco import MuJoCoPlant
from envs.reaching import ReachingEnv
from models.controllers import RNNController, ControllerConfig

# %% Setup: create plant and calibrate
xml_path = str(project_root / "mujoco" / "two-joint-planar-arm.xml")

plant = MuJoCoPlant(xml_path)
print(f"Model: {plant.model_name}")
print(f"  Muscles: {plant.num_muscles}, Sensors: {plant.num_sensors}")

print("\nCalibrating sensors...")
sensor_stats = plant.calibrate(num_episodes=50, max_steps=100)
print("Done.")

# %% Create controller and environment
config = ControllerConfig(
    num_muscles=plant.num_muscles,
    num_core_units=32,
    nonlinearity="tanh",
    target_grid_size=4,
    target_sigma=0.5,
)
controller = RNNController(config)
print(f"\nController parameters: {controller.count_parameters()}")

env = ReachingEnv(xml_path, sensor_stats=sensor_stats)


# %% Evaluation function
def evaluate(ctrl, env, params, num_episodes=3, max_steps=300):
    """Evaluate parameters on the reaching task."""
    ctrl.set_flat_params(params)
    ctrl.eval()

    total_reward = 0.0
    for _ in range(num_episodes):
        obs, _ = env.reset()
        ctrl._reset_state()

        for _ in range(max_steps):
            action, _ = ctrl.predict(obs, deterministic=True)
            obs, reward, done, trunc, _ = env.step(action)
            total_reward += reward

            if done or trunc:
                break

    return total_reward / num_episodes


# %% Training loop
num_generations = 20  # Use 200+ for real training
population_size = 16  # Use 32+ for real training

initial_params = controller.get_flat_params()
optimizer = CMA(mean=initial_params, sigma=0.3, population_size=population_size)

print(f"\nStarting CMA-ES training")
print(
    f"  Generations: {num_generations}, Population: {population_size}, Parameters: {len(initial_params)}"
)

best_fitness = -np.inf
best_params = None
history = {"best": [], "mean": []}

for gen in range(num_generations):
    solutions = [optimizer.ask() for _ in range(population_size)]
    fitness = np.array([evaluate(controller, env, s) for s in solutions])

    optimizer.tell(list(zip(solutions, -fitness)))  # CMA minimizes

    gen_best = fitness.max()
    if gen_best > best_fitness:
        best_fitness = gen_best
        best_params = solutions[np.argmax(fitness)].copy()

    history["best"].append(gen_best)
    history["mean"].append(fitness.mean())

    print(
        f"Gen {gen:3d}: best={gen_best:7.2f}, mean={fitness.mean():7.2f}, best_so_far={best_fitness:.2f}"
    )

# %% Plot training progress
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history["best"], label="Generation best")
plt.plot(history["mean"], alpha=0.7, label="Generation mean")
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(np.maximum.accumulate(history["best"]), "g-", linewidth=2)
plt.xlabel("Generation")
plt.ylabel("Best fitness so far")
plt.grid(True)
plt.tight_layout()
plt.show()

# %% Evaluate the best controller
print("\n--- Evaluating best controller ---")
controller.set_flat_params(best_params)
controller.eval()

env_render = ReachingEnv(xml_path, render_mode="rgb_array", sensor_stats=sensor_stats)

successes = 0
for ep in range(10):
    obs, info = env_render.reset()
    controller._reset_state()

    episode_reward = 0
    for step in range(300):
        action, _ = controller.predict(obs, deterministic=True)
        action = np.clip(action, 0.0, 1.0)

        obs, reward, done, trunc, info = env_render.step(action)
        episode_reward += reward

        if done or trunc:
            if info.get("phase") == "done":
                successes += 1
            break

    print(
        f"  Episode {ep+1}: reward={episode_reward:.1f}, phase={info.get('phase', 'running')}"
    )

env_render.close()
print(f"\nSuccess rate: {successes}/10 = {successes*10}%")

# %% Save trained controller
output_dir = project_root / "outputs" / "cmaes_example"
output_dir.mkdir(parents=True, exist_ok=True)

torch.save(
    {
        "state_dict": controller.state_dict(),
        "params": best_params,
        "fitness": best_fitness,
        "num_muscles": plant.num_muscles,
        "num_sensors": plant.num_sensors,
    },
    output_dir / "controller.pt",
)

print(f"\nSaved to {output_dir / 'controller.pt'}")

# %% Cleanup
env.close()
plant.close()
