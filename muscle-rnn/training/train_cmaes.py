"""
CMA-ES Training for Muscle-Driven Arm Controller

Uses Covariance Matrix Adaptation Evolution Strategy to optimize
the RNN controller weights for the reaching task.
"""

import numpy as np
import torch
import json
import pickle
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Callable
from dataclasses import dataclass, asdict
import time
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

    def tqdm(iterable, **kwargs):
        return iterable


import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from envs.reaching_env import ReachingEnv, calibrate_sensors
from models.controllers import (
    RNNController,
    MLPController,
    ModelConfig,
    create_controller,
)
from utils.model_parser import parse_mujoco_xml


@dataclass
class CMAESConfig:
    """Configuration for CMA-ES training."""

    # CMA-ES parameters
    population_size: int = 64
    sigma_init: float = 0.1  # Reduced from 0.5 to prevent large weight perturbations

    # Training parameters
    num_generations: int = 500
    num_eval_episodes: int = 5
    max_steps_per_episode: int = 300

    # Early stopping
    target_fitness: float = 50.0
    patience: int = 50

    # Checkpointing and inspection periods (in generations)
    checkpoint_period: int = 25  # Save checkpoint every N generations
    inspection_period: int = 0  # Run full inspection every N generations (0=disabled)

    # Parallelization
    num_workers: int = mp.cpu_count() - 1  # Default to number of CPUs minus one
    use_multiprocessing: bool = True  # Explicitly enabled by default

    # Paths
    xml_path: str = ""
    output_dir: str = "outputs/cmaes"


class CMAES:
    """Simple CMA-ES implementation."""

    def __init__(
        self,
        dim: int,
        population_size: int = None,
        sigma: float = 0.5,
        mean: np.ndarray = None,
    ):
        self.dim = dim

        if population_size is None:
            self.lambda_ = 4 + int(3 * np.log(dim))
        else:
            self.lambda_ = population_size

        self.mu = self.lambda_ // 2

        weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = weights / weights.sum()
        self.mueff = 1.0 / (self.weights**2).sum()

        self.cc = (4 + self.mueff / dim) / (dim + 4 + 2 * self.mueff / dim)
        self.cs = (self.mueff + 2) / (dim + self.mueff + 5)
        self.c1 = 2 / ((dim + 1.3) ** 2 + self.mueff)
        self.cmu = min(
            1 - self.c1,
            2 * (self.mueff - 2 + 1 / self.mueff) / ((dim + 2) ** 2 + self.mueff),
        )
        self.damps = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (dim + 1)) - 1) + self.cs

        self.mean = mean if mean is not None else np.zeros(dim)
        self.sigma = sigma
        self.pc = np.zeros(dim)
        self.ps = np.zeros(dim)
        self.B = np.eye(dim)
        self.D = np.ones(dim)
        self.C = np.eye(dim)
        self.invsqrtC = np.eye(dim)

        self.chiN = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))

        self.generation = 0
        self.best_fitness_history = []
        self.mean_fitness_history = []
        self.sigma_history = []

    def sample_population(self) -> np.ndarray:
        z = np.random.randn(self.lambda_, self.dim)
        y = z @ (self.B * self.D).T
        x = self.mean + self.sigma * y
        return x

    def update(self, population: np.ndarray, fitness: np.ndarray) -> None:
        """Update CMA-ES state based on fitness."""
        # Replace NaN fitness with very bad values
        fitness = np.nan_to_num(fitness, nan=-1e6, posinf=-1e6, neginf=-1e6)

        indices = np.argsort(-fitness)  # Sort descending (higher is better)
        selected = population[indices[: self.mu]]

        old_mean = self.mean.copy()
        self.mean = self.weights @ selected

        # Check for NaN in mean
        if np.any(np.isnan(self.mean)):
            self.mean = old_mean  # Revert to old mean
            self.sigma *= 0.5  # Reduce step size
            self.generation += 1
            self.best_fitness_history.append(fitness[indices[0]])
            self.mean_fitness_history.append(np.nanmean(fitness))
            self.sigma_history.append(self.sigma)
            return

        y_w = (self.mean - old_mean) / self.sigma
        z_w = self.invsqrtC @ y_w

        self.ps = (1 - self.cs) * self.ps + np.sqrt(
            self.cs * (2 - self.cs) * self.mueff
        ) * z_w

        hsig = np.linalg.norm(self.ps) / np.sqrt(
            1 - (1 - self.cs) ** (2 * (self.generation + 1))
        ) / self.chiN < 1.4 + 2 / (self.dim + 1)

        self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(
            self.cc * (2 - self.cc) * self.mueff
        ) * y_w

        c1a = self.c1 * (1 - (1 - hsig**2) * self.cc * (2 - self.cc))
        y_selected = (selected - old_mean) / self.sigma

        self.C = (
            (1 - c1a - self.cmu * self.weights.sum()) * self.C
            + self.c1 * np.outer(self.pc, self.pc)
            + self.cmu * (self.weights[:, None] * y_selected).T @ y_selected
        )

        self.sigma *= np.exp(
            (self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chiN - 1)
        )

        # Clamp sigma to reasonable range
        self.sigma = np.clip(self.sigma, 1e-10, 10.0)

        if self.generation % (self.dim // 10 + 1) == 0:
            self.C = np.triu(self.C) + np.triu(self.C, 1).T
            D, B = np.linalg.eigh(self.C)
            D = np.sqrt(np.maximum(D, 1e-20))
            self.D = D
            self.B = B
            self.invsqrtC = B @ np.diag(1 / D) @ B.T

        self.best_fitness_history.append(fitness[indices[0]])
        self.mean_fitness_history.append(np.nanmean(fitness))
        self.sigma_history.append(self.sigma)

        self.generation += 1

    def get_best(self) -> np.ndarray:
        return self.mean.copy()

    def save_state(self, path: str) -> None:
        state = {
            "dim": self.dim,
            "lambda_": self.lambda_,
            "mu": self.mu,
            "mean": self.mean,
            "sigma": self.sigma,
            "pc": self.pc,
            "ps": self.ps,
            "C": self.C,
            "B": self.B,
            "D": self.D,
            "generation": self.generation,
            "best_fitness_history": self.best_fitness_history,
            "mean_fitness_history": self.mean_fitness_history,
            "sigma_history": self.sigma_history,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    def load_state(self, path: str) -> None:
        with open(path, "rb") as f:
            state = pickle.load(f)
        for key, value in state.items():
            setattr(self, key, value)
        self.invsqrtC = self.B @ np.diag(1 / self.D) @ self.B.T


def _evaluate_single(args):
    """Worker function for parallel evaluation."""
    params, xml_path, model_config_dict, sensor_stats, n_episodes, max_steps = args

    # Check for NaN in params
    if np.any(np.isnan(params)):
        return -1000.0  # Return bad fitness for NaN params

    config = ModelConfig(**model_config_dict)
    controller = RNNController(config)
    controller.set_flat_params(params)
    controller.eval()

    env = ReachingEnv(xml_path, sensor_stats=sensor_stats)

    total_reward = 0
    successes = 0

    with torch.no_grad():
        for ep in range(n_episodes):
            obs, info = env.reset()
            controller.init_hidden(1, torch.device("cpu"))

            episode_reward = 0
            for step in range(max_steps):
                # Check for NaN in observation
                if np.any(np.isnan(obs)):
                    episode_reward = -1000.0
                    break

                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                action, _, _ = controller.forward(obs_tensor)
                action = action.squeeze(0).numpy()

                # Check for NaN in action
                if np.any(np.isnan(action)):
                    episode_reward = -1000.0
                    break

                # Clip action to valid range
                action = np.clip(action, 0.0, 2.0)

                obs, reward, terminated, truncated, info = env.step(action)

                # Check for NaN in reward
                if np.isnan(reward):
                    reward = -10.0

                episode_reward += reward

                if terminated or truncated:
                    if info.get("phase") == "done":
                        successes += 1
                    break

            total_reward += episode_reward

    env.close()

    fitness = total_reward / n_episodes

    # Final NaN check
    if np.isnan(fitness):
        fitness = -1000.0

    return fitness


def evaluate_controller(
    params: np.ndarray,
    xml_path: str,
    model_config_dict: Dict,
    sensor_stats: Dict,
    n_episodes: int,
    max_steps: int,
    render: bool = False,
    return_trajectories: bool = False,
) -> Tuple[float, Dict[str, Any]]:
    """Evaluate a controller with given parameters."""
    config = ModelConfig(**model_config_dict)
    controller = RNNController(config)
    controller.set_flat_params(params)
    controller.eval()

    render_mode = "rgb_array" if render else None
    env = ReachingEnv(xml_path, render_mode=render_mode, sensor_stats=sensor_stats)

    total_reward = 0
    episode_rewards = []
    successes = 0
    trajectories = [] if return_trajectories else None
    frames = [] if render else None

    with torch.no_grad():
        for ep in range(n_episodes):
            obs, info = env.reset()
            controller.init_hidden(1, torch.device("cpu"))

            episode_reward = 0
            episode_traj = (
                {"obs": [], "action": [], "reward": [], "info": []}
                if return_trajectories
                else None
            )

            for step in range(max_steps):
                if return_trajectories:
                    episode_traj["obs"].append(obs.copy())

                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                action, _, _ = controller.forward(obs_tensor)
                action = action.squeeze(0).numpy()

                obs, reward, terminated, truncated, step_info = env.step(action)
                episode_reward += reward

                if return_trajectories:
                    episode_traj["action"].append(action.copy())
                    episode_traj["reward"].append(reward)
                    episode_traj["info"].append(step_info)

                if render:
                    frame = env.render()
                    if frame is not None:
                        frames.append(frame)

                if terminated or truncated:
                    if step_info.get("phase") == "done":
                        successes += 1
                    break

            episode_rewards.append(episode_reward)
            total_reward += episode_reward

            if return_trajectories:
                trajectories.append(episode_traj)

    env.close()

    fitness = total_reward / n_episodes
    info_out = {
        "episode_rewards": episode_rewards,
        "successes": successes,
        "success_rate": successes / n_episodes,
    }

    if return_trajectories:
        info_out["trajectories"] = trajectories
    if render:
        info_out["frames"] = frames

    return fitness, info_out


class CMAESTrainer:
    """Trainer class for CMA-ES optimization."""

    def __init__(
        self, config: CMAESConfig, model_config: ModelConfig, sensor_stats: Dict
    ):
        self.config = config
        self.model_config = model_config
        self.sensor_stats = sensor_stats

        # Create output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize controller to get parameter count
        self.controller = RNNController(model_config)
        self.n_params = self.controller.count_parameters()
        print(f"Controller has {self.n_params} parameters")

        # Initialize CMA-ES
        initial_params = self.controller.get_flat_params()
        self.cmaes = CMAES(
            dim=self.n_params,
            population_size=config.population_size,
            sigma=config.sigma_init,
            mean=initial_params,
        )

        # Tracking
        self.best_fitness = -np.inf
        self.best_params = None
        self.no_improvement_count = 0

    def train(self) -> Dict[str, Any]:
        """Run CMA-ES training loop."""
        config = self.config
        model_config_dict = asdict(self.model_config)

        print(f"Starting CMA-ES training for {config.num_generations} generations")
        print(f"Population size: {self.cmaes.lambda_}")
        print(
            f"Parallel workers: {config.num_workers if config.use_multiprocessing else 1} (sequential)"
        )

        start_time = time.time()

        for gen in range(config.num_generations):
            gen_start = time.time()

            # Sample population
            population = self.cmaes.sample_population()

            # Evaluate population
            if config.use_multiprocessing and config.num_workers > 1:
                # Parallel evaluation (may not work on Windows/notebooks)
                try:
                    args_list = [
                        (
                            params,
                            config.xml_path,
                            model_config_dict,
                            self.sensor_stats,
                            config.num_eval_episodes,
                            config.max_steps_per_episode,
                        )
                        for params in population
                    ]

                    with ProcessPoolExecutor(
                        max_workers=config.num_workers
                    ) as executor:
                        fitness_values = list(executor.map(_evaluate_single, args_list))
                    fitness = np.array(fitness_values)
                except Exception as e:
                    print(
                        f"Warning: Parallel evaluation failed ({e}), falling back to sequential"
                    )
                    config.use_multiprocessing = False
                    fitness = np.array(
                        [
                            _evaluate_single(
                                (
                                    params,
                                    config.xml_path,
                                    model_config_dict,
                                    self.sensor_stats,
                                    config.num_eval_episodes,
                                    config.max_steps_per_episode,
                                )
                            )
                            for params in population
                        ]
                    )
            else:
                # Sequential evaluation (default, always works)
                if HAS_TQDM:
                    fitness = np.array(
                        [
                            _evaluate_single(
                                (
                                    params,
                                    config.xml_path,
                                    model_config_dict,
                                    self.sensor_stats,
                                    config.num_eval_episodes,
                                    config.max_steps_per_episode,
                                )
                            )
                            for params in tqdm(
                                population, desc=f"Gen {gen}", leave=False
                            )
                        ]
                    )
                else:
                    fitness = np.array(
                        [
                            _evaluate_single(
                                (
                                    params,
                                    config.xml_path,
                                    model_config_dict,
                                    self.sensor_stats,
                                    config.num_eval_episodes,
                                    config.max_steps_per_episode,
                                )
                            )
                            for params in population
                        ]
                    )

            # Update CMA-ES
            self.cmaes.update(population, fitness)

            # Track best
            gen_best_idx = np.argmax(fitness)
            gen_best_fitness = fitness[gen_best_idx]

            if gen_best_fitness > self.best_fitness:
                self.best_fitness = gen_best_fitness
                self.best_params = population[gen_best_idx].copy()
                self.no_improvement_count = 0
            else:
                self.no_improvement_count += 1

            # Logging
            gen_time = time.time() - gen_start
            print(
                f"Gen {gen}: best={gen_best_fitness:.3f}, mean={fitness.mean():.3f}, "
                f"sigma={self.cmaes.sigma:.4f}, time={gen_time:.1f}s"
            )

            # Checkpointing
            if (gen + 1) % config.checkpoint_period == 0:
                self._save_checkpoint(gen)
                self._render_and_plot(gen)

            # Full inspection (episode summary, simulation video, network activity)
            if config.inspection_period > 0 and (gen + 1) % config.inspection_period == 0:
                self._run_inspection(gen)

            # Early stopping
            if self.best_fitness >= config.target_fitness:
                print(f"Target fitness reached at generation {gen}")
                break

            if self.no_improvement_count >= config.patience:
                print(f"No improvement for {config.patience} generations, stopping")
                break

        total_time = time.time() - start_time
        print(f"Training complete in {total_time:.1f}s")

        # Final checkpoint
        self._save_checkpoint(gen, final=True)
        self._render_and_plot(gen, final=True)

        return {
            "best_fitness": self.best_fitness,
            "best_params": self.best_params,
            "generations": gen + 1,
            "total_time": total_time,
        }

    def _save_checkpoint(self, gen: int, final: bool = False):
        """Save training checkpoint."""
        suffix = "final" if final else f"gen{gen}"

        # Save CMA-ES state
        self.cmaes.save_state(self.output_dir / f"cmaes_state_{suffix}.pkl")

        # Save best controller
        if self.best_params is not None:
            self.controller.set_flat_params(self.best_params)
            torch.save(
                {
                    "model_state_dict": self.controller.state_dict(),
                    "model_config": asdict(self.model_config),
                    "fitness": self.best_fitness,
                    "generation": gen,
                },
                self.output_dir / f"best_controller_{suffix}.pt",
            )

        # Save training history
        history = {
            "best_fitness_history": self.cmaes.best_fitness_history,
            "mean_fitness_history": self.cmaes.mean_fitness_history,
            "sigma_history": self.cmaes.sigma_history,
        }
        with open(self.output_dir / f"history_{suffix}.json", "w") as f:
            json.dump(history, f)

        print(f"Checkpoint saved: {suffix}")

    def _render_and_plot(self, gen: int, final: bool = False):
        """Render evaluation and plot training curves."""
        suffix = "final" if final else f"gen{gen}"

        if self.best_params is None:
            return

        # Plot training curves
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        axes[0].plot(self.cmaes.best_fitness_history, label="Best")
        axes[0].plot(self.cmaes.mean_fitness_history, label="Mean", alpha=0.7)
        axes[0].set_xlabel("Generation")
        axes[0].set_ylabel("Fitness")
        axes[0].set_title("Fitness History")
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(self.cmaes.sigma_history)
        axes[1].set_xlabel("Generation")
        axes[1].set_ylabel("Sigma")
        axes[1].set_title("Step Size")
        axes[1].grid(True)

        # Evaluate and get trajectory
        fitness, info = evaluate_controller(
            self.best_params,
            self.config.xml_path,
            asdict(self.model_config),
            self.sensor_stats,
            n_episodes=1,
            max_steps=self.config.max_steps_per_episode,
            render=False,
            return_trajectories=True,
        )

        if info["trajectories"]:
            traj = info["trajectories"][0]
            rewards = traj["reward"]
            axes[2].plot(np.cumsum(rewards))
            axes[2].set_xlabel("Step")
            axes[2].set_ylabel("Cumulative Reward")
            axes[2].set_title(f"Episode Reward (fitness={fitness:.2f})")
            axes[2].grid(True)

        plt.tight_layout()
        plt.savefig(self.output_dir / f"training_curves_{suffix}.png", dpi=150)
        plt.close()

        print(f"Plots saved: {suffix}")

    def _run_inspection(self, gen: int):
        """Run full inspection with episode summary, simulation video, and network activity."""
        if self.best_params is None:
            return

        print(f"\n--- Running inspection at generation {gen} ---")

        # Import visualization functions
        from utils.visualization import (
            record_episode,
            plot_episode_summary,
            print_weight_summary,
            plot_reflex_connections,
            save_video,
        )

        # Set best params on controller
        self.controller.set_flat_params(self.best_params)
        self.controller.eval()

        # Create inspection directory
        inspect_dir = self.output_dir / "inspections"
        inspect_dir.mkdir(exist_ok=True)

        # Record episode with rendering
        trajectory = record_episode(
            controller=self.controller,
            xml_path=self.config.xml_path,
            sensor_stats=self.sensor_stats,
            max_steps=self.config.max_steps_per_episode,
        )

        # Episode stats
        total_reward = sum(trajectory["rewards"])
        final_phase = (
            trajectory["infos"][-1].get("phase", "unknown")
            if trajectory["infos"]
            else "unknown"
        )
        final_dist = (
            trajectory["infos"][-1].get("distance_to_target", -1)
            if trajectory["infos"]
            else -1
        )

        print(
            f"  Episode: {len(trajectory['rewards'])} steps, reward={total_reward:.2f}"
        )
        print(f"  Final phase: {final_phase}, distance: {final_dist:.4f}")

        # Save simulation video
        if "frames" in trajectory and trajectory["frames"]:
            video_path = str(inspect_dir / f"simulation_gen{gen}.mp4")
            save_video(trajectory["frames"], video_path, fps=30)
            print(f"  Simulation video saved: {video_path}")

        # Save episode summary plot
        plot_episode_summary(
            trajectory,
            output_path=str(inspect_dir / f"episode_summary_gen{gen}.png"),
            show=False,
        )

        # Save reflex connections
        plot_reflex_connections(
            self.controller,
            output_path=str(inspect_dir / f"reflex_gen{gen}.png"),
            show=False,
        )

        # Network activity visualization
        try:
            from utils.network_visualizer import record_episode_with_network

            print(f"  Recording network activity visualization...")
            network_traj = record_episode_with_network(
                controller=self.controller,
                xml_path=self.config.xml_path,
                sensor_stats=self.sensor_stats,
                max_steps=self.config.max_steps_per_episode,
                output_video=str(inspect_dir / f"network_activity_gen{gen}.mp4"),
                fps=30,
            )
            print(f"  Network activity video saved")
        except Exception as e:
            print(f"  Warning: Could not create network activity video: {e}")

        print(f"  Inspection outputs saved to {inspect_dir}")
        print(f"--- End inspection ---\n")


def run_cmaes_training(
    xml_path: str,
    output_dir: str = "outputs/cmaes",
    num_generations: int = 500,
    population_size: int = 64,
    sigma_init: float = 0.1,
    num_workers: int = None,
    use_multiprocessing: bool = True,
    calibration_episodes: int = 50,
    checkpoint_period: int = 25,
    inspection_period: int = 0,
) -> Dict[str, Any]:
    """
    Main entry point for CMA-ES training.

    Args:
        xml_path: Path to MuJoCo XML file
        output_dir: Output directory for checkpoints
        num_generations: Number of generations
        population_size: CMA-ES population size
        sigma_init: Initial CMA-ES step size
        num_workers: Number of parallel workers (default: cpu_count - 1)
        use_multiprocessing: Whether to use parallel evaluation
        calibration_episodes: Episodes for sensor calibration
        checkpoint_period: Save checkpoint every N generations
        inspection_period: Run full inspection every N generations (0=disabled)

    Returns:
        Training results
    """
    # Default workers to cpu_count - 1
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)

    # Parse model
    parsed_model = parse_mujoco_xml(xml_path)
    print(f"Model: {parsed_model.model_name}")
    print(f"Joints: {parsed_model.n_joints}")
    print(f"Muscles: {parsed_model.n_muscles}")
    print(f"Sensors: {parsed_model.n_sensors}")

    # Calibrate sensors
    print("Calibrating sensors...")
    sensor_stats = calibrate_sensors(xml_path, n_episodes=calibration_episodes)

    # Create model config
    model_config = ModelConfig(
        num_muscles=parsed_model.n_muscles,
        num_sensors=parsed_model.n_sensors,
        num_target_units=16,  # 4x4 grid
        rnn_hidden_size=32,
    )

    # Create training config
    train_config = CMAESConfig(
        xml_path=xml_path,
        output_dir=output_dir,
        num_generations=num_generations,
        population_size=population_size,
        sigma_init=sigma_init,
        num_workers=num_workers,
        use_multiprocessing=use_multiprocessing,
        checkpoint_period=checkpoint_period,
        inspection_period=inspection_period,
    )

    # Save configs
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(output_path / "sensor_stats.pkl", "wb") as f:
        pickle.dump(sensor_stats, f)

    with open(output_path / "model_config.json", "w") as f:
        json.dump(asdict(model_config), f, indent=2)

    with open(output_path / "train_config.json", "w") as f:
        json.dump(asdict(train_config), f, indent=2)

    # Train
    trainer = CMAESTrainer(train_config, model_config, sensor_stats)
    results = trainer.train()

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CMA-ES Training for Muscle Arm")
    parser.add_argument("xml_path", help="Path to MuJoCo XML file")
    parser.add_argument(
        "--output-dir", default="outputs/cmaes", help="Output directory"
    )
    parser.add_argument(
        "--generations", type=int, default=500, help="Number of generations"
    )
    parser.add_argument("--population", type=int, default=64, help="Population size")
    parser.add_argument(
        "--workers", type=int, default=mp.cpu_count() - 1, help="Number of workers"
    )

    args = parser.parse_args()

    results = run_cmaes_training(
        xml_path=args.xml_path,
        output_dir=args.output_dir,
        n_generations=args.generations,
        population_size=args.population,
        n_workers=args.workers,
    )

    print(f"\nTraining complete!")
    print(f"Best fitness: {results['best_fitness']:.3f}")
    print(f"Generations: {results['generations']}")
