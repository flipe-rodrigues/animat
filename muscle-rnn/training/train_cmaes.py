"""CMA-ES Training for Muscle-Driven Arm Controller."""

import numpy as np
import torch
import json
import pickle
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass, asdict
import time
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

    def tqdm(iterable, **kwargs):
        return iterable


from envs.reaching import ReachingEnv
from envs.plant import parse_mujoco_xml, calibrate_sensors
from models.controllers import RNNController
from core.config import ModelConfig
from core.constants import (
    DEFAULT_POPULATION_SIZE,
    DEFAULT_NUM_GENERATIONS,
    DEFAULT_CMAES_SIGMA,
    DEFAULT_NUM_EVAL_EPISODES,
    DEFAULT_MAX_EPISODE_STEPS,
    DEFAULT_TARGET_FITNESS,
    DEFAULT_PATIENCE,
    DEFAULT_CHECKPOINT_EVERY,
    DEFAULT_INSPECTION_EVERY,
    DEFAULT_CALIBRATION_EPISODES,
    DEFAULT_NUM_TARGET_UNITS,
    DEFAULT_TARGET_GRID_SIZE,
    DEFAULT_TARGET_SIGMA,
    DEFAULT_RNN_HIDDEN_SIZE,
    DEFAULT_VIDEO_FPS,
    DEFAULT_PLOT_DPI,
)


@dataclass
class CMAESConfig:
    """Configuration for CMA-ES training."""

    population_size: int = DEFAULT_POPULATION_SIZE
    sigma_init: float = DEFAULT_CMAES_SIGMA
    num_generations: int = DEFAULT_NUM_GENERATIONS
    num_eval_episodes: int = DEFAULT_NUM_EVAL_EPISODES
    max_steps_per_episode: int = DEFAULT_MAX_EPISODE_STEPS
    target_fitness: float = DEFAULT_TARGET_FITNESS
    patience: int = DEFAULT_PATIENCE
    save_checkpoint_every: int = DEFAULT_CHECKPOINT_EVERY
    inspection_every: int = DEFAULT_INSPECTION_EVERY
    num_workers: int = max(1, mp.cpu_count() - 1)
    use_multiprocessing: bool = True
    xml_path: str = ""
    output_dir: str = "outputs/cmaes"


class CMAES:
    """Simple CMA-ES implementation."""

    def __init__(
        self,
        dim: int,
        population_size: Optional[int] = None,
        sigma: float = 0.5,
        mean: Optional[np.ndarray] = None,
    ):
        self.dim = dim
        self.lambda_ = population_size or 4 + int(3 * np.log(dim))
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
        self.chiN = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))

        self.mean = mean if mean is not None else np.zeros(dim)
        self.sigma = sigma
        self.pc = np.zeros(dim)
        self.ps = np.zeros(dim)
        self.C = np.eye(dim)
        self.B = np.eye(dim)
        self.D = np.ones(dim)
        self.invsqrtC = np.eye(dim)

        self.generation = 0
        self.best_fitness_history = []
        self.mean_fitness_history = []
        self.sigma_history = []

    def sample_population(self) -> np.ndarray:
        z = np.random.randn(self.lambda_, self.dim)
        return self.mean + self.sigma * (z @ (self.B * self.D).T)

    def update(self, population: np.ndarray, fitness: np.ndarray) -> None:
        fitness = np.nan_to_num(fitness, nan=-1e6, posinf=-1e6, neginf=-1e6)
        indices = np.argsort(-fitness)
        selected = population[indices[: self.mu]]

        old_mean = self.mean.copy()
        self.mean = self.weights @ selected

        if np.any(np.isnan(self.mean)):
            self.mean = old_mean
            self.sigma *= 0.5
        else:
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
            self.sigma = np.clip(self.sigma, 1e-10, 10.0)

            if self.generation % (self.dim // 10 + 1) == 0:
                self.C = np.triu(self.C) + np.triu(self.C, 1).T
                D, B = np.linalg.eigh(self.C)
                self.D = np.sqrt(np.maximum(D, 1e-20))
                self.B = B
                self.invsqrtC = B @ np.diag(1 / self.D) @ B.T

        self.best_fitness_history.append(fitness[indices[0]])
        self.mean_fitness_history.append(np.nanmean(fitness))
        self.sigma_history.append(self.sigma)
        self.generation += 1

    def get_best(self) -> np.ndarray:
        return self.mean.copy()

    def save_state(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(
                {
                    k: getattr(self, k)
                    for k in [
                        "dim",
                        "lambda_",
                        "mu",
                        "mean",
                        "sigma",
                        "pc",
                        "ps",
                        "C",
                        "B",
                        "D",
                        "generation",
                        "best_fitness_history",
                        "mean_fitness_history",
                        "sigma_history",
                    ]
                },
                f,
            )

    def load_state(self, path: str) -> None:
        with open(path, "rb") as f:
            state = pickle.load(f)
        for k, v in state.items():
            setattr(self, k, v)
        self.invsqrtC = self.B @ np.diag(1 / self.D) @ self.B.T


def _evaluate_single(args) -> float:
    """Worker function for parallel evaluation."""
    params, xml_path, model_config_dict, sensor_stats, num_episodes, max_steps = args

    if np.any(np.isnan(params)):
        return -1000.0

    config = ModelConfig(**model_config_dict)
    controller = RNNController(config)
    controller.set_flat_params(params)
    controller.eval()

    env = ReachingEnv(xml_path, sensor_stats=sensor_stats)
    env.set_network_params(params)
    total_reward = 0

    with torch.no_grad():
        for _ in range(num_episodes):
            obs, _ = env.reset()
            controller.init_hidden(1, torch.device("cpu"))

            for step in range(max_steps):
                if np.any(np.isnan(obs)):
                    obs = np.nan_to_num(obs, nan=0.0)

                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                action, _, _ = controller.forward(obs_tensor)
                action = action.squeeze(0).numpy()

                # Sanitize NaN/Inf actions instead of breaking
                if np.any(~np.isfinite(action)):
                    action = np.nan_to_num(action, nan=0.5, posinf=1.0, neginf=0.0)
                action = np.clip(action, 0.0, 1.0)

                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward if not np.isnan(reward) else -10.0

                if terminated or truncated:
                    break

    env.close()
    fitness = total_reward / num_episodes
    return -1000.0 if np.isnan(fitness) else fitness


def evaluate_controller(
    params: np.ndarray,
    xml_path: str,
    model_config_dict: Dict,
    sensor_stats: Dict,
    num_episodes: int,
    max_steps: int,
    render: bool = False,
    return_trajectories: bool = False,
) -> Tuple[float, Dict[str, Any]]:
    """Evaluate a controller with given parameters."""
    config = ModelConfig(**model_config_dict)
    controller = RNNController(config)
    controller.set_flat_params(params)
    controller.eval()

    env = ReachingEnv(
        xml_path, render_mode="rgb_array" if render else None, sensor_stats=sensor_stats
    )

    total_reward = 0
    episode_rewards = []
    successes = 0
    trajectories = [] if return_trajectories else None
    frames = [] if render else None

    with torch.no_grad():
        for _ in range(num_episodes):
            obs, _ = env.reset()
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

                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward

                if return_trajectories:
                    episode_traj["action"].append(action.copy())
                    episode_traj["reward"].append(reward)
                    episode_traj["info"].append(info)

                if render:
                    frame = env.render()
                    if frame is not None:
                        frames.append(frame)

                if terminated or truncated:
                    if info.get("phase") == "done":
                        successes += 1
                    break

            episode_rewards.append(episode_reward)
            total_reward += episode_reward
            if return_trajectories:
                trajectories.append(episode_traj)

    env.close()

    return total_reward / num_episodes, {
        "episode_rewards": episode_rewards,
        "successes": successes,
        "success_rate": successes / num_episodes,
        "trajectories": trajectories,
        "frames": frames,
    }


class CMAESTrainer:
    """Trainer class for CMA-ES optimization."""

    def __init__(
        self, config: CMAESConfig, model_config: ModelConfig, sensor_stats: Dict
    ):
        self.config = config
        self.model_config = model_config
        self.sensor_stats = sensor_stats

        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.controller = RNNController(model_config)
        self.num_params = self.controller.count_parameters()
        print(f"Controller has {self.num_params} parameters")

        self.cmaes = CMAES(
            self.num_params,
            config.population_size,
            config.sigma_init,
            self.controller.get_flat_params(),
        )

        self.best_fitness = -np.inf
        self.best_params = None
        self.no_improvement_count = 0

    def train(self) -> Dict[str, Any]:
        """Run CMA-ES training loop."""
        config = self.config
        model_config_dict = asdict(self.model_config)

        print(f"Starting CMA-ES training for {config.num_generations} generations")
        print(f"Population size: {self.cmaes.lambda_}")

        start_time = time.time()

        for gen in range(config.num_generations):
            gen_start = time.time()
            population = self.cmaes.sample_population()

            # Evaluate population
            args_list = [
                (
                    p,
                    config.xml_path,
                    model_config_dict,
                    self.sensor_stats,
                    config.num_eval_episodes,
                    config.max_steps_per_episode,
                )
                for p in population
            ]

            if config.use_multiprocessing and config.num_workers > 1:
                try:
                    with ProcessPoolExecutor(
                        max_workers=config.num_workers
                    ) as executor:
                        fitness = np.array(
                            list(executor.map(_evaluate_single, args_list))
                        )
                except Exception as e:
                    print(f"Parallel failed ({e}), using sequential")
                    config.use_multiprocessing = False
                    fitness = np.array(
                        [
                            _evaluate_single(a)
                            for a in (
                                tqdm(args_list, desc=f"Gen {gen}", leave=False)
                                if HAS_TQDM
                                else args_list
                            )
                        ]
                    )
            else:
                fitness = np.array(
                    [
                        _evaluate_single(a)
                        for a in (
                            tqdm(args_list, desc=f"Gen {gen}", leave=False)
                            if HAS_TQDM
                            else args_list
                        )
                    ]
                )

            self.cmaes.update(population, fitness)

            gen_best_idx = np.argmax(fitness)
            gen_best_fitness = fitness[gen_best_idx]

            if gen_best_fitness > self.best_fitness:
                self.best_fitness = gen_best_fitness
                self.best_params = population[gen_best_idx].copy()
                self.no_improvement_count = 0
            else:
                self.no_improvement_count += 1

            print(
                f"Gen {gen}: best={gen_best_fitness:.3f}, mean={fitness.mean():.3f}, sigma={self.cmaes.sigma:.4f}, time={time.time() - gen_start:.1f}s"
            )

            if (gen + 1) % config.save_checkpoint_every == 0:
                self._save_checkpoint(gen)

            if config.inspection_every > 0 and (gen + 1) % config.inspection_every == 0:
                self._run_inspection(gen)

            if self.best_fitness >= config.target_fitness:
                print(f"Target fitness reached at generation {gen}")
                break

            if self.no_improvement_count >= config.patience:
                print(f"No improvement for {config.patience} generations, stopping")
                break

        self._save_checkpoint(gen, final=True)

        return {
            "best_fitness": self.best_fitness,
            "best_params": self.best_params,
            "generations": gen + 1,
            "total_time": time.time() - start_time,
        }

    def _save_checkpoint(self, gen: int, final: bool = False) -> None:
        suffix = "final" if final else f"gen{gen}"

        self.cmaes.save_state(self.output_dir / f"cmaes_state_{suffix}.pkl")

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

        with open(self.output_dir / f"history_{suffix}.json", "w") as f:
            json.dump(
                {
                    "best_fitness_history": self.cmaes.best_fitness_history,
                    "mean_fitness_history": self.cmaes.mean_fitness_history,
                    "sigma_history": self.cmaes.sigma_history,
                },
                f,
            )

        # Plot training curves
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(self.cmaes.best_fitness_history, label="Best")
        axes[0].plot(self.cmaes.mean_fitness_history, label="Mean", alpha=0.7)
        axes[0].set_xlabel("Generation")
        axes[0].set_ylabel("Fitness")
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(self.cmaes.sigma_history)
        axes[1].set_xlabel("Generation")
        axes[1].set_ylabel("Sigma")
        axes[1].grid(True)

        plt.tight_layout()
        plt.savefig(
            self.output_dir / f"training_curves_{suffix}.png", dpi=DEFAULT_PLOT_DPI
        )
        plt.close()

        print(f"Checkpoint saved: {suffix}")

    def _run_inspection(self, gen: int) -> None:
        if self.best_params is None:
            return

        print(f"\n--- Inspection at generation {gen} ---")
        self.controller.set_flat_params(self.best_params)

        try:
            from utils.episode_recorder import record_and_save

            inspect_dir = self.output_dir / "inspections" / f"gen{gen}"
            inspect_dir.mkdir(parents=True, exist_ok=True)
            record_and_save(
                self.controller,
                self.config.xml_path,
                self.sensor_stats,
                str(inspect_dir),
                self.config.max_steps_per_episode,
                gen * 1000,
                DEFAULT_VIDEO_FPS,
            )
        except Exception as e:
            print(f"  Warning: Could not record episode: {e}")

        print(f"--- End inspection ---\n")


def run_cmaes_training(
    xml_path: str,
    output_dir: str = "outputs/cmaes",
    num_generations: int = DEFAULT_NUM_GENERATIONS,
    population_size: int = DEFAULT_POPULATION_SIZE,
    sigma_init: float = DEFAULT_CMAES_SIGMA,
    num_workers: Optional[int] = None,
    use_multiprocessing: bool = True,
    calibration_episodes: int = DEFAULT_CALIBRATION_EPISODES,
    save_checkpoint_every: int = DEFAULT_CHECKPOINT_EVERY,
    inspection_every: int = DEFAULT_INSPECTION_EVERY,
) -> Dict[str, Any]:
    """Main entry point for CMA-ES training."""
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)

    parsed_model = parse_mujoco_xml(xml_path)
    print(
        f"Model: {parsed_model.model_name} (Joints: {parsed_model.num_joints}, Muscles: {parsed_model.num_muscles})"
    )

    print("Calibrating sensors...")
    sensor_stats = calibrate_sensors(xml_path, num_episodes=calibration_episodes)

    model_config = ModelConfig(
        num_muscles=parsed_model.num_muscles,
        num_sensors=parsed_model.num_sensors,
        num_target_units=DEFAULT_NUM_TARGET_UNITS,
        target_grid_size=DEFAULT_TARGET_GRID_SIZE,
        target_sigma=DEFAULT_TARGET_SIGMA,
        rnn_hidden_size=DEFAULT_RNN_HIDDEN_SIZE,
    )

    train_config = CMAESConfig(
        xml_path=xml_path,
        output_dir=output_dir,
        num_generations=num_generations,
        population_size=population_size,
        sigma_init=sigma_init,
        num_workers=num_workers,
        use_multiprocessing=use_multiprocessing,
        save_checkpoint_every=save_checkpoint_every,
        inspection_every=inspection_every,
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(output_path / "sensor_stats.pkl", "wb") as f:
        pickle.dump(sensor_stats, f)
    with open(output_path / "model_config.json", "w") as f:
        json.dump(asdict(model_config), f, indent=2)
    with open(output_path / "train_config.json", "w") as f:
        json.dump(asdict(train_config), f, indent=2)

    trainer = CMAESTrainer(train_config, model_config, sensor_stats)
    return trainer.train()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CMA-ES Training for Muscle Arm")
    parser.add_argument("xml_path", help="Path to MuJoCo XML file")
    parser.add_argument("--output-dir", default="outputs/cmaes")
    parser.add_argument("--generations", type=int, default=DEFAULT_NUM_GENERATIONS)
    parser.add_argument("--population", type=int, default=DEFAULT_POPULATION_SIZE)
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))

    args = parser.parse_args()

    results = run_cmaes_training(
        xml_path=args.xml_path,
        output_dir=args.output_dir,
        num_generations=args.generations,
        population_size=args.population,
        num_workers=args.workers,
    )

    print(f"\nTraining complete! Best fitness: {results['best_fitness']:.3f}")
