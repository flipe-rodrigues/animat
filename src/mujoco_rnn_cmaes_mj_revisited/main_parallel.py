# %%
"""
.####.##.....##.########...#######..########..########..######.
..##..###...###.##.....##.##.....##.##.....##....##....##....##
..##..####.####.##.....##.##.....##.##.....##....##....##......
..##..##.###.##.########..##.....##.########.....##.....######.
..##..##.....##.##........##.....##.##...##......##..........##
..##..##.....##.##........##.....##.##....##.....##....##....##
.####.##.....##.##.........#######..##.....##....##.....######.
"""
import pickle
import time
import os
from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, List, Optional
import multiprocessing as mp
from multiprocessing.pool import Pool

import numpy as np
from cmaes import CMA

from plants import SequentialReacher
from encoders import GridTargetEncoder
from environments import SequentialReachingEnv
from networks import NeuroMuscularRNN
from utils import relu, tanh, alpha_from_tau

# Import optimized workers
from workers import (
    init_worker,
    evaluate_worker,
    cleanup_worker,
)


"""
..######...#######..##....##.########.####..######..
.##....##.##.....##.###...##.##........##..##....##.
.##.......##.....##.####..##.##........##..##.......
.##.......##.....##.##.##.##.######....##..##...####
.##.......##.....##.##..####.##........##..##....##.
.##....##.##.....##.##...###.##........##..##....##.
..######...#######..##....##.##.......####..######..
"""


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""

    num_generations: int = 10000
    initial_sigma: float = 1.3
    checkpoint_interval: int = 1000
    eval_interval: int = 10
    num_workers: int = mp.cpu_count()
    checkpoint_dir: str = "../../models"

    # New: Load balancing
    chunk_size_multiplier: int = 4  # chunksize = pop_size // (workers * multiplier)

    # New: Checkpoint management
    keep_last_n_checkpoints: int = 5


@dataclass
class RNNConfig:
    """Configuration for RNN architecture."""

    hidden_size: int = 25
    tau: float = 10e-3
    activation: Callable = tanh
    use_bias: bool = True


@dataclass
class EnvConfig:
    """Configuration for environment."""

    plant_xml_file: str = "arm.xml"
    grid_size: int = 8
    sigma: float = 0.25
    target_duration_distro: Dict = None
    iti_distro: Dict = None
    num_targets: int = 10
    randomize_gravity: bool = True
    loss_weights: Dict = None

    def __post_init__(self):
        if self.target_duration_distro is None:
            self.target_duration_distro = {"mean": 3, "min": 1, "max": 6}
        if self.iti_distro is None:
            self.iti_distro = {"mean": 1, "min": 0, "max": 3}
        if self.loss_weights is None:
            self.loss_weights = {
                "distance": 1.0,
                "energy": 0.1,
            }


@dataclass
class PerformanceMetrics:
    """Track training performance metrics."""

    generation_times: List[float] = field(default_factory=list)
    best_losses: List[float] = field(default_factory=list)
    mean_losses: List[float] = field(default_factory=list)

    def add_generation(self, gen_time, best_loss, mean_loss):
        """Record metrics for a generation."""
        self.generation_times.append(gen_time)
        self.best_losses.append(best_loss)
        self.mean_losses.append(mean_loss)

    def get_summary(self):
        """Get summary statistics."""
        if not self.generation_times:
            return {}

        return {
            "mean_gen_time": np.mean(self.generation_times[-100:]),  # Last 100 gens
            "total_time": sum(self.generation_times),
            "best_loss_overall": min(self.best_losses),
            "best_loss_recent": (
                min(self.best_losses[-10:])
                if len(self.best_losses) >= 10
                else min(self.best_losses)
            ),
        }


"""
..######..########.########.##.....##.########.
.##....##.##..........##....##.....##.##.....##
.##.......##..........##....##.....##.##.....##
..######..######......##....##.....##.########.
.......##.##..........##....##.....##.##.......
.##....##.##..........##....##.....##.##.......
..######..########....##.....#######..##.......
"""


def create_rnn_config(
    reacher: SequentialReacher, target_encoder: GridTargetEncoder, rnn_config: RNNConfig
) -> Dict:
    """Create RNN configuration dictionary for workers."""
    return {
        "target_size": target_encoder.size,
        "length_size": reacher.num_sensors_len,
        "velocity_size": reacher.num_sensors_vel,
        "force_size": reacher.num_sensors_frc,
        "hidden_size": rnn_config.hidden_size,
        "output_size": reacher.num_actuators,
        "activation": rnn_config.activation,
        "smoothing_factor": alpha_from_tau(
            tau=rnn_config.tau, dt=reacher.model.opt.timestep
        ),
        "use_bias": rnn_config.use_bias,
    }


def create_env_config(reacher: SequentialReacher, env_config: EnvConfig) -> Dict:
    """Create environment configuration dictionary for workers."""
    workspace_bounds = reacher.get_workspace_bounds()

    return {
        "plant": {"plant_xml_file": env_config.plant_xml_file},
        "encoder": {
            "grid_size": env_config.grid_size,
            "x_bounds": workspace_bounds[0],
            "y_bounds": workspace_bounds[1],
            "sigma": env_config.sigma,
        },
        "env": {
            "target_duration_distro": env_config.target_duration_distro,
            "iti_distro": env_config.iti_distro,
            "num_targets": env_config.num_targets,
            "randomize_gravity": env_config.randomize_gravity,
            "loss_weights": env_config.loss_weights,
        },
    }


def setup_components(env_config: EnvConfig, rnn_config: RNNConfig):
    """Initialize all components needed for training."""
    # Create plant
    reacher = SequentialReacher(plant_xml_file=env_config.plant_xml_file)

    # Create target encoder
    workspace_bounds = reacher.get_workspace_bounds()
    target_encoder = GridTargetEncoder(
        grid_size=env_config.grid_size,
        x_bounds=workspace_bounds[0],
        y_bounds=workspace_bounds[1],
        sigma=env_config.sigma,
    )

    # Create RNN
    rnn = NeuroMuscularRNN(
        target_size=target_encoder.size,
        length_size=reacher.num_sensors_len,
        velocity_size=reacher.num_sensors_vel,
        force_size=reacher.num_sensors_frc,
        hidden_size=rnn_config.hidden_size,
        output_size=reacher.num_actuators,
        activation=rnn_config.activation,
        smoothing_factor=alpha_from_tau(
            tau=rnn_config.tau, dt=reacher.model.opt.timestep
        ),
        use_bias=rnn_config.use_bias,
    )

    # Create evaluation environment
    eval_env = SequentialReachingEnv(
        plant=reacher,
        target_encoder=target_encoder,
        target_duration_distro=env_config.target_duration_distro,
        iti_distro=env_config.iti_distro,
        num_targets=env_config.num_targets,
        randomize_gravity=env_config.randomize_gravity,
        loss_weights=env_config.loss_weights,
    )

    return reacher, target_encoder, rnn, eval_env


"""
..######..##.....##....###............########..######.
.##....##.###...###...##.##...........##.......##....##
.##.......####.####..##...##..........##.......##......
.##.......##.###.##.##.....##.#######.######....######.
.##.......##.....##.#########.........##.............##
.##....##.##.....##.##.....##.........##.......##....##
..######..##.....##.##.....##.........########..######.
"""


def evaluate_population(
    pool: Pool, population, generation: int, training_config: TrainingConfig
):
    """
    Evaluate a population of individuals in parallel with load balancing.

    Uses chunking to improve load balancing across workers.
    """
    # Prepare arguments (params, seed)
    args_list = [(params, generation) for params in population]

    # Calculate chunk size for better load balancing
    pop_size = len(population)
    num_workers = training_config.num_workers
    chunk_size = max(
        1, pop_size // (num_workers * training_config.chunk_size_multiplier)
    )

    # Evaluate with chunking
    results = pool.starmap(evaluate_worker, args_list, chunksize=chunk_size)

    return results


def print_generation_stats(
    generation: int,
    losses,
    gen_time: float,
    total_time: float,
    pop_size: int,
    metrics: PerformanceMetrics,
):
    """Print statistics for current generation."""
    best_loss = min(losses)
    mean_loss = np.mean(losses)
    std_loss = np.std(losses)
    evals_per_sec = pop_size / gen_time

    # Get recent performance
    summary = metrics.get_summary()
    recent_best = summary.get("best_loss_recent", best_loss)

    print(
        f"Gen {generation:4d} | "
        f"Best: {best_loss:7.3f} (Recent: {recent_best:7.3f}) | "
        f"Mean: {mean_loss:7.3f} ± {std_loss:6.3f} | "
        f"Speed: {evals_per_sec:5.1f} eval/s | "
        f"ΔT: {gen_time:6.1f}s | "
        f"T: {total_time:6.1f}s"
    )


def evaluate_best_solution(
    optimizer: CMA,
    rnn: NeuroMuscularRNN,
    eval_env: SequentialReachingEnv,
    verbose: bool = True,
):
    """Evaluate the current best solution."""
    best_rnn = rnn.from_params(optimizer.mean)

    if verbose:
        print(f"  → Evaluating best solution...")

    eval_loss = -eval_env.evaluate(best_rnn, seed=0, render=True, log=True)

    if verbose:
        print(f"  → Evaluation loss: {eval_loss:.3f}")
        eval_env.plot()

    return eval_loss


def save_checkpoint(
    optimizer: CMA,
    generation: int,
    metrics: PerformanceMetrics,
    configs: Dict,
    checkpoint_dir: str,
):
    """Save complete training checkpoint with atomic write."""
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint = {
        "optimizer": optimizer,
        "generation": generation,
        "metrics": metrics,
        "configs": configs,
        "timestamp": time.time(),
    }

    filepath = os.path.join(checkpoint_dir, f"checkpoint_gen_{generation}.pkl")
    filepath_tmp = filepath + ".tmp"

    # Atomic write
    try:
        with open(filepath_tmp, "wb") as f:
            pickle.dump(checkpoint, f)
        os.rename(filepath_tmp, filepath)
        print(f"  → Checkpoint saved: {filepath}")
    except Exception as e:
        print(f"  → Warning: Checkpoint save failed: {e}")
        if os.path.exists(filepath_tmp):
            os.remove(filepath_tmp)


def cleanup_old_checkpoints(checkpoint_dir: str, keep_last_n: int = 5):
    """Keep only the most recent N checkpoints."""
    if not os.path.exists(checkpoint_dir):
        return

    # Find all checkpoint files
    checkpoint_files = [
        f
        for f in os.listdir(checkpoint_dir)
        if f.startswith("checkpoint_gen_") and f.endswith(".pkl")
    ]

    if len(checkpoint_files) <= keep_last_n:
        return

    # Sort by generation number
    def extract_gen(filename):
        try:
            return int(filename.split("_")[-1].replace(".pkl", ""))
        except:
            return -1

    checkpoint_files.sort(key=extract_gen)

    # Remove old checkpoints
    for old_checkpoint in checkpoint_files[:-keep_last_n]:
        try:
            os.remove(os.path.join(checkpoint_dir, old_checkpoint))
        except Exception as e:
            print(f"  → Warning: Could not remove old checkpoint {old_checkpoint}: {e}")


"""
.########.########.....###....####.##....##
....##....##.....##...##.##....##..###...##
....##....##.....##..##...##...##..####..##
....##....########..##.....##..##..##.##.##
....##....##...##...#########..##..##..####
....##....##....##..##.....##..##..##...###
....##....##.....##.##.....##.####.##....##
"""


def train(
    training_config: TrainingConfig,
    env_config: EnvConfig,
    rnn_config: RNNConfig,
    resume_from: str = None,
):
    """
    Main training loop with optimizations.

    Args:
        training_config: Training configuration
        env_config: Environment configuration
        rnn_config: RNN configuration
        resume_from: Optional path to checkpoint to resume from
    """
    print("=" * 80)
    print("PARALLEL CMA-ES TRAINING")
    print("=" * 80)

    # Setup components
    reacher, target_encoder, rnn, eval_env = setup_components(env_config, rnn_config)

    # Create configuration dictionaries for workers
    rnn_config_dict = create_rnn_config(reacher, target_encoder, rnn_config)
    env_config_dict = create_env_config(reacher, env_config)

    # Initialize or resume
    if resume_from:
        print(f"\nResuming from checkpoint: {resume_from}")
        with open(resume_from, "rb") as f:
            checkpoint = pickle.load(f)
        optimizer = checkpoint["optimizer"]
        start_generation = checkpoint["generation"] + 1
        metrics = checkpoint.get("metrics", PerformanceMetrics())
    else:
        optimizer = CMA(
            mean=rnn.get_params(),
            sigma=training_config.initial_sigma,
            # bounds=rnn.get_bounds(),
        )
        start_generation = 0
        metrics = PerformanceMetrics()

    # Store configs for checkpointing
    configs = {
        "training": asdict(training_config),
        "env": asdict(env_config),
        "rnn": asdict(rnn_config),
    }

    print(f"\nUsing {training_config.num_workers} parallel workers")
    print(f"Population size: {optimizer.population_size}")
    print(f"Chunk size multiplier: {training_config.chunk_size_multiplier}")
    print(f"Expected speedup: ~{training_config.num_workers}x")
    print(f"Optimizing {rnn.num_weights} weights and {rnn.num_biases} biases ")
    print("=" * 80)

    # Create worker pool with initialization
    print("\nInitializing worker pool...")
    pool = Pool(
        processes=training_config.num_workers,
        initializer=init_worker,
        initargs=(rnn_config_dict, env_config_dict),
    )
    print("Worker pool ready!")

    # Training state
    start_time = time.time()

    try:
        for generation in range(start_generation, training_config.num_generations):
            gen_start = time.time()

            # Generate population
            population = [optimizer.ask() for _ in range(optimizer.population_size)]

            # Evaluate population in parallel (OPTIMIZED)
            pop_losses = evaluate_population(
                pool, population, generation, training_config
            )

            # Update optimizer
            solutions = list(zip(population, pop_losses))
            optimizer.tell(solutions)

            # Track metrics
            gen_time = time.time() - gen_start
            total_time = time.time() - start_time
            best_loss = min(pop_losses)
            mean_loss = np.mean(pop_losses)

            metrics.add_generation(gen_time, best_loss, mean_loss)

            # Print statistics
            print_generation_stats(
                generation,
                pop_losses,
                gen_time,
                total_time,
                optimizer.population_size,
                metrics,
            )

            # Periodic evaluation
            if generation % training_config.eval_interval == 0:
                evaluate_best_solution(optimizer, rnn, eval_env)

            # Save checkpoints
            if generation % training_config.checkpoint_interval == 0 and generation > 0:
                save_checkpoint(
                    optimizer,
                    generation,
                    metrics,
                    configs,
                    training_config.checkpoint_dir,
                )
                cleanup_old_checkpoints(
                    training_config.checkpoint_dir,
                    training_config.keep_last_n_checkpoints,
                )

    except KeyboardInterrupt:
        print("\n" + "=" * 80)
        print("Training interrupted by user")
        print("=" * 80)

        # Save interrupt checkpoint
        save_checkpoint(
            optimizer, generation, metrics, configs, training_config.checkpoint_dir
        )

    finally:
        # Cleanup
        print("\nCleaning up...")
        pool.close()
        pool.join()
        eval_env.plant.close()
        print("Done!")

        # Final statistics
        summary = metrics.get_summary()
        print("\n" + "=" * 80)
        print("TRAINING COMPLETE")
        print("=" * 80)
        print(f"Total time: {summary.get('total_time', 0):.1f}s")
        print(f"Total generations: {generation + 1}")
        print(
            f"Best loss achieved: {summary.get('best_loss_overall', float('inf')):.3f}"
        )
        print(f"Mean generation time: {summary.get('mean_gen_time', 0):.2f}s")


"""
.##.....##....###....####.##....##
.###...###...##.##....##..###...##
.####.####..##...##...##..####..##
.##.###.##.##.....##..##..##.##.##
.##.....##.#########..##..##..####
.##.....##.##.....##..##..##...###
.##.....##.##.....##.####.##....##
"""


def main():
    """Main entry point."""
    # Configuration
    training_config = TrainingConfig(
        num_generations=10000,
        initial_sigma=1.3,
        checkpoint_interval=1000,
        eval_interval=10,
        chunk_size_multiplier=4,  # Better load balancing
        keep_last_n_checkpoints=5,  # Save space
    )

    env_config = EnvConfig(
        loss_weights={
            "distance": 1.0,
            "energy": 0.05,
        },
        randomize_gravity=False,
    )

    rnn_config = RNNConfig(
        hidden_size=25,
        tau=10e-3,
        activation=tanh,
        use_bias=False,
    )

    # Run training
    train(training_config, env_config, rnn_config)


if __name__ == "__main__":
    main()
