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
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Type
import multiprocessing as mp
from multiprocessing.pool import Pool
import numpy as np
from cmaes import CMA
from plants import SequentialReacher
from encoders import GridTargetEncoder
from environments import SequentialReachingEnv
from networks import NeuroMuscularRNN, AlphaOnlyRNN, FullRNN
from utils import relu, tanh, alpha_from_tau
from workers import init_worker, evaluate_worker


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
    num_generations: int = 10000
    initial_sigma: float = 1.3
    checkpoint_interval: int = 1000
    eval_interval: int = 10
    num_workers: int = mp.cpu_count() - 1
    checkpoint_dir: str = "../../models"
    chunk_size_multiplier: int = 4
    keep_last_n_checkpoints: int = 5
    seed: Optional[int] = None


@dataclass
class RNNConfig:
    rnn_class: Type[NeuroMuscularRNN] = FullRNN
    hidden_size: int = 25
    tau: float = 10e-3
    activation: Callable = tanh
    use_bias: bool = True


@dataclass
class EnvConfig:
    plant_xml_file: str = "arm.xml"
    grid_size: int = 8
    sigma: float = 0.25
    num_targets: int = 10
    randomize_gravity: bool = True

    # Use field with default_factory for mutable defaults
    target_duration_distro: Dict = field(
        default_factory=lambda: {"mean": 3, "min": 1, "max": 6}
    )
    iti_distro: Dict = field(default_factory=lambda: {"mean": 1, "min": 0, "max": 3})
    loss_weights: Dict = field(
        default_factory=lambda: {
            "distance": 1.0,
            "energy": 0.1,
            "ridge": 0.001,
            "lasso": 0.001,
        }
    )


@dataclass
class PerformanceMetrics:
    generation_times: List[float] = field(default_factory=list)
    best_losses: List[float] = field(default_factory=list)
    mean_losses: List[float] = field(default_factory=list)

    def add_generation(self, gen_time, best_loss, mean_loss):
        self.generation_times.append(gen_time)
        self.best_losses.append(best_loss)
        self.mean_losses.append(mean_loss)

    def get_summary(self):
        if not self.generation_times:
            return {}

        recent_times = self.generation_times[-100:]
        recent_losses = (
            self.best_losses[-10:] if len(self.best_losses) >= 10 else self.best_losses
        )

        return {
            "mean_gen_time": np.mean(recent_times),
            "total_time": sum(self.generation_times),
            "best_loss_overall": min(self.best_losses),
            "best_loss_recent": min(recent_losses),
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


def create_configs_for_workers(
    reacher: SequentialReacher,
    target_encoder: GridTargetEncoder,
    rnn_config: RNNConfig,
    env_config: EnvConfig,
) -> tuple[Dict, Dict]:
    """
    Create configuration dictionaries for worker initialization.

    This creates lightweight config dicts that workers use to initialize
    their own plant, encoder, and RNN instances.
    """
    workspace_bounds = reacher.get_workspace_bounds()

    rnn_config_dict = {
        "rnn_class": rnn_config.rnn_class,
        "target_size": target_encoder.size,
        "length_size": reacher.num_sensors_len,
        "velocity_size": reacher.num_sensors_vel,
        "force_size": reacher.num_sensors_frc,
        "hidden_size": rnn_config.hidden_size,
        "output_size": reacher.num_actuators,
        "activation": rnn_config.activation,
        "smoothing_factor": (
            1.0
            if rnn_config.tau == 0
            else alpha_from_tau(tau=rnn_config.tau, dt=reacher.model.opt.timestep)
        ),
        "use_bias": rnn_config.use_bias,
    }

    env_config_dict = {
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

    return rnn_config_dict, env_config_dict


def setup_components(env_config: EnvConfig, rnn_config: RNNConfig):
    """
    Setup main process components for evaluation and checkpointing.

    These components are used in the main process for periodic evaluations
    and visualizations, not for parallel training.
    """
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

    # Create RNN using the specified class
    rnn = rnn_config.rnn_class(
        target_size=target_encoder.size,
        length_size=reacher.num_sensors_len,
        velocity_size=reacher.num_sensors_vel,
        force_size=reacher.num_sensors_frc,
        hidden_size=rnn_config.hidden_size,
        output_size=reacher.num_actuators,
        activation=rnn_config.activation,
        smoothing_factor=(
            1.0
            if rnn_config.tau == 0
            else alpha_from_tau(tau=rnn_config.tau, dt=reacher.model.opt.timestep)
        ),
        use_bias=rnn_config.use_bias,
    )

    # Create lightweight evaluation environment
    eval_env = SequentialReachingEnv(
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
    pool: Pool, population, generation: int, chunk_size: int, seed: Optional[int] = None
):
    """Evaluate entire population in parallel"""
    args_list = [
        (params, generation if seed is None else seed) for params in population
    ]
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
    """Print formatted generation statistics"""
    best_loss = min(losses)
    mean_loss = np.mean(losses)
    std_loss = np.std(losses)
    evals_per_sec = pop_size / gen_time

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
    plant: SequentialReacher,
    target_encoder: GridTargetEncoder,
    eval_env: SequentialReachingEnv,
    verbose: bool = True,
    seed: int = 42,
):
    """
    Evaluate and visualize the best solution.

    Now passes plant and encoder as arguments to evaluate().
    """
    best_rnn = rnn.from_params(optimizer.mean)

    if verbose:
        print(f"  → Evaluating best solution...")

    # Pass plant and encoder as arguments
    eval_loss = -eval_env.evaluate(
        best_rnn,
        plant,
        target_encoder,
        seed=seed,
        render=True,
        render_speed=1.0,
        log=True,
    )

    if verbose:
        print(f"  → Evaluation loss: {eval_loss:.3f}")
        eval_env.plot()

    return eval_loss


def save_checkpoint(
    optimizer: CMA,
    generation: int,
    metrics: PerformanceMetrics,
    rnn_class_name: str,
    checkpoint_dir: str,
):
    """Save training checkpoint"""
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint = {
        "optimizer": optimizer,
        "generation": generation,
        "metrics": metrics,
        "rnn_class_name": rnn_class_name,
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
    """Remove old checkpoints, keeping only the most recent N"""
    if not os.path.exists(checkpoint_dir):
        return

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
    resume_from: Optional[str] = None,
):
    """Main training loop with parallel CMA-ES"""
    print("=" * 80)
    print("PARALLEL CMA-ES TRAINING")
    print("=" * 80)

    # Setup main process components
    reacher, target_encoder, rnn, eval_env = setup_components(env_config, rnn_config)

    # Create configuration dictionaries for workers
    rnn_config_dict, env_config_dict = create_configs_for_workers(
        reacher, target_encoder, rnn_config, env_config
    )

    # Initialize or resume
    if resume_from:
        print(f"\nResuming from checkpoint: {resume_from}")
        with open(resume_from, "rb") as f:
            checkpoint = pickle.load(f)
        optimizer = checkpoint["optimizer"]
        start_generation = checkpoint["generation"] + 1
        metrics = checkpoint.get("metrics", PerformanceMetrics())
    else:
        optimizer = CMA(mean=rnn.get_params(), sigma=training_config.initial_sigma)
        start_generation = 0
        metrics = PerformanceMetrics()

    # Calculate chunk size once
    chunk_size = max(
        1,
        optimizer.population_size
        // (training_config.num_workers * training_config.chunk_size_multiplier),
    )

    print(f"\nRNN Type: {rnn_config.rnn_class.__name__}")
    print(f"Using {training_config.num_workers} parallel workers")
    print(f"Population size: {optimizer.population_size}")
    print(f"Chunk size: {chunk_size}")
    print(f"Seed: {training_config.seed}")
    print(f"Optimizing {rnn.num_weights} weights and {rnn.num_biases} biases")
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

            # Generate and evaluate population
            population = [optimizer.ask() for _ in range(optimizer.population_size)]
            pop_losses = evaluate_population(
                pool, population, generation, chunk_size, seed=training_config.seed
            )

            # Update optimizer
            optimizer.tell(list(zip(population, pop_losses)))

            # Track metrics
            gen_time = time.time() - gen_start
            total_time = time.time() - start_time
            metrics.add_generation(gen_time, min(pop_losses), np.mean(pop_losses))

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
                evaluate_best_solution(
                    optimizer,
                    rnn,
                    reacher,
                    target_encoder,
                    eval_env,
                    seed=training_config.seed,
                )

            # Save checkpoints
            if generation % training_config.checkpoint_interval == 0 and generation > 0:
                save_checkpoint(
                    optimizer,
                    generation,
                    metrics,
                    rnn_config.rnn_class.__name__,
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
        save_checkpoint(
            optimizer,
            generation,
            metrics,
            rnn_config.rnn_class.__name__,
            training_config.checkpoint_dir,
        )

    finally:
        # Cleanup
        print("\nCleaning up...")
        pool.close()
        pool.join()
        reacher.close()
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
    training_config = TrainingConfig(
        num_generations=10000,
        initial_sigma=1.3,
        checkpoint_interval=1000,
        eval_interval=10,
        chunk_size_multiplier=4,
        keep_last_n_checkpoints=5,
        seed=None,
    )
    env_config = EnvConfig(
        # plant_xml_file="C:\\Users\\flipe\\Documents\\GitHub\\myosuite\\myosuite\\simhive\\myo_sim\\arm\\myoarm.xml",
        loss_weights={"distance": 1.0, "energy": 0.05, "ridge": 0.001, "lasso": 0.001},
        randomize_gravity=False,
    )
    rnn_config = RNNConfig(
        rnn_class=AlphaOnlyRNN,  # Change to AlphaOnlyRNN or FullRNN
        hidden_size=25,
        tau=0,
        activation=tanh,
        use_bias=True,
    )
    train(training_config, env_config, rnn_config)


if __name__ == "__main__":
    main()
