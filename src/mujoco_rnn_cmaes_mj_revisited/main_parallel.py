# %%
"""
Parallel CMA-ES optimization for neuromuscular control.

This script trains a recurrent neural network controller using CMA-ES
with parallel fitness evaluation across multiple worker processes.

Shows how to use both simple shaping and potential-based shaping.
"""

import pickle
import time
from dataclasses import dataclass
from typing import Callable, Dict
import multiprocessing as mp
from multiprocessing.pool import Pool

import numpy as np
from cmaes import CMA

from plants import SequentialReacher
from encoders import GridTargetEncoder
from environments import SequentialReachingEnv  # Use modified version
from networks import NeuroMuscularRNN
from utils import tanh, alpha_from_tau

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    num_generations: int = 10000
    initial_sigma: float = 1.3
    checkpoint_interval: int = 1000
    eval_interval: int = 10
    num_workers: int = mp.cpu_count()
    checkpoint_dir: str = "../../models"


@dataclass
class RNNConfig:
    """Configuration for RNN architecture."""
    hidden_size: int = 25
    tau: float = 10e-3
    activation: Callable = tanh


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
    
    # NEW: Shaping configuration
    use_potential_shaping: bool = True  # Set to True for potential-based
    gamma: float = 0.99  # Discount factor (for potential shaping)

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


# ============================================================================
# WORKER FUNCTIONS
# ============================================================================

def evaluate_worker(params: np.ndarray, seed: int, 
                   rnn_config: Dict, env_config: Dict) -> float:
    """Evaluate a single individual's fitness."""
    try:
        # Create RNN
        rnn = NeuroMuscularRNN(**rnn_config)
        rnn.set_params(params)

        # Create environment components
        reacher = SequentialReacher(**env_config["plant"])
        target_encoder = GridTargetEncoder(**env_config["encoder"])
        env = SequentialReachingEnv(
            plant=reacher,
            target_encoder=target_encoder,
            **env_config["env"]
        )

        # Evaluate fitness
        fitness = -env.evaluate(rnn, seed=seed, render=False, log=False)

        # Cleanup
        env.plant.close()
        del env, reacher, rnn

        return fitness

    except Exception as e:
        print(f"Worker error (seed {seed}): {e}")
        import traceback
        traceback.print_exc()
        return -1e10


# ============================================================================
# SETUP FUNCTIONS
# ============================================================================

def create_rnn_config(reacher: SequentialReacher, 
                     target_encoder: GridTargetEncoder,
                     rnn_config: RNNConfig) -> Dict:
    """Create RNN configuration dictionary for workers."""
    return {
        "input_size_tgt": target_encoder.size,
        "input_size_len": reacher.num_sensors_len,
        "input_size_vel": reacher.num_sensors_vel,
        "input_size_frc": reacher.num_sensors_frc,
        "hidden_size": rnn_config.hidden_size,
        "output_size": reacher.num_actuators,
        "activation": rnn_config.activation,
        "smoothing_factor": alpha_from_tau(
            tau=rnn_config.tau, 
            dt=reacher.model.opt.timestep
        ),
    }


def create_env_config(reacher: SequentialReacher,
                     env_config: EnvConfig) -> Dict:
    """Create environment configuration dictionary for workers."""
    workspace_bounds = reacher.get_workspace_bounds()

    return {
        "plant": {
            "plant_xml_file": env_config.plant_xml_file
        },
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
            "use_potential_shaping": env_config.use_potential_shaping,
            "gamma": env_config.gamma,
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
        input_size_tgt=target_encoder.size,
        input_size_len=reacher.num_sensors_len,
        input_size_vel=reacher.num_sensors_vel,
        input_size_frc=reacher.num_sensors_frc,
        hidden_size=rnn_config.hidden_size,
        output_size=reacher.num_actuators,
        activation=rnn_config.activation,
        smoothing_factor=alpha_from_tau(
            tau=rnn_config.tau,
            dt=reacher.model.opt.timestep
        ),
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
        use_potential_shaping=env_config.use_potential_shaping,
        gamma=env_config.gamma,
    )

    return reacher, target_encoder, rnn, eval_env


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def evaluate_population(pool: Pool, population, generation: int,
                       rnn_config_dict: Dict, env_config_dict: Dict):
    """Evaluate a population of individuals in parallel."""
    args_list = [
        (params, generation, rnn_config_dict, env_config_dict)
        for params in population
    ]
    return pool.starmap(evaluate_worker, args_list)


def print_generation_stats(generation: int, fitnesses, gen_time: float,
                          total_time: float, pop_size: int):
    """Print statistics for current generation."""
    best_fitness = min(fitnesses)
    mean_fitness = np.mean(fitnesses)
    std_fitness = np.std(fitnesses)
    evals_per_sec = pop_size / gen_time

    print(
        f"Gen {generation:4d} | "
        f"Best: {best_fitness:7.3f} | "
        f"Mean: {mean_fitness:7.3f} ± {std_fitness:6.3f} | "
        f"Speed: {evals_per_sec:5.1f} eval/s | "
        f"ΔT: {gen_time:6.1f}s | "
        f"T: {total_time:6.1f}s"
    )


def evaluate_best_solution(optimizer: CMA, rnn: NeuroMuscularRNN,
                          eval_env: SequentialReachingEnv):
    """Evaluate the current best solution."""
    best_rnn = rnn.from_params(optimizer.mean)
    print(f"  → Evaluating best solution...")
    eval_fitness = -eval_env.evaluate(best_rnn, seed=0, render=True, log=True)
    print(f"  → Evaluation fitness: {eval_fitness:.3f}")
    eval_env.plot()


def save_checkpoint(optimizer: CMA, generation: int, checkpoint_dir: str):
    """Save optimizer checkpoint."""
    filepath = f"{checkpoint_dir}/optimizer_gen_{generation}_parallel.pkl"
    with open(filepath, "wb") as f:
        pickle.dump(optimizer, f)
    print(f"  → Checkpoint saved: {filepath}")


def train(training_config: TrainingConfig, env_config: EnvConfig,
          rnn_config: RNNConfig):
    """Main training loop."""
    print("=" * 80)
    print("PARALLEL CMA-ES OPTIMIZATION")
    print("=" * 80)
    
    # Print shaping configuration
    if env_config.use_potential_shaping:
        print("\n🎯 Using POTENTIAL-BASED SHAPING")
        print("   → Rewards improvement (policy invariant)")
    else:
        print("\n📊 Using SIMPLE SHAPING")
        print("   → Direct cost on distance")
    
    print("=" * 80)

    # Setup components
    reacher, target_encoder, rnn, eval_env = setup_components(
        env_config, rnn_config
    )

    # Create configuration dictionaries for workers
    rnn_config_dict = create_rnn_config(reacher, target_encoder, rnn_config)
    env_config_dict = create_env_config(reacher, env_config)

    # Initialize optimizer
    optimizer = CMA(mean=rnn.get_params(), sigma=training_config.initial_sigma)

    print(f"\nUsing {training_config.num_workers} parallel workers")
    print(f"Population size: {optimizer.population_size}")
    print(f"Expected speedup: ~{training_config.num_workers}x")
    print("=" * 80)

    # Create worker pool
    pool = Pool(processes=training_config.num_workers)

    # Training state
    all_fitnesses = []
    start_time = time.time()

    try:
        for generation in range(training_config.num_generations):
            gen_start = time.time()

            # Generate population
            population = [optimizer.ask() for _ in range(optimizer.population_size)]

            # Evaluate population in parallel
            pop_fitnesses = evaluate_population(
                pool, population, generation, rnn_config_dict, env_config_dict
            )

            # Update optimizer
            solutions = list(zip(population, pop_fitnesses))
            optimizer.tell(solutions)

            # Track fitnesses
            for idx, fitness in enumerate(pop_fitnesses):
                all_fitnesses.append((generation, idx, fitness))

            # Print statistics
            gen_time = time.time() - gen_start
            total_time = time.time() - start_time
            print_generation_stats(
                generation, pop_fitnesses, gen_time, total_time,
                optimizer.population_size
            )

            # Periodic evaluation
            if generation % training_config.eval_interval == 0 and generation > 0:
                evaluate_best_solution(optimizer, rnn, eval_env)

            # Save checkpoints
            if generation % training_config.checkpoint_interval == 0 and generation > 0:
                save_checkpoint(optimizer, generation, training_config.checkpoint_dir)

    except KeyboardInterrupt:
        print("\n" + "=" * 80)
        print("Training interrupted by user")
        print("=" * 80)

    finally:
        # Cleanup
        print("\nCleaning up...")
        pool.close()
        pool.join()
        eval_env.plant.close()
        print("Done!")

        # Final statistics
        print("\n" + "=" * 80)
        print("TRAINING COMPLETE")
        print("=" * 80)
        print(f"Total time: {time.time() - start_time:.1f}s")
        print(f"Total generations: {generation + 1}")
        if all_fitnesses:
            best_overall = min(f[2] for f in all_fitnesses)
            print(f"Best fitness achieved: {best_overall:.3f}")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point."""
    # Configuration
    training_config = TrainingConfig(
        num_generations=10000,
        initial_sigma=1.3,
        checkpoint_interval=1000,
        eval_interval=10,
    )
    
    env_config = EnvConfig(
        loss_weights={
            "distance": 1.0,
            "energy": 0.1,
        },
        randomize_gravity=False,
        use_potential_shaping=True,  # 🎯 SET THIS FOR POTENTIAL-BASED SHAPING
        gamma=0.99,
    )
    
    rnn_config = RNNConfig()

    # Run training
    train(training_config, env_config, rnn_config)


if __name__ == "__main__":
    main()