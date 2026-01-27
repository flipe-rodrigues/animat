"""
Usage example and comparison: CMA-ES vs RL+Distillation

This script demonstrates how to use both approaches and compares their performance.
"""

import numpy as np
import matplotlib.pyplot as plt
from plants import SequentialReacher
from encoders import GridTargetEncoder
from environments import SequentialReachingEnv
from networks import NeuroMuscularRNN
from utils import tanh, alpha_from_tau

# Import the new RL+Distillation approach
from rl_distillation import (
    train_with_rl_distillation,
    RLConfig,
    DistillationConfig,
    PPOTrainer,
    MLPActor,
    MLPCritic,
)

# Import CMA-ES
from cmaes import CMA


# ============================================================================
# SETUP ENVIRONMENT (same for both approaches)
# ============================================================================
"""
..######..########.########.##.....##.########.
.##....##.##..........##....##.....##.##.....##
.##.......##..........##....##.....##.##.....##
..######..######......##....##.....##.########.
.......##.##..........##....##.....##.##.......
.##....##.##..........##....##.....##.##.......
..######..########....##.....#######..##.......
"""


def setup_environment():
    """Setup the standard environment"""
    # Initialize the plant
    reacher = SequentialReacher(plant_xml_file="arm.xml")

    # Initialize the target encoder
    target_encoder = GridTargetEncoder(
        grid_size=8,
        x_bounds=reacher.get_workspace_bounds()[0],
        y_bounds=reacher.get_workspace_bounds()[1],
        sigma=0.25,
    )

    # Initialize the environment/task
    env = SequentialReachingEnv(
        plant=reacher,
        target_encoder=target_encoder,
        target_duration_distro={"mean": 3, "min": 1, "max": 6},
        iti_distro={"mean": 1, "min": 0, "max": 3},
        num_targets=10,
        randomize_gravity=True,
        loss_weights={
            "distance": 1,
            "energy": 0.1,
            "ridge": 0,
            "lasso": 0,
        },
    )

    return reacher, target_encoder, env


def create_rnn(reacher, target_encoder):
    """Create a standard RNN"""
    rnn = NeuroMuscularRNN(
        input_size_tgt=target_encoder.size,
        input_size_len=reacher.num_sensors_len,
        input_size_vel=reacher.num_sensors_vel,
        input_size_frc=reacher.num_sensors_frc,
        hidden_size=25,
        output_size=reacher.num_actuators,
        activation=tanh,
        smoothing_factor=alpha_from_tau(tau=10e-3, dt=reacher.model.opt.timestep),
    )
    return rnn


# ============================================================================
# APPROACH 1: CMA-ES (Original)
# ============================================================================

"""
..######..##.....##....###............########..######.
.##....##.###...###...##.##...........##.......##....##
.##.......####.####..##...##..........##.......##......
.##.......##.###.##.##.....##.#######.######....######.
.##.......##.....##.#########.........##.............##
.##....##.##.....##.##.....##.........##.......##....##
..######..##.....##.##.....##.........########..######.
"""


def train_with_cmaes(env, rnn, num_generations=1000):
    """
    Train RNN using CMA-ES

    Args:
        env: Environment
        rnn: RNN to train
        num_generations: Number of CMA-ES generations

    Returns:
        Trained RNN and fitness history
    """
    print("=" * 80)
    print("Training with CMA-ES")
    print("=" * 80)

    optimizer = CMA(mean=rnn.get_params(), sigma=1.3)
    fitnesses = []

    for gg in range(num_generations):
        solutions = []

        for ii in range(optimizer.population_size):
            x = optimizer.ask()
            fitness = -env.evaluate(rnn.from_params(x), seed=gg)
            solutions.append((x, fitness))
            fitnesses.append((gg, ii, fitness))

            if ii == 0:  # Print first of each generation
                print(f"Generation {gg}: Fitness = {fitness:.4f}")

        optimizer.tell(solutions)

        # Periodic evaluation
        if gg % 100 == 0:
            best_rnn = rnn.from_params(optimizer.mean)
            eval_fitness = -env.evaluate(best_rnn, seed=0, render=False, log=False)
            print(f"  Evaluation fitness: {eval_fitness:.4f}")

    # Get best RNN
    best_rnn = rnn.from_params(optimizer.mean)
    return best_rnn, fitnesses


# ============================================================================
# APPROACH 2: RL + Distillation (New)
# ============================================================================

"""
.########..##......
.##.....##.##......
.##.....##.##......
.########..##......
.##...##...##......
.##....##..##......
.##.....##.########
"""


def train_with_rl(env, rnn, rl_iterations=1000, distillation_epochs=100):
    """
    Train RNN using RL + Distillation

    Args:
        env: Environment
        rnn: RNN to train
        rl_iterations: Number of PPO iterations
        distillation_epochs: Number of distillation epochs

    Returns:
        Trained RNN and statistics
    """
    print("=" * 80)
    print("Training with RL + Distillation")
    print("=" * 80)

    # Configure RL training
    rl_config = RLConfig(
        num_iterations=rl_iterations,
        buffer_size=2048,
        batch_size=64,
        num_epochs=10,
        log_interval=10,
    )

    # Configure distillation
    distillation_config = DistillationConfig(
        num_epochs=distillation_epochs,
        num_trajectories=500,
        batch_size=128,
        log_interval=10,
    )

    # Train
    trained_rnn, teacher, stats = train_with_rl_distillation(
        env=env,
        rnn=rnn,
        rl_config=rl_config,
        distillation_config=distillation_config,
    )

    return trained_rnn, stats


# ============================================================================
# COMPARISON AND VISUALIZATION
# ============================================================================

"""
..######...#######..##.....##.########.....###....########..####..######...#######..##....##
.##....##.##.....##.###...###.##.....##...##.##...##.....##..##..##....##.##.....##.###...##
.##.......##.....##.####.####.##.....##..##...##..##.....##..##..##.......##.....##.####..##
.##.......##.....##.##.###.##.########..##.....##.########...##...######..##.....##.##.##.##
.##.......##.....##.##.....##.##........#########.##...##....##........##.##.....##.##..####
.##....##.##.....##.##.....##.##........##.....##.##....##...##..##....##.##.....##.##...###
..######...#######..##.....##.##........##.....##.##.....##.####..######...#######..##....##
"""


def evaluate_policy(env, rnn, num_trials=10):
    """
    Evaluate a policy over multiple trials

    Args:
        env: Environment
        rnn: Policy to evaluate
        num_trials: Number of evaluation trials

    Returns:
        Mean and std of fitness
    """
    fitnesses = []

    for trial in range(num_trials):
        fitness = -env.evaluate(rnn, seed=trial, render=False, log=False)
        fitnesses.append(fitness)

    return np.mean(fitnesses), np.std(fitnesses)


def plot_comparison(cmaes_fitnesses, rl_stats):
    """
    Plot comparison of both approaches

    Args:
        cmaes_fitnesses: List of (generation, individual, fitness) from CMA-ES
        rl_stats: Statistics from RL training
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: CMA-ES convergence
    cmaes_array = np.array(cmaes_fitnesses)
    generations = np.unique(cmaes_array[:, 0])

    gen_means = []
    gen_stds = []
    for gen in generations:
        gen_fitness = cmaes_array[cmaes_array[:, 0] == gen][:, 2]
        gen_means.append(np.mean(gen_fitness))
        gen_stds.append(np.std(gen_fitness))

    gen_means = np.array(gen_means)
    gen_stds = np.array(gen_stds)

    axes[0].plot(generations, gen_means, label="CMA-ES", color="blue")
    axes[0].fill_between(
        generations, gen_means - gen_stds, gen_means + gen_stds, alpha=0.2, color="blue"
    )
    axes[0].set_xlabel("Generation")
    axes[0].set_ylabel("Fitness")
    axes[0].set_title("CMA-ES Training")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: RL training
    if rl_stats["rl_stats"] is not None:
        iterations = rl_stats["rl_stats"]["iteration"]
        rewards = rl_stats["rl_stats"]["episode_reward_mean"]
        reward_stds = rl_stats["rl_stats"]["episode_reward_std"]

        axes[1].plot(iterations, rewards, label="PPO", color="green")
        axes[1].fill_between(
            iterations,
            np.array(rewards) - np.array(reward_stds),
            np.array(rewards) + np.array(reward_stds),
            alpha=0.2,
            color="green",
        )
        axes[1].set_xlabel("Iteration")
        axes[1].set_ylabel("Episode Reward")
        axes[1].set_title("PPO Training (Teacher)")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    # Plot 3: Distillation training
    epochs = rl_stats["distillation_stats"]["epoch"]
    action_loss = rl_stats["distillation_stats"]["action_loss"]
    kl_loss = rl_stats["distillation_stats"]["kl_loss"]

    axes[2].plot(epochs, action_loss, label="Action Loss", color="orange")
    axes[2].plot(epochs, kl_loss, label="KL Loss", color="red")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Loss")
    axes[2].set_title("Distillation Training (Student RNN)")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].set_yscale("log")

    plt.tight_layout()
    plt.savefig("training_comparison.png", dpi=150)
    plt.show()


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
    """Main comparison script"""

    # Setup
    reacher, target_encoder, env = setup_environment()

    # ========================================================================
    # Train with both approaches
    # ========================================================================

    # Approach 1: CMA-ES
    print("\n" + "=" * 80)
    print("APPROACH 1: CMA-ES")
    print("=" * 80 + "\n")

    rnn_cmaes = create_rnn(reacher, target_encoder)
    trained_rnn_cmaes, cmaes_fitnesses = train_with_cmaes(
        env, rnn_cmaes, num_generations=10  # Reduced for demo
    )

    # Approach 2: RL + Distillation
    print("\n" + "=" * 80)
    print("APPROACH 2: RL + Distillation")
    print("=" * 80 + "\n")

    rnn_rl = create_rnn(reacher, target_encoder)
    trained_rnn_rl, rl_stats = train_with_rl(
        env, rnn_rl, rl_iterations=10, distillation_epochs=50  # Reduced for demo
    )

    # ========================================================================
    # Evaluate both
    # ========================================================================

    print("\n" + "=" * 80)
    print("EVALUATION")
    print("=" * 80)

    print("\nEvaluating CMA-ES policy...")
    cmaes_mean, cmaes_std = evaluate_policy(env, trained_rnn_cmaes, num_trials=10)
    print(f"CMA-ES: {cmaes_mean:.4f} ± {cmaes_std:.4f}")

    print("\nEvaluating RL+Distillation policy...")
    rl_mean, rl_std = evaluate_policy(env, trained_rnn_rl, num_trials=10)
    print(f"RL+Distillation: {rl_mean:.4f} ± {rl_std:.4f}")

    # ========================================================================
    # Comparison
    # ========================================================================

    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)

    print(f"\nCMA-ES:             {cmaes_mean:.4f} ± {cmaes_std:.4f}")
    print(f"RL+Distillation:    {rl_mean:.4f} ± {rl_std:.4f}")

    if rl_mean > cmaes_mean:
        improvement = ((rl_mean - cmaes_mean) / abs(cmaes_mean)) * 100
        print(f"\nRL+Distillation is {improvement:.1f}% better")
    else:
        improvement = ((cmaes_mean - rl_mean) / abs(rl_mean)) * 100
        print(f"\nCMA-ES is {improvement:.1f}% better")

    # Plot
    plot_comparison(cmaes_fitnesses, rl_stats)

    # ========================================================================
    # Visual Comparison
    # ========================================================================

    print("\n" + "=" * 80)
    print("VISUAL COMPARISON")
    print("=" * 80)
    print("\nRendering CMA-ES policy...")
    env.evaluate(trained_rnn_cmaes, seed=0, render=True, log=True)
    env.plot()

    print("\nRendering RL+Distillation policy...")
    env.evaluate(trained_rnn_rl, seed=0, render=True, log=True)
    env.plot()


# ============================================================================
# PROS AND CONS
# ============================================================================

"""
CMA-ES APPROACH:
Pros:
  + No gradient computation needed
  + Works well for small parameter spaces
  + Robust to noisy objectives
  + Simple to implement
  + No hyperparameter tuning for learning rates

Cons:
  - Slow for large networks (scales poorly with parameters)
  - Requires many evaluations per generation
  - No knowledge transfer between generations
  - Cannot leverage neural network structure
  - Sample inefficient

RL + DISTILLATION APPROACH:
Pros:
  + Gradient-based learning is much faster
  + Sample efficient (reuses data via replay)
  + Can train larger networks
  + Knowledge distillation allows architecture flexibility
  + Can leverage GPU acceleration
  + Teacher can be larger/more expressive than student

Cons:
  - More complex implementation
  - Requires hyperparameter tuning
  - May need careful initialization
  - Potential for overfitting
  - Two-stage process adds complexity

HYBRID APPROACH (Best of Both):
  1. Train teacher with RL (fast, gradient-based)
  2. Distill to RNN student (preserves recurrent structure)
  3. Fine-tune RNN with CMA-ES (polish the solution)
  
This combines:
  - Fast initial learning (RL)
  - Knowledge transfer (Distillation)  
  - Robust fine-tuning (CMA-ES)
"""


if __name__ == "__main__":
    # Quick demo
    print("This script compares CMA-ES vs RL+Distillation approaches")
    print("Uncomment main() to run the full comparison (takes time!)")

    main()  # Uncomment to run

    print("\nTo use the RL+Distillation approach in your code:")
    print("=" * 80)
    print(
        """
from rl_distillation import train_with_rl_distillation, RLConfig, DistillationConfig

# Setup your environment as usual
reacher = SequentialReacher(plant_xml_file="arm.xml")
target_encoder = GridTargetEncoder(...)
env = SequentialReachingEnv(...)
rnn = NeuroMuscularRNN(...)

# Train with RL + Distillation
trained_rnn, teacher, stats = train_with_rl_distillation(
    env=env,
    rnn=rnn,
    rl_config=RLConfig(num_iterations=1000),
    distillation_config=DistillationConfig(num_epochs=100)
)

# Use the trained RNN
env.evaluate(trained_rnn, render=True, log=True)
env.plot()
    """
    )
