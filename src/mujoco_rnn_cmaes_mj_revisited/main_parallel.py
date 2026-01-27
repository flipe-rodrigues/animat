from plants import SequentialReacher
from encoders import GridTargetEncoder
from environments import SequentialReachingEnv
from networks import NeuroMuscularRNN
from utils import tanh, alpha_from_tau
from cmaes import CMA
import pickle
import numpy as np
import multiprocessing as mp
from multiprocessing import Pool
import time


# ============================================================================
# WORKER FUNCTION - AT MODULE LEVEL
# ============================================================================


def evaluate_worker(params, seed, rnn_class_info, env_info):
    """
    Evaluate one individual in parallel

    Args:
        params: RNN parameters to evaluate
        seed: Random seed
        rnn_class_info: Dict with RNN configuration
        env_info: Dict with environment configuration

    Note: Changed argument order - params and seed first (from starmap),
          then rnn_class_info and env_info (fixed arguments)
    """
    try:
        # Recreate RNN
        rnn = NeuroMuscularRNN(**rnn_class_info)
        rnn.set_params(params)

        # Recreate environment
        reacher = SequentialReacher(**env_info["plant"])
        target_encoder = GridTargetEncoder(**env_info["encoder"])
        env = SequentialReachingEnv(
            plant=reacher, target_encoder=target_encoder, **env_info["env"]
        )

        # Evaluate
        fitness = -env.evaluate(rnn, seed=seed, render=False, log=False)

        # Cleanup
        env.plant.close()
        del env
        del reacher
        del rnn

        return fitness

    except Exception as e:
        print(f"Worker error with seed {seed}: {e}")
        import traceback

        traceback.print_exc()
        return -1e10  # Return very bad fitness on error


# ============================================================================
# ALTERNATIVE: Wrapper function (simpler approach)
# ============================================================================


def evaluate_worker_wrapper(args):
    """
    Alternative wrapper that unpacks all arguments
    Use this with pool.map() instead of pool.starmap()
    """
    params, seed, rnn_class_info, env_info = args
    return evaluate_worker(params, seed, rnn_class_info, env_info)


# ============================================================================
# MAIN CODE
# ============================================================================

if __name__ == "__main__":

    print("=" * 80)
    print("PARALLEL CMA-ES OPTIMIZATION")
    print("=" * 80)

    # ========================================================================
    # SETUP
    # ========================================================================

    # Initialize the plant
    reacher = SequentialReacher(plant_xml_file="arm.xml")

    # Initialize the target encoder
    target_encoder = GridTargetEncoder(
        grid_size=8,
        x_bounds=reacher.get_workspace_bounds()[0],
        y_bounds=reacher.get_workspace_bounds()[1],
        sigma=0.25,
    )

    # Specify policy
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

    # Store info for workers
    rnn_info = {
        "input_size_tgt": target_encoder.size,
        "input_size_len": reacher.num_sensors_len,
        "input_size_vel": reacher.num_sensors_vel,
        "input_size_frc": reacher.num_sensors_frc,
        "hidden_size": 25,
        "output_size": reacher.num_actuators,
        "activation": tanh,
        "smoothing_factor": alpha_from_tau(tau=10e-3, dt=reacher.model.opt.timestep),
    }

    env_info = {
        "plant": {"plant_xml_file": "arm.xml"},
        "encoder": {
            "grid_size": 8,
            "x_bounds": reacher.get_workspace_bounds()[0],
            "y_bounds": reacher.get_workspace_bounds()[1],
            "sigma": 0.25,
        },
        "env": {
            "target_duration_distro": {"mean": 3, "min": 1, "max": 6},
            "iti_distro": {"mean": 1, "min": 0, "max": 3},
            "num_targets": 10,
            "randomize_gravity": True,
            "loss_weights": {
                "distance": 1,
                "energy": 0.1,
                "ridge": 0,
                "lasso": 0,
            },
        },
    }

    # ========================================================================
    # OPTIMIZATION SETUP
    # ========================================================================

    optimizer = CMA(mean=rnn.get_params(), sigma=1.3)
    num_generations = 10000
    fitnesses = []

    # Create worker pool
    num_workers = mp.cpu_count()
    print(f"\nUsing {num_workers} parallel workers")
    print(f"Population size: {optimizer.population_size}")
    print(f"Expected speedup: ~{num_workers}x")
    print("=" * 80)

    pool = Pool(processes=num_workers)

    # Create environment for periodic evaluation
    eval_env = SequentialReachingEnv(
        plant=reacher,
        target_encoder=target_encoder,
        target_duration_distro={"mean": 3, "min": 1, "max": 6},
        iti_distro={"mean": 1, "min": 0, "max": 3},
        num_targets=10,
        randomize_gravity=True,
        loss_weights={"distance": 1, "energy": 0.1, "ridge": 0, "lasso": 0},
    )

    # ========================================================================
    # TRAINING LOOP
    # ========================================================================

    start_time = time.time()

    try:
        for gg in range(num_generations):
            gen_start = time.time()

            # Collect population
            population = []
            for ii in range(optimizer.population_size):
                x = optimizer.ask()
                population.append(x)

            # ================================================================
            # FIXED: Proper argument passing
            # ================================================================

            # METHOD 1: Using starmap with explicit argument construction
            # Each item should be: (params, seed, rnn_info, env_info)
            args_list = [(params, gg, rnn_info, env_info) for params in population]
            pop_fitnesses = pool.starmap(evaluate_worker, args_list)

            # ================================================================
            # ALTERNATIVE METHOD 2: Using map with wrapper
            # ================================================================
            # args_list = [(params, gg, rnn_info, env_info) for params in population]
            # pop_fitnesses = pool.map(evaluate_worker_wrapper, args_list)

            # ================================================================
            # Create solutions and update optimizer
            # ================================================================
            solutions = list(zip(population, pop_fitnesses))

            # Track fitnesses
            for ii, fitness in enumerate(pop_fitnesses):
                fitnesses.append((gg, ii, fitness))

            # Tell optimizer
            optimizer.tell(solutions)

            # ================================================================
            # LOGGING
            # ================================================================
            gen_time = time.time() - gen_start
            total_time = time.time() - start_time

            # Calculate statistics
            best_fitness = max(pop_fitnesses)
            mean_fitness = np.mean(pop_fitnesses)
            std_fitness = np.std(pop_fitnesses)
            evals_per_sec = optimizer.population_size / gen_time

            # Print progress
            print(
                f"Gen {gg:4d} | "
                f"Best: {best_fitness:7.3f} | "
                f"Mean: {mean_fitness:7.3f} ± {std_fitness:6.3f} | "
                f"Speed: {evals_per_sec:5.1f} eval/s | "
                f"Time: {total_time:6.1f}s"
            )

            # ================================================================
            # PERIODIC EVALUATION
            # ================================================================
            if gg % 10 == 0 and gg > 0:
                best_rnn = rnn.from_params(optimizer.mean)
                print(f"  → Evaluating best solution...")
                eval_fitness = -eval_env.evaluate(
                    best_rnn, seed=0, render=True, log=True
                )
                print(f"  → Evaluation fitness: {eval_fitness:.3f}")
                eval_env.plot()

            # ================================================================
            # SAVE CHECKPOINTS
            # ================================================================
            if gg % 1000 == 0 and gg > 0:
                file = f"../../models/optimizer_gen_{gg}_parallel.pkl"
                with open(file, "wb") as f:
                    pickle.dump(optimizer, f)
                print(f"  → Checkpoint saved: {file}")

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
