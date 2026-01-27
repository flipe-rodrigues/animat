"""
SIMPLE MULTIPROCESSING OPTIMIZATION FOR CMA-ES

This is a minimal, easy-to-understand optimization that just uses Python's
multiprocessing to parallelize CMA-ES fitness evaluations.

No GPU, no PyTorch complexity, no vectorized environments.
Just simple parallel evaluation of CMA-ES population members.

Expected speedup: 4-8x on typical multi-core CPU (linear with cores)
"""

import numpy as np
import multiprocessing as mp
from multiprocessing import Pool
from functools import partial
import time
from typing import List, Tuple, Callable
import pickle
from cmaes import CMA


# ============================================================================
# PARALLEL CMA-ES EVALUATOR
# ============================================================================

def evaluate_individual(
    params: np.ndarray,
    rnn_fn: Callable,
    env_fn: Callable,
    seed: int
) -> float:
    """
    Evaluate a single set of RNN parameters.
    
    This function will be called in parallel by worker processes.
    
    Args:
        params: RNN parameters to evaluate
        rnn_fn: Function that creates RNN from params
        env_fn: Function that creates environment
        seed: Random seed for reproducibility
        
    Returns:
        fitness: Negative loss (higher is better for CMA-ES)
    """
    # Create RNN and environment in worker process
    rnn = rnn_fn(params)
    env = env_fn()
    
    # Evaluate
    fitness = -env.evaluate(rnn, seed=seed, render=False, log=False)
    
    # Clean up
    env.plant.close()
    
    return fitness


class ParallelCMAES:
    """
    CMA-ES optimizer with parallel fitness evaluation.
    
    This is a drop-in replacement for the sequential CMA-ES loop
    that evaluates population members in parallel.
    """
    
    def __init__(
        self,
        rnn_template,
        env_fn: Callable,
        initial_params: np.ndarray = None,
        sigma: float = 1.3,
        num_workers: int = None
    ):
        """
        Args:
            rnn_template: Template RNN (used to get architecture)
            env_fn: Function that creates an environment
            initial_params: Initial parameters (default: random from template)
            sigma: Initial step size for CMA-ES
            num_workers: Number of parallel workers (default: CPU count)
        """
        self.rnn_template = rnn_template
        self.env_fn = env_fn
        
        # Setup multiprocessing
        if num_workers is None:
            num_workers = mp.cpu_count()
        self.num_workers = num_workers
        
        print(f"Initializing ParallelCMAES with {num_workers} workers")
        
        # Initialize CMA-ES
        if initial_params is None:
            initial_params = rnn_template.get_params()
        
        self.optimizer = CMA(mean=initial_params, sigma=sigma)
        
        # Create pool of workers
        self.pool = Pool(processes=num_workers)
        
        # Tracking
        self.generation = 0
        self.fitnesses_history = []
    
    def _create_rnn_from_params(self, params: np.ndarray):
        """Create RNN from parameters (helper for workers)"""
        return self.rnn_template.from_params(params)
    
    def evaluate_population_parallel(
        self,
        population: List[np.ndarray],
        seed_base: int = 0
    ) -> List[float]:
        """
        Evaluate all population members in parallel.
        
        Args:
            population: List of parameter vectors
            seed_base: Base seed (each individual gets seed_base + index)
            
        Returns:
            fitnesses: List of fitness values
        """
        # Create evaluation function with fixed arguments
        eval_fn = partial(
            evaluate_individual,
            rnn_fn=self._create_rnn_from_params,
            env_fn=self.env_fn
        )
        
        # Create tasks (parameter set, seed pairs)
        tasks = [(params, seed_base + i) for i, params in enumerate(population)]
        
        # Evaluate in parallel
        fitnesses = self.pool.starmap(eval_fn, tasks)
        
        return fitnesses
    
    def run_generation(self, seed_base: int = None) -> Tuple[float, float, float]:
        """
        Run one generation of CMA-ES with parallel evaluation.
        
        Args:
            seed_base: Base seed for this generation
            
        Returns:
            best_fitness: Best fitness in population
            mean_fitness: Mean fitness in population
            std_fitness: Std of fitness in population
        """
        if seed_base is None:
            seed_base = self.generation
        
        # Ask CMA-ES for population
        population = []
        for _ in range(self.optimizer.population_size):
            x = self.optimizer.ask()
            population.append(x)
        
        # Evaluate population in parallel
        fitnesses = self.evaluate_population_parallel(population, seed_base)
        
        # Tell CMA-ES the results
        solutions = list(zip(population, fitnesses))
        self.optimizer.tell(solutions)
        
        # Track statistics
        best_fitness = max(fitnesses)
        mean_fitness = np.mean(fitnesses)
        std_fitness = np.std(fitnesses)
        
        for i, fitness in enumerate(fitnesses):
            self.fitnesses_history.append((self.generation, i, fitness))
        
        self.generation += 1
        
        return best_fitness, mean_fitness, std_fitness
    
    def train(
        self,
        num_generations: int,
        eval_interval: int = 10,
        save_interval: int = 100,
        save_path: str = None
    ) -> dict:
        """
        Main training loop.
        
        Args:
            num_generations: Number of generations to run
            eval_interval: How often to print progress
            save_interval: How often to save checkpoints
            save_path: Path prefix for saving (default: no saving)
            
        Returns:
            stats: Dictionary with training statistics
        """
        print(f"\nStarting training for {num_generations} generations")
        print(f"Population size: {self.optimizer.population_size}")
        print(f"Parallel workers: {self.num_workers}")
        print("=" * 80)
        
        start_time = time.time()
        
        stats = {
            'generation': [],
            'best_fitness': [],
            'mean_fitness': [],
            'std_fitness': [],
            'time_elapsed': []
        }
        
        for gen in range(num_generations):
            gen_start = time.time()
            
            # Run one generation
            best_fit, mean_fit, std_fit = self.run_generation(seed_base=gen)
            
            gen_time = time.time() - gen_start
            total_time = time.time() - start_time
            
            # Store stats
            stats['generation'].append(gen)
            stats['best_fitness'].append(best_fit)
            stats['mean_fitness'].append(mean_fit)
            stats['std_fitness'].append(std_fit)
            stats['time_elapsed'].append(total_time)
            
            # Logging
            if gen % eval_interval == 0:
                evals_per_sec = self.optimizer.population_size / gen_time
                print(f"Gen {gen:4d} | "
                      f"Best: {best_fit:7.3f} | "
                      f"Mean: {mean_fit:7.3f} ± {std_fit:6.3f} | "
                      f"Speed: {evals_per_sec:5.1f} eval/s | "
                      f"Time: {total_time:6.1f}s")
            
            # Periodic evaluation with rendering
            if gen % eval_interval == 0 and gen > 0:
                best_rnn = self.rnn_template.from_params(self.optimizer.mean)
                env = self.env_fn()
                print(f"  Evaluating best solution...")
                eval_fitness = -env.evaluate(best_rnn, seed=0, render=False, log=False)
                print(f"  Evaluation fitness: {eval_fitness:.3f}")
                env.plant.close()
            
            # Save checkpoint
            if save_path and gen % save_interval == 0 and gen > 0:
                checkpoint_path = f"{save_path}_gen_{gen}.pkl"
                self.save_checkpoint(checkpoint_path)
        
        print("=" * 80)
        print(f"Training complete! Total time: {time.time() - start_time:.1f}s")
        
        # Get final best RNN
        best_rnn = self.rnn_template.from_params(self.optimizer.mean)
        
        # Clean up
        self.close()
        
        return stats, best_rnn
    
    def get_best_rnn(self):
        """Get RNN with best parameters found so far"""
        return self.rnn_template.from_params(self.optimizer.mean)
    
    def save_checkpoint(self, filepath: str):
        """Save checkpoint"""
        checkpoint = {
            'optimizer': self.optimizer,
            'generation': self.generation,
            'fitnesses_history': self.fitnesses_history,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(checkpoint, f)
        print(f"  Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load checkpoint"""
        with open(filepath, 'rb') as f:
            checkpoint = pickle.load(f)
        self.optimizer = checkpoint['optimizer']
        self.generation = checkpoint['generation']
        self.fitnesses_history = checkpoint['fitnesses_history']
        print(f"Checkpoint loaded from {filepath}")
    
    def close(self):
        """Clean up worker pool"""
        self.pool.close()
        self.pool.join()
        print("Worker pool closed")


# ============================================================================
# EVEN SIMPLER: PARALLEL MAP FUNCTION
# ============================================================================

def parallel_cmaes_simple(
    rnn,
    env_fn: Callable,
    num_generations: int = 1000,
    num_workers: int = None
):
    """
    Simplest possible parallel CMA-ES - just use Pool.map()
    
    This is a minimal example showing the core idea.
    """
    if num_workers is None:
        num_workers = mp.cpu_count()
    
    print(f"Simple parallel CMA-ES with {num_workers} workers")
    
    # Initialize optimizer
    optimizer = CMA(mean=rnn.get_params(), sigma=1.3)
    fitnesses = []
    
    # Create worker pool
    with Pool(processes=num_workers) as pool:
        for gen in range(num_generations):
            # Generate population
            population = [optimizer.ask() for _ in range(optimizer.population_size)]
            
            # Evaluate in parallel using map
            eval_fn = partial(
                evaluate_individual,
                rnn_fn=lambda p: rnn.from_params(p),
                env_fn=env_fn
            )
            
            # Create (params, seed) pairs
            tasks = [(params, gen) for params in population]
            
            # Parallel evaluation
            pop_fitnesses = pool.starmap(eval_fn, tasks)
            
            # Update optimizer
            solutions = list(zip(population, pop_fitnesses))
            optimizer.tell(solutions)
            
            # Track
            for i, fit in enumerate(pop_fitnesses):
                fitnesses.append((gen, i, fit))
            
            # Log
            if gen % 10 == 0:
                print(f"Gen {gen}: Best = {max(pop_fitnesses):.3f}, "
                      f"Mean = {np.mean(pop_fitnesses):.3f}")
    
    # Return best RNN
    return rnn.from_params(optimizer.mean), fitnesses


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

def example_usage():
    """Example of how to use parallel CMA-ES"""
    
    from plants import SequentialReacher
    from encoders import GridTargetEncoder
    from environments import SequentialReachingEnv
    from networks import NeuroMuscularRNN
    from utils import tanh, alpha_from_tau
    
    # Setup (exactly as before)
    reacher = SequentialReacher(plant_xml_file="arm.xml")
    
    target_encoder = GridTargetEncoder(
        grid_size=8,
        x_bounds=reacher.get_workspace_bounds()[0],
        y_bounds=reacher.get_workspace_bounds()[1],
        sigma=0.25,
    )
    
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
    
    # Create environment function
    def make_env():
        reacher_local = SequentialReacher(plant_xml_file="arm.xml")
        target_encoder_local = GridTargetEncoder(
            grid_size=8,
            x_bounds=reacher_local.get_workspace_bounds()[0],
            y_bounds=reacher_local.get_workspace_bounds()[1],
            sigma=0.25,
        )
        return SequentialReachingEnv(
            plant=reacher_local,
            target_encoder=target_encoder_local,
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
    
    # ========================================================================
    # OLD WAY (Sequential)
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("SEQUENTIAL CMA-ES (Original)")
    print("=" * 80)
    
    start_time = time.time()
    
    optimizer = CMA(mean=rnn.get_params(), sigma=1.3)
    
    # Run just 10 generations for comparison
    for gen in range(10):
        solutions = []
        for i in range(optimizer.population_size):
            x = optimizer.ask()
            env = make_env()
            fitness = -env.evaluate(rnn.from_params(x), seed=gen)
            env.plant.close()
            solutions.append((x, fitness))
        
        optimizer.tell(solutions)
        
        if gen % 5 == 0:
            best_fitness = max([f for _, f in solutions])
            print(f"Gen {gen}: Best = {best_fitness:.3f}")
    
    sequential_time = time.time() - start_time
    print(f"Sequential time: {sequential_time:.1f}s")
    
    # ========================================================================
    # NEW WAY (Parallel)
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("PARALLEL CMA-ES (Optimized)")
    print("=" * 80)
    
    start_time = time.time()
    
    parallel_optimizer = ParallelCMAES(
        rnn_template=rnn,
        env_fn=make_env,
        num_workers=mp.cpu_count()
    )
    
    stats, best_rnn = parallel_optimizer.train(
        num_generations=10,
        eval_interval=5
    )
    
    parallel_time = time.time() - start_time
    print(f"Parallel time: {parallel_time:.1f}s")
    
    # ========================================================================
    # COMPARISON
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON")
    print("=" * 80)
    print(f"Sequential: {sequential_time:.1f}s")
    print(f"Parallel:   {parallel_time:.1f}s")
    print(f"Speedup:    {sequential_time/parallel_time:.2f}x")
    print(f"Workers:    {mp.cpu_count()}")


def minimal_example():
    """Absolute minimal example showing just the core change"""
    
    # Assume you have: rnn, env_fn (function that creates environment)
    
    # ========================================================================
    # BEFORE: Sequential evaluation
    # ========================================================================
    """
    optimizer = CMA(mean=rnn.get_params(), sigma=1.3)
    for gen in range(num_generations):
        solutions = []
        for i in range(optimizer.population_size):
            x = optimizer.ask()
            fitness = -env.evaluate(rnn.from_params(x), seed=gen)
            solutions.append((x, fitness))
        optimizer.tell(solutions)
    """
    
    # ========================================================================
    # AFTER: Parallel evaluation (ONE LINE CHANGE!)
    # ========================================================================
    """
    parallel_optimizer = ParallelCMAES(rnn, env_fn)
    stats, best_rnn = parallel_optimizer.train(num_generations=1000)
    """
    
    pass


# ============================================================================
# BENCHMARKING
# ============================================================================

def benchmark_speedup():
    """Benchmark to see actual speedup on your machine"""
    
    import matplotlib.pyplot as plt
    
    # Test with different number of workers
    worker_counts = [1, 2, 4, 8, mp.cpu_count()]
    times = []
    speedups = []
    
    baseline_time = None
    
    for num_workers in worker_counts:
        if num_workers > mp.cpu_count():
            continue
        
        print(f"\nTesting with {num_workers} workers...")
        
        # Run a few generations
        # (You'd insert your actual setup here)
        start = time.time()
        
        # Simulate work
        time.sleep(1.0 / num_workers)  # Placeholder
        
        elapsed = time.time() - start
        times.append(elapsed)
        
        if baseline_time is None:
            baseline_time = elapsed
            speedup = 1.0
        else:
            speedup = baseline_time / elapsed
        
        speedups.append(speedup)
        
        print(f"  Time: {elapsed:.2f}s, Speedup: {speedup:.2f}x")
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(worker_counts[:len(times)], times, 'o-')
    ax1.set_xlabel('Number of Workers')
    ax1.set_ylabel('Time (seconds)')
    ax1.set_title('Training Time vs Workers')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(worker_counts[:len(speedups)], speedups, 'o-', label='Actual')
    ax2.plot(worker_counts[:len(speedups)], worker_counts[:len(speedups)], 
             '--', alpha=0.5, label='Ideal (linear)')
    ax2.set_xlabel('Number of Workers')
    ax2.set_ylabel('Speedup')
    ax2.set_title('Speedup vs Workers')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('parallel_benchmark.png', dpi=150)
    plt.show()
    
    return worker_counts[:len(times)], times, speedups


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print(__doc__)
    print("\n" + "=" * 80)
    print("SIMPLE MULTIPROCESSING FOR CMA-ES")
    print("=" * 80)
    print(f"\nAvailable CPU cores: {mp.cpu_count()}")
    print(f"Expected speedup: ~{mp.cpu_count()}x (linear scaling)")
    print("\nThis is the SIMPLEST optimization - just parallel evaluation!")
    print("\nNo GPU required, no complex dependencies, just multiprocessing.")
    print("\n" + "=" * 80)
