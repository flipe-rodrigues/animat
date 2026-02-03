"""CMA-ES Training for Muscle-Driven Arm Controller.

Uses the `cmaes` PyPI package (https://pypi.org/project/cmaes/).
"""

import numpy as np
import torch
from cmaes import CMA
from typing import Callable, Optional
from concurrent.futures import ProcessPoolExecutor


def evaluate_controller(
    controller,
    env,
    num_episodes: int = 5,
    max_steps: int = 300,
) -> float:
    """Evaluate a controller on the environment.
    
    Args:
        controller: Neural network controller with predict() and _reset_state()
        env: Gym environment
        num_episodes: Number of episodes to average over
        max_steps: Maximum steps per episode
        
    Returns:
        Average reward across episodes
    """
    controller.eval()
    total_reward = 0.0
    
    for _ in range(num_episodes):
        obs, _ = env.reset()
        controller._reset_state()
        
        for _ in range(max_steps):
            action, _ = controller.predict(obs, deterministic=True)
            action = np.clip(action, 0.0, 1.0)
            
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                break
    
    return total_reward / num_episodes


def cmaes_optimize(
    create_controller_fn: Callable,
    create_env_fn: Callable,
    num_generations: int = 100,
    population_size: int = 32,
    sigma: float = 0.25,
    num_eval_episodes: int = 5,
    max_steps: int = 300,
    num_workers: int = 1,
    callback: Optional[Callable] = None,
) -> dict:
    """Run CMA-ES optimization to train a controller.
    
    Args:
        create_controller_fn: Factory function that returns a fresh controller
        create_env_fn: Factory function that returns an environment
        num_generations: Number of CMA-ES generations
        population_size: Population size for CMA-ES
        sigma: Initial step size
        num_eval_episodes: Episodes to average for fitness
        max_steps: Max steps per episode
        num_workers: Number of parallel workers (1 = sequential)
        callback: Optional callback(generation, best_fitness, mean_fitness, best_params)
        
    Returns:
        Dict with best_params, best_fitness, history
    """
    # Create a template controller to get parameter count
    template = create_controller_fn()
    initial_params = template.get_flat_params()
    num_params = len(initial_params)
    
    print(f"Optimizing {num_params} parameters")
    print(f"Population: {population_size}, Generations: {num_generations}")
    
    # Initialize CMA-ES optimizer
    optimizer = CMA(mean=initial_params, sigma=sigma, population_size=population_size)
    
    best_fitness = -np.inf
    best_params = None
    fitness_history = []
    
    def evaluate_single(params):
        """Evaluate a single parameter vector."""
        controller = create_controller_fn()
        controller.set_flat_params(params)
        env = create_env_fn()
        fitness = evaluate_controller(controller, env, num_eval_episodes, max_steps)
        env.close()
        return fitness
    
    for generation in range(num_generations):
        # Ask for candidate solutions
        solutions = [optimizer.ask() for _ in range(population_size)]
        
        # Evaluate population
        if num_workers > 1:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                fitness_values = list(executor.map(evaluate_single, solutions))
        else:
            fitness_values = [evaluate_single(params) for params in solutions]
        
        fitness_array = np.array(fitness_values)
        
        # Tell CMA-ES (it minimizes, so negate fitness)
        optimizer.tell(list(zip(solutions, -fitness_array)))
        
        # Track best
        gen_best_idx = np.argmax(fitness_array)
        gen_best = fitness_array[gen_best_idx]
        gen_mean = np.mean(fitness_array)
        
        if gen_best > best_fitness:
            best_fitness = gen_best
            best_params = solutions[gen_best_idx].copy()
        
        fitness_history.append({
            "generation": generation,
            "best": gen_best,
            "mean": gen_mean,
            "best_so_far": best_fitness,
        })
        
        print(f"Gen {generation}: best={gen_best:.2f}, mean={gen_mean:.2f}, best_so_far={best_fitness:.2f}")
        
        if callback:
            callback(generation, best_fitness, gen_mean, best_params)
    
    return {
        "best_params": best_params,
        "best_fitness": best_fitness,
        "history": fitness_history,
    }
