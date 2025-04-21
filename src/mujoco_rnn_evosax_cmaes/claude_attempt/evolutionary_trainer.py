"""
Evolutionary training using CMA-ES for the RNN controller.
"""
import os
import time
import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple, Dict, Any, Callable, List
import pickle
from functools import partial

# Import EvoSAX for CMA-ES (compatible with 0.2.0)
from evosax.algorithms import CMA_ES
from evosax.problems import Problem

from rnn_model import SimpleRNN
from environment import ArmReachingEnv

class ArmReachingProblem(Problem):
    """Custom problem definition for the arm reaching task."""
    
    def __init__(self, num_dims):
        super().__init__()
    
    def __call__(self, x, random_key):
        # Simply return the parameters without modification
        # (actual evaluation is done in the EvolutionaryTrainer class)
        return x

class EvolutionaryTrainer:
    def __init__(
        self,
        env: ArmReachingEnv,
        popsize: int = 64,
        hidden_size: int = 32,
        n_targets: int = 10,
        steps_per_target: int = 100,
        save_path: str = "models",
        seed: int = 42
    ):
        """
        Initialize the evolutionary trainer for the RNN controller.
        
        Args:
            env: Arm reaching environment
            popsize: Population size for CMA-ES
            hidden_size: Hidden layer size for the RNN
            n_targets: Number of targets per evaluation
            steps_per_target: Number of steps per target
            save_path: Path to save models
            seed: Random seed
        """
        self.env = env
        self.popsize = popsize
        self.hidden_size = hidden_size
        self.n_targets = n_targets
        self.steps_per_target = steps_per_target
        self.save_path = save_path
        
        # Create directory for saving models
        os.makedirs(save_path, exist_ok=True)
        
        # Initialize random keys
        self.key = jax.random.PRNGKey(seed)
        self.key, subkey = jax.random.split(self.key)
        
        # Initialize RNN
        self.rnn = SimpleRNN(
            input_size=env.input_dim,
            hidden_size=hidden_size,
            output_size=env.output_dim
        )
        self.params = self.rnn.init_params(subkey)
        self.param_count = self.rnn.param_count
        
        # Initialize CMA-ES for evosax 0.2.0
        self.key, subkey = jax.random.split(self.key)
        
        # Create a custom problem for our arm reaching task
        arm_problem = ArmReachingProblem(num_dims=self.param_count)
        
        # Initialize CMA-ES with the correct interface
        self.es = CMA_ES(population_size=popsize, solution=arm_problem.sample())
        
        # Initialize ES state
        self.key, subkey = jax.random.split(self.key)
        self.es_state = self.es.initialize(subkey)
        
        # Performance tracking
        self.best_fitness = -float('inf')
        self.best_params = None
        self.fitness_history = []

# class EvolutionaryTrainer:
#     def __init__(
#         self,
#         env: ArmReachingEnv,
#         popsize: int = 64,
#         hidden_size: int = 32,
#         n_targets: int = 10,
#         steps_per_target: int = 100,
#         save_path: str = "models",
#         seed: int = 42
#     ):
#         """
#         Initialize the evolutionary trainer for the RNN controller.
        
#         Args:
#             env: Arm reaching environment
#             popsize: Population size for CMA-ES
#             hidden_size: Hidden layer size for the RNN
#             n_targets: Number of targets per evaluation
#             steps_per_target: Number of steps per target
#             save_path: Path to save models
#             seed: Random seed
#         """
#         self.env = env
#         self.popsize = popsize
#         self.hidden_size = hidden_size
#         self.n_targets = n_targets
#         self.steps_per_target = steps_per_target
#         self.save_path = save_path
        
#         # Create directory for saving models
#         os.makedirs(save_path, exist_ok=True)
        
#         # Initialize random keys
#         self.key = jax.random.PRNGKey(seed)
#         self.key, subkey = jax.random.split(self.key)
        
#         # Initialize RNN
#         self.rnn = SimpleRNN(
#             input_size=env.input_dim,
#             hidden_size=hidden_size,
#             output_size=env.output_dim
#         )
#         self.params = self.rnn.init_params(subkey)
#         self.param_count = self.rnn.param_count
        
#         # Initialize CMA-ES for evosax 0.2.0
#         self.key, subkey = jax.random.split(self.key)
        
#         # Properly initialize CMA-ES for evosax 0.2.0
#         # From the source code we can see it needs population_size and num_dims
#         self.es = CMA_ES(population_size=popsize, solution=pro)
        
#         # Initialize ES state
#         self.key, subkey = jax.random.split(self.key)
#         self.es_state = self.es._init(subkey, self.es._default_params)
        
#         # Initialize the mean with zeros instead of NaN
#         self.es_state = self.es_state.replace(mean=jnp.zeros(self.param_count))
        
#         # Set best_solution and best_fitness based on the SolutionFeaturesState class
#         self.es_state = self.es_state.replace(
#             best_solution=jnp.zeros(self.param_count),
#             best_fitness=jnp.inf
#         )
        
#         # Performance tracking
#         self.best_fitness = -float('inf')
#         self.best_params = None
#         self.fitness_history = []
    
#     def evaluate_params(self, params_flat: jnp.ndarray, key: jnp.ndarray) -> float:
#         """
#         Evaluate a single set of parameters.
        
#         Args:
#             params_flat: Flattened RNN parameters
#             key: JAX random key
            
#         Returns:
#             Fitness score (negative mean distance)
#         """
#         # Unflatten parameters
#         params = self.rnn.unflatten_params(params_flat)
        
#         # Evaluate on multiple targets
#         total_reward = 0.0
        
#         # Define RNN forward pass
#         def policy_fn(params, obs, h_state):
#             action, h_state = self.rnn.predict(params, obs, h_state)
#             return action, h_state
        
#         # Single-target evaluation
#         def evaluate_single_target(carry, _):
#             cumulative_reward, subkey = carry
            
#             # Generate new key
#             subkey, rollout_key = jax.random.split(subkey)
            
#             # Perform rollout
#             results = self.env.rollout_jax(
#                 rollout_key, 
#                 policy_fn, 
#                 params, 
#                 n_steps=self.steps_per_target
#             )
            
#             # Update reward
#             cumulative_reward += results["total_reward"]
            
#             return (cumulative_reward, subkey), None
        
#         # Evaluate across multiple targets
#         initial_state = (0.0, key)
#         (total_reward, _), _ = jax.lax.scan(
#             evaluate_single_target, 
#             initial_state, 
#             jnp.arange(self.n_targets)
#         )
        
#         # Return average reward
#         return total_reward / self.n_targets
    
#     def evaluate_population(self, params_array: jnp.ndarray, keys: jnp.ndarray) -> jnp.ndarray:
#         """
#         Evaluate an entire population of parameters.
        
#         Args:
#             params_array: Array of flattened parameters [popsize, param_count]
#             keys: Random keys for each evaluation [popsize]
            
#         Returns:
#             Array of fitness scores [popsize]
#         """
#         return jax.vmap(self.evaluate_params)(params_array, keys)
    
#     def train(self, n_generations: int = 100, log_interval: int = 10) -> Dict[str, Any]:
#         """
#         Train the RNN controller using CMA-ES.
        
#         Args:
#             n_generations: Number of generations to train
#             log_interval: Interval for logging progress
            
#         Returns:
#             Dictionary with training results
#         """
#         # Compile the evaluation function
#         evaluate_population_jit = jax.jit(self.evaluate_population)
        
#         # Get default parameters
#         default_params = self.es._default_params
        
#         # Training loop
#         start_time = time.time()
#         for gen in range(n_generations):
#             # Get a new set of random keys for each member of the population
#             self.key, *subkeys = jax.random.split(self.key, self.popsize + 1)
#             subkeys = jnp.array(subkeys)
            
#             # Ask for new parameter candidates using the correct API for evosax 0.2.0
#             self.key, ask_key = jax.random.split(self.key)
#             population, self.es_state = self.es._ask(ask_key, self.es_state, default_params)
            
#             # Evaluate the population
#             fitnesses = evaluate_population_jit(population, subkeys)
            
#             # For CMA-ES in evosax 0.2.0, we need to negate the fitness since it minimizes by default
#             # but our reward function is better when higher (negative distance)
#             neg_fitnesses = -fitnesses
            
#             # Update the ES with the results using the correct API for evosax 0.2.0
#             self.key, tell_key = jax.random.split(self.key)
#             self.es_state = self.es._tell(tell_key, population, neg_fitnesses, self.es_state, default_params)
            
#             # Track the best fitness
#             best_idx = jnp.argmax(fitnesses)
#             current_best_fitness = fitnesses[best_idx]
            
#             if current_best_fitness > self.best_fitness:
#                 self.best_fitness = current_best_fitness
#                 self.best_params = population[best_idx]
                
#                 # Save the best parameters
#                 params = self.rnn.unflatten_params(self.best_params)
#                 self.save_model(params, f"best_model_gen_{gen}")
            
#             # Record history
#             self.fitness_history.append({
#                 'generation': gen,
#                 'mean_fitness': float(jnp.mean(fitnesses)),
#                 'best_fitness': float(current_best_fitness),
#                 'best_overall': float(self.best_fitness)
#             })
            
#             # Log progress
#             if gen % log_interval == 0 or gen == n_generations - 1:
#                 elapsed = time.time() - start_time
#                 print(f"Gen {gen}/{n_generations} | "
#                       f"Best: {current_best_fitness:.4f} | "
#                       f"Mean: {jnp.mean(fitnesses):.4f} | "
#                       f"Time: {elapsed:.2f}s")
        
#         # Final save
#         if self.best_params is not None:
#             best_params = self.rnn.unflatten_params(self.best_params)
#             self.save_model(best_params, "final_best_model")
        
#         # Return results
#         return {
#             'best_fitness': self.best_fitness,
#             'best_params': self.best_params,
#             'history': self.fitness_history
#         }
    
#     def save_model(self, params: Dict[str, jnp.ndarray], name: str) -> None:
#         """
#         Save model parameters to disk.
        
#         Args:
#             params: RNN parameters
#             name: Name of the save file
#         """
#         # Convert params to numpy for saving
#         params_np = {k: np.array(v) for k, v in params.items()}
        
#         # Save to disk
#         path = os.path.join(self.save_path, f"{name}.pkl")
#         with open(path, 'wb') as f:
#             pickle.dump(params_np, f)
    
#     def load_model(self, path: str) -> Dict[str, jnp.ndarray]:
#         """
#         Load model parameters from disk.
        
#         Args:
#             path: Path to the saved model
            
#         Returns:
#             RNN parameters
#         """
#         with open(path, 'rb') as f:
#             params_np = pickle.load(f)
        
#         # Convert numpy arrays to JAX arrays
#         params = {k: jnp.array(v) for k, v in params_np.items()}
        
#         return params