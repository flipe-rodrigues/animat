# main.py
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from evosax.algorithms import CMA_ES
from problem import RNNReachProblem
from env import create_env

# === Hyperparameters ===
INPUT_DIM = 12 + 3
HIDDEN_DIM = 32
OUTPUT_DIM = 4
POP_SIZE = 128
GENERATIONS = 500
SEED = 42

key = jax.random.key(SEED)

# === Setup ===
problem = RNNReachProblem(
    model=create_env("mujoco/arm_model.xml"),
    input_dim=INPUT_DIM,
    hidden_dim=HIDDEN_DIM,
    output_dim=OUTPUT_DIM,
    episode_len=200,
    batch_size=POP_SIZE
)

key, subkey = jax.random.split(key)
problem_state = problem.init(subkey)

# Instantiate evolution strategy
key, subkey = jax.random.split(key)
solution = problem.sample(subkey)
es = CMA_ES(
    population_size=POP_SIZE,
    solution=solution,  # requires a dummy solution
)

# Use default parameters
params = es.default_params

# es = CMA_ES(population_size=POP_SIZE, solution=problem.sample(), sigma=1)
# rng = jax.random.PRNGKey(SEED)
# es_state = es.initialize(rng, es.default_params)

key, subkey = jax.random.split(key)
state = es.init(subkey, solution, params)

metrics_log = []
for i in range(GENERATIONS):
    key, subkey = jax.random.split(key)
    key_ask, key_eval, key_tell = jax.random.split(subkey, 3)

    population, state = es.ask(key_ask, state, params)
    fitness, problem_state, info = problem.eval(key_eval, population, problem_state)
    state, metrics = es.tell(key_tell, population, fitness, state, params)

    # Log metrics
    metrics_log.append(metrics)

# === Evolution Loop ===
# for gen in range(GENERATIONS):
#     rng, ask_key, eval_key = jax.random.split(rng, 3)
#     solutions, es_state = es.ask(ask_key, es_state)

#     fitness = problem.evaluate(solutions, eval_key)  # Vectorized over population
#     es_state = es.tell(solutions, fitness, es_state)

#     best_fitness = jnp.min(fitness)
#     print(f"[Gen {gen:03d}] Best Loss: {best_fitness:.4f}")

# Optional: save best candidate
# best_idx = jnp.argmin(fitness)
# best_solution = solutions[best_idx]
# jnp.save("best_solution.npy", best_solution)

# Extract the best fitness values across generations
generations = [metrics["generation_counter"] for metrics in metrics_log]
best_fitness = [metrics["best_fitness"] for metrics in metrics_log]

plt.figure(figsize=(10, 5))
plt.plot(generations, best_fitness, label="Best Fitness", marker="o", markersize=3)

plt.title("Best fitness over generations")
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()