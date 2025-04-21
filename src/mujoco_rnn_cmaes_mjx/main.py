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
import matplotlib.pyplot as plt
from plants import SequentialReacher
from environments import SequentialReachingEnv
from networks import RNN
from utils import *
from cmaes import CMA
import numpy as np
import jax
import jax.numpy as jnp

# %%
"""
.##.....##....###....####.##....##
.###...###...##.##....##..###...##
.####.####..##...##...##..####..##
.##.###.##.##.....##..##..##.##.##
.##.....##.#########..##..##..####
.##.....##.##.....##..##..##...###
.##.....##.##.....##.####.##....##
"""
if __name__ == "__main__":
    
    # Number of parallel environments to run
    num_parallel_envs = 8

    # Set up JAX device count for parallel processing
    print(f"Number of JAX devices: {jax.device_count()}")

    # Initialize reacher, RNN, and environment
    reacher = SequentialReacher(plant_xml_file="arm_model.xml")
    rnn = RNN(
        input_size=3 + reacher.num_sensors,
        hidden_size=25,
        output_size=reacher.num_actuators,
        activation=tanh,
        alpha=reacher.mj_model.opt.timestep / 10e-3,
    )
    env = SequentialReachingEnv(
        plant=reacher,
        target_duration={"mean": 3, "min": 1, "max": 6},
        num_targets=10,
        loss_weights={
            "euclidean": 1,
            "manhattan": 0,
            "energy": 0,
            "ridge": 0,
            "lasso": 0,
        },
    )
    optimizer = CMA(mean=rnn.get_params(), sigma=1.3)

    num_generations = 10000
    fitnesses = []

    # Start timer to measure performance improvement
    start_time = time.time()

    for gg in range(num_generations):
        # Create batch of solutions
        solutions = []
        all_params = []

        # Generate population
        for ii in range(optimizer.population_size):
            x = optimizer.ask()
            all_params.append(x)

        # Batch all parameters into a single array for parallel evaluation
        all_params_batch = jnp.array(all_params)

        # Evaluate all solutions in parallel
        # Split evaluation into batches if population size is large
        batch_size = min(optimizer.population_size, num_parallel_envs)
        num_batches = (optimizer.population_size + batch_size - 1) // batch_size

        all_fitnesses = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, optimizer.population_size)
            batch_params = all_params_batch[start_idx:end_idx]

            # Evaluate batch in parallel
            batch_fitnesses = -env.evaluate_parallel(
                batch_params,
                len(batch_params),
                seeds=jnp.array(range(start_idx, end_idx)),
            )
            all_fitnesses.extend(batch_fitnesses)

        # Process results
        for ii, (x, fitness) in enumerate(zip(all_params, all_fitnesses)):
            solutions.append((x, fitness))
            fitnesses.append((gg, ii, fitness))
            print(f"#{gg}.{ii} {fitness}")

        # Update optimizer
        optimizer.tell(solutions)

        # Evaluate best policy
        if gg % 10 == 0:
            best_rnn = rnn.from_params(optimizer.mean)
            env.evaluate(best_rnn, seed=0, render=True, log=True)
            env.plot()

            # Print timing information
            elapsed_time = time.time() - start_time
            print(f"Generation {gg}: Elapsed time = {elapsed_time:.2f} seconds")

        # Save optimizer state
        if gg % 1000 == 0:
            file = f"../../models/optimizer_gen_{gg}_cmaesv2_parallel.pkl"
            with open(file, "wb") as f:
                pickle.dump(optimizer, f)

# %%
"""
..######...#######..##....##.##.....##.########.########...######...########.##....##..######..########
.##....##.##.....##.###...##.##.....##.##.......##.....##.##....##..##.......###...##.##....##.##......
.##.......##.....##.####..##.##.....##.##.......##.....##.##........##.......####..##.##.......##......
.##.......##.....##.##.##.##.##.....##.######...########..##...####.######...##.##.##.##.......######..
.##.......##.....##.##..####..##...##..##.......##...##...##....##..##.......##..####.##.......##......
.##....##.##.....##.##...###...##.##...##.......##....##..##....##..##.......##...###.##....##.##......
..######...#######..##....##....###....########.##.....##..######...########.##....##..######..########
"""
# Extract fitness values for each generation
fitnesses = np.array(fitnesses)
generations = np.unique(fitnesses[:, 0])
avg_fitness = []
std_fitness = []

for gen in generations:
    gen_fitness = fitnesses[fitnesses[:, 0] == gen][:, 2]
    avg_fitness.append(np.mean(gen_fitness))
    std_fitness.append(np.std(gen_fitness))

avg_fitness = np.array(avg_fitness)
std_fitness = np.array(std_fitness)

# Plot average fitness with standard deviation
plt.figure(figsize=(10, 6))
plt.plot(generations, avg_fitness, label="Average Fitness")
plt.fill_between(
    generations,
    avg_fitness - std_fitness,
    avg_fitness + std_fitness,
    color="blue",
    alpha=0.2,
    label="Standard Deviation",
)
plt.legend()
plt.xlabel("Iterations")
plt.ylabel("Objective Function Value (Loss)")
plt.title("Fitness During CMA-ES Optimization (Parallel MJX)")
plt.savefig("cmaes_fitness_parallel.png")
plt.show()
