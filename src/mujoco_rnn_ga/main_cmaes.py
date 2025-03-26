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
import matplotlib.pyplot as plt
from tqdm import tqdm
from plants import SequentialReacher
from environments import SequentialReachingEnv
from networks import RNN
from cmaes import CMA
from utils import *
import numpy as np

# %%
"""
.########.##.....##..#######..##.......##.....##.########
.##.......##.....##.##.....##.##.......##.....##.##......
.##.......##.....##.##.....##.##.......##.....##.##......
.######...##.....##.##.....##.##.......##.....##.######..
.##........##...##..##.....##.##........##...##..##......
.##.........##.##...##.....##.##.........##.##...##......
.########....###.....#######..########....###....########
"""


# def evolve(env, rnn, num_individuals=25, num_generations=10, mutation_rate=0.1):
#     """Run evolutionary learning process"""
#     cmaes = CMAES(
#         num_parameters=rnn.num_params,
#         lambda_=num_individuals,
#         mean0=rnn.get_params(),
#         sigma0=mutation_rate,
#     )

#     best_rnn = []

#     for gg in range(num_generations):

#         # Generate new population
#         Z = np.random.randn(cmaes.num_perturbations, cmaes.num_parameters)
#         X = cmaes.mean + cmaes.sigma * Z @ np.linalg.cholesky(cmaes.C).T
#         fitness = np.array(
#             [
#                 env.evaluate(rnn.from_params(x), seed=0)
#                 for x in tqdm(X, desc="Evaluating")
#             ]
#         )

#         # Sort by fitness and update mean
#         sorted_idcs = np.argsort(fitness)[::-1]
#         X = X[sorted_idcs]
#         Z = Z[sorted_idcs]
#         cmaes.mean = np.sum(
#             cmaes.weights[:, np.newaxis] * X[: cmaes.num_perturbations],
#             axis=0,
#         )

#         # Update evolution paths
#         z_mean = np.sum(
#             cmaes.weights[:, np.newaxis] * Z[: cmaes.num_perturbations],
#             axis=0,
#         )
#         cmaes.ps = (1 - 0.5) * cmaes.ps + np.sqrt(0.5 * (2 - 0.5)) * z_mean
#         hsig = (
#             np.linalg.norm(cmaes.ps) / np.sqrt(1 - (1 - 0.5) ** (2 * (gg + 1)))
#             < (1.4 + 2 / (cmaes.num_parameters + 1)) * cmaes.chiN
#         )
#         cmaes.pc = (1 - cmaes.c1) * cmaes.pc + hsig * np.sqrt(
#             cmaes.c1 * (2 - cmaes.c1)
#         ) * z_mean

#         # Update covariance matrix
#         C_new = (1 - cmaes.c1 - cmaes.cmu) * cmaes.C + cmaes.c1 * np.outer(
#             cmaes.pc, cmaes.pc
#         )
#         for i in range(cmaes.num_perturbations):
#             C_new += cmaes.cmu * cmaes.weights[i] * np.outer(Z[i], Z[i])
#         cmaes.C = C_new

#         # Update step size
#         cmaes.sigma *= np.exp(
#             (np.linalg.norm(cmaes.ps) / cmaes.chiN - 1) * 0.5 / cmaes.damps
#         )

#         # Print best fitness in each generation
#         print(f"Generation {gg+1}: Best fitness = {fitness.max()}")

#         best_rnn = rnn.from_params(cmaes.mean)

#         if gg % 10 == 0:
#             env.evaluate(best_rnn, seed=0, render=True, log=True)
#             env.plot()
#         if gg % 100 == 0:
#             file = f"../../models/best_rnn_gen_{gg}_myCMAES.pkl"
#         else:
#             file = f"../../models/best_rnn_gen_curr_myCMAES.pkl"
#         with open(file, "wb") as f:
#             pickle.dump(best_rnn, f)

#     return best_rnn


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
    reacher = SequentialReacher(plant_xml_file="arm_model.xml")
    rnn = RNN(
        input_size=3 + reacher.num_sensors,
        hidden_size=25,
        output_size=reacher.num_actuators,
        activation=tanh,
        alpha=reacher.model.opt.timestep / 25e-3,
    )
    env = SequentialReachingEnv(
        plant=reacher, target_duration=(3, 2, 6), num_targets=10
    )
    optimizer = CMA(mean=rnn.get_params(), sigma=1.3)

    num_generations = 1000
    fitnesses = []
    for gg in range(num_generations):
        solutions = []
        for ii in range(optimizer.population_size):
            x = optimizer.ask()
            value = -env.evaluate(rnn.from_params(x), seed=gg)
            solutions.append((x, value))
            fitnesses.append((gg, ii, value))
            print(f"#{gg}.{ii} {value}")
        optimizer.tell(solutions)

        best_rnn = rnn.from_params(optimizer.mean)
        if gg % 10 == 0:
            env.evaluate(best_rnn, seed=0, render=True, log=True)
            env.plot()
        if gg % 100 == 0:
            file = f"../../models/best_rnn_gen_{gg}_cmaesv2.pkl"
        else:
            file = f"../../models/best_rnn_gen_curr_cmaesv2.pkl"
        with open(file, "wb") as f:
            pickle.dump(best_rnn, f)

# %%
# Plot the loss over iterations
# fitness_values = [solution[1] for solution in solutions]
# plt.plot(fitness_values)
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
plt.title("Fitness During CMA-ES Optimization")
plt.show()

# %%
"""
.########..########.##....##.########..########.########.
.##.....##.##.......###...##.##.....##.##.......##.....##
.##.....##.##.......####..##.##.....##.##.......##.....##
.########..######...##.##.##.##.....##.######...########.
.##...##...##.......##..####.##.....##.##.......##...##..
.##....##..##.......##...###.##.....##.##.......##....##.
.##.....##.########.##....##.########..########.##.....##
"""
models_dir = "../../models"
gen_idx = 200  # Specify the generation index you want to plot
model_file = f"best_rnn_gen_{gen_idx}_cmaesv2.pkl"
# model_file = "best_rnn_gen_curr.pkl"
with open(os.path.join(models_dir, model_file), "rb") as f:
    best_rnn = pickle.load(f)
# best_rnn = rnn.from_params(optimizer.mean)
env.evaluate(best_rnn, seed=0, render=True, log=True)
env.plot()
