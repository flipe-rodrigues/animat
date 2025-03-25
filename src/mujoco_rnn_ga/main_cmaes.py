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


# %%
"""
..######..##.....##....###............########..######.
.##....##.###...###...##.##...........##.......##....##
.##.......####.####..##...##..........##.......##......
.##.......##.###.##.##.....##.#######.######....######.
.##.......##.....##.#########.........##.............##
.##....##.##.....##.##.....##.........##.......##....##
..######..##.....##.##.....##.........########..######.
"""


# class CMAES:
#     def __init__(
#         self,
#         num_parameters,
#         lambda_,
#         mean0,
#         sigma0,
#     ):
#         self.num_parameters = num_parameters
#         self.num_perturbations = lambda_

#         self.mean = mean0
#         self.sigma = sigma0

#         self.C = np.eye(num_parameters)  # Covariance matrix
#         self.pc = np.zeros(num_parameters)  # Evolution path
#         self.ps = np.zeros(num_parameters)
#         self.weights = np.log(lambda_ + 0.5) - np.log(np.arange(1, lambda_ + 1))
#         self.weights /= self.weights.sum()
#         self.mu_eff = 1 / np.sum(self.weights**2)  # Effective number of parents
#         self.c1 = 2 / ((num_parameters + 1.3) ** 2 + self.mu_eff)
#         self.cmu = min(
#             1 - self.c1,
#             2
#             * (self.mu_eff - 2 + 1 / self.mu_eff)
#             / ((num_parameters + 2) ** 2 + self.mu_eff),
#         )
#         self.damps = (
#             1
#             + 2 * max(0, np.sqrt((self.mu_eff - 1) / (num_parameters + 1)) - 1)
#             + self.c1
#             + self.cmu
#         )
#         self.chiN = np.sqrt(num_parameters) * (
#             1 - 1 / (4 * num_parameters) + 1 / (21 * num_parameters**2)
#         )


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
        alpha=reacher.model.opt.timestep / 10e-3,
    )
    env = SequentialReachingEnv(
        plant=reacher, 
        target_duration=(3, 2, 6), 
        num_targets=10
    )
    optimizer = CMA(mean=rnn.get_params(), sigma=1.3)        
    
    num_generations = 1000
    fitnesses = []
    for gg in range(num_generations):
        solutions = []
        for ii in range(optimizer.population_size):
            x = optimizer.ask()
            value = -env.evaluate(rnn.from_params(x), seed=0) # change seed to gg!!!
            solutions.append((x, value))
            fitnesses.append((gg, ii, value))
            print(f"#{gg}.{ii} {value}")
        optimizer.tell(solutions)

#%%
# Plot the loss over iterations
fitness_values = [solution[1] for solution in solutions]
plt.plot(fitness_values)
plt.xlabel('Iterations')
plt.ylabel('Objective Function Value (Loss)')
plt.title('Fitness During CMA-ES Optimization')
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
# models_dir = "../models"
# gen_idx = 500  # Specify the generation index you want to plot
# model_file = f"best_rnn_gen_{gen_idx}.pkl"
# model_file = "best_rnn_gen_curr.pkl"
# with open(os.path.join(models_dir, model_file), "rb") as f:
#     best_rnn = pickle.load(f)
best_rnn = rnn.from_params(optimizer.mean)
env.evaluate(best_rnn, seed=0, render=True, log=True)
env.plot()
