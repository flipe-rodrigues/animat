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
from plants import SequentialReacher
from environments import SequentialReachingEnv
from networks import RNN
from utils import *
from cmaes import CMA
import numpy as np

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

    # Initialize the plant
    reacher = SequentialReacher(plant_xml_file="arm_model.xml")

    # Specify policy
    rnn = RNN(
        input_size=3 + reacher.num_sensors,
        hidden_size=25,
        output_size=reacher.num_actuators,
        activation=tanh,
        alpha=reacher.model.opt.timestep / 10e-3,
    )

    # Initialize the environment/task
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

    # Optimization setup
    optimizer = CMA(mean=rnn.get_params(), sigma=1.3)
    num_generations = 10000
    fitnesses = []
    for gg in range(num_generations):
        solutions = []
        for ii in range(optimizer.population_size):
            x = optimizer.ask()
            fitness = -env.evaluate(rnn.from_params(x), seed=gg)
            solutions.append((x, fitness))
            fitnesses.append((gg, ii, fitness))
            print(f"#{gg}.{ii} {fitness}")
        optimizer.tell(solutions)

        best_rnn = rnn.from_params(optimizer.mean)
        if gg % 10 == 0:
            env.evaluate(best_rnn, seed=0, render=True, log=True)
            env.plot()
        if gg % 1000 == 0:
            file = f"../../models/optimizer_gen_{gg}_cmaesv2.pkl"
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
