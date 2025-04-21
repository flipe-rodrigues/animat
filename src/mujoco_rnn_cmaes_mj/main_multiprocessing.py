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
from multiprocessing import Pool
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

    with Pool() as pool:
        for gg in range(num_generations):

            def evaluate_candidate(params):
                loss = -env.evaluate(
                    RNN.from_params_static(params, 15, 25, 4, tanh, 1),
                    seed=gg,
                    render=False,
                )
                return params, loss

            candidates = [optimizer.ask() for _ in range(optimizer.population_size)]
            solutions = pool.map(evaluate_candidate, candidates)
            optimizer.tell(solutions)

            best_loss = min([loss for _, loss in solutions])
            print(f"Generation {gg}: Best Loss = {best_loss:.5f}")

            best_rnn = rnn.from_params(optimizer.mean)
            if gg % 10 == 0:
                env.evaluate(best_rnn, seed=0, render=True, log=True)
                env.plot()
            if gg % 1000 == 0:
                file = f"../../models/best_rnn_gen_{gg}_cma-es.pkl"
                with open(file, "wb") as f:
                    pickle.dump(best_rnn, f)

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
