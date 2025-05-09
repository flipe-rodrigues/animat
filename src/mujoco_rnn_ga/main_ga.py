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
from utils import *

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


def evolve(env, rnn, num_individuals=25, num_generations=10, mutation_rate=0.1):
    """Run evolutionary learning process"""
    population = [
        RNN(
            rnn.input_size,
            rnn.hidden_size,
            rnn.output_size,
            rnn.activation,
            rnn.alpha,
        )
        for _ in range(num_individuals)
    ]

    best_rnn = []
    best_fitness = -np.inf

    # Mutation rate adaptation parameters
    too_low_counter = 0
    too_high_counter = 0
    adaptation_threshold = 5

    for gg in range(num_generations):
        fitnesses = np.array(
            [
                env.evaluate(individual, seed=gg, render=False)
                for individual in tqdm(population, desc="Evaluating")
            ]
        )
        best_idx = np.argmax(fitnesses)
        worst_idx = np.argmin(fitnesses)

        # Adapt mutation rate
        # if best_rnn == population[best_idx]:
        #     too_low_counter = 0
        #     too_high_counter += 1
        # else:
        #     too_low_counter += 1
        #     too_high_counter = 0
        # if too_low_counter >= adaptation_threshold:
        #     mutation_rate *= 1.1
        #     too_low_counter = 0
        # if too_high_counter >= adaptation_threshold:
        #     mutation_rate *= 0.9
        #     too_high_counter = 0

        if fitnesses[best_idx] > best_fitness:
            best_fitness = fitnesses[best_idx]
            best_rnn = population[best_idx]

        # Current generation statistics
        print(
            f"Generation {gg+1}, Best: {fitnesses[best_idx]:.2f}, Worst: {fitnesses[worst_idx]:.2f}, Sigma:{mutation_rate:.2f}"
        )

        # Select top individuals
        sorted_indices = np.argsort(fitnesses)[::-1]
        population = [population[i] for i in sorted_indices[: num_individuals // 2]]

        # Mutate top performers to create offspring
        for ii in range(len(population)):
            if np.random.rand() >= 0.5:
                parent1 = np.random.choice(population)
                parent2 = np.random.choice(population)
                child = RNN.recombine(parent1, parent2)
            else:
                parent = population[ii]
                child = parent.mutate(mutation_rate)
            population.append(child)

        if gg % 10 == 0:
            env.evaluate(best_rnn, seed=0, render=True, log=True)
            env.plot()
        if gg % 100 == 0:
            file = f"../../models/best_rnn_gen_{gg}_GA.pkl"
        else:
            file = f"../../models/best_rnn_gen_curr_GA.pkl"
        with open(file, "wb") as f:
            pickle.dump(best_rnn, f)

    return best_rnn


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
        activation=softpus,
        alpha=reacher.model.opt.timestep / 10e-3,
    )
    env = SequentialReachingEnv(
        plant=reacher,
        target_duration={"mean": 3, "min": 1, "max": 6},
        num_targets=10,
        loss_weights={"euclidean": 1, "manhattan": 0, "energy": 0},
    )
    best_rnn = evolve(
        env=env,
        rnn=rnn,
        num_individuals=100,
        num_generations=1000,
        mutation_rate=0.05,
    )

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
gen_idx = 400  # Specify the generation index you want to plot
model_file = f"best_rnn_gen_{gen_idx}_GA.pkl"
# model_file = "best_rnn_gen_curr_GA.pkl"
with open(os.path.join(models_dir, model_file), "rb") as f:
    best_rnn = pickle.load(f)

env.evaluate(best_rnn, seed=0, render=True, log=True)
env.plot()
