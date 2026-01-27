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
from plants import *
from encoders import *
from environments import *
from networks import *
from utils import *
from cmaes import CMA
import numpy as np

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


if __name__ == "__main__":

    # Initialize the plant
    reacher = SequentialReacher(plant_xml_file="arm.xml")

    # Initialize the target encoder
    target_encoder = GridTargetEncoder(
        grid_size=8,
        x_bounds=reacher.get_workspace_bounds()[0],
        y_bounds=reacher.get_workspace_bounds()[1],
        sigma=0.25,
    )

    # Specify policy
    rnn = NeuroMuscularRNN(
        target_size=target_encoder.size,
        length_size=reacher.num_sensors_len,
        velocity_size=reacher.num_sensors_vel,
        force_size=reacher.num_sensors_frc,
        hidden_size=25,
        output_size=reacher.num_actuators,
        hidden_activation=tanh,
        smoothing_factor=alpha_from_tau(tau=10e-3, dt=reacher.model.opt.timestep),
    )

    # Initialize the environment/task
    env = SequentialReachingEnv(
        plant=reacher,
        target_encoder=target_encoder,
        target_duration_distro={"mean": 3, "min": 1, "max": 6},
        iti_distro={"mean": 1, "min": 0, "max": 3},
        num_targets=10,
        randomize_gravity=True,
        shaping_functions={
            "distance": lambda x: x,
            "energy": lambda x: x,
        },
        loss_weights={
            "distance": 1,
            "energy": 0.1,
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
.########..####..######..########.####.##.......##..........###....########.####..#######..##....##
.##.....##..##..##....##....##.....##..##.......##.........##.##......##.....##..##.....##.###...##
.##.....##..##..##..........##.....##..##.......##........##...##.....##.....##..##.....##.####..##
.##.....##..##...######.....##.....##..##.......##.......##.....##....##.....##..##.....##.##.##.##
.##.....##..##........##....##.....##..##.......##.......#########....##.....##..##.....##.##..####
.##.....##..##..##....##....##.....##..##.......##.......##.....##....##.....##..##.....##.##...###
.########..####..######.....##....####.########.########.##.....##....##....####..#######..##....##
"""
from rl_distillation import *

reacher = SequentialReacher(plant_xml_file="arm.xml")
target_encoder = GridTargetEncoder(
    grid_size=8,
    x_bounds=reacher.get_workspace_bounds()[0],
    y_bounds=reacher.get_workspace_bounds()[1],
    sigma=0.25,
)
env = SequentialReachingEnv(
    plant=reacher,
    target_encoder=target_encoder,
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
rnn = NeuroMuscularRNN(
    target_size=target_encoder.size,
    length_size=reacher.num_sensors_len,
    velocity_size=reacher.num_sensors_vel,
    force_size=reacher.num_sensors_frc,
    hidden_size=25,
    output_size=reacher.num_actuators,
    hidden_activation=tanh,
    smoothing_factor=alpha_from_tau(tau=10e-3, dt=reacher.model.opt.timestep),
)

rl_config = RLConfig(num_iterations=1000, lr_actor=3e-4, buffer_size=2048)
distillation_config = DistillationConfig(
    num_epochs=100, num_trajectories=1000, temperature=1.0
)
trained_rnn, teacher, stats = train_with_rl_distillation(
    env=env, rnn=rnn, rl_config=rl_config, distillation_config=distillation_config
)
env.evaluate(trained_rnn, render=True, log=True)
env.plot()

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
