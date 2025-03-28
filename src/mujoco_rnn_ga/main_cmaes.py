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
import seaborn as sns

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
        alpha=1,  # reacher.model.opt.timestep / 25e-3,
    )
    env = SequentialReachingEnv(
        plant=reacher,
        target_duration={"mean": 3, "min": 1, "max": 6},
        num_targets=10,
        loss_weights={"euclidean": 1, "manhattan": 0, "energy": 0},
    )
    optimizer = CMA(mean=rnn.get_params(), sigma=1.3)

    num_generations = 10000
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
# models_dir = "../../models"
# gen_idx = 200  # Specify the generation index you want to plot
# model_file = f"best_rnn_gen_{gen_idx}_cmaesv2.pkl"
# model_file = "best_rnn_gen_curr.pkl"
# with open(os.path.join(models_dir, model_file), "rb") as f:
#     best_rnn = pickle.load(f)
best_rnn = rnn.from_params(optimizer.mean)
env.evaluate(best_rnn, seed=0, render=True, log=True)
env.plot()

# %%
reacher = SequentialReacher(plant_xml_file="arm_model.xml")
env = SequentialReachingEnv(
    plant=reacher,
    target_duration={"mean": 3, "min": 1, "max": 6},
    num_targets=10,
    loss_weights={"euclidean": 1, "manhattan": 0, "energy": 0},
)
models_dir = "../../models"
model_file = "best_rnn_gen_curr_CMA-ES_v2.pkl"
with open(os.path.join(models_dir, model_file), "rb") as f:
    best_rnn = pickle.load(f)
env.evaluate(best_rnn, seed=0, render=True, log=True)
env.plot()

# %%
"""
.########..####..######...######..########..######..########
.##.....##..##..##....##.##....##.##.......##....##....##...
.##.....##..##..##.......##.......##.......##..........##...
.##.....##..##...######...######..######...##..........##...
.##.....##..##........##.......##.##.......##..........##...
.##.....##..##..##....##.##....##.##.......##....##....##...
.########..####..######...######..########..######.....##...
"""

# Extract weights and biases from the RNN
weights_input = best_rnn.W_in
weights_hidden = best_rnn.W_h
weights_output = best_rnn.W_out.T
bias_hidden = best_rnn.b_h
bias_output = best_rnn.b_out

# Combine all weights and biases into a single figure
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# # Adjust aspect ratio for each heatmap based on matrix dimensions
# def set_aspect_ratio(ax, matrix):
#     rows, cols = matrix.shape
#     ax.set_aspect(rows / cols)


# # Plot input weights
# sns.heatmap(weights_input, cmap="viridis", cbar=True, ax=axes[0, 0])
# set_aspect_ratio(axes[0, 0], weights_input)
# axes[0, 0].set_title("Input Weights")
# axes[0, 0].set_xlabel("Input Features")
# axes[0, 0].set_ylabel("Hidden Units")

# # Plot hidden weights
# sns.heatmap(weights_hidden, cmap="viridis", cbar=True, ax=axes[0, 1])
# set_aspect_ratio(axes[0, 1], weights_hidden)
# axes[0, 1].set_title("Hidden Weights")
# axes[0, 1].set_xlabel("Hidden Units")
# axes[0, 1].set_ylabel("Hidden Units")

# # Plot output weights
# sns.heatmap(weights_output, cmap="viridis", cbar=True, ax=axes[0, 2])
# set_aspect_ratio(axes[0, 2], weights_output)
# axes[0, 2].set_title("Output Weights")
# axes[0, 2].set_xlabel("Hidden Units")
# axes[0, 2].set_ylabel("Output Units")

# # Plot hidden biases
# sns.heatmap(bias_hidden, cmap="viridis", cbar=True, ax=axes[1, 1])
# set_aspect_ratio(axes[1, 1], bias_hidden)
# axes[1, 1].set_title("Hidden Biases")
# axes[1, 1].set_xlabel("Hidden Units")
# axes[1, 1].set_yticks([])

# # Plot output biases
# sns.heatmap(bias_output, cmap="viridis", cbar=True, ax=axes[1, 2])
# set_aspect_ratio(axes[1, 2], bias_output)
# axes[1, 2].set_title("Output Biases")
# axes[1, 2].set_xlabel("Output Units")
# axes[1, 2].set_yticks([])

# # Plot input weights
sns.heatmap(weights_input, cmap="viridis", cbar=True, ax=axes[0, 0])
axes[0, 0].set_title("Input Weights")
axes[0, 0].set_xlabel("Input Features")
axes[0, 0].set_ylabel("Hidden Units")

# Plot hidden weights
sns.heatmap(weights_hidden, cmap="viridis", cbar=True, ax=axes[0, 1])
axes[0, 1].set_title("Hidden Weights")
axes[0, 1].set_xlabel("Hidden Units")
axes[0, 1].set_ylabel("Hidden Units")

# Plot output weights
sns.heatmap(weights_output, cmap="viridis", cbar=True, ax=axes[0, 2])
axes[0, 2].set_title("Output Weights")
axes[0, 2].set_xlabel("Hidden Units")
axes[0, 2].set_ylabel("Output Units")

# Plot hidden biases
sns.heatmap(
    bias_hidden.reshape(1, -1), cmap="viridis", cbar=True, annot=False, ax=axes[1, 1]
)
axes[1, 1].set_title("Hidden Biases")
axes[1, 1].set_xlabel("Hidden Units")
axes[1, 1].set_yticks([])

# Plot output biases
sns.heatmap(
    bias_output.reshape(1, -1), cmap="viridis", cbar=True, annot=False, ax=axes[1, 2]
)
axes[1, 2].set_title("Output Biases")
axes[1, 2].set_xlabel("Output Units")
axes[1, 2].set_yticks([])

# Remove the last unused subplot
axes[1, 0].axis("off")

# Adjust layout
plt.tight_layout()
plt.show()
