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
import plotly.graph_objects as go
from plants import SequentialReacher
from environments import SequentialReachingEnv
from networks import RNN
from utils import *
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA

# %%
"""
.##........#######.....###....########.
.##.......##.....##...##.##...##.....##
.##.......##.....##..##...##..##.....##
.##.......##.....##.##.....##.##.....##
.##.......##.....##.#########.##.....##
.##.......##.....##.##.....##.##.....##
.########..#######..##.....##.########.
"""
reacher = SequentialReacher(plant_xml_file="arm_with_pulley.xml")
rnn = RNN(
    input_size=3 + reacher.num_sensors,
    hidden_size=25,
    output_size=reacher.num_actuators,
    activation=tanh,
    alpha=1,  # reacher.model.opt.timestep / 10e-3,
)
env = SequentialReachingEnv(
    plant=reacher,
    target_duration={"mean": 60, "min": 60, "max": 60},
    num_targets=2,
    loss_weights={
        "euclidean": 1,
        "manhattan": 0,
        "energy": 0,
        "ridge": 0.001,
        "lasso": 0,
    },
)
models_dir = "../../models"
gen_idx = 9000  # Specify the generation index you want to load
model_file = f"optimizer_gen_{gen_idx}_cmaesv2.pkl"
# model_file = "optimizer_gen_5000_tau10_rnn50.pkl"
with open(os.path.join(models_dir, model_file), "rb") as f:
    optimizer = pickle.load(f)
best_rnn = rnn.from_params(optimizer.mean)

# Swap input weights at indices 1 and 2 in best_rnn
best_rnn.W_in[:, [1, 2]] = best_rnn.W_in[:, [2, 1]]

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
# best_rnn.W_in[:,:3] = 0
# plt.figure(figsize=(10, 10))
# sns.heatmap(best_rnn.W_in, cmap="viridis", cbar=True)
# plt.title("Input Weights")
# plt.xlabel("Input Features")
# plt.ylabel("Hidden Units")
env.feldman(best_rnn, seed=0, render=True, log=True)
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

# Plot input weights
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

# %%
"""
.####.##....##.########..##.....##.########.....######..########.....###.....######..########
..##..###...##.##.....##.##.....##....##.......##....##.##.....##...##.##...##....##.##......
..##..####..##.##.....##.##.....##....##.......##.......##.....##..##...##..##.......##......
..##..##.##.##.########..##.....##....##........######..########..##.....##.##.......######..
..##..##..####.##........##.....##....##.............##.##........#########.##.......##......
..##..##...###.##........##.....##....##.......##....##.##........##.....##.##....##.##......
.####.##....##.##.........#######.....##........######..##........##.....##..######..########
"""
num_targets = 1000

# Sample 10 target positions from the reacher
sampled_targets = reacher.sample_targets(num_targets)

# Extract x and z coordinates from the sampled targets
x_coords = sampled_targets[:, 0]
y_coords = sampled_targets[:, 1]

# Plot the x and z coordinates
plt.figure(figsize=(6, 6))
plt.scatter(x_coords, y_coords, c="red", s=25, alpha=0.7)
plt.title("Sampled Targets: X vs Z Coordinates")
plt.xlabel("X Coordinate")
plt.ylabel("Z Coordinate")
plt.grid(True)
plt.show()

# Project the sampled targets through the corresponding input weights (first 3 columns)
input_weights_targets = weights_input[:, :3]
projections = np.dot(sampled_targets, input_weights_targets.T)

# Perform PCA on the projections
pca = PCA(n_components=3)
pca_projections = pca.fit_transform(projections)

# Select specific rows (e.g., 6, 19, 24) from the 2D array and stack them
selected_weights = input_weights_targets[[6, 19, 24], :]
selected_projections = np.dot(sampled_targets, selected_weights.T)

# Visualize the PCA projections in 3D using Plotly
fig = go.Figure(
    data=[
        go.Scatter3d(
            x=selected_projections[:, 0],
            y=selected_projections[:, 1],
            z=selected_projections[:, 2],
            mode="markers",
            marker=dict(
                size=2,
                color="black",
                opacity=0.8,
            ),
        )
    ]
)

fig.update_layout(
    title="Selected Projections of Sampled Targets Through Input Weights (3D)",
    scene=dict(xaxis_title="Unit 6", yaxis_title="Unit 19", zaxis_title="Unit 24"),
)

fig.show()

# Visualize the PCA projections in 3D
fig = go.Figure(
    data=[
        go.Scatter3d(
            x=pca_projections[:, 0],
            y=pca_projections[:, 1],
            z=pca_projections[:, 2],
            mode="markers",
            marker=dict(
                size=2,
                color=pca_projections[:, 0],  # Color by the first principal component
                colorscale="Viridis",
                opacity=0.8,
            ),
        )
    ]
)

fig.update_layout(
    title="PCA Projections of Sampled Targets Through Input Weights (3D)",
    scene=dict(xaxis_title="PC 1", yaxis_title="PC 2", zaxis_title="PC 3"),
)

fig.show()

# %%
"""
..######..##.......##.....##..######..########.########.########.
.##....##.##.......##.....##.##....##....##....##.......##.....##
.##.......##.......##.....##.##..........##....##.......##.....##
.##.......##.......##.....##..######.....##....######...########.
.##.......##.......##.....##.......##....##....##.......##...##..
.##....##.##.......##.....##.##....##....##....##.......##....##.
..######..########..#######...######.....##....########.##.....##
"""

# Calculate the total absolute output weights for each hidden unit
total_abs_output_weights = np.sum(np.abs(weights_output), axis=1)

# Plot the distribution of total absolute output weights
plt.figure(figsize=(10, 6))
sns.histplot(total_abs_output_weights, kde=True, bins=20, color="blue")
plt.title("Distribution of Total Absolute Output Weights for Each Hidden Unit")
plt.xlabel("Total Absolute Output Weights")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

# Calculate the total absolute sensory weights for each hidden unit
total_abs_input_weights = np.sum(np.abs(weights_input), axis=1)

# Plot the distribution of total absolute sensory weights
plt.figure(figsize=(10, 6))
sns.histplot(total_abs_input_weights, kde=True, bins=20, color="green")
plt.title("Distribution of Total Absolute Sensory Weights for Each Hidden Unit")
plt.xlabel("Total Absolute Sensory Weights")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

total_abs_hidden_weights = np.sum(np.abs(weights_hidden), axis=1)

# Plot the distribution of total absolute hidden weights
plt.figure(figsize=(10, 6))
sns.histplot(total_abs_hidden_weights, kde=True, bins=20, color="orange")
plt.title("Distribution of Total Absolute Hidden Weights for Each Hidden Unit")
plt.xlabel("Total Absolute Hidden Weights")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()


# %%
"""
.......##..#######..####.##....##.########
.......##.##.....##..##..###...##....##...
.......##.##.....##..##..####..##....##...
.......##.##.....##..##..##.##.##....##...
.##....##.##.....##..##..##..####....##...
.##....##.##.....##..##..##...###....##...
..######...#######..####.##....##....##...
"""

# Create a 2D joint distribution plot of total sensory and total output weights
plt.figure(figsize=(10, 8))
joint_plot = sns.jointplot(
    x=total_abs_input_weights,
    y=total_abs_output_weights,
    kind="scatter",
    cmap="viridis",
    marginal_kws=dict(bins=20, fill=True),
)

# Overlay the index of each neuron as text
for i, (x, y) in enumerate(zip(total_abs_input_weights, total_abs_output_weights)):
    joint_plot.ax_joint.text(
        x, y + 0.1, str(i), fontsize=8, color="black", ha="center", va="center"
    )

plt.suptitle("2D Joint Distribution of Total Input and Output Weights", y=1.02)
joint_plot.set_axis_labels(
    "Total Absolute Input Weights", "Total Absolute Output Weights"
)
plt.show()

# Create a 2D joint distribution plot of total sensory and total output weights
plt.figure(figsize=(10, 8))
joint_plot = sns.jointplot(
    x=total_abs_input_weights,
    y=total_abs_hidden_weights,
    kind="scatter",
    cmap="viridis",
    marginal_kws=dict(bins=20, fill=True),
)

# Overlay the index of each neuron as text
for i, (x, y) in enumerate(zip(total_abs_input_weights, total_abs_hidden_weights)):
    joint_plot.ax_joint.text(
        x, y + 0.1, str(i), fontsize=8, color="black", ha="center", va="center"
    )

plt.suptitle("2D Joint Distribution of Total Input and Output Weights", y=1.02)
joint_plot.set_axis_labels(
    "Total Absolute Input Weights", "Total Absolute Hidden Weights"
)
plt.show()

# Create a 2D joint distribution plot of total sensory and total output weights
plt.figure(figsize=(10, 8))
joint_plot = sns.jointplot(
    x=total_abs_hidden_weights,
    y=total_abs_output_weights,
    kind="scatter",
    cmap="viridis",
    marginal_kws=dict(bins=20, fill=True),
)

# Overlay the index of each neuron as text
for i, (x, y) in enumerate(zip(total_abs_hidden_weights, total_abs_output_weights)):
    joint_plot.ax_joint.text(
        x, y + 0.1, str(i), fontsize=8, color="black", ha="center", va="center"
    )

plt.suptitle("2D Joint Distribution of Total Hidden and Output Weights", y=1.02)
joint_plot.set_axis_labels(
    "Total Absolute Hidden Weights", "Total Absolute Output Weights"
)
plt.show()

# %%
"""
..######..########.####.##.....##.##.....##.##..........###....########.########
.##....##....##.....##..###...###.##.....##.##.........##.##......##....##......
.##..........##.....##..####.####.##.....##.##........##...##.....##....##......
..######.....##.....##..##.###.##.##.....##.##.......##.....##....##....######..
.......##....##.....##..##.....##.##.....##.##.......#########....##....##......
.##....##....##.....##..##.....##.##.....##.##.......##.....##....##....##......
..######.....##....####.##.....##..#######..########.##.....##....##....########
"""
import pickle
import matplotlib.pyplot as plt
from plants import SequentialReacher
from environments import SequentialReachingEnv
from networks import RNN
from utils import *
from sklearn.decomposition import PCA

reacher = SequentialReacher(plant_xml_file="arm.xml")
print("Number of sensors:", reacher.num_sensors)
print("Number of actuators:", reacher.num_actuators)

rnn = RNN(
    input_size=3 + reacher.num_sensors,
    hidden_size=25,
    output_size=reacher.num_actuators,
    activation=tanh,
    alpha=1,  # reacher.model.opt.timestep / 10e-3,
)
env = SequentialReachingEnv(
    plant=reacher,
    target_duration={"mean": 3, "min": 1, "max": 6},
    num_targets=10,
    loss_weights={
        "euclidean": 1,
        "manhattan": 0,
        "energy": 0,
        "ridge": 0.001,
        "lasso": 0,
    },
)

models_dir = "../../models"
gen_idx = 9999
model_file = f"optimizer_gen_{gen_idx}_cmaesv2.pkl"
with open(os.path.join(models_dir, model_file), "rb") as f:
    optimizer = pickle.load(f)
best_rnn = rnn.from_params(optimizer.mean)

# Zero out the first 3 columns of the input weights
# rnn.W_in[:, :3] = 0
# rnn.W_in[:, :] = 0

num_targets = 25
sampled_targets = reacher.sample_targets(num_targets)
x_coords = sampled_targets[:, 0]
y_coords = sampled_targets[:, 1]

# Plot the x and z coordinates
plt.figure(figsize=(6, 6))
plt.scatter(x_coords, y_coords, c="red", s=25, alpha=0.7)
plt.title("Sampled Targets: X vs Z Coordinates")
plt.xlabel("X Coordinate")
plt.ylabel("Z Coordinate")
plt.grid(True)
plt.show()

for unit_idx in range(0, rnn.hidden_size):

    force_data = env.stimulate(
        best_rnn,
        units=np.array([unit_idx]),
        action_modifier=1,
        delay=1,
        seed=0,
        render=True if unit_idx == 0 else False,
    )

    # Plot force vectors over time
    position_vecs = np.array(force_data["position"])
    force_vecs = np.array(force_data["force"])

    # Replace NaNs with zeros in force_vecs
    position_vecs = np.nan_to_num(position_vecs)
    force_vecs = np.nan_to_num(force_vecs)

    time = np.linspace(0, reacher.data.time, len(force_vecs))

    if unit_idx == 0:
        plt.figure(figsize=(25, 5))
        for i in range(force_vecs.shape[1] - 1):
            plt.plot(time, force_vecs[:, i], label=f"Force Component {i+1}")
        for t in np.arange(0.5, reacher.data.time, 1.0):
            plt.axvline(x=t, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
        lower_percentile = np.percentile(force_vecs, 1)
        upper_percentile = np.percentile(force_vecs, 99)
        plt.ylim([lower_percentile, upper_percentile])
        plt.xlabel("Time (s)")
        plt.ylabel("Force (a.u.)")
        plt.title("Force Vectors Over Time")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # Define the time window for averaging (100 ms)
    time_window = 0.1  # 100 ms

    # Initialize a list to store average force vectors
    average_positions = []
    rest_average_forces = []
    stim_average_forces = []

    # Iterate second by second
    for t in range(1, int(reacher.data.time) + 1):

        # Find indices corresponding to the last 100 ms of the current second
        start_time = t - .5 - time_window
        stop_time = t - .5
        indices = (time > start_time) & (time <= stop_time)

        # Compute the average force vector within the 100-ms period
        avg_force_vec = np.mean(force_vecs[indices], axis=0)
        rest_average_forces.append(avg_force_vec)

        # Find indices corresponding to the last 100 ms of the current second
        start_time = t - time_window
        stop_time = t
        indices = (time > start_time) & (time <= stop_time)

        # Compute the average position vector within the 100-ms period
        avg_position_vec = np.mean(position_vecs[indices], axis=0)
        average_positions.append(avg_position_vec)

        # Compute the average force vector within the 100-ms period
        avg_force_vec = np.mean(force_vecs[indices], axis=0)
        stim_average_forces.append(avg_force_vec)

    # Convert the list to a numpy array for further analysis
    average_positions = np.array(average_positions)
    rest_average_forces = np.array(rest_average_forces)
    stim_average_forces = np.array(stim_average_forces)

    # print(average_positions.shape)
    # print(rest_average_forces.shape)
    # print(stim_average_forces.shape)

    plt.figure(figsize=(8, 8))

    # Extract x and y components from average positions and forces
    x_positions = [pos[0] for pos in average_positions]
    y_positions = [pos[1] for pos in average_positions]
    x_forces = [force[0] for force in stim_average_forces]
    y_forces = [force[1] for force in stim_average_forces]

    # Plot the 2D vector field
    plt.quiver(
        x_positions,
        y_positions,
        x_forces,
        y_forces,
        angles="xy",
        scale_units="xy",
        scale=500,
        linewidth=1,
        color="red",
        edgecolor="red",
        facecolor='none',
        label="Stimulated",
    )

    # Extract x and y components from average positions and forces
    x_positions = [pos[0] for pos in average_positions]
    y_positions = [pos[1] for pos in average_positions]
    x_forces = [force[0] for force in rest_average_forces]
    y_forces = [force[1] for force in rest_average_forces]

    # Plot the 2D vector field
    plt.quiver(
        x_positions,
        y_positions,
        x_forces,
        y_forces,
        angles="xy",
        scale_units="xy",
        scale=500,
        linewidth=1,
        color="black",
        edgecolor="black",
        facecolor='none',
        label="Rest",
    )

    # Calculate the convergence point (mean of positions weighted by force magnitudes)
    force_magnitudes = np.sqrt(np.array(x_forces) ** 2 + np.array(y_forces) ** 2)
    convergence_x = np.average(x_positions, weights=np.abs(x_forces))
    convergence_y = np.average(y_positions, weights=np.abs(y_forces))

    # Add a dot at the convergence point
    # plt.scatter(convergence_x, convergence_y, color="red", s=100, label="Convergence Point")
    plt.legend()

    plt.title(f"Convergence force field (CFF) stimulating unit {unit_idx}")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.grid(True)
    plt.axis("equal")
    plt.xlim(
        reacher.hand_position_stats["min"][0], reacher.hand_position_stats["max"][0]
    )
    plt.ylim(
        reacher.hand_position_stats["min"][1], reacher.hand_position_stats["max"][1]
    )
    plt.show()
