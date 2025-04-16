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
reacher = SequentialReacher(plant_xml_file="arm_model.xml")
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
gen_idx = 9999  # Specify the generation index you want to load
model_file = f"optimizer_gen_{gen_idx}_cmaesv2.pkl"
with open(os.path.join(models_dir, model_file), "rb") as f:
    optimizer = pickle.load(f)
best_rnn = rnn.from_params(optimizer.mean)

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
num_targets = 100

# Sample 10 target positions from the reacher
sampled_targets = reacher.sample_targets(num_targets)

# Extract x and z coordinates from the sampled targets
x_coords = sampled_targets[:, 0]
z_coords = sampled_targets[:, 2]

# Plot the x and z coordinates
plt.figure(figsize=(6, 6))
plt.scatter(x_coords, z_coords, c="red", s=25, alpha=0.7)
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
                size=3,
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
                size=3,
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
from plants import SequentialReacher
from environments import SequentialReachingEnv
from networks import RNN

reacher = SequentialReacher(plant_xml_file="arm_model_nailed.xml")
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
forces = env.stimulate(best_rnn, units=np.array([8]), delay=3, seed=0, render=True)

# %%
import mujoco

model = reacher.model
data = reacher.data

# Assume model and data are already loaded
target_eq_name = "nail"

# Step 1: Find the equality constraint ID by name
eq_id = None
for i in range(model.neq):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_EQUALITY, i)
    if name == target_eq_name:
        eq_id = i
        break
if eq_id is None:
    raise RuntimeError(f"Equality constraint '{target_eq_name}' not found.")

print("Equality constraint ID:", eq_id)

# Step 2: Get the constraint type to determine its size
eq_type = model.eq_type[eq_id]
eq_sizes = {
    mujoco.mjtEq.mjEQ_CONNECT: 3,
    mujoco.mjtEq.mjEQ_WELD: 6,
    mujoco.mjtEq.mjEQ_JOINT: 1,
    mujoco.mjtEq.mjEQ_TENDON: 1,
    mujoco.mjtEq.mjEQ_DISTANCE: 1,
}
constraint_dim = eq_sizes[eq_type]

# Step 3: Sum dimensions of all previous equality constraints to find start index
efc_start = 0
for i in range(eq_id):
    prev_type = model.eq_type[i]
    efc_start += eq_sizes[prev_type]

# Step 4: Extract the force vector (usually length 3 for CONNECT)
force_vec = data.efc_force[efc_start : efc_start + constraint_dim]

print(f"Force from equality constraint '{target_eq_name}':", force_vec)
