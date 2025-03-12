# %%
"""
.####.##.....##.########...#######..########..########
..##..###...###.##.....##.##.....##.##.....##....##...
..##..####.####.##.....##.##.....##.##.....##....##...
..##..##.###.##.########..##.....##.########.....##...
..##..##.....##.##........##.....##.##...##......##...
..##..##.....##.##........##.....##.##....##.....##...
.####.##.....##.##.........#######..##.....##....##...
"""
import os
import mujoco
import mujoco.viewer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%
"""
.##.....##.##.....##.......##..#######...######...#######.
.###...###.##.....##.......##.##.....##.##....##.##.....##
.####.####.##.....##.......##.##.....##.##.......##.....##
.##.###.##.##.....##.......##.##.....##.##.......##.....##
.##.....##.##.....##.##....##.##.....##.##.......##.....##
.##.....##.##.....##.##....##.##.....##.##....##.##.....##
.##.....##..#######...######...#######...######...#######.
"""

os.chdir(os.path.dirname(__file__))
MODEL_XML_PATH = "mujoco/arm_model.xml"
model = mujoco.MjModel.from_xml_path(MODEL_XML_PATH)
data = mujoco.MjData(model)

num_actuators = model.nu
hand_id = model.geom("hand").id

dur2run = 10  # seconds
time_data = []
hand_position_data = {
    "x": [],
    "y": [],
    "z": [],
}
sensor_data = {
    "deltoid_length": [],
    "latissimus_length": [],
    "biceps_length": [],
    "triceps_length": [],
    "deltoid_velocity": [],
    "latissimus_velocity": [],
    "biceps_velocity": [],
    "triceps_velocity": [],
    "deltoid_force": [],
    "latissimus_force": [],
    "biceps_force": [],
    "triceps_force": [],
}

ctrl_increment = 0.05

# Simulate and save data
mujoco.mj_resetData(model, data)
while data.time < dur2run:
    mujoco.mj_step(model, data)

    # Random actuator control
    data.ctrl[:] = np.clip(
        data.ctrl + np.random.randn(num_actuators) * ctrl_increment, 0, 1
    )

    # Store time data
    time_data.append(data.time)

    # Store sensor data
    for i, key in enumerate(sensor_data.keys()):
        sensor_data[key].append(data.sensordata[i])

    # Store hand position data
    hand_position = data.geom_xpos[hand_id, :].copy()
    for i, key in enumerate(hand_position_data.keys()):
        hand_position_data[key].append(hand_position[i])

# %%
"""
.########..##........#######..########
.##.....##.##.......##.....##....##...
.##.....##.##.......##.....##....##...
.########..##.......##.....##....##...
.##........##.......##.....##....##...
.##........##.......##.....##....##...
.##........########..#######.....##...
"""

# Plot data
fig, axes = plt.subplots(2, 2)

# Determine the index up to which to plot
dur2plot = min(10, dur2run)
idcs2plot = np.searchsorted(time_data, dur2plot)

# Length
axes[0, 0].plot(
    time_data[:idcs2plot], sensor_data["deltoid_length"][:idcs2plot], label="Deltoid"
)
axes[0, 0].plot(
    time_data[:idcs2plot],
    sensor_data["latissimus_length"][:idcs2plot],
    label="Latissimus",
)
axes[0, 0].plot(
    time_data[:idcs2plot], sensor_data["biceps_length"][:idcs2plot], label="Biceps"
)
axes[0, 0].plot(
    time_data[:idcs2plot], sensor_data["triceps_length"][:idcs2plot], label="Triceps"
)
axes[0, 0].legend()
axes[0, 0].set_title("Length")
axes[0, 0].set_xlabel("Time (s)")
axes[0, 0].set_ylabel("Length (a.u.)")

# Velocity
axes[0, 1].plot(
    time_data[:idcs2plot], sensor_data["deltoid_velocity"][:idcs2plot], label="Deltoid"
)
axes[0, 1].plot(
    time_data[:idcs2plot],
    sensor_data["latissimus_velocity"][:idcs2plot],
    label="Latissimus",
)
axes[0, 1].plot(
    time_data[:idcs2plot], sensor_data["biceps_velocity"][:idcs2plot], label="Biceps"
)
axes[0, 1].plot(
    time_data[:idcs2plot], sensor_data["triceps_velocity"][:idcs2plot], label="Triceps"
)
axes[0, 1].legend()
axes[0, 1].set_title("Velocity")
axes[0, 1].set_xlabel("Time (s)")
axes[0, 1].set_ylabel("Velocity (a.u.)")

# Force
axes[1, 0].plot(
    time_data[:idcs2plot], sensor_data["deltoid_force"][:idcs2plot], label="Deltoid"
)
axes[1, 0].plot(
    time_data[:idcs2plot],
    sensor_data["latissimus_force"][:idcs2plot],
    label="Latissimus",
)
axes[1, 0].plot(
    time_data[:idcs2plot], sensor_data["biceps_force"][:idcs2plot], label="Biceps"
)
axes[1, 0].plot(
    time_data[:idcs2plot], sensor_data["triceps_force"][:idcs2plot], label="Triceps"
)
axes[1, 0].legend()
axes[1, 0].set_title("Force")
axes[1, 0].set_xlabel("Time (s)")
axes[1, 0].set_ylabel("Force (a.u.)")

# Position
axes[1, 1].plot(
    hand_position_data["x"][:idcs2plot],
    hand_position_data["z"][:idcs2plot],
    color="black",
    marker=".",
    markersize=0.1,
)
axes[1, 1].set_title("Hand position")
axes[1, 1].set_xlabel("x (a.u.)")
axes[1, 1].set_ylabel("y (a.u.)")

plt.tight_layout()
plt.show()

# %%
"""
..######..########....###....########..######.
.##....##....##......##.##......##....##....##
.##..........##.....##...##.....##....##......
..######.....##....##.....##....##.....######.
.......##....##....#########....##..........##
.##....##....##....##.....##....##....##....##
..######.....##....##.....##....##.....######.
"""

# Convert sensor_data and hand_position_data to pandas DataFrames
sensor_df = pd.DataFrame(sensor_data)
hand_position_df = pd.DataFrame(hand_position_data)

# Compute statistics
sensor_stats_df = pd.DataFrame(
    {
        "Range": sensor_df.max() - sensor_df.min(),
        "Min": sensor_df.min(),
        "Max": sensor_df.max(),
        "Mean": sensor_df.mean(),
    }
)

target_stats_df = pd.DataFrame(
    {
        "Range": hand_position_df.max() - hand_position_df.min(),
        "Min": hand_position_df.min(),
        "Max": hand_position_df.max(),
        "Mean": hand_position_df.mean(),
    }
)

print(sensor_stats_df)
print(target_stats_df)

# %%
"""
.########..########....###.....######..##.....##
.##.....##.##.........##.##...##....##.##.....##
.##.....##.##........##...##..##.......##.....##
.########..######...##.....##.##.......#########
.##...##...##.......#########.##.......##.....##
.##....##..##.......##.....##.##....##.##.....##
.##.....##.########.##.....##..######..##.....##
"""
x, y = hand_position_data["x"], hand_position_data["z"]
counts2d, x_edges, y_edges = np.histogram2d(x, y, bins=100)

plt.figure()
plt.imshow(
    counts2d, extent=(x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]), origin="lower"
)

x_centers = (x_edges[:-1] + x_edges[1:]) / 2
y_centers = (y_edges[:-1] + y_edges[1:]) / 2
nonzero_idcs = np.argwhere(counts2d > 0)
reachable_positions = [(x_centers[i], 0, y_centers[j]) for i, j in nonzero_idcs]

plt.figure()
plt.imshow(
    counts2d > 0,
    extent=(x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]),
    origin="lower",
)

# Convert reachable_positions to a DataFrame
reachable_positions_df = pd.DataFrame(reachable_positions, columns=["x", "y", "z"])

print(reachable_positions_df)


# %%
"""
..######.....###....##.....##.########
.##....##...##.##...##.....##.##......
.##........##...##..##.....##.##......
..######..##.....##.##.....##.######..
.......##.#########..##...##..##......
.##....##.##.....##...##.##...##......
..######..##.....##....###....########
"""
save_dir = "../mujoco"

# Save sensor_data and hand_position_data to the mujoco folder
sensor_stats_df.to_pickle(f"{save_dir}/sensor_stats.pkl")
target_stats_df.to_pickle(f"{save_dir}/target_stats.pkl")

# Save reachable_positions_df to the mujoco folder
reachable_positions_df.to_pickle(f"{save_dir}/reachable_positions.pkl")
