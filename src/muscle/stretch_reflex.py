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
MODEL_XML_PATH = "../../mujoco/muscle.xml"
model = mujoco.MjModel.from_xml_path(MODEL_XML_PATH)
data = mujoco.MjData(model)

num_actuators = model.nu
stretcher_id = model.actuator("stretcher").id
alpha_id = model.actuator("alpha").id
gamma_static_id = model.actuator("gamma_static").id
gamma_dynamic_id = model.actuator("gamma_dynamic").id
soleus_id = model.actuator("soleus").id
length_sensor_id = model.sensor("soleus_length").id
velocity_sensor_id = model.sensor("soleus_velocity").id
force_sensor_id = model.sensor("soleus_force").id

dur2run = 5  # seconds
time_data = []
sensor_data = {
    "soleus_length": [],
    "soleus_velocity": [],
    "soleus_force": [],
}
afferent_data = {
    "Ia": [],
    "II": [],
}

# Get actuator length range
[length_min, length_max] = model.actuator_lengthrange[soleus_id]
length_range = length_max - length_min
print(f"Muscle length range: {length_min:.2f} to {length_max:.2f} (range: {length_range:.2f})")

# Define lambda range
lambda_extra = .5
lambda_range = length_range * (1 + lambda_extra)
lambda_min = length_min - lambda_extra / 2 * length_range
lambda_max = length_max + lambda_extra / 2 * length_range
print(f"Lambda range: {lambda_min:.2f} to {lambda_min + lambda_range:.2f} (range: {lambda_range:.2f})")

# Simulate and save data
mujoco.mj_resetData(model, data)
while data.time < dur2run:
    mujoco.mj_step(model, data)

    #
    data.ctrl[stretcher_id] = 0.1
    data.ctrl[alpha_id] = .5
    data.ctrl[gamma_static_id] = 0.25
    data.ctrl[gamma_dynamic_id] = 0.0

    # stretch reflex
    alpha_ = data.ctrl[alpha_id]
    x = data.sensordata[length_sensor_id]
    v = data.sensordata[velocity_sensor_id]
    lambda_static = data.ctrl[gamma_static_id] * lambda_range + lambda_min
    mu_ = data.ctrl[gamma_dynamic_id]
    lambda_dynamic = lambda_static - mu_ * v
    F = max(0, x - lambda_dynamic) + alpha_
    data.ctrl[soleus_id] = F

    # store afferent data
    Ia = mu_ * v
    II = max(0, x - lambda_static)
    afferent_data["Ia"].append(Ia)
    afferent_data["II"].append(II)

    # Store time data
    time_data.append(data.time)

    # Store sensor data
    for i, key in enumerate(sensor_data.keys()):
        sensor_data[key].append(data.sensordata[i])

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
fig, axes = plt.subplots(4, 1, figsize=(8, 10))

# Determine the index up to which to plot
dur2plot = min(10, dur2run)
idcs2plot = np.searchsorted(time_data, dur2plot)

# Length
axes[0].plot(
    time_data[:idcs2plot], sensor_data["soleus_length"][:idcs2plot], label="Soleus"
)
axes[0].legend()
axes[0].set_title("Length")
axes[0].set_xlabel("Time (s)")
axes[0].set_ylabel("Length (a.u.)")

# Velocity
axes[1].plot(
    time_data[:idcs2plot], sensor_data["soleus_velocity"][:idcs2plot], label="Soleus"
)
axes[1].legend()
axes[1].set_title("Velocity")
axes[1].set_xlabel("Time (s)")
axes[1].set_ylabel("Velocity (a.u.)")

# Force
axes[2].plot(
    time_data[:idcs2plot], sensor_data["soleus_force"][:idcs2plot], label="Soleus"
)
axes[2].legend()
axes[2].set_title("Force")
axes[2].set_xlabel("Time (s)")
axes[2].set_ylabel("Force (a.u.)")

# Spindle afferent Ia & II
axes[3].plot(
    time_data[:idcs2plot], afferent_data["Ia"][:idcs2plot], label="Afferent Ia"
)
axes[3].plot(
    time_data[:idcs2plot], afferent_data["II"][:idcs2plot], label="Afferent II"
)
axes[3].legend()
axes[3].set_title("Afferent Ia")
axes[3].set_xlabel("Time (s)")
axes[3].set_ylabel("Firing Rate (a.u.)")

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

# Compute statistics
sensor_stats_df = pd.DataFrame(
    {
        "min": sensor_df.min(),
        "max": sensor_df.max(),
        "mean": sensor_df.mean(),
        "std": sensor_df.std(),
    }
)

print(sensor_stats_df)
