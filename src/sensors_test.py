#%%
import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt

#%%
MODEL_XML_PATH = "../mujoco/arm_model.xml"  
model = mujoco.MjModel.from_xml_path(MODEL_XML_PATH)
data = mujoco.MjData(model)

duration = 10000
time_data = []
hand_id = model.geom('hand').id
hand_positions = []
sensor_data = {
    "deltoid_length": [],
    "deltoid_velocity": [],
    "latissimus_length": [],
    "latissimus_velocity": [],
    "biceps_length": [],
    "biceps_velocity": [],
    "triceps_length": [],
    "triceps_velocity": [],
}

ctrl_increment = 0.01

# Simulate and save data
mujoco.mj_resetData(model, data)
while data.time < duration:
    mujoco.mj_step(model, data)

    # Random actuator control
    data.ctrl[:] = np.clip(data.ctrl + np.random.uniform(-1, 1, size=4) * ctrl_increment, 0, 1)

    time_data.append(data.time)

    # Store sensor data
    for i, key in enumerate(sensor_data.keys()):
        sensor_data[key].append(data.sensordata[i])

    hand_positions.append(data.geom_xpos[hand_id,:].copy())

#%%
# Plot data
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
# time_vals = np.arange(num_steps)

# deltoid
axes[0, 0].plot(time_data, sensor_data["deltoid_length"], label="Length")
axes[0, 0].plot(time_data, sensor_data["deltoid_velocity"], label="Velocity")
axes[0, 0].set_title("deltoid Tendon")
axes[0, 0].legend()

# latissimus
axes[0, 1].plot(time_data, sensor_data["latissimus_length"], label="Length")
axes[0, 1].plot(time_data, sensor_data["latissimus_velocity"], label="Velocity")
axes[0, 1].set_title("latissimus Tendon")
axes[0, 1].legend()

# Biceps
axes[1, 0].plot(time_data, sensor_data["biceps_length"], label="Length")
axes[1, 0].plot(time_data, sensor_data["biceps_velocity"], label="Velocity")
axes[1, 0].set_title("Biceps Tendon")
axes[1, 0].legend()

# Triceps
axes[1, 1].plot(time_data, sensor_data["triceps_length"], label="Length")
axes[1, 1].plot(time_data, sensor_data["triceps_velocity"], label="Velocity")
axes[1, 1].set_title("Triceps Tendon")
axes[1, 1].legend()

# Set axis labels
for ax in axes.flat:
    ax.set_xlabel("Steps")
    ax.set_ylabel("arb.")

plt.tight_layout()
plt.show()

#%%
fig = plt.figure
plt.title('Hand position')
plt.xlabel('x')
plt.ylabel('y')
hand_position = np.array(hand_positions)
plt.plot(hand_position[:, 0], hand_position[:, 2], '.')
plt.show()

# %%
np.savetxt("targets.csv", hand_position, delimiter=",", header="x,y,z", comments="", fmt="%.3f")