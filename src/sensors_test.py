import mujoco
import mujoco.viewer
import numpy as np
import time
import matplotlib.pyplot as plt


MODEL_XML_PATH = "mujoco/arm_model.xml"  
model = mujoco.MjModel.from_xml_path(MODEL_XML_PATH)
data = mujoco.MjData(model)

# Simulation settings
num_steps = 5000  # How long to record data
increment = 0.01

# Storage for sensor data
sensor_data = {
    "flexor_length": [],
    "flexor_velocity": [],
    "extensor_length": [],
    "extensor_velocity": [],
    "biceps_length": [],
    "biceps_velocity": [],
    "triceps_length": [],
    "triceps_velocity": [],
}


with mujoco.viewer.launch_passive(model, data) as viewer:
    print("Simulation running...")

    for _ in range(num_steps):
        step_start = time.time()

        # Random actuator control
        data.ctrl[:] = np.clip(data.ctrl + np.random.uniform(-1, 1, size=4) * increment, 0, 1)

        # Step simulation
        mujoco.mj_step(model, data)

        # Store sensor data
        for i, key in enumerate(sensor_data.keys()):
            sensor_data[key].append(data.sensordata[i])

        # Render updated simulation
        viewer.sync()

        # Timing control
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

# Plot data
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
time_axis = np.arange(num_steps)

# Flexor
axes[0, 0].plot(time_axis, sensor_data["flexor_length"], label="Length")
axes[0, 0].plot(time_axis, sensor_data["flexor_velocity"], label="Velocity")
axes[0, 0].set_title("Flexor Tendon")
axes[0, 0].legend()

# Extensor
axes[0, 1].plot(time_axis, sensor_data["extensor_length"], label="Length")
axes[0, 1].plot(time_axis, sensor_data["extensor_velocity"], label="Velocity")
axes[0, 1].set_title("Extensor Tendon")
axes[0, 1].legend()

# Biceps
axes[1, 0].plot(time_axis, sensor_data["biceps_length"], label="Length")
axes[1, 0].plot(time_axis, sensor_data["biceps_velocity"], label="Velocity")
axes[1, 0].set_title("Biceps Tendon")
axes[1, 0].legend()

# Triceps
axes[1, 1].plot(time_axis, sensor_data["triceps_length"], label="Length")
axes[1, 1].plot(time_axis, sensor_data["triceps_velocity"], label="Velocity")
axes[1, 1].set_title("Triceps Tendon")
axes[1, 1].legend()

# Set axis labels
for ax in axes.flat:
    ax.set_xlabel("Steps")
    ax.set_ylabel("arb.")

plt.tight_layout()
plt.ion()  
plt.show()
plt.pause(10)  
plt.savefig('../figs/sensors.png')

