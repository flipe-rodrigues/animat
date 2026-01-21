#%%
import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

MODEL_XML_PATH = "mujoco/arm.xml"

# Load the MuJoCo model
model = mujoco.MjModel.from_xml_path(MODEL_XML_PATH)  # Replace with your model file
data = mujoco.MjData(model)

buffer_size = 1000

# Start the viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    
    # Create figure and axis for plotting
    fig, ax = plt.subplots()
    ax.set_xlim(0, buffer_size)  # Number of simulation steps to show
    ax.set_ylim(0, 1)   # Adjust based on your sensor range

    sensor_data = []
    x_data = list(range(buffer_size))  # X-axis (time steps)
    line, = ax.plot(x_data, np.zeros(buffer_size))  # Initialize plot

    def update_plot(frame):
        mujoco.mj_step(model, data)  # Step simulation
        viewer.sync()  # Update the viewer

        # Append new sensor data (assuming single sensor)
        sensor_value = data.sensordata[0]  # Modify index based on your sensor
        sensor_data.append(sensor_value)

        # Keep only the latest 100 points
        if len(sensor_data) > buffer_size:
            sensor_data.pop(0)

        # Update the plot
        line.set_ydata(sensor_data + [0] * (buffer_size - len(sensor_data)))  # Padding if < 100 points
        return line,

    # Run animation
    ani = FuncAnimation(fig, update_plot, interval=10, blit=True)
    plt.show()