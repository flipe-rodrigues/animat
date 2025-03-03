import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Load the MuJoCo model
MODEL_XML_PATH = "mujoco/arm_model.xml"
model = mujoco.MjModel.from_xml_path(MODEL_XML_PATH)  # Replace with your model file
data = mujoco.MjData(model)

# Identify the geom you want to track
geom_name = 'hand'  # Change this to your geom name
geom_id = model.geom(geom_name).id

# Initialize storage for trajectory
trajectory = []

# Setup Matplotlib figure
fig, ax = plt.subplots()
ax.set_xlim(-1, 1)  # Adjust limits based on expected motion
ax.set_ylim(-1, 1)
line, = ax.plot([], [], 'bo-', markersize=5, label=geom_name)  # Blue dot with line

# Start the viewer
with mujoco.viewer.launch_passive(model, data) as viewer:

    def update_plot(frame):
        mujoco.mj_step(model, data)  # Step simulation
        viewer.sync()  # Update the viewer

        # Extract geom position in world coordinates
        geom_pos = data.geom_xpos[geom_id][:]  # Only (x, y) for 2D plotting

        # Append to trajectory
        trajectory.append(geom_pos)

        # Convert trajectory to NumPy array for plotting
        traj_array = np.array(trajectory)

        # Update plot data
        line.set_data(traj_array[:, 0], traj_array[:, 2])

        return line,

    # Run animation
    ani = FuncAnimation(fig, update_plot, interval=10, blit=True)
    
    # Show the plot
    plt.legend()
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title(f"Live Trajectory of {geom_name}")
    plt.show()
