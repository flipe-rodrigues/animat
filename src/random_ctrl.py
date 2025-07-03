# %%
import mujoco
import mujoco.viewer
import numpy as np
import time
import os

def get_root_path():
    root_path = os.path.abspath(os.path.dirname(__file__))
    while root_path != os.path.dirname(root_path):
        if os.path.exists(os.path.join(root_path, ".git")):
            break
        root_path = os.path.dirname(root_path)
    return root_path

# Load the model
plant_xml_file = "arm.xml" 
mj_dir = os.path.join(get_root_path(), "mujoco")
MODEL_XML_PATH = os.path.join(mj_dir, plant_xml_file)
model = mujoco.MjModel.from_xml_path(MODEL_XML_PATH)
data = mujoco.MjData(model)

# Define actuator indices
flexor_shoulder_idx = 0
extensor_shoulder_idx = 1
flexor_elbow_idx = 2
extensor_elbow_idx = 3

# Define the increment for control signal changes
increment = .01

# Set the number of data points to store for plotting
buffer_size = 100
sensor_data_buffer = np.zeros(buffer_size)  # Buffer for sensor values

# Start the MuJoCo viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    print("Simulation running...")
    
    # Example modification of viewer options
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = True
    viewer.sync()
    
    viewer.cam.lookat[:] = [0, 0, -.5]
    viewer.cam.azimuth = 90
    viewer.cam.elevation = 0

    while viewer.is_running():
        
        # Step simulation
        mujoco.mj_step(model, data)

        step_start = time.time()

        data.ctrl[flexor_shoulder_idx] = np.clip(
            data.ctrl[flexor_shoulder_idx] + np.random.uniform(-1, 1) * increment, 0, 1
        )
        data.ctrl[extensor_shoulder_idx] = np.clip(
            data.ctrl[extensor_shoulder_idx] + np.random.uniform(-1, 1) * increment, 0, 1
        )
        data.ctrl[flexor_elbow_idx] = np.clip(
            data.ctrl[flexor_elbow_idx] + np.random.uniform(-1, 1) * increment, 0, 1
        )
        data.ctrl[extensor_elbow_idx] = np.clip(
            data.ctrl[extensor_elbow_idx] + np.random.uniform(-1, 1) * increment, 0, 1
        )

        # Update sensor data (assuming the first sensor)
        new_sensor_value = data.sensordata[0]  # Modify index for multiple sensors
        sensor_data_buffer = np.roll(sensor_data_buffer, -1)  # Shift buffer
        sensor_data_buffer[-1] = new_sensor_value  # Add new value

        # Render the updated simulation
        viewer.sync()

        # Rudimentary time keeping, will drift relative to wall clock.
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)