import mujoco
import mujoco.viewer
import pygame
import numpy as np
import time

# Load the model
MODEL_XML_PATH = "mujoco/arm_model.xml"  # Replace with your actual XML file
model = mujoco.MjModel.from_xml_path(MODEL_XML_PATH)
data = mujoco.MjData(model)

# Define actuator indices
flexor_shoulder_idx = 0
extensor_shoulder_idx = 1
flexor_elbow_idx = 2
extensor_elbow_idx = 3

increment = .01

# Start the MuJoCo viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    print("Simulation running...")
    
    # Example modification of viewer options
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = True
    viewer.sync()
    
    while viewer.is_running():
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

        # Step simulation
        mujoco.mj_step(model, data)

        # Render the updated simulation
        viewer.sync()

        # Rudimentary time keeping, will drift relative to wall clock.
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

    pygame.quit()
