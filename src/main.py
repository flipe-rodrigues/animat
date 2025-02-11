import time

import mujoco
import mujoco.viewer

m = mujoco.MjModel.from_xml_path("mujoco/arm_model.xml")
d = mujoco.MjData(m)

with mujoco.viewer.launch_passive(m, d) as viewer:
  # Close the viewer automatically after 30 wall-seconds.
  start = time.time()
  while viewer.is_running() and time.time() - start < 30:
    step_start = time.time()

    # mj_step can be replaced with code that also evaluates
    # a policy and applies a control signal before stepping the physics.
    mujoco.mj_step(m, d)

    # Example modification of a viewer option: toggle contact points every two seconds.
    with viewer.lock():
      viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)

    # Pick up changes to the physics state, apply perturbations, update options from GUI.
    viewer.sync()

    # Rudimentary time keeping, will drift relative to wall clock.
    time_until_next_step = m.opt.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
      time.sleep(time_until_next_step)

# import mujoco
# import numpy as np
# import time
# from arm_simulation import ArmSimulation

# def main():
#     # Initialize the MuJoCo environment
#     model = mujoco.load_model_from_path("mujoco/arm_model.xml")
#     sim = mujoco.MjSim(model)
#     viewer = mujoco.MjViewer(sim)

#     # Create an instance of the ArmSimulation class
#     arm_simulation = ArmSimulation(sim)

#     # Initialize the arm
#     arm_simulation.initialize_arm()

#     # Start the simulation loop
#     while True:
#         # Step the simulation
#         arm_simulation.step_simulation()

#         # Render the simulation
#         viewer.render()
#         time.sleep(0.01)  # Control the simulation speed

# if __name__ == "__main__":
#     main()