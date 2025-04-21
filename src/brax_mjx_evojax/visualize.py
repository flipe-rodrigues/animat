import mujoco
import numpy as np
import time
import argparse
import pickle
import os

def visualize_model(xml_file, policy_file=None):
    """Visualize the model with an optional trained policy."""
    # Load model
    model = mujoco.MjModel.from_xml_path(xml_file)
    data = mujoco.MjData(model)
    
    # Initialize viewer
    viewer = mujoco.viewer.launch_passive(model, data)
    
    # Load policy if provided
    policy = None
    if policy_file is not None and os.path.exists(policy_file):
        try:
            with open(policy_file, 'rb') as f:
                policy_params = pickle.load(f)
            print(f"Loaded policy from {policy_file}")
            
            # Here we would need to initialize the policy
            # This depends on how the policy is saved
            # policy = TanhRNNPolicy(...)
            # policy_state = policy.init_states(1)
        except Exception as e:
            print(f"Failed to load policy: {e}")
            policy = None
    
    # Set initial target position
    target_pos = np.array([0.3, -0.2, 0.1])
    data.mocap_pos[0] = target_pos
    
    # Main simulation loop
    t0 = time.time()
    while viewer.is_running():
        # Step physics
        mujoco.mj_step(model, data)
        
        # Apply policy if available
        if policy is not None:
            # In a real implementation, we would:
            # 1. Get observations (sensor data + target)
            # 2. Compute actions using the policy
            # 3. Apply actions to data.ctrl
            pass
        
        # Update viewer
        viewer.sync()
        
        # Control simulation speed
        time_until_next_step = 0.01 - (time.time() - t0)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
        t0 = time.time()
        
        # Maybe change target position every few seconds
        if np.random.random() < 0.001:  # Low probability per step
            target_pos = np.array([
                np.random.uniform(-0.5, 0.5),
                np.random.uniform(-0.5, 0.5),
                np.random.uniform(0.0, 0.3)
            ])
            data.mocap_pos[0] = target_pos
            print(f"New target position: {target_pos}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--xml_file', type=str, default='arm_model.xml',
                      help='Path to the MuJoCo XML file')
    parser.add_argument('--policy_file', type=str, default=None,
                      help='Path to the trained policy file (optional)')
    args = parser.parse_args()
    
    visualize_model(args.xml_file, args.policy_file)