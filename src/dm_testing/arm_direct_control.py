import numpy as np
from dm_testing.arm_env import load
from dm_testing.dm_control_test import display_video

def reach_top_right_corner():
    """Direct the arm to reach the top right corner with fixed actions."""
    # Load the environment
    env = load()
    
    # Reset the environment
    time_step = env.reset()
    
    # Set target position to top right corner (assuming positive x and y are "top right")
    # Note: In the real system, you might need to modify these coordinates depending on your setup
    target_position = np.array([0.5, 0.5, 0.5])  # Example coordinates for top-right
    
    # If your environment allows direct setting of the target:
    env.physics.named.data.mocap_pos["target"] = target_position
    
    # Fixed action for each muscle (values between 0-1)
    # These values are just examples - you'll need to adjust them for your specific arm model
    # Higher values mean stronger muscle contraction
    fixed_action = np.array([0.7, 0.3, 0.2, 0.8])  # Example: deltoid, latissimus, biceps, triceps
    
    # Parameters for simulation
    duration = 5.0  # Simulate for 5 seconds
    frames = []
    
    step = 0
    
    # Run the simulation with the fixed action
    while not time_step.last():
        step += 1
        
        # Apply the fixed action
        time_step = env.step(fixed_action)
        
        # Capture frames every 10 steps
        if step % 10 == 0 or step == 1:
            frames.append(env.physics.render(camera_id=-1, width=640, height=480))
        
        # Print current hand position every 50 steps
        if step % 50 == 0:
            hand_position = env.physics.named.data.geom_xpos['hand']
            target_position = env.physics.named.data.mocap_pos['target']
            distance = np.linalg.norm(hand_position - target_position)
            print(f"Step {step}: Hand Position: {hand_position}, Distance to Target: {distance:.4f}")
    
    # Save the animation
    display_video(frames, filename='reach_corner.gif', framerate=30)
    print(f"Animation saved as 'reach_corner.gif' with {len(frames)} frames")
    
    return frames

if __name__ == "__main__":
    print("Commanding arm to reach top right corner...")
    reach_top_right_corner()
    print("Done!")