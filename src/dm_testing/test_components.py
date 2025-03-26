from dm_testing.dm_control_test import ArmEntity, CreatureObservables, ReachTargetTask, run_simulation
from dm_control import mjcf, composer
import numpy as np
import PIL.Image
import matplotlib.pyplot as plt

def test_arm_entity():
    """Test ArmEntity class"""
    # Load model
    arm_model = mjcf.from_path("mujoco/arm_model.xml")
    arm_entity = ArmEntity(arm_model)
    
    # Test properties
    print("Testing ArmEntity...")
    print(f"MJCF Model exists: {arm_entity.mjcf_model is not None}")
    print(f"Number of actuators: {len(arm_entity.actuators)}")
    print(f"Actuator names: {[act.name for act in arm_entity.actuators]}")
    return arm_entity

def test_observables(arm_entity):
    """Test CreatureObservables class"""
    print("\nTesting CreatureObservables...")
    obs = arm_entity._build_observables()
    
    # Test each observable
    print("Available observables:")
    for name, observable in obs._observables.items():
        print(f"- {name}")
    return obs

def test_reach_task(arm_entity):
    """Test ReachTargetTask class"""
    print("\nTesting ReachTargetTask...")
    
    # Create task
    task = ReachTargetTask(arm_entity)
    
    # Create environment
    env = composer.Environment(task)
    
    # Test initialization
    time_step = env.reset()
    print("Environment reset successful")
    
    # Test single step
    action = np.random.uniform(0, 1, size=4)  # 4 muscles
    time_step = env.step(action)
    
    # Print task state
    hand_pos = env.physics.named.data.geom_xpos["hand"]
    target_pos = env.physics.named.data.mocap_pos["target"]
    print(f"Hand position: {hand_pos}")
    print(f"Target position: {target_pos}")
    print(f"Reward: {time_step.reward}")
    
    # Render frame
    frame = env.physics.render()
    PIL.Image.fromarray(frame).save("test_frame.png")
    print("Test frame saved as test_frame.png")
    
    return task, env

def visualize_test_results(env):
    """Visualize a test simulation"""
    frames, rewards, observations, ticks = run_simulation(env, duration=2.0)
    
    # Save a frame
    PIL.Image.fromarray(frames[0]).save("test_frame.png")
    print("Test frame saved as test_frame.png")
    
    # Plot reward
    plt.figure()
    plt.plot(ticks, rewards)
    plt.title("Test Simulation Reward")
    plt.xlabel("Time")
    plt.ylabel("Reward")
    plt.savefig("test_reward.png")
    print("Test reward plot saved as test_reward.png")

if __name__ == "__main__":
    # Run tests
    arm_entity = test_arm_entity()
    observables = test_observables(arm_entity)
    task, env = test_reach_task(arm_entity)
    
    # Run visualization test
    visualize_test_results(env)