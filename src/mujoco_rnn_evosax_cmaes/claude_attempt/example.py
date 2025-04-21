"""
Example script showing how to run the arm reaching task.
"""
import os
import argparse
import jax
import jax.numpy as jnp
import pickle
import time
import matplotlib.pyplot as plt
import numpy as np

from environment import ArmReachingEnv
from rnn_model import SimpleRNN
from evolutionary_trainer import EvolutionaryTrainer

def run_example():
    """
    Run a basic example of the arm reaching task.
    """
    # Model path - now correctly configured with all attachment sites
    def get_root_path():
        root_path = os.path.abspath(os.path.dirname(__file__))
        while root_path != os.path.dirname(root_path):
            if os.path.exists(os.path.join(root_path, ".git")):
                break
            root_path = os.path.dirname(root_path)
        return root_path
    mj_dir = os.path.join(get_root_path(), "mujoco")
    model_path = os.path.join(mj_dir, "arm_model.xml")
    
    # Set random seed - compatible with JAX 0.5.2
    seed = 42
    key = jax.random.PRNGKey(seed)
    
    # Create environment with rendering
    try:
        env = ArmReachingEnv(model_path=model_path, episode_length=200, render=True)
    except Exception as e:
        print(f"Warning: Could not initialize environment with rendering: {e}")
        print("Trying without rendering...")
        env = ArmReachingEnv(model_path=model_path, episode_length=200, render=False)
    
    # Create RNN
    hidden_size = 32
    rnn = SimpleRNN(
        input_size=env.input_dim,
        hidden_size=hidden_size,
        output_size=env.output_dim
    )
    
    # Initialize RNN parameters randomly
    key, subkey = jax.random.split(key)
    params = rnn.init_params(subkey)
    
    # Training setup
    popsize = 16  # Small population for quick example
    n_targets = 3  # Small number of targets for quick example
    steps_per_target = 100
    
    # Create trainer with smaller population for quick example
    trainer = EvolutionaryTrainer(
        env=env,
        popsize=popsize,
        hidden_size=hidden_size,
        n_targets=n_targets,
        steps_per_target=steps_per_target,
        save_path="models",
        seed=seed
    )
    
    # Train for a few generations
    print("Training for 5 generations (quick example)...")
    results = trainer.train(n_generations=5, log_interval=1)
    
    # Load the best model
    best_params = rnn.unflatten_params(results['best_params'])
    
    # Test the trained model
    print("\nTesting the trained model...")
    
    # Initialize hidden state
    h_state = rnn.init_hidden()
    
    # Generate a target - compatible with JAX 0.5.2
    key, subkey = jax.random.split(key)
    observation = env.reset(subkey)
    
    # Store data for plotting
    distances = []
    positions = []
    
    # Run for a while
    for step in range(200):
        # Get action from policy
        action, h_state = rnn.predict(best_params, observation, h_state)
        
        # Step the environment
        next_observation, reward, done, info = env.step(action)
        
        # Store data
        distances.append(info["distance"])
        try:
            # Convert to numpy to ensure compatibility
            positions.append(np.array(info["end_effector_pos"], dtype=np.float64))
        except (TypeError, ValueError) as e:
            print(f"Warning: Could not store position: {e}")
            positions.append(np.zeros(3))
        
        # Update observation
        observation = next_observation
        
        # Sleep to slow down rendering
        time.sleep(0.01)
    
    # Plot the distance over time
    plt.figure(figsize=(10, 6))
    plt.plot(distances)
    plt.title("Distance to Target over Time")
    plt.xlabel("Step")
    plt.ylabel("Distance")
    plt.grid(True)
    plt.savefig("distance_plot.png")
    plt.close()
    
    # Plot the trajectory
    try:
        positions = np.array(positions)
        plt.figure(figsize=(10, 10))
        plt.plot(positions[:, 0], positions[:, 2], 'b-', linewidth=2)
        
        # Get target from the environment
        target_pos = np.array(env.current_target)
        plt.scatter([target_pos[0]], [target_pos[2]], color='purple', s=100, label='Target')
        
        plt.scatter([positions[0, 0]], [positions[0, 2]], color='green', s=100, label='Start')
        plt.scatter([positions[-1, 0]], [positions[-1, 2]], color='red', s=100, label='End')
        plt.title("End Effector Trajectory (X-Z Plane)")
        plt.xlabel("X Position")
        plt.ylabel("Z Position")
        plt.grid(True)
        plt.legend()
        plt.savefig("trajectory_plot.png")
    except Exception as e:
        print(f"Warning: Could not plot trajectory: {e}")
    finally:
        plt.close()
    
    print("Example completed! Plots saved to distance_plot.png and trajectory_plot.png")
    
    # Clean up
    env.close()

if __name__ == "__main__":
    run_example()