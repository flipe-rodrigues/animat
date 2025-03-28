import numpy as np
import tensorflow as tf
import sonnet as snt
import matplotlib.pyplot as plt
import reverb
import os
import acme
from acme import adders
from acme import specs
from acme.agents.tf import ddpg
from acme.tf import networks
from acme.tf import utils as tf_utils
from acme.utils import loggers
from dm_testing.arm_env import load, _DEFAULT_TIME_LIMIT
from dm_testing.dm_control_test import display_video
import imageio

# Set random seeds
tf.random.set_seed(42)
np.random.seed(42)

def make_networks(action_spec, observation_spec):
    """Creates networks used by the DDPG agent."""
    
    # Get the input shapes
    input_shape = sum([np.prod(s.shape) for s in observation_spec.values()])
    action_shape = action_spec.shape[0]
    
    # Create the critic network
    critic_network = snt.Sequential([
        networks.CriticMultiplexer(
            observation_network=snt.nets.MLP([256, 256]),
            action_network=snt.nets.MLP([256]),
            critic_network=snt.nets.MLP([256, 128, 1])
        )
    ])
    
    # Create the policy network
    policy_network = snt.Sequential([
        snt.nets.MLP([256, 256, 256, action_shape], activate_final=False),
        networks.NearZeroInitializedLinear(action_shape),
        tf.nn.sigmoid  # Output in [0, 1] for muscle activations
    ])
    
    return {
        'policy': policy_network,
        'critic': critic_network,
        'observation_network': tf.identity,
    }

def normalize_observations(timestep, obs_mean, obs_std):
    """Normalize observations for stability."""
    flat_obs = np.concatenate([
        obs.flatten() if name != 'target_position' else obs.flatten()[0:3]
        for name, obs in timestep.observation.items()
    ])
    normalized_obs = (flat_obs - obs_mean) / (obs_std + 1e-8)
    return normalized_obs

def train_ddpg(num_episodes=2000, evaluation_interval=100):
    """Train a DDPG agent on the arm environment."""
    # Load the environment
    environment = load()
    
    # Get environment specs
    environment_spec = specs.make_environment_spec(environment)
    
    # Create networks
    agent_networks = make_networks(
        action_spec=environment_spec.actions,
        observation_spec=environment_spec.observations,
    )
    
    # Configure DDPG agent
    agent = ddpg.DDPG(
        environment_spec=environment_spec,
        policy_network=agent_networks['policy'],
        critic_network=agent_networks['critic'],
        observation_network=agent_networks['observation_network'],
        # DDPG parameters
        batch_size=256,
        prefetch_size=4,
        target_update_period=100,
        min_replay_size=1000,
        max_replay_size=100000,
        # Optimizer parameters
        policy_optimizer=snt.optimizers.Adam(learning_rate=1e-4),
        critic_optimizer=snt.optimizers.Adam(learning_rate=1e-4),
        # Noise parameters for exploration
        sigma=0.3,
    )
    
    # Track progress
    returns = []
    best_return = -float('inf')
    running_avgs = []
    
    # Observation statistics for normalization
    obs_count = 0
    obs_mean = None
    obs_var = None
    
    # Begin training loop
    for episode in range(num_episodes):
        # Reset environment
        timestep = environment.reset()
        
        # Evaluate and render at specific intervals
        record_video = (episode == 0 or episode % evaluation_interval == 0 or 
                         episode == num_episodes - 1)
        frames = [] if record_video else None
        
        episode_return = 0
        steps = 0
        
        # Run episode
        while not timestep.last():
            # Select action
            action = agent.select_action(timestep.observation)
            
            # Apply action and observe next state
            timestep = environment.step(action)
            
            # Observe reward
            episode_return += timestep.reward
            steps += 1
            
            # Record frame for visualization if needed
            if frames is not None and steps % 10 == 0:
                frames.append(environment.physics.render(camera_id=-1, width=640, height=480))
            
            # Update agent with experience
            agent.update()
        
        # Record episode return
        returns.append(episode_return)
        
        # Track statistics
        if episode >= 100:
            running_avg = np.mean(returns[-100:])
            running_avgs.append(running_avg)
            
        # Check if this is a new best episode
        if episode_return > best_return:
            best_return = episode_return
            print(f"New best! Episode {episode}, Return: {best_return:.2f}")
            
            # Save best policy checkpoint
            # TODO: Add policy saving functionality
        
        # Print progress
        if episode % 10 == 0:
            avg_return = np.mean(returns[-100:]) if len(returns) >= 100 else np.mean(returns)
            print(f"Episode {episode}/{num_episodes}, Return: {episode_return:.2f}, "
                 f"Avg Return: {avg_return:.2f}, Best: {best_return:.2f}")
            
        # Save video for visualization
        if frames is not None and len(frames) > 1:
            if episode % evaluation_interval == 0:
                # Save episode animation
                filename = f"episode_{episode}_animation.gif"
                display_video(frames, filename=filename, framerate=30)
                print(f"Animation saved as {filename}")
        
    # Final evaluation with no exploration noise
    print("\nFinal Evaluation:")
    timestep = environment.reset()
    final_return = 0
    eval_frames = []
    steps = 0
    
    # Run evaluation episode
    while not timestep.last():
        # Select action without exploration noise
        action = agent.select_action(timestep.observation, evaluation=True)
        
        # Apply action and observe next state
        timestep = environment.step(action)
        
        # Observe reward
        final_return += timestep.reward
        steps += 1
        
        # Capture frame
        if steps % 5 == 0:  # Higher frame rate for final animation
            eval_frames.append(environment.physics.render(camera_id=-1, width=640, height=480))
    
    print(f"Final Evaluation Return: {final_return:.2f}")
    
    # Save final animation
    if len(eval_frames) > 1:
        display_video(eval_frames, filename="final_animation.gif", framerate=30)
        print("Final animation saved as final_animation.gif")
    
    # Plot training curve
    plt.figure(figsize=(10, 6))
    plt.plot(returns, alpha=0.3, label='Episode Returns')
    if len(running_avgs) > 0:
        plt.plot(range(99, 99 + len(running_avgs)), running_avgs, label='100-Episode Moving Average')
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.title('DDPG Training Progress')
    plt.legend()
    plt.grid(True)
    plt.savefig('ddpg_training_progress.png')
    plt.close()
    
    print("Training completed. Progress plot saved as 'ddpg_training_progress.png'")
    
    return agent, returns

if __name__ == "__main__":
    # Install Acme if not already installed
    try:
        import acme
    except ImportError:
        print("Acme not found. Installing required packages...")
        os.system("pip install dm-acme[reverb,tf]")
        os.system("pip install dm-sonnet")
        print("Packages installed. Please restart and run this script again.")
        exit()
    
    # Train the agent
    agent, returns = train_ddpg(num_episodes=2000)
    print(f"Time limit for the environment: {_DEFAULT_TIME_LIMIT} seconds")