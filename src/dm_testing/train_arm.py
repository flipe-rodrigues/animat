import numpy as np
from dm_testing.arm_env import load
from dm_testing.dm_control_test import display_video
import matplotlib.pyplot as plt
from collections import deque
import os

# Create environment
env = load()

# Print specs to understand the interface
print("Action spec:", env.action_spec())
print("Observation spec:", env.observation_spec())

# Training parameters
num_episodes = 1  # Just one episode to test the visualization
max_steps_per_episode = 500
frames = []  # Store frames for video

# Run one episode
time_step = env.reset()
episode_reward = 0

for step in range(max_steps_per_episode):
    # Random action
    action = np.random.uniform(
        env.action_spec().minimum,
        env.action_spec().maximum,
        size=env.action_spec().shape
    )
    
    # Step the environment
    time_step = env.step(action)
    episode_reward += time_step.reward
    
    # Capture frame
    frames.append(env.physics.render())
    
    if time_step.last():
        break

print(f"Episode finished with reward: {episode_reward:.2f}")

# Save video using the existing display_video function
display_video(frames, framerate=30)