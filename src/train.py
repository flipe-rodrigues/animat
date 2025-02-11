import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from arm_env import ArmEnv
from rnn_model import RNNController

# Hyperparameters
input_size = 10  # Adjust based on observation size
hidden_size = 128
output_size = 4  # Number of actuators
num_episodes = 1000
max_steps = 200
learning_rate = 0.001

# Initialize environment and model
env = ArmEnv("mujoco/arm_model.xml")
model = RNNController(input_size, hidden_size, output_size)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

# Training loop
for episode in range(num_episodes):
    obs = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    hidden = model.init_hidden(1, hidden_size)
    total_reward = 0

    for step in range(max_steps):
        action, hidden = model(obs, hidden)
        action = action.detach().numpy().flatten()
        obs, reward, done = env.step(action)
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        total_reward += reward

        if done:
            break

    print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")

    # Update model
    optimizer.zero_grad()
    hand_pos = env.data.geom('hand').xpos[:2]
    loss = criterion(torch.tensor(hand_pos, dtype=torch.float32), torch.tensor(env.target, dtype=torch.float32))
    loss.backward()
    optimizer.step()

    # Render the environment
    env.render()