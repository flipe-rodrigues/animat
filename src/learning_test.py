'''import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

# Define the network with N random neurons
class RandomNeuralNetwork(nn.Module):
    def __init__(self, N=32):
        super(RandomNeuralNetwork, self).__init__()

        input_size = 14  # 2 (target) + 12 (sensor)
        output_size = 4   # 4 muscle activations

        # Random hidden layer weights (N neurons)
        self.hidden_layer = nn.Linear(input_size, N)
        self.output_layer = nn.Linear(N, output_size)

        # Initialize weights randomly
        with torch.no_grad():
            self.hidden_layer.weight = nn.Parameter(torch.randn(N, input_size))
            self.output_layer.weight = nn.Parameter(torch.randn(output_size, N))

    def forward(self, x):
        x = torch.relu(self.hidden_layer(x))  # Random transformation
        x = torch.sigmoid(self.output_layer(x))  # Output between 0 and 1
        return x
    
    # Create the network
N = 32  # Number of hidden neurons
net = RandomNeuralNetwork(N)

# Optimizer and loss function
optimizer = optim.Adam(net.parameters(), lr=0.01)
loss_function = nn.MSELoss()  # Mean Squared Error for regression

# Dummy training data (replace with real data from simulation)
num_samples = 1000
X_train = torch.randn(num_samples, 14)  # 14D input (target + sensor)
y_train = torch.rand(num_samples, 4)    # 4D output (muscle activation)

# Training loop
epochs = 50
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = net(X_train)
    loss = loss_function(outputs, y_train)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

import torch
import torch.nn as nn
import torch.optim as optim

# Define the neural network with N random neurons
class RandomNeuralNetwork(nn.Module):
    def __init__(self, N=16):  # Start with 16 neurons for simplicity
        super(RandomNeuralNetwork, self).__init__()

        input_size = 14  # 2 (target) + 12 (sensor)
        output_size = 4   # 4 muscle activations

        # Hidden layer with N neurons
        self.hidden_layer = nn.Linear(input_size, N)
        self.output_layer = nn.Linear(N, output_size)

        # Initialize random weights
        with torch.no_grad():
            self.hidden_layer.weight = nn.Parameter(torch.randn(N, input_size))
            self.output_layer.weight = nn.Parameter(torch.randn(output_size, N))

    def forward(self, x):
        x = torch.relu(self.hidden_layer(x))  # Hidden layer with activation
        x = torch.sigmoid(self.output_layer(x))  # Output in range [0, 1]
        return x

# Create a small network with 16 neurons
net = RandomNeuralNetwork(N=16)

# Example input: (random target position + random sensor values)
example_input = torch.randn(1, 14)  # 1 sample, 14 features
output = net(example_input)

print("Neural Network Output:", output)'''

import gym
import numpy as np
import torch
import mujoco
from gym import spaces

class ArmEnv(gym.Env):
    def __init__(self):
        super(ArmEnv, self).__init__()

        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path("../mujuco/arm_model.xml")
        self.data = mujoco.MjData(self.model)

        # Define state space (target position + sensor data)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(14,), dtype=np.float32)

        # Define action space (muscle activations)
        self.action_space = spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)

        # Random target position in a 1x1 square around the origin
        self.target = np.random.uniform(-0.5, 0.5, size=(2,))

        # Randomize initial sensor values
        sensor_data = np.random.uniform(-1, 1, size=(12,))
        return np.concatenate((self.target, sensor_data))

    def step(self, action):
        # Apply muscle activations
        self.data.ctrl[:] = action

        # Step the simulation
        mujoco.mj_step(self.model, self.data)

        # Compute new sensor values
        sensor_data = np.array([
            self.data.sensordata[0], self.data.sensordata[1], self.data.sensordata[2],  # Deltoid
            self.data.sensordata[3], self.data.sensordata[4], self.data.sensordata[5],  # Latissimus
            self.data.sensordata[6], self.data.sensordata[7], self.data.sensordata[8],  # Biceps
            self.data.sensordata[9], self.data.sensordata[10], self.data.sensordata[11] # Triceps
        ])

        # Get hand position (end-effector)
        hand_pos = self.data.geom_xpos[self.model.geom_name2id("hand")][:2]  # Only x, y

        # Compute reward (negative distance to target)
        reward = -np.linalg.norm(hand_pos - self.target)

        # Check if the hand is close enough to the target
        done = np.linalg.norm(hand_pos - self.target) < 0.05

        return np.concatenate((self.target, sensor_data)), reward, done, {}

    def render(self):
        pass  # Use a separate MuJoCo viewer if needed



from stable_baselines3 import PPO

# Create environment
env = ArmEnv()

# Train using PPO (Proximal Policy Optimization)
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)

# Save the trained model
model.save("arm_controller")

from stable_baselines3 import PPO

# Load trained model
model = PPO.load("arm_controller")

while viewer.is_running():
    mujoco.mj_step(model, data)

    # Random target position
    target_pos = np.random.uniform(-0.5, 0.5, size=(2,))

    # Get sensor data from MuJoCo
    sensor_data = np.array([
        data.sensordata[0], data.sensordata[1], data.sensordata[2],
        data.sensordata[3], data.sensordata[4], data.sensordata[5],
        data.sensordata[6], data.sensordata[7], data.sensordata[8],
        data.sensordata[9], data.sensordata[10], data.sensordata[11]
    ])

    # Predict action
    input_data = np.concatenate((target_pos, sensor_data))
    action, _ = model.predict(input_data)

    # Apply muscle activations
    data.ctrl[:] = action

    viewer.sync()



