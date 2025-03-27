import torch
import torch.nn as nn
import torch.optim as optim
import mujoco
import mujoco.viewer
import numpy as np
import time

# Define the neural network
class MuscleControlNN(nn.Module):
    def __init__(self, input_dim=14, hidden_dim=256, output_dim=4):
        super(MuscleControlNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.Tanh()
        self.output_activation = nn.Tanh()  # Outputs will be scaled from [-1,1] to [0,1]

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.output_activation(self.fc3(x))
        return (x + 1) / 2  # Scale to [0,1]

# Load MuJoCo model
XML_PATH = "mujoco/arm_model.xml"  # Ensure this file exists in the working directory
model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

# Neural network and optimizer
net = MuscleControlNN()
optimizer = optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-5)  # Add weight decay
loss_fn = nn.SmoothL1Loss()  # Huber loss

'''
# Simulation parameters
epochs = 3000  # Increased training epochs for better convergence
for epoch in range(epochs):
    mujoco.mj_step(model, data)  # Step simulation forward

    # Get sensor data
    sensors = np.array(data.sensordata)  # 12 sensor values
    target_pos = np.array([data.mocap_pos[0][0], data.mocap_pos[0][2]])  # Target 2D position
    
    # Normalize inputs separately
    sensors = (sensors - np.min(sensors)) / (np.max(sensors) - np.min(sensors) + 1e-8)
    target_pos = target_pos / np.array([1.0, 1.0])  # Assuming target is within [-1, 1] range
    
    input_data = np.concatenate([target_pos, sensors])
    input_data = torch.tensor(input_data, dtype=torch.float32)

    # Forward pass
    muscle_controls = net(input_data)
    
    # Apply muscle control
    data.ctrl[:] = muscle_controls.detach().numpy()
    
    # Compute loss (distance from hand to target)
    hand_pos = torch.tensor(
        np.array([data.xpos[model.body('hand').id][0], data.xpos[model.body('hand').id][2]]),
        dtype=torch.float32, requires_grad=True)  # Set requires_grad=True

    target_pos = torch.tensor(target_pos, dtype=torch.float32, requires_grad=True)  # Set requires_grad=True

    loss = loss_fn(hand_pos, target_pos)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

print("Training complete!")'''

# Simulation parameters
epochs = 3000  # Increased training epochs for better convergence
steps_per_epoch = 100  # Number of steps per epoch for better visualization

# Start the MuJoCo viewer with passive mode
with mujoco.viewer.launch_passive(model, data) as viewer:
    print("Simulation running...")

    # Modify visualization settings if needed
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = True
    viewer.sync()

    viewer.cam.lookat[:] = [0, 0, -.5]
    viewer.cam.azimuth = 90
    viewer.cam.elevation = 0

    for epoch in range(epochs):
        for _ in range(steps_per_epoch):
            mujoco.mj_step(model, data)  # Step the simulation forward

            # Get sensor data
            sensors = np.array(data.sensordata)  # 12 sensor values
            target_pos = np.array([data.mocap_pos[0][0], data.mocap_pos[0][2]])  # Target 2D position

            # Normalize inputs separately
            sensors = (sensors - np.min(sensors)) / (np.max(sensors) - np.min(sensors) + 1e-8)
            target_pos = target_pos / np.array([1.0, 1.0])  # Assuming target is within [-1, 1] range

            input_data = np.concatenate([target_pos, sensors])
            input_data = torch.tensor(input_data, dtype=torch.float32)

            # Forward pass
            muscle_controls = net(input_data)

            # Apply muscle control
            data.ctrl[:] = muscle_controls.detach().numpy()

            # Compute loss (distance from hand to target)
            hand_pos = torch.tensor(
                np.array([data.xpos[model.body('hand').id][0], data.xpos[model.body('hand').id][2]]), 
                dtype=torch.float32, requires_grad=True)  # Ensure hand_pos requires gradients
            target_pos = torch.tensor(target_pos, dtype=torch.float32, requires_grad=False)  # Target does not need gradients

            loss = loss_fn(hand_pos, target_pos)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update the viewer and sync it
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            step_start = time.time()
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

    print("Training complete!")