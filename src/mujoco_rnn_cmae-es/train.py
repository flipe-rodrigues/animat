# %%
# Import the required libraries
"""
.####.##.....##.########...#######..########..########..######.
..##..###...###.##.....##.##.....##.##.....##....##....##....##
..##..####.####.##.....##.##.....##.##.....##....##....##......
..##..##.###.##.########..##.....##.########.....##.....######.
..##..##.....##.##........##.....##.##...##......##..........##
..##..##.....##.##........##.....##.##....##.....##....##....##
.####.##.....##.##.........#######..##.....##....##.....######.
"""
import os
import torch
import torch.nn as nn
import cma
import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces

# %%
# Define the RNN controller
"""
.########..##....##.##....##
.##.....##.###...##.###...##
.##.....##.####..##.####..##
.########..##.##.##.##.##.##
.##...##...##..####.##..####
.##....##..##...###.##...###
.##.....##.##....##.##....##
"""


class RNNController(nn.Module):
    """RNN Controller for 2-joint limb with 4 muscles and 12 sensors + target pos"""

    def __init__(self, input_size=15, hidden_size=15, output_size=4):
        super(RNNController, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(
            input_size, hidden_size, batch_first=True, nonlinearity="tanh"
        )
        self.fc = nn.Linear(hidden_size, output_size)
        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(self, x, hidden):
        x[-4:] /= 100
        out, hidden = self.rnn.forward(x, hidden)
        out = self.fc(out)
        return torch.sigmoid(out), hidden

    def get_params(self):
        params = np.concatenate(
            [p.detach().cpu().numpy().ravel() for p in self.parameters()]
        )
        return params

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def set_params(self, params):
        start = 0
        for p in self.parameters():
            end = start + p.numel()
            p.data = torch.tensor(params[start:end], dtype=torch.float64).view(p.shape)
            start = end


# %%
# Define the environment
"""
.########.##....##.##.....##.####.########...#######..##....##.##.....##.########.##....##.########
.##.......###...##.##.....##..##..##.....##.##.....##.###...##.###...###.##.......###...##....##...
.##.......####..##.##.....##..##..##.....##.##.....##.####..##.####.####.##.......####..##....##...
.######...##.##.##.##.....##..##..########..##.....##.##.##.##.##.###.##.######...##.##.##....##...
.##.......##..####..##...##...##..##...##...##.....##.##..####.##.....##.##.......##..####....##...
.##.......##...###...##.##....##..##....##..##.....##.##...###.##.....##.##.......##...###....##...
.########.##....##....###....####.##.....##..#######..##....##.##.....##.########.##....##....##...
"""


class SequentialReachingEnv(gym.Env):
    """Custom 2-Joint Limb with 4 Muscles, 12 Sensors, and a Target Position"""

    def __init__(
        self,
        xml_path="path/to/your_model.xml",
        max_num_targets=10,
        max_target_duration=3,
    ):
        super().__init__()

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.max_num_targets = max_num_targets
        self.max_target_duration = max_target_duration
        self.viewer = None

        # Observation space: 12 sensor readings + 3D target position
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(15,), dtype=np.float64
        )

        # Action space: 4 muscle activations
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float64)

        # Get the site ID using the name of your end effector
        self.hand_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "hand")

        # Parse target positions from CSV file
        self.reachable_positions = self.parse_targets("../../src/targets.csv")

    def parse_targets(self, targets_path="path/to/targets.csv", bins=100):
        target_positions = np.loadtxt(
            targets_path, delimiter=",", skiprows=1, usecols=(0, 2)
        )
        x, y = target_positions[:, 0], target_positions[:, 1]
        counts2d, x_edges, y_edges = np.histogram2d(x, y, bins=bins)
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2
        nonzero_indices = np.argwhere(counts2d > 0)
        return [(x_centers[i], 0, y_centers[j]) for i, j in nonzero_indices]

    def sample_targets(self, num_samples=10):
        sampled_positions = np.random.choice(
            len(self.reachable_positions), num_samples, replace=False
        )
        return [self.reachable_positions[i] for i in sampled_positions]

    def update_target(self, position):
        self.data.mocap_pos = position
        mujoco.mj_forward(self.model, self.data)

    def step(self, action):
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)

        sensor_data = self.data.sensordata.copy()
        hand_position = self.data.site_xpos[self.hand_id]
        distance = np.linalg.norm(
            hand_position - self.target_positions[self.target_idx]
        )
        reward = -distance

        done = self.data.time > self.max_target_duration * self.max_num_targets
        if distance < 0.05 or self.data.time > self.max_target_duration * (
            self.target_idx + 1
        ):
            if self.target_idx < self.max_num_targets - 1:
                self.target_idx += 1
                self.update_target(self.target_positions[self.target_idx])
            else:
                done = True

        obs = np.concatenate([self.target_positions[self.target_idx], sensor_data])
        return obs, reward, done

    def reset(self, seed=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        self.target_positions = self.sample_targets(self.max_num_targets)
        self.target_idx = 0
        self.update_target(self.target_positions[self.target_idx])

        sensor_data = self.data.sensordata.copy()
        obs = np.concatenate([self.target_positions[self.target_idx], sensor_data])
        return obs, {}

    def render(self):
        if self.viewer is not None:
            self.viewer.sync()
        else:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
            self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = True
            self.viewer.cam.lookat[:] = [0, -1.5, -0.5]
            self.viewer.cam.azimuth = 90
            self.viewer.cam.elevation = 0

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


# %%
# Define the evaluation function
"""
.########.##.....##....###....##.......##.....##....###....########.########
.##.......##.....##...##.##...##.......##.....##...##.##......##....##......
.##.......##.....##..##...##..##.......##.....##..##...##.....##....##......
.######...##.....##.##.....##.##.......##.....##.##.....##....##....######..
.##........##...##..#########.##.......##.....##.#########....##....##......
.##.........##.##...##.....##.##.......##.....##.##.....##....##....##......
.########....###....##.....##.########..#######..##.....##....##....########
"""


def evaluate(params, seed=None, render=False):
    """Runs an episode using the given RNN parameters and returns the negative distance."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = SequentialReachingEnv(
        xml_path="../../mujoco/arm_model.xml",
        max_num_targets=10,
        max_target_duration=3,
    )

    rnn = RNNController()
    rnn.set_params(params)

    obs, _ = env.reset()
    hidden = torch.zeros(1, 1, rnn.hidden_size, dtype=torch.float64)
    total_reward = 0

    done = False
    while not done:

        obs_tensor = torch.tensor(obs, dtype=torch.float64).unsqueeze(0).unsqueeze(0)
        action, hidden = rnn(obs_tensor, hidden)
        action = action.squeeze().detach().numpy()

        obs, reward, done = env.step(action)
        total_reward += reward

        if render:
            env.render()

        if done:
            break

    env.close()
    return -total_reward  # CMA-ES minimizes, so negate reward


# %%
# Optimize using CMA-ES
"""
..######..##.....##....###............########..######.
.##....##.###...###...##.##...........##.......##....##
.##.......####.####..##...##..........##.......##......
.##.......##.###.##.##.....##.#######.######....######.
.##.......##.....##.#########.........##.............##
.##....##.##.....##.##.....##.........##.......##....##
..######..##.....##.##.....##.........########..######.
"""

os.chdir(os.path.dirname(__file__))

rnn = RNNController()

es = cma.CMAEvolutionStrategy(x0=rnn.get_params(), sigma0=0.5)

while not es.stop():
    solutions = es.ask()
    rewards = [evaluate(sol, seed=0) for sol in solutions]
    es.tell(solutions, rewards)
    es.disp()
    es.logger.add()

    if es.countiter % 10 == 0:
        evaluate(es.result.xbest, seed=0, render=True)

es.result_pretty()
es.logger.plot()

torch.save(es.result.xbest, "best_rnn_params.pth")


# %%
