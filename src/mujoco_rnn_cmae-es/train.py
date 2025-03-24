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
import time
import torch
import torch.nn as nn
import cma
import pickle
import mujoco
import mujoco.viewer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces

# %%
# Define utility functions
"""
.##.....##.########.####.##.......####.########.##....##
.##.....##....##.....##..##........##.....##.....##..##.
.##.....##....##.....##..##........##.....##......####..
.##.....##....##.....##..##........##.....##.......##...
.##.....##....##.....##..##........##.....##.......##...
.##.....##....##.....##..##........##.....##.......##...
..#######.....##....####.########.####....##.......##...
"""


def get_root_path():
    root_path = os.path.abspath(os.path.dirname(__file__))
    while root_path != os.path.dirname(root_path):
        if os.path.exists(os.path.join(root_path, ".git")):
            break
        root_path = os.path.dirname(root_path)
    return root_path


def zscore(x, xmean, xstd, default=0):
    valid = xstd > 0
    xnorm = np.full_like(x, default)
    xnorm[valid] = (x[valid] - xmean[valid]) / xstd[valid]
    return xnorm


def l1_norm(x):
    return np.sum(np.abs(x))


def l2_norm(x):
    return np.sqrt(np.sum(x**2))


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

    def __init__(self, input_size=15, hidden_size=25, output_size=4):
        super(RNNController, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(
            input_size, hidden_size, batch_first=True, nonlinearity="tanh"
        )
        self.fc = nn.Linear(hidden_size, output_size)
        self.init_weights()

    def init_weights(self):
        for name, param in self.rnn.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(self, input, hidden):
        output, hidden = self.rnn.forward(input, hidden)
        output = self.fc(output)
        return torch.sigmoid(output), hidden

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
        xml_file="your_model.xml",
        max_num_targets=10,
        max_target_duration=3,
    ):
        super().__init__()

        mj_dir = os.path.join(get_root_path(), "mujoco")
        xml_path = os.path.join(mj_dir, xml_file)
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.max_num_targets = max_num_targets
        self.max_target_duration = max_target_duration
        self.max_trial_duration = max_num_targets * max_target_duration
        self.viewer = None
        self.logger = None

        # Get the site ID using the name of your end effector
        self.hand_id = self.model.geom("hand").id

        # Load sensor stats
        sensor_stats_path = os.path.join(mj_dir, "sensor_stats.pkl")
        with open(sensor_stats_path, "rb") as f:
            self.sensor_stats = pickle.load(f)

        # Load target stats
        target_stats_path = os.path.join(mj_dir, "target_stats.pkl")
        with open(target_stats_path, "rb") as f:
            self.target_stats = pickle.load(f)

        # Define the lower and upper bounds for each feature (15 features)
        low_values = np.concatenate(
            [
                self.sensor_stats["min"].values,
                self.target_stats["min"].values,
            ]
        )
        high_values = np.concatenate(
            [
                self.sensor_stats["max"].values,
                self.target_stats["max"].values,
            ]
        )

        # Observation space: 12 sensor readings + 3D target position
        self.observation_space = spaces.Box(
            low=low_values, high=high_values, dtype=np.float64
        )

        # not sure this matters in the end..
        # be sure to convince yourself that it does..
        # the hand position bug may have been the reason why this wasn't going anywhere..
        self.observation_space = spaces.Box(low=-3, high=3, dtype=np.float64)

        # Action space: 4 muscle activations
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float64)

        # Load valid target positions
        reachable_positions_path = os.path.join(mj_dir, "reachable_positions.pkl")
        with open(reachable_positions_path, "rb") as f:
            self.reachable_positions = pickle.load(f)

    def sample_targets(self, num_samples=10):
        return self.reachable_positions.sample(num_samples).values

    def update_target(self, position):
        self.data.mocap_pos = position
        mujoco.mj_forward(self.model, self.data)

    def get_obs(self):
        target_position = self.data.mocap_pos[0].copy()
        sensor_data = self.data.sensordata.copy()
        norm_target_position = zscore(
            target_position,
            self.target_stats["mean"].values,
            self.target_stats["std"].values,
        )
        norm_sensor_data = zscore(
            sensor_data,
            self.sensor_stats["mean"].values,
            self.sensor_stats["std"].values,
        )
        return norm_target_position, norm_sensor_data
        # return target_position, sensor_data

    def get_hand_pos(self):
        return self.data.geom_xpos[self.hand_id].copy()

    def step(self, action):
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)

        context, feedback = self.get_obs()
        hand_position = self.get_hand_pos()
        target_position = self.target_positions[self.target_idx]
        manhattan_distance = l1_norm(target_position - hand_position)
        euclidean_distance = l2_norm(target_position - hand_position)
        energy = np.mean(action)
        reward = -(euclidean_distance + manhattan_distance + energy) / 3

        self.log_data(
            time=self.data.time,
            sensors=feedback,
            target=target_position,
            manhattan_distance=manhattan_distance,
            euclidean_distance=euclidean_distance,
            energy=energy,
            reward=reward,
            fitness=0,
        )

        done = self.data.time > self.max_target_duration * self.max_num_targets
        if self.data.time > self.max_target_duration * (self.target_idx + 1):
            if self.target_idx < self.max_num_targets - 1:
                self.target_idx += 1
                self.update_target(self.target_positions[self.target_idx])
            else:
                done = True

        obs = np.concatenate([context, feedback])
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
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
            self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = True
            self.viewer.cam.lookat[:] = [0, -1.5, -0.5]
            self.viewer.cam.azimuth = 90
            self.viewer.cam.elevation = 0
        else:
            self.viewer.sync()
            time.sleep(self.model.opt.timestep)

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def log_data(
        self,
        time,
        sensors,
        target,
        manhattan_distance,
        euclidean_distance,
        energy,
        reward,
        fitness,
    ):
        if self.logger is None:
            self.logger = {}
            self.logger["time"] = []
            self.logger["sensors"] = {
                "deltoid_len": [],
                "latissimus_len": [],
                "biceps_len": [],
                "triceps_len": [],
                "deltoid_vel": [],
                "latissimus_vel": [],
                "biceps_vel": [],
                "triceps_vel": [],
                "deltoid_frc": [],
                "latissimus_frc": [],
                "biceps_frc": [],
                "triceps_frc": [],
            }
            self.logger["target"] = []
            self.logger["manhattan_distance"] = []
            self.logger["euclidean_distance"] = []
            self.logger["energy"] = []
            self.logger["reward"] = []
            self.logger["fitness"] = []

        self.logger["time"].append(time)
        for i, key in enumerate(self.logger["sensors"].keys()):
            self.logger["sensors"][key].append(sensors[i])
        self.logger["target"].append(target)
        self.logger["manhattan_distance"].append(manhattan_distance)
        self.logger["euclidean_distance"].append(euclidean_distance)
        self.logger["energy"].append(energy)
        self.logger["reward"].append(reward)
        self.logger["fitness"].append(fitness)

    def plot(self):
        _, axes = plt.subplots(2, 2)

        # Length
        axes[0, 0].plot(
            self.logger["time"], self.logger["sensors"]["deltoid_len"], label="Deltoid"
        )
        axes[0, 0].plot(
            self.logger["time"],
            self.logger["sensors"]["latissimus_len"],
            label="Latissimus",
        )
        axes[0, 0].plot(
            self.logger["time"], self.logger["sensors"]["biceps_len"], label="Biceps"
        )
        axes[0, 0].plot(
            self.logger["time"], self.logger["sensors"]["triceps_len"], label="Triceps"
        )
        axes[0, 0].set_title("Length")

        # Velocity
        axes[0, 1].plot(
            self.logger["time"], self.logger["sensors"]["deltoid_vel"], label="Deltoid"
        )
        axes[0, 1].plot(
            self.logger["time"],
            self.logger["sensors"]["latissimus_vel"],
            label="Latissimus",
        )
        axes[0, 1].plot(
            self.logger["time"], self.logger["sensors"]["biceps_vel"], label="Biceps"
        )
        axes[0, 1].plot(
            self.logger["time"], self.logger["sensors"]["triceps_vel"], label="Triceps"
        )
        axes[0, 1].set_title("Velocity")
        axes[0, 1].legend(loc="center left", bbox_to_anchor=(1, 0.5))

        # Force
        axes[1, 0].plot(
            self.logger["time"], self.logger["sensors"]["deltoid_frc"], label="Deltoid"
        )
        axes[1, 0].plot(
            self.logger["time"],
            self.logger["sensors"]["latissimus_frc"],
            label="Latissimus",
        )
        axes[1, 0].plot(
            self.logger["time"], self.logger["sensors"]["biceps_frc"], label="Biceps"
        )
        axes[1, 0].plot(
            self.logger["time"], self.logger["sensors"]["triceps_frc"], label="Triceps"
        )
        axes[1, 0].set_title("Force")

        # Fitness
        axes[1, 1].plot(
            self.logger["time"],
            self.logger["manhattan_distance"],
            label="Manhattan distance",
        )
        axes[1, 1].plot(
            self.logger["time"],
            self.logger["euclidean_distance"],
            label="Euclidean distance",
        )
        axes[1, 1].plot(self.logger["time"], self.logger["reward"], label="Reward")
        axes[1, 1].plot(self.logger["time"], self.logger["energy"], label="Energy")
        axes[1, 1].set_title("Fitness")
        axes[1, 1].set_ylim([-1.05, 1.05])
        axes[1, 1].legend(loc="center left", bbox_to_anchor=(1, 0.5))

        # Create a twin axis (right y-axis)
        r, g, b = np.array([1, 1, 1]) * 0.25
        fitness_clr = (r, g, b)
        ax_right = axes[1, 1].twinx()
        fitness = np.cumsum(self.logger["reward"]) / self.max_trial_duration
        ax_right.plot(self.logger["time"], fitness, color=fitness_clr)
        ax_right.set_ylabel("Cumulative Reward", color=fitness_clr)
        ax_right.tick_params(axis="y", labelcolor=fitness_clr)

        # Set axis labels
        for ax in axes.flat:
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Arb.")

        plt.tight_layout()
        plt.show()

        self.logger = None


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


def evaluate(params, seed=None, render=False, plot=False):
    """Runs an episode using the given RNN parameters and returns the negative distance."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = SequentialReachingEnv(
        xml_file="arm_model.xml",
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
        if render:
            env.render()

        obs_tensor = torch.tensor(obs, dtype=torch.float64).unsqueeze(0).unsqueeze(0)
        action, hidden = rnn(obs_tensor, hidden)
        action = action.squeeze().detach().numpy()

        obs, reward, done = env.step(action)
        total_reward += reward

        if done:
            break

    env.close()

    if plot:
        env.plot()

    return -total_reward / env.max_trial_duration  # CMA-ES minimizes, so negate reward


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
rnn = RNNController()
es = cma.CMAEvolutionStrategy(x0=rnn.get_params(), sigma0=0.5)

while not es.stop():
    solutions = es.ask()
    # rewards = [evaluate(sol, seed=0, render=True, plot=True) for sol in solutions]
    # rewards = [evaluate(sol, seed=0) for sol in solutions]
    rewards = [evaluate(sol, seed=es.countiter) for sol in solutions]
    es.tell(solutions, rewards)
    es.disp()
    es.logger.add()

    plot = es.countiter % 10 == 0
    render = es.countiter % 100 == 0

    if es.countiter % 10 == 0:
        evaluate(es.result.xbest, seed=0, render=False, plot=True)
    if es.countiter % 100 == 0:
        with open(f"outcmaes/xbest_{es.countiter}.pkl", "wb") as f:
            pickle.dump(es, f)

with open("outcmaes/xbest_converged.pkl", "wb") as f:
    pickle.dump(es, f)

# %%
# Plot the latest xbest pickle file from outcmaes
"""
.########..##........#######..########
.##.....##.##.......##.....##....##...
.##.....##.##.......##.....##....##...
.########..##.......##.....##....##...
.##........##.......##.....##....##...
.##........##.......##.....##....##...
.##........########..#######.....##...
"""

xbest_path = os.path.join("outcmaes", "xbest_340.pkl")
with open(xbest_path, "rb") as f:
    xbest = pickle.load(f)

evaluate(xbest, seed=0, render=True, plot=True)

# %%

# Plot the loss over time
plt.figure()
plt.plot(es.logger.f, label="Loss")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Loss over Time")
plt.legend()
plt.show()
