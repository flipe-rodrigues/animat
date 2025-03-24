"""
.####.##.....##.########...#######..########..########..######.
..##..###...###.##.....##.##.....##.##.....##....##....##....##
..##..####.####.##.....##.##.....##.##.....##....##....##......
..##..##.###.##.########..##.....##.########.....##.....######.
..##..##.....##.##........##.....##.##...##......##..........##
..##..##.....##.##........##.....##.##....##.....##....##....##
.####.##.....##.##.........#######..##.....##....##.....######.
"""

import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
from plants import SequentialReacher
from networks import RNN
from utils import *


"""
.####.##....##.####.########
..##..###...##..##.....##...
..##..####..##..##.....##...
..##..##.##.##..##.....##...
..##..##..####..##.....##...
..##..##...###..##.....##...
.####.##....##.####....##...
"""


class SequentialReachingEnv:
    def __init__(
        self,
        plant,
        target_duration,
        num_targets,
    ):
        self.plant = plant
        self.target_duration = target_duration
        self.num_targets = num_targets
        self.logger = None

    """
    .##........#######...######..
    .##.......##.....##.##....##.
    .##.......##.....##.##.......
    .##.......##.....##.##...####
    .##.......##.....##.##....##.
    .##.......##.....##.##....##.
    .########..#######...######..
    """

    def log(
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

    """
    .########.##.....##....###....##.......##.....##....###....########.########
    .##.......##.....##...##.##...##.......##.....##...##.##......##....##......
    .##.......##.....##..##...##..##.......##.....##..##...##.....##....##......
    .######...##.....##.##.....##.##.......##.....##.##.....##....##....######..
    .##........##...##..#########.##.......##.....##.#########....##....##......
    .##.........##.##...##.....##.##.......##.....##.##.....##....##....##......
    .########....###....##.....##.########..#######..##.....##....##....########
    """

    def evaluate(self, rnn, seed=0, render=False, log=False):
        """Evaluate fitness of a given RNN policy"""
        np.random.seed(seed)

        rnn.init_state()
        self.plant.reset()

        target_positions = self.plant.sample_targets(self.num_targets)
        target_durations = truncated_exponential(
            mu=self.target_duration[0],
            a=self.target_duration[1],
            b=self.target_duration[2],
            size=self.num_targets,
        )
        target_offset_times = target_durations.cumsum()
        trial_duration = target_durations.sum()
        total_reward = 0

        target_idx = 0
        self.plant.update_target(target_positions[target_idx])

        while target_idx < self.num_targets:
            if render:
                self.plant.render()

            context, feedback = self.plant.get_obs()
            obs = np.concatenate([context, feedback])
            muscle_activations = rnn.step(obs)
            self.plant.step(muscle_activations)
            hand_position = self.plant.get_hand_pos()
            target_position = target_positions[target_idx]
            manhattan_distance = l1_norm(target_position - hand_position)
            euclidean_distance = l2_norm(target_position - hand_position)
            energy = np.mean(muscle_activations)
            reward = -(euclidean_distance + manhattan_distance + energy) / 3
            total_reward += reward

            if log:
                self.log(
                    time=self.plant.data.time,
                    sensors=feedback,
                    target=target_position,
                    manhattan_distance=manhattan_distance,
                    euclidean_distance=euclidean_distance,
                    energy=energy,
                    reward=reward,
                    fitness=total_reward / trial_duration,
                )

            if self.plant.data.time > target_offset_times[target_idx]:
                target_idx += 1
                if target_idx < self.num_targets:
                    self.plant.update_target(target_positions[target_idx])

        self.plant.close()

        return total_reward / trial_duration

    """
    .########..##........#######..########
    .##.....##.##.......##.....##....##...
    .##.....##.##.......##.....##....##...
    .########..##.......##.....##....##...
    .##........##.......##.....##....##...
    .##........##.......##.....##....##...
    .##........########..#######.....##...
    """

    def plot(self):
        log = self.logger

        _, axes = plt.subplots(2, 2)

        # Length
        axes[0, 0].plot(log["time"], log["sensors"]["deltoid_len"], label="Deltoid")
        axes[0, 0].plot(
            log["time"],
            log["sensors"]["latissimus_len"],
            label="Latissimus",
        )
        axes[0, 0].plot(log["time"], log["sensors"]["biceps_len"], label="Biceps")
        axes[0, 0].plot(log["time"], log["sensors"]["triceps_len"], label="Triceps")
        axes[0, 0].set_title("Length")

        # Velocity
        axes[0, 1].plot(log["time"], log["sensors"]["deltoid_vel"], label="Deltoid")
        axes[0, 1].plot(
            log["time"],
            log["sensors"]["latissimus_vel"],
            label="Latissimus",
        )
        axes[0, 1].plot(log["time"], log["sensors"]["biceps_vel"], label="Biceps")
        axes[0, 1].plot(log["time"], log["sensors"]["triceps_vel"], label="Triceps")
        axes[0, 1].set_title("Velocity")
        axes[0, 1].legend(loc="center left", bbox_to_anchor=(1, 0.5))

        # Force
        axes[1, 0].plot(log["time"], log["sensors"]["deltoid_frc"], label="Deltoid")
        axes[1, 0].plot(
            log["time"], log["sensors"]["latissimus_frc"], label="Latissimus"
        )
        axes[1, 0].plot(log["time"], log["sensors"]["biceps_frc"], label="Biceps")
        axes[1, 0].plot(log["time"], log["sensors"]["triceps_frc"], label="Triceps")
        axes[1, 0].set_title("Force")

        # Fitness
        axes[1, 1].plot(
            log["time"], log["manhattan_distance"], label="Manhattan distance"
        )
        axes[1, 1].plot(
            log["time"], log["euclidean_distance"], label="Euclidean distance"
        )
        axes[1, 1].plot(log["time"], log["reward"], label="Reward")
        axes[1, 1].plot(log["time"], log["energy"], label="Energy")
        axes[1, 1].set_title("Fitness")
        axes[1, 1].set_ylim([-1.05, 1.05])
        axes[0, 1].legend(loc="center left", bbox_to_anchor=(1, 0.5))

        # Create a twin axis (right y-axis)
        r, g, b = np.array([1, 1, 1]) * 0.25
        fitness_clr = (r, g, b)
        ax_right = axes[1, 1].twinx()
        ax_right.plot(log["time"], log["fitness"], color=fitness_clr)
        ax_right.set_ylabel("Cumulative Reward", color=fitness_clr)
        ax_right.tick_params(axis="y", labelcolor=fitness_clr)

        # Set axis labels
        for ax in axes.flat:
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Arb.")

        plt.tight_layout()
        plt.show()

        self.logger = None
