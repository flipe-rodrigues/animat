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
        loss_weights,
    ):
        self.plant = plant
        self.target_duration = target_duration
        self.num_targets = num_targets
        self.loss_weights = loss_weights
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
        entropy,
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
            self.logger["entropy"] = []
            self.logger["reward"] = []
            self.logger["fitness"] = []

        self.logger["time"].append(time)
        for i, key in enumerate(self.logger["sensors"].keys()):
            self.logger["sensors"][key].append(sensors[i])
        self.logger["target"].append(target)
        self.logger["manhattan_distance"].append(manhattan_distance)
        self.logger["euclidean_distance"].append(euclidean_distance)
        self.logger["energy"].append(energy)
        self.logger["entropy"].append(entropy)
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
            mu=self.target_duration["mean"],
            a=self.target_duration["min"],
            b=self.target_duration["max"],
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
            action = rnn.step(obs)
            self.plant.step(action)
            hand_position = self.plant.get_hand_pos()
            target_position = target_positions[target_idx]
            manhattan_distance = l1_norm(target_position - hand_position)
            euclidean_distance = l2_norm(target_position - hand_position)
            energy = np.mean(action)
            entropy = action_entropy(action)

            reward = -(
                euclidean_distance * self.loss_weights["euclidean"]
                + manhattan_distance * self.loss_weights["manhattan"]
                + energy * entropy * self.loss_weights["energy"]
                + .001 * l1_norm(rnn.get_params())
            )
            total_reward += reward

            if log:
                self.log(
                    time=self.plant.data.time,
                    sensors=feedback,
                    target=target_position,
                    manhattan_distance=manhattan_distance,
                    euclidean_distance=euclidean_distance,
                    energy=energy,
                    entropy=entropy,
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

        _, axes = plt.subplots(3, 2, figsize=(10, 10))

        # Targets (NOT WORKING BECAUSE NON-CONSTANT TARGET DURATIONS!!!)
        # for t in range(0, int(self.logger["time"][-1]), 3):
        #     axes[0, 0].axvline(x=t, color="gray", linestyle="--", linewidth=0.5)
        #     axes[0, 1].axvline(x=t, color="gray", linestyle="--", linewidth=0.5)
        #     axes[1, 0].axvline(x=t, color="gray", linestyle="--", linewidth=0.5)
        #     axes[1, 1].axvline(x=t, color="gray", linestyle="--", linewidth=0.5)
        #     axes[2, 0].axvline(x=t, color="gray", linestyle="--", linewidth=0.5)
        #     axes[2, 1].axvline(x=t, color="gray", linestyle="--", linewidth=0.5)
    

        linewidth = 1

        # Length
        axes[0, 0].plot(log["time"], log["sensors"]["deltoid_len"], label="Deltoid")
        axes[0, 0].plot(
            log["time"],
            log["sensors"]["latissimus_len"],
            linewidth=linewidth,
            label="Latissimus",
        )
        axes[0, 0].plot(
            log["time"],
            log["sensors"]["biceps_len"],
            linewidth=linewidth,
            label="Biceps",
        )
        axes[0, 0].plot(
            log["time"],
            log["sensors"]["triceps_len"],
            linewidth=linewidth,
            label="Triceps",
        )
        axes[0, 0].set_title("Length")

        # Velocity
        axes[0, 1].plot(
            log["time"],
            log["sensors"]["deltoid_vel"],
            linewidth=linewidth,
            label="Deltoid",
        )
        axes[0, 1].plot(
            log["time"],
            log["sensors"]["latissimus_vel"],
            linewidth=linewidth,
            label="Latissimus",
        )
        axes[0, 1].plot(
            log["time"],
            log["sensors"]["biceps_vel"],
            linewidth=linewidth,
            label="Biceps",
        )
        axes[0, 1].plot(
            log["time"],
            log["sensors"]["triceps_vel"],
            linewidth=linewidth,
            label="Triceps",
        )
        axes[0, 1].set_title("Velocity")
        axes[0, 1].legend(loc="center left", bbox_to_anchor=(1, 0.5))

        # Force
        axes[1, 0].plot(
            log["time"],
            log["sensors"]["deltoid_frc"],
            linewidth=linewidth,
            label="Deltoid",
        )
        axes[1, 0].plot(
            log["time"],
            log["sensors"]["latissimus_frc"],
            linewidth=linewidth,
            label="Latissimus",
        )
        axes[1, 0].plot(
            log["time"],
            log["sensors"]["biceps_frc"],
            linewidth=linewidth,
            label="Biceps",
        )
        axes[1, 0].plot(
            log["time"],
            log["sensors"]["triceps_frc"],
            linewidth=linewidth,
            label="Triceps",
        )
        axes[1, 0].set_title("Force")

        # Distance
        axes[1, 1].plot(
            log["time"],
            log["manhattan_distance"],
            linewidth=linewidth,
            label="Manhattan",
        )
        axes[1, 1].plot(
            log["time"],
            log["euclidean_distance"],
            linewidth=linewidth,
            label="Euclidean",
        )
        axes[1, 1].set_title("Distance")
        axes[1, 1].set_ylim([-0.05, 2.05])
        axes[1, 1].legend()

        # Energy
        axes[2, 0].plot(log["time"], log["entropy"], linewidth=0.1, label="Entropy")
        axes[2, 0].plot(log["time"], log["energy"], linewidth=0.1, label="Energy")
        axes[2, 0].set_title("Energy")
        axes[2, 0].set_ylim([-0.05, 2.05])
        axes[2, 0].legend()

        # Fitness
        axes[2, 1].plot(log["time"], log["reward"], linewidth=linewidth, label="Reward")
        axes[2, 1].set_title("Loss")
        axes[2, 1].set_ylim([-2.05, 0.05])

        # Create a twin axis (right y-axis)
        r, g, b = np.array([1, 1, 1]) * 0.25
        fitness_clr = (r, g, b)
        ax_right = axes[2, 1].twinx()
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
