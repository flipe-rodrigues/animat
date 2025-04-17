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
        target_position,
        hand_position,
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
            self.logger["target_position"] = []
            self.logger["hand_position"] = []
            self.logger["manhattan_distance"] = []
            self.logger["euclidean_distance"] = []
            self.logger["energy"] = []
            self.logger["entropy"] = []
            self.logger["reward"] = []
            self.logger["fitness"] = []

        self.logger["time"].append(time)
        for i, key in enumerate(self.logger["sensors"].keys()):
            self.logger["sensors"][key].append(sensors[i])
        self.logger["target_position"].append(target_position)
        self.logger["hand_position"].append(hand_position)
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
                + l1_norm(rnn.get_params() * self.loss_weights["ridge"])
                + l2_norm(rnn.get_params() * self.loss_weights["lasso"])
            )
            total_reward += reward

            if log:
                self.log(
                    time=self.plant.data.time,
                    sensors=feedback,
                    target_position=target_position,
                    hand_position=hand_position,
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
    ..######..########.####.##.....##.##.....##.##..........###....########.########
    .##....##....##.....##..###...###.##.....##.##.........##.##......##....##......
    .##..........##.....##..####.####.##.....##.##........##...##.....##....##......
    ..######.....##.....##..##.###.##.##.....##.##.......##.....##....##....######..
    .......##....##.....##..##.....##.##.....##.##.......#########....##....##......
    .##....##....##.....##..##.....##.##.....##.##.......##.....##....##....##......
    ..######.....##....####.##.....##..#######..########.##.....##....##....########
    """

    def stimulate(self, rnn, units, delay, seed=0, render=False):
        """Evaluate fitness of a given RNN policy"""
        np.random.seed(seed)

        rnn.init_state()
        self.plant.reset()

        # Zero out the first 3 columns of the input weights
        rnn.W_in[:, :3] = 0

        if render:
            self.plant.render()

        self.plant.model.eq_active0 = 1

        force_vecs = []

        total_delay = delay
            
        pos = self.plant.sample_targets(1)
        self.plant.update_nail(pos)
            
        while self.plant.viewer.is_running():
            if render:
                self.plant.render()

            context, feedback = self.plant.get_obs()
            obs = np.concatenate([context, feedback])

            if self.plant.data.time > total_delay:
                # rnn.h[units] = rnn.activation(np.inf)  # Stimulate the specified units
                total_delay += delay
                pos = self.plant.sample_targets(1)
                self.plant.update_nail(pos)
                # self.plant.update_target(pos)
                # self.plant.randomize_configuration()
                # self.plant.update_nail(self.plant.get_hand_pos())
                # self.plant.model.body_mass[self.plant.hand_id] = 1e3
            action = rnn.step(obs)
            self.plant.step(action)

            # sensor_id = self.plant.hand_force_id
            # force = self.plant.data.sensordata[sensor_id : sensor_id + 3]
            # # Recompute force assuming a hand mass of 1
            # hand_mass = 1
            # acceleration = force / (1e3 * self.plant.hand_default_mass)
            # force = acceleration * hand_mass
            # # force = self.plant.data.cfrc_ext[self.plant.hand_id, :3].copy()
            force = self.plant.data.efc_force.copy()
            # force_vecs.append(self.plant.data.efc_force.copy())
            force_vecs.append(force)

        self.plant.close()

        # Plot force vectors over time
        force_vecs = np.array(force_vecs)
        time = np.linspace(0, self.plant.data.time, len(force_vecs))

        plt.figure(figsize=(10, 5))
        for i in range(force_vecs.shape[1]):
            plt.plot(
                time[time > 1], force_vecs[time > 1, i], label=f"Force Component {i+1}"
            )
        plt.xlabel("Time (s)")
        plt.ylabel("Force (a.u.)")
        plt.title("Force Vectors Over Time")
        plt.legend()
        plt.tight_layout()
        plt.show()

        return force_vecs

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

        # Targets
        target_onset_idcs = np.where(
            np.any(np.diff(np.array(log["target_position"]), axis=0) != 0, axis=1)
        )[0]
        target_onset_idcs = np.insert(target_onset_idcs, 0, 0)
        target_onset_times = np.array(
            [log["time"][target_onset_idx] for target_onset_idx in target_onset_idcs]
        )
        for t in target_onset_times:
            axes[0, 0].axvline(x=t, color="gray", linestyle="--", linewidth=0.5)
            axes[0, 1].axvline(x=t, color="gray", linestyle="--", linewidth=0.5)
            axes[1, 0].axvline(x=t, color="gray", linestyle="--", linewidth=0.5)
            axes[1, 1].axvline(x=t, color="gray", linestyle="--", linewidth=0.5)
            axes[2, 0].axvline(x=t, color="gray", linestyle="--", linewidth=0.5)
            axes[2, 1].axvline(x=t, color="gray", linestyle="--", linewidth=0.5)

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

        # Hand Velocity
        plt.figure(figsize=(10, 1))
        # Annotate target change times with vertical lines
        for idx in target_onset_idcs:
            plt.axvline(
                x=log["time"][idx],
                color="blue",
                linestyle="--",
                linewidth=0.5,
                label="Target Change" if idx == target_onset_idcs[0] else None,
            )
        hand_positions = np.array(log["hand_position"])
        hand_velocities = np.linalg.norm(np.diff(hand_positions, axis=0), axis=1)
        time = np.array(
            log["time"][:-1]
        )  # Exclude the last time step to match velocity array length
        plt.plot(
            time,
            hand_velocities,
            linewidth=linewidth,
            label="Hand Velocity",
            color="black",
        )
        plt.xlabel("Time (s)")
        plt.ylabel("Hand velocity (a.u.)")
        ax_right = plt.gca().twinx()  # Create a twin axis (right y-axis)
        ax_right.plot(
            log["time"][:-1],
            log["euclidean_distance"][:-1],
            linewidth=linewidth,
            label="Euclidean Distance",
            color="red",
        )
        ax_right.set_ylabel("Euclidean Distance", color="red")
        ax_right.tick_params(axis="y", labelcolor="red")

        self.logger = None
