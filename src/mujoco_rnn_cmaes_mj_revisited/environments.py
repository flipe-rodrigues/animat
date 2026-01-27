import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from plants import SequentialReacher
from encoders import *
from networks import *
from utils import *

MUSCLES = ["deltoid", "latissimus", "biceps", "triceps"]
SENSORS = ["len", "vel", "frc"]


class SequentialReachingEnv:
    def __init__(
        self,
        plant,
        target_encoder: TargetEncoder,
        target_duration_distro,
        iti_distro,
        num_targets,
        randomize_gravity=False,
        loss_weights={
            "distance": 1.0,
            "energy": 0.1,
        },
    ):
        self.plant = plant
        self.target_encoder = target_encoder
        self.target_duration_distro = target_duration_distro
        self.iti_distro = iti_distro
        self.num_targets = num_targets
        self.randomize_gravity = randomize_gravity
        self.loss_weights = loss_weights
        self.logger = None

    """
    .##........#######...######....######...########.########.
    .##.......##.....##.##....##..##....##..##.......##.....##
    .##.......##.....##.##........##........##.......##.....##
    .##.......##.....##.##...####.##...####.######...########.
    .##.......##.....##.##....##..##....##..##.......##...##..
    .##.......##.....##.##....##..##....##..##.......##....##.
    .########..#######...######....######...########.##.....##
    """

    def _init_logger(self):
        self.logger = {
            "time": [],
            "sensors": {f"{m}_{s}": [] for s in SENSORS for m in MUSCLES},
            "target_position": [],
            "hand_position": [],
            "gravity": [],
            "distance": [],
            "energy": [],
            "reward": [],
            "fitness": [],
        }

    def log(
        self,
        time,
        sensors,
        target_position,
        hand_position,
        gravity,
        distance,
        energy,
        reward,
        fitness,
    ):
        if self.logger is None:
            self._init_logger()

        self.logger["time"].append(time)
        for key, value in zip(self.logger["sensors"], sensors):
            self.logger["sensors"][key].append(value)
        self.logger["target_position"].append(target_position)
        self.logger["hand_position"].append(hand_position)
        self.logger["gravity"].append(gravity)
        self.logger["distance"].append(distance)
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
        self.plant.disable_target()

        # Sample targets and durations
        target_positions = self.plant.sample_targets(self.num_targets)
        target_durations = truncated_exponential(
            mu=self.target_duration_distro["mean"],
            a=self.target_duration_distro["min"],
            b=self.target_duration_distro["max"],
            size=self.num_targets,
        )
        itis = truncated_exponential(
            mu=self.iti_distro["mean"],
            a=self.iti_distro["min"],
            b=self.iti_distro["max"],
            size=self.num_targets,
        )

        target_onset_times = np.concatenate(
            [[0], (target_durations[:-1] + itis[:-1]).cumsum()]
        )
        target_offset_times = target_onset_times + target_durations
        trial_duration = target_durations.sum() + itis.sum()

        total_reward = 0
        target_idx = 0

        while target_idx < self.num_targets:
            if render:
                self.plant.render()

            # Disable target if past offset
            if self.plant.data.time >= target_offset_times[target_idx]:
                self.plant.disable_target()
                target_idx += 1

            # Enable target if past onset
            if (
                target_idx < self.num_targets
                and self.plant.data.time >= target_onset_times[target_idx]
                and not self.plant.target_is_active
            ):
                if self.randomize_gravity:
                    self.plant.randomize_gravity_direction()
                target_pos = target_positions[target_idx]
                self.plant.update_target(target_pos)
                self.plant.enable_target()

            # Observations
            tgt_obs = self.target_encoder.encode(
                x=target_pos[0], y=target_pos[1]
            ).flatten() * (1 if self.plant.target_is_active else 0)
            len_obs = self.plant.get_len_obs()
            vel_obs = self.plant.get_vel_obs()
            frc_obs = self.plant.get_frc_obs()

            # Motor commands
            ctrl = rnn.step(tgt_obs, len_obs, vel_obs, frc_obs)
            self.plant.step(ctrl)

            distance = self.plant.get_distance_to_target()
            energy = sum(ctrl**2)

            # Rewards
            distance_reward = -distance * self.loss_weights["distance"]
            energy_reward = -energy * self.loss_weights["energy"]
            reward = distance_reward + energy_reward
            total_reward += reward

            # Logging
            if log:
                self.log(
                    time=self.plant.data.time,
                    sensors=np.concatenate([len_obs, vel_obs, frc_obs]),
                    target_position=target_pos,
                    hand_position=self.plant.get_hand_pos(),
                    gravity=self.plant.get_gravity(),
                    distance=distance,
                    energy=energy,
                    reward=reward,
                    fitness=total_reward / trial_duration,
                )

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
        if log is None:
            raise RuntimeError("No data logged. Run `evaluate(log=True)` first.")

        _, axes = plt.subplots(3, 2, figsize=(10, 10))

        # Target change lines
        target_onset_idcs = np.where(
            np.any(np.diff(np.array(log["target_position"]), axis=0) != 0, axis=1)
        )[0]
        target_onset_idcs = np.insert(target_onset_idcs, 0, 0)
        target_onset_times = [log["time"][idx] for idx in target_onset_idcs]
        self._draw_target_lines(axes, target_onset_times)

        linewidth = 1

        # Plot sensors
        self._plot_sensor_group(
            axes[0, 0],
            log["time"],
            log["sensors"],
            ["deltoid_len", "latissimus_len", "biceps_len", "triceps_len"],
            "Length",
        )
        self._plot_sensor_group(
            axes[0, 1],
            log["time"],
            log["sensors"],
            ["deltoid_vel", "latissimus_vel", "biceps_vel", "triceps_vel"],
            "Velocity",
        )
        axes[0, 1].legend(loc="center left", bbox_to_anchor=(1, 0.5))

        self._plot_sensor_group(
            axes[1, 0],
            log["time"],
            log["sensors"],
            ["deltoid_frc", "latissimus_frc", "biceps_frc", "triceps_frc"],
            "Force",
        )

        # Distance
        axes[1, 1].plot(
            log["time"], log["distance"], linewidth=linewidth, label="Distance"
        )
        axes[1, 1].set_title("Distance")
        axes[1, 1].set_ylim([-0.05, 2.05])
        axes[1, 1].legend()

        # Energy
        axes[2, 0].plot(log["time"], log["energy"], linewidth=linewidth, label="Energy")
        axes[2, 0].set_title("Energy")
        axes[2, 0].set_ylim([-0.05, 2.05])
        axes[2, 0].legend()

        # Reward / fitness
        axes[2, 1].plot(log["time"], log["reward"], linewidth=linewidth, label="Reward")
        axes[2, 1].set_title("Loss")
        axes[2, 1].set_ylim([-2.05, 0.05])
        ax_right = axes[2, 1].twinx()
        ax_right.plot(log["time"], log["fitness"], color=(0.25, 0.25, 0.25))
        ax_right.set_ylabel("Cumulative Reward", color=(0.25, 0.25, 0.25))
        ax_right.tick_params(axis="y", labelcolor=(0.25, 0.25, 0.25))

        # Axis labels
        for ax in axes.flat:
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Arb.")

        plt.tight_layout()
        plt.show()

        # Hand velocity plot
        plt.figure(figsize=(10, 1))
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
        time = np.array(log["time"][:-1])
        plt.plot(
            time,
            hand_velocities,
            linewidth=linewidth,
            label="Hand Velocity",
            color="black",
        )
        plt.xlabel("Time (s)")
        plt.ylabel("Hand velocity (a.u.)")
        ax_right = plt.gca().twinx()
        ax_right.plot(
            time,
            log["distance"][:-1],
            linewidth=linewidth,
            color="red",
            label="Distance",
        )
        ax_right.set_ylabel("Distance", color="red")
        ax_right.tick_params(axis="y", labelcolor="red")
        plt.show()

        self.logger = None

    def _draw_target_lines(self, axes, times):
        for t in times:
            for ax in axes.flatten():
                ax.axvline(x=t, color="gray", linestyle="--", linewidth=0.5)

    def _plot_sensor_group(self, ax, time, sensors, keys, title):
        for key in keys:
            ax.plot(time, sensors[key], label=key.replace("_", " ").title())
        ax.set_title(title)
