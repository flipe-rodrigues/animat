"""
.####.##.....##.########...#######..########..########..######.
..##..###...###.##.....##.##.....##.##.....##....##....##....##
..##..####.####.##.....##.##.....##.##.....##....##....##......
..##..##.###.##.########..##.....##.########.....##.....######.
..##..##.....##.##........##.....##.##...##......##..........##
..##..##.....##.##........##.....##.##....##.....##....##....##
.####.##.....##.##.........#######..##.....##....##.....######.
"""

import matplotlib.pyplot as plt
from tqdm import tqdm
from plants import SequentialReacher
from encoders import *
from networks import *
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
        target_encoder: TargetEncoder,
        target_duration_distro,
        iti_distro,
        num_targets,
        randomize_gravity,
        loss_weights,
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
        gravity,
        distance,
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
            self.logger["target_position"] = []
            self.logger["hand_position"] = []
            self.logger["gravity"] = []
            self.logger["distance"] = []
            self.logger["energy"] = []
            self.logger["reward"] = []
            self.logger["fitness"] = []

        self.logger["time"].append(time)
        for i, key in enumerate(self.logger["sensors"].keys()):
            self.logger["sensors"][key].append(sensors[i])
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

            if self.plant.data.time >= target_offset_times[target_idx]:
                self.plant.disable_target()
                target_idx += 1

            if (
                target_idx < self.num_targets
                and self.plant.data.time >= target_onset_times[target_idx]
            ):
                if self.randomize_gravity:
                    self.plant.randomize_gravity_direction()
                target_position = target_positions[target_idx]
                self.plant.update_target(target_position)
                self.plant.enable_target()

            tgt_obs = self.target_encoder.encode(
                target_position[0], target_position[1]
            ).flatten()
            tgt_obs *= 1 if self.plant.target_is_active else 0

            len_obs = self.plant.get_len_obs()
            vel_obs = self.plant.get_vel_obs()
            frc_obs = self.plant.get_frc_obs()

            # Compute motor commands (i.e., alpha-MN activations)
            motor_commands = rnn.step(tgt_obs, len_obs, vel_obs, frc_obs)
            self.plant.step(motor_commands)

            hand_position = self.plant.get_hand_pos()
            distance = (
                l2_norm(target_position - hand_position)
                if self.plant.target_is_active
                else 0
            )
            energy = sum(motor_commands**2)

            reward = -(
                distance * self.loss_weights["distance"]
                + energy * self.loss_weights["energy"]
                + l1_norm(rnn.get_params() * self.loss_weights["ridge"])
                + l2_norm(rnn.get_params() * self.loss_weights["lasso"])
            )
            total_reward += reward

            if log:
                self.log(
                    time=self.plant.data.time,
                    sensors=np.concatenate([len_obs, vel_obs, frc_obs]),
                    target_position=target_position,
                    hand_position=hand_position,
                    gravity=self.plant.get_gravity(),
                    distance=distance,
                    energy=energy,
                    reward=reward,
                    fitness=total_reward / trial_duration,
                )
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

    def stimulate(self, rnn, units, action_modifier=1, delay=1, seed=0, render=False):
        """Evaluate fitness of a given RNN policy"""
        np.random.seed(seed)

        rnn.init_state()
        self.plant.reset()

        if render:
            self.plant.render()

        # Turn on the "nail"
        self.plant.model.eq_active0 = 1

        force_data = {"position": [], "force": []}

        total_delay = 0

        grid_positions = np.array(self.plant.grid_positions.copy())
        grid_pos_idx = 0
        self.plant.update_nail(grid_positions[grid_pos_idx])

        # Update the target position
        target_position = self.plant.sample_targets(1)
        self.plant.update_target(target_position)

        while grid_pos_idx < len(grid_positions) - 1:
            if render:
                self.plant.render()

            context, feedback = self.plant.get_obs()
            obs = np.concatenate([context, feedback])

            # Stimulate the specified units
            if self.plant.data.time > total_delay - delay / 2:
                rnn.h[units] = rnn.activation(np.inf)

            # Update nail position
            if self.plant.data.time > total_delay:
                grid_pos_idx += 1
                self.plant.update_nail(grid_positions[grid_pos_idx])
                total_delay += delay

            action = rnn.step(obs) * action_modifier
            self.plant.step(action)
            force = self.plant.data.efc_force.copy()

            if force.shape != (3,):
                force = np.full(3, np.nan)

            force_data["position"].append(grid_positions[grid_pos_idx])
            force_data["force"].append(force)

        self.plant.close()

        return force_data

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
            log["distance"],
            linewidth=linewidth,
            label="Distance",
        )
        axes[1, 1].set_title("Distance")
        axes[1, 1].set_ylim([-0.05, 2.05])
        axes[1, 1].legend()

        # Energy
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
            log["distance"][:-1],
            linewidth=linewidth,
            label="Distance",
            color="red",
        )
        ax_right.set_ylabel("Distance", color="red")
        ax_right.tick_params(axis="y", labelcolor="red")

        self.logger = None
