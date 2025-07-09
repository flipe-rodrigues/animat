"""
.####.##.....##.########...#######..########..########..######.
..##..###...###.##.....##.##.....##.##.....##....##....##....##
..##..####.####.##.....##.##.....##.##.....##....##....##......
..##..##.###.##.########..##.....##.########.....##.....######.
..##..##.....##.##........##.....##.##...##......##..........##
..##..##.....##.##........##.....##.##....##.....##....##....##
.####.##.....##.##.........#######..##.....##....##.....######.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
from plants import SequentialReacher
from networks import RNN
from utils import *
import matplotlib as mpl
import os


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
    .########.....###.....######...######..####.##.....##.########
    .##.....##...##.##...##....##.##....##..##..##.....##.##......
    .##.....##..##...##..##.......##........##..##.....##.##......
    .########..##.....##..######...######...##..##.....##.######..
    .##........#########.......##.......##..##...##...##..##......
    .##........##.....##.##....##.##....##..##....##.##...##......
    .##........##.....##..######...######..####....###....########
    """

    def passive(
        self, rnn, seed=0, weight_mod=1, weight_density=100, render=False, log=False
    ):
        """Move the forearm passively through its range of motion"""
        np.random.seed(seed)

        # Points to move the hand to in space.
        passive_hand_positions = []
        num_passive_targets = 10
        max_angle = (
            -80
        )  # In world space coordinates. Right is 0 degrees. Down is 270 degrees.
        min_angle = 70
        passive_target_angles = np.linspace(min_angle, max_angle, num_passive_targets)
        print(passive_target_angles)

        for i in range(len(passive_target_angles)):
            angle = np.radians(passive_target_angles[i])  # Convert angle to radians
            x = 0 + 0.5 * np.cos(angle)  # x-coordinate
            y = -0.4 + 0.5 * np.sin(angle)  # y-coordinate
            z = 0  # z-coordinate remains the same
            coordinates = [x, y, z]
            passive_hand_positions.append(coordinates)

        print("Passive target positions:", passive_hand_positions)

        # Plot and save the passive target positions (x and y only)
        import matplotlib.pyplot as plt

        # Convert passive_target_positions to a numpy array for easier manipulation
        passive_hand_positions = np.array(passive_hand_positions)

        # Extract x and y coordinates
        x_coords = passive_hand_positions[:, 0]
        y_coords = passive_hand_positions[:, 1]

        # ----------------------------------------------------------------------- MOVE DOWN TO PLOT PASSIVE SOMEHOW
        # Create a scatter plot
        plt.figure(figsize=(5, 5))
        plt.scatter(x_coords, y_coords, c="blue", marker="o", label="Passive Targets")

        # Label axes
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.title("Passive Target Positions")
        plt.legend()

        # Save the plot
        save_path = "/Users/joseph/My Drive/Champalimaud/rotations/Joe/figures/passive_hand_positions_xy.png"
        # save_path = "C:\\Users\\User\\Desktop\\tests"
        plt.savefig(save_path, dpi=900)
        plt.show()
        # -----------------------------------------------------------------------

        # Keep track of elbow torque and angle
        elbow_joint_id = self.plant.model.joint("elbow").id
        elbow_angle_log = []
        elbow_torque_log = []
        passive_hand_positions_log = []
        hand_position_log = []

        rnn.init_state()
        self.plant.reset()

        # Run it
        if render:
            self.plant.render()

        # Turn on the "nail"
        #  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        self.plant.model.eq_active0 = 1
        #  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        total_delay = 0
        delay = 5

        pos_idx = 0
        self.plant.update_nail(passive_hand_positions[pos_idx])

        while pos_idx < len(passive_hand_positions) - 1:
            if render:
                self.plant.render()

            context, feedback = self.plant.get_obs()
            obs = np.concatenate([context, feedback])

            # Store elbow data
            elbow_torque_log.append(
                self.plant.data.qfrc_actuator[elbow_joint_id]
            )  # Elbow torque as read out like this.
            elbow_angle_log.append(
                np.pi - self.plant.data.qpos[elbow_joint_id]
            )  # This adjustment is needed to give the intuitive absolute elbow angle (e.g. arm extended = 180 degrees).

            # Update nail position !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            if self.plant.data.time > total_delay:
                pos_idx += 1
                self.plant.update_nail(passive_hand_positions[pos_idx])
                passive_hand_positions_log.append(passive_hand_positions[pos_idx])
                total_delay += delay
            #  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

            hand_position = self.plant.get_hand_pos()
            hand_position_log.append(hand_position)

            action = rnn.step(obs)
            self.plant.step(action)

        self.plant.close()

        self.num_passive_targets = num_passive_targets
        self.delay = delay
        self.passive_hand_positions = passive_hand_positions
        self.elbow_torque_log = elbow_torque_log
        self.elbow_angle_log = elbow_angle_log
        self.hand_position_log = hand_position_log

        return

    """
    .########.########.##.......########..##.....##....###....##....##
    .##.......##.......##.......##.....##.###...###...##.##...###...##
    .##.......##.......##.......##.....##.####.####..##...##..####..##
    .######...######...##.......##.....##.##.###.##.##.....##.##.##.##
    .##.......##.......##.......##.....##.##.....##.#########.##..####
    .##.......##.......##.......##.....##.##.....##.##.....##.##...###
    .##.......########.########.########..##.....##.##.....##.##....##
    """

    def feldman(
        self, rnn, seed=0, weight_mod=1, weight_density=100, render=False, log=False
    ):
        """Based on Asatryan and Feldman, 1965"""
        np.random.seed(seed)

        rnn.init_state()
        self.plant.reset()

        target_positions = self.plant.sample_targets(self.num_targets)

        # For bicep
        if "flexor" in self.plant.xml:
            angles = [325, 345]

            for i, angle in enumerate(angles):
                angle = np.radians(angle)  # Convert angle to radians
                x = 0 + 0.5 * np.cos(angle)  # x-coordinate
                y = -0.4 + 0.5 * np.sin(angle)  # y-coordinate
                z = 0  # z-coordinate remains the same
                coordinates = [x, y, z]
                target_positions[i] = coordinates

            print("Flexor!")

        # For tricep
        elif "extensor" in self.plant.xml:
            angles = [315, 335]

            for i, angle in enumerate(angles):
                angle = np.radians(angle)  # Convert angle to radians
                x = 0 + 0.5 * np.cos(angle)  # x-coordinate
                y = -0.4 + 0.5 * np.sin(angle)  # y-coordinate
                z = 0  # z-coordinate remains the same
                coordinates = [x, y, z]
                target_positions[i] = coordinates

            print("Extensor!")

        target_durations = truncated_exponential(
            mu=self.target_duration["mean"],
            a=self.target_duration["min"],
            b=self.target_duration["max"],
            size=self.num_targets,
        )
        target_offset_times = target_durations.cumsum()
        target_onset_times = target_offset_times - target_durations[0]
        trial_duration = target_durations.sum()
        total_reward = 0

        target_idx = 0
        self.plant.update_target(target_positions[target_idx])

        # Keep track of elbow torque
        elbow_joint_id = self.plant.model.joint("elbow").id
        elbow_angle_log = []
        elbow_torque_log = []
        elbow_torque_sensor_log = []
        action_log = []
        weight_log = []

        max_pulley_weight = 0.5 * weight_mod
        min_pulley_weight = 1e-3 * weight_mod
        num_unique_weights = 5
        pulley_weights = np.linspace(
            min_pulley_weight, max_pulley_weight, num_unique_weights
        )
        alternating_weights = []
        for i in range(num_unique_weights - 1):
            alternating_weights.append(max_pulley_weight)
            alternating_weights.append(pulley_weights[i])
        num_pulley_weights = len(alternating_weights)
        weight_update_times = (
            np.linspace(0, 1, num_pulley_weights + 1) * target_durations[0]
        )
        weight_idx = 0

        print(f"Number of pulley weights: {num_pulley_weights}")
        print(f"Weight update period: {weight_update_times} seconds")
        print(f"Alternating weights: {alternating_weights}")

        weight_update_times[-1] = 1e6  # hack!!!

        while target_idx < self.num_targets:
            if render:
                self.plant.render()

            context, feedback = self.plant.get_obs()
            obs = np.concatenate([context, feedback])

            action = rnn.step(
                obs
            )  # action seems to be the muscle inputs for the timestep.
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

            target_aligned_time = self.plant.data.time - target_onset_times[target_idx]

            # Run the weight update protocol
            if target_aligned_time > weight_update_times[weight_idx]:
                weight_mass = alternating_weights[weight_idx]
                weight_idx += 1

            # print(f"Target {target_idx + 1}, Time: {target_aligned_time:.2f}s, Weight Index: {weight_idx}, Weight: {weight_mass}")

            # Update the mass of the cylinder
            self.plant.model.body(self.plant.weight_id).mass[0] = weight_mass

            # Keep cylinder height constant
            cylinder_height = 0.2
            self.plant.model.geom(self.plant.weight_id).size[1] = (
                cylinder_height / 2
            )  # This actually codes 'half length along cylinder's local z axis'. So length = twice this.

            # Make cylinder height reflect mass
            cylinder_radius = np.sqrt(
                weight_mass / (weight_density * cylinder_height * np.pi)
            )
            self.plant.model.geom(self.plant.weight_id).size[0] = cylinder_radius

            # Log the elbow torque
            # elbow_id = self.model.joint("elbow").id # self.plant.model.joint_name2id("elbow")
            # elbow_dof = self.plant.model.jnt_dofadr[elbow_id]
            # elbow_torque = self.plant.data.qfrc_actuator[elbow_dof]
            # elbow_torque_log.append(elbow_torque)

            # Store elbow data
            elbow_torque_log.append(
                self.plant.data.qfrc_actuator[elbow_joint_id]
            )  # Elbow torque as read out like this.
            elbow_angle_log.append(
                np.pi - self.plant.data.qpos[elbow_joint_id]
            )  # This adjustment is needed to give the intuitive absolute elbow angle (e.g. arm extended = 180 degrees).
            elbow_torque_sensor_log.append(
                self.plant.data.sensordata[-1]
            )  # Elbow torque as read out from a sensor.
            action_log.append(action)
            weight_log.append(weight_mass)

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
                weight_idx = 0
                target_idx += 1
                if target_idx < self.num_targets:
                    self.plant.update_target(target_positions[target_idx])

        self.plant.close()

        self.elbow_torque_log = elbow_torque_log
        self.elbow_angle_log = elbow_angle_log
        self.elbow_torque_sensor_log = elbow_torque_sensor_log
        self.num_pulley_weights = num_pulley_weights
        self.weight_update_period = np.diff(weight_update_times)[0]
        self.action_log = action_log
        self.target_offset_times = target_offset_times
        self.weight_log = weight_log

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

    # Set default font to Helvetica
    # mpl.rcParams['font.family'] = 'Helvetica'

    def plot(self, save_path):
        #

        timesteps = np.arange(len(self.elbow_torque_log))
        time_sec = timesteps * self.plant.model.opt.timestep

        if hasattr(self, "elbow_torque_log"):

            muscle_id = "flexor" if "flexor" in self.plant.xml else "extensor"

            # Plotting for flexor only
            session_fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(
                5, 1, figsize=(8, 7), sharex=True
            )

            # Plot weight log on the first subplot
            ax1.plot(
                time_sec, self.weight_log, color="green", linewidth=2, label="Weight"
            )
            ax1.set_ylabel("Weight (kg)", color="green")
            ax1.tick_params(axis="y", labelcolor="green")
            ax1.legend(loc="center right", frameon=False)  # Remove grey box
            ax1.spines["top"].set_visible(False)
            ax1.spines["right"].set_visible(False)
            ax1.spines["bottom"].set_visible(False)
            ax1.xaxis.set_ticks_position("none")
            ax1.grid(False)

            # Plot x and y target position over time on the second subplot
            target_positions = np.array(self.logger["target_position"])
            ax2.plot(
                time_sec,
                target_positions[:, 0],
                label="Target X",
                color="red",
                linewidth=2,
            )
            ax2.plot(
                time_sec,
                target_positions[:, 1],
                label="Target Y",
                color="blue",
                linewidth=2,
            )
            ax2.set_ylabel("Target Position")
            ax2.legend(loc="center right", frameon=False)  # Remove grey box
            ax2.spines["top"].set_visible(False)
            ax2.spines["right"].set_visible(False)
            ax2.spines["bottom"].set_visible(False)
            ax2.xaxis.set_ticks_position("none")
            ax2.grid(False)

            # Plot angle on the third subplot
            smoothed_angle = np.convolve(
                np.degrees(self.elbow_angle_log), np.ones(50) / 50, mode="same"
            )  # Simple moving average
            ax3.plot(time_sec, smoothed_angle, color="black", linewidth=2)
            ax3.set_ylabel("Elbow Angle (°)", color="black")
            ax3.tick_params(axis="y", labelcolor="black")
            ax3.spines["top"].set_visible(False)
            ax3.spines["right"].set_visible(False)
            ax3.spines["bottom"].set_visible(False)
            ax3.xaxis.set_ticks_position("none")
            ax3.grid(False)

            # Set y-limits to 5th and 95th percentiles of smoothed_angle
            ymin = np.percentile(smoothed_angle, 1)
            ymax = np.percentile(smoothed_angle, 99)
            ax3.set_ylim([ymin, ymax])

            # Plot bicep and tricep activation on the fourth subplot
            bicep_activation = np.array(self.action_log)[:, 2]
            tricep_activation = np.array(self.action_log)[:, 3]

            # Plot raw and smoothed bicep activation
            # ax4.plot(
            #     time_sec,
            #     muscle_activation,
            #     color="purple",
            #     alpha=0.5,
            #     linewidth=0.5,
            #     label="Raw",
            # )
            smoothed_bicep = np.convolve(
                bicep_activation, np.ones(100) / 100, mode="same"
            )  # Larger smoothing window
            ax4.plot(
                time_sec,
                smoothed_bicep,
                color="purple",
                linewidth=1,
                label="Bicep (smoothed)",
            )

            # Plot raw and smoothed tricep activation
            # ax4.plot(
            #     time_sec,
            #     tricep_activation,
            #     color="orange",
            #     alpha=0.5,
            #     linewidth=0.5,
            #     label="Tricep (Raw)",
            # )
            smoothed_tricep = np.convolve(
                tricep_activation, np.ones(100) / 100, mode="same"
            )  # Larger smoothing window
            ax4.plot(
                time_sec,
                smoothed_tricep,
                color="orange",
                linewidth=1,
                label="Tricep (Smoothed)",
            )

            # Plot the difference between bicep and tricep activation
            smoothed_diff = smoothed_bicep - smoothed_tricep
            ax4.plot(
                time_sec,
                smoothed_diff,
                color="gray",
                linewidth=1,
                label="Bicep - Tricep (Smoothed)",
            )

            ax4.set_ylabel("Muscle Activation")
            ax4.legend(loc="upper right", frameon=False)  # Remove grey box
            ax4.spines["top"].set_visible(False)
            ax4.spines["right"].set_visible(False)
            ax4.spines["bottom"].set_visible(False)
            ax4.xaxis.set_ticks_position("none")
            ax4.grid(False)

            # Plot torque on the fifth subplot
            ax5.plot(
                time_sec,
                self.elbow_torque_log,
                color="blue",
                alpha=0.5,
                linewidth=0.5,
                label="Raw",
            )
            smoothed_torque = np.convolve(
                self.elbow_torque_log, np.ones(100) / 100, mode="same"
            )  # Larger smoothing window
            ax5.plot(
                time_sec, smoothed_torque, color="blue", linewidth=2, label="Smoothed"
            )
            ax5.set_xlabel("Time (s)")
            ax5.set_ylabel("Elbow Torque (Nm)", color="blue")
            ax5.tick_params(axis="y", labelcolor="blue")
            ax5.legend(loc="upper right", frameon=False)  # Remove grey box
            ax5.spines["top"].set_visible(False)
            ax5.spines["right"].set_visible(False)
            ax5.grid(False)

            # Set y-limits to 5th and 95th percentiles of smoothed_angle
            ymin = np.percentile(smoothed_torque, 0.1)
            ymax = np.percentile(smoothed_torque, 99.9)
            ax5.set_ylim([ymin, ymax])

            session_fig.tight_layout()

            file_name = f"elbow_torque_angle_weight_{muscle_id}"
            png_file = os.path.join(save_path, f"{file_name}.png")
            plt.savefig(png_file, dpi=900)
            plt.show()

            # Split logs based on weight_update_period
            torque_segments = []
            angle_segments = []
            action_segments = []
            bicep_segments = []
            tricep_segments = []
            for target_idx in range(self.num_targets):
                for i in range(self.num_pulley_weights):
                    start_time = (
                        target_idx * self.num_pulley_weights * self.weight_update_period
                        + i * self.weight_update_period
                    )
                    end_time = start_time + self.weight_update_period
                    segment_indices = (time_sec >= start_time) & (time_sec < end_time)
                    torque_segments.append(
                        np.array(self.elbow_torque_log)[segment_indices]
                    )
                    angle_segments.append(
                        np.array(self.elbow_angle_log)[segment_indices]
                    )
                    action_segments.append(np.array(self.action_log)[segment_indices])

            bicep_segments = [segment[:, 2] for segment in action_segments]
            tricep_segments = [segment[:, 3] for segment in action_segments]

            # Find the average torque and angle for the second half of each segment
            avg_torque = [
                np.mean(segment[len(segment) // 2 :]) for segment in torque_segments
            ]
            avg_angle = [
                np.mean(segment[len(segment) // 2 :]) for segment in angle_segments
            ]
            avg_angle_degrees = [np.degrees(angle) for angle in avg_angle]
            avg_bicep = [
                np.mean(segment[len(segment) // 2 :]) for segment in bicep_segments
            ]
            avg_tricep = [
                np.mean(segment[len(segment) // 2 :]) for segment in tricep_segments
            ]

            # Save settings
            flexor_only_file_name = "avg_torque_vs_avg_angle_with_muscle_activity"
            flexor_only_pickle_file = os.path.join(
                save_path, f"{flexor_only_file_name}.fig.pickle"
            )
            flexor_extensor_file_name = (
                "avg_torque_vs_avg_angle_with_flexor_and_extensor"
            )
            flexor_extensor_pickle_file = os.path.join(
                save_path, f"{flexor_extensor_file_name}.fig.pickle"
            )

            # BICEP
            if "flexor" in self.plant.xml:

                # Scatter plot of avg_torque vs avg_angle with bicep muscle activity as error bars
                fig = plt.figure(figsize=(5, 5))  # Make the plot square
                num_segments_to_plot = len(torque_segments) / self.num_targets
                plt.xlabel("Average Elbow Angle (°)")
                colors = [
                    "purple",
                    "green",
                    "red",
                    "blue",
                    "orange",
                ]  # Add more colors if needed

                # Define error values (example: replace with your own error values)
                torque_errors = avg_bicep  # Example: constant error for torques

                for target_idx in range(self.num_targets):
                    start_idx = target_idx * int(num_segments_to_plot)
                    end_idx = (target_idx + 1) * int(num_segments_to_plot)
                    odd_indices = list(range(start_idx + 1, end_idx, 2))
                    indices_to_plot = [start_idx] + odd_indices

                    # Extract data for plotting
                    angles = [avg_angle_degrees[i] for i in indices_to_plot]
                    torques = [avg_torque[i] for i in indices_to_plot]
                    errors = [torque_errors[i] for i in indices_to_plot]

                    # Normalised muscle activation
                    normalised_errors = (errors - np.min(errors)) / (
                        np.max(errors) - np.min(errors)
                    )
                    cmap = plt.get_cmap("gray")
                    colors = cmap(normalised_errors)

                    # Sort points by average elbow torque
                    sorted_indices = np.argsort(torques)
                    sorted_angles = [angles[i] for i in sorted_indices]
                    sorted_torques = [torques[i] for i in sorted_indices]

                    # Join points with straight lines in sorted order
                    plt.plot(
                        sorted_angles,
                        sorted_torques,
                        color="black",
                        linewidth=2,
                        alpha=1,
                        zorder=2,
                    )

                    # Plot error bars
                    plt.errorbar(
                        angles,
                        torques,
                        yerr=[[0] * len(indices_to_plot), errors],
                        fmt="o",
                        markerfacecolor="white",  # Grayscale based on normalized errors
                        markeredgecolor="black",  # Black edge
                        ecolor="lightgrey",  # Set error bars to lighter grey
                        alpha=1,
                        zorder=1,
                    )

                    plt.scatter(
                        angles,
                        torques,
                        c=colors,  # RGBA or RGB colors per point
                        edgecolors="black",  # Black edge for better visibility
                        zorder=3,
                    )

                # Add a horizontal line at y = 0
                plt.axhline(y=0, color="black", linestyle="--", linewidth=0.8)

                plt.ylabel("Average Elbow Torque (Nm)")
                plt.title("Bicep and Tricep Unloading Responses")
                plt.grid(False)

                # Remove top and right axes
                plt.gca().spines["top"].set_visible(False)
                plt.gca().spines["right"].set_visible(False)

                # Only show ticks on the left and bottom axes
                plt.gca().yaxis.set_ticks_position("left")
                plt.gca().xaxis.set_ticks_position("bottom")

                # Set ticks every 0.5
                plt.gca().yaxis.set_major_locator(plt.MultipleLocator(0.5))
                plt.ylim([-1.5, 1.5])

                # Save the figure to a pickle file
                with open(flexor_only_pickle_file, "wb") as f:
                    pickle.dump(fig, f)
                print(f"Figure saved to {flexor_only_pickle_file}")

                # Option to save the figure
                png_file = os.path.join(save_path, f"{flexor_only_file_name}.png")
                plt.savefig(png_file, dpi=900)
                print(f"Figure saved to {png_file}")

                plt.show()

            # # EXTENSOR
            elif "extensor" in self.plant.xml:

                # If now wanting to plot tricep (extensor) data
                with open(flexor_only_pickle_file, "rb") as f:
                    fig = pickle.load(f)
                plt.figure(fig.number)

                # Scatter plot of avg_torque vs avg_angle with tricep muscle activity as error bars
                num_segments_to_plot = len(torque_segments) / self.num_targets
                plt.xlabel("Average Elbow Angle (°)")
                colors = [
                    "purple",
                    "green",
                    "red",
                    "blue",
                    "orange",
                ]  # Add more colors if needed

                # Define error values (example: replace with your own error values)
                torque_errors = avg_tricep  # Example: constant error for torques

                for target_idx in range(self.num_targets):
                    start_idx = target_idx * int(num_segments_to_plot)
                    end_idx = (target_idx + 1) * int(num_segments_to_plot)
                    odd_indices = list(range(start_idx + 1, end_idx, 2))
                    indices_to_plot = [start_idx] + odd_indices

                    # Extract data for plotting
                    angles = [avg_angle_degrees[i] for i in indices_to_plot]
                    torques = [avg_torque[i] for i in indices_to_plot]
                    errors = [torque_errors[i] for i in indices_to_plot]

                    # Normalised muscle activation
                    normalised_errors = (errors - np.min(errors)) / (
                        np.max(errors) - np.min(errors)
                    )
                    cmap = plt.get_cmap("gray")
                    colors = cmap(normalised_errors)

                    # Sort points by average elbow torque
                    sorted_indices = np.argsort(torques)
                    sorted_angles = [angles[i] for i in sorted_indices]
                    sorted_torques = [torques[i] for i in sorted_indices]

                    # Join points with straight lines in sorted order
                    plt.plot(
                        sorted_angles,
                        sorted_torques,
                        color="black",
                        linewidth=2,
                        alpha=1,
                        zorder=2,
                    )

                    # Plot error bars
                    plt.errorbar(
                        angles,
                        torques,
                        yerr=[
                            errors,
                            [0] * len(indices_to_plot),
                        ],  # Flip: errors go down, not up
                        fmt="o",
                        markerfacecolor="white",  # Grayscale based on normalized errors
                        markeredgecolor="black",  # Black edge
                        ecolor="lightgrey",  # Set error bars to lighter grey
                        alpha=1,
                        zorder=1,
                    )

                    plt.scatter(
                        angles,
                        torques,
                        c=colors,  # RGBA or RGB colors per point
                        edgecolors="black",  # Black edge for better visibility
                        zorder=3,
                    )

                plt.ylabel("Average Elbow Torque (Nm)")
                plt.grid(False)

                # Remove top and right axes
                plt.gca().spines["top"].set_visible(False)
                plt.gca().spines["right"].set_visible(False)

                # Only show ticks on the left and bottom axes
                plt.gca().yaxis.set_ticks_position("left")
                plt.gca().xaxis.set_ticks_position("bottom")

                # Set ticks every 0.5
                plt.gca().yaxis.set_major_locator(plt.MultipleLocator(0.5))

                with open(flexor_extensor_pickle_file, "wb") as f:
                    pickle.dump(fig, f)

                # Option to save the figure
                png_file = os.path.join(save_path, f"{flexor_extensor_file_name}.png")
                plt.savefig(png_file, dpi=900)

                plt.show()

    """
    .########..##........#######..########....########.....###.....######...######..####.##.....##.########
    .##.....##.##.......##.....##....##.......##.....##...##.##...##....##.##....##..##..##.....##.##......
    .##.....##.##.......##.....##....##.......##.....##..##...##..##.......##........##..##.....##.##......
    .########..##.......##.....##....##.......########..##.....##..######...######...##..##.....##.######..
    .##........##.......##.....##....##.......##........#########.......##.......##..##...##...##..##......
    .##........##.......##.....##....##.......##........##.....##.##....##.##....##..##....##.##...##......
    .##........########..#######.....##.......##........##.....##..######...######..####....###....########
    """

    def plot_passive(
        self,
        save_path="/Users/joseph/My Drive/Champalimaud/rotations/Joe/figures",
        pickle_path="/Users/joseph/My Drive/Champalimaud/rotations/Joe/figures/avg_torque_vs_avg_angle_with_flexor_and_extensor.fig.pickle",
    ):

        timesteps = np.arange(len(self.elbow_torque_log))
        time_sec = timesteps * self.plant.model.opt.timestep

        # Split logs based on weight_update_period
        torque_segments = []
        angle_segments = []
        hand_position_segments = []
        for i in range(self.num_passive_targets):
            start_time = self.delay * i
            end_time = self.delay * (i + 1)
            print("Start time:", start_time, "End time:", end_time)
            segment_indices = (time_sec >= start_time) & (time_sec < end_time)
            torque_segments.append(np.array(self.elbow_torque_log)[segment_indices])
            angle_segments.append(np.array(self.elbow_angle_log)[segment_indices])
            hand_positions = np.stack(self.hand_position_log)  # ensures shape (N, 3)
            hand_position_segments.append(hand_positions[segment_indices])

        # Find the average torque and angle for the second half of each segment
        avg_torque = [
            np.mean(segment[len(segment) // 2 :]) for segment in torque_segments
        ]
        avg_angle = [
            np.mean(segment[len(segment) // 2 :]) for segment in angle_segments
        ]
        avg_angle_degrees = [np.degrees(angle) for angle in avg_angle]
        avg_hand_position = [
            np.mean(segment[len(segment) // 2 :], axis=0)
            for segment in hand_position_segments
        ]

        # # Scatter plot of average hand position
        # avg_hand_position = np.array(avg_hand_position)
        # plt.figure(figsize=(3, 3))
        # plt.scatter(avg_hand_position[:, 0], avg_hand_position[:, 1], c='blue', marker='o', label='Average Hand Position')
        # plt.xlabel("X")
        # plt.ylabel("Y")
        # plt.ylim([-0.8, 0])
        # plt.xlim([0, 0.8])
        # plt.title("Passive Movement of Hand")
        # plt.legend(frameon=False)
        # plt.grid(False)

        # # Save the scatter plot
        # save_path = '/Users/joseph/My Drive/Champalimaud/rotations/Joe/figures/avg_hand_position_scatter.png'
        # plt.savefig(save_path, dpi=900)
        # plt.show()

        # Big raw data plot
        if hasattr(self, "elbow_torque_log"):
            session_fig, (ax2, ax3, ax5) = plt.subplots(
                3, 1, figsize=(8, 5), sharex=True
            )

            # Plot x and y hand (nailed) position over time on the second subplot
            ax2.plot(
                time_sec,
                np.array(self.hand_position_log)[:, 0],
                label="X",
                color="red",
                linewidth=2,
            )
            ax2.plot(
                time_sec,
                np.array(self.hand_position_log)[:, 1],
                label="Y",
                color="blue",
                linewidth=2,
            )
            ax2.set_ylabel("Hand Position")
            ax2.legend(loc="center right", frameon=False)  # Remove grey box
            ax2.spines["top"].set_visible(False)
            ax2.spines["right"].set_visible(False)
            ax2.spines["bottom"].set_visible(False)
            ax2.xaxis.set_ticks_position("none")
            ax2.grid(False)

            # Plot angle on the third subplot
            smoothed_angle = np.convolve(
                np.degrees(self.elbow_angle_log), np.ones(50) / 50, mode="same"
            )  # Simple moving average
            ax3.plot(time_sec, smoothed_angle, color="black", linewidth=2)
            ax3.set_ylabel("Elbow Angle (°)", color="black")
            ax3.tick_params(axis="y", labelcolor="black")
            ax3.spines["top"].set_visible(False)
            ax3.spines["right"].set_visible(False)
            ax3.spines["bottom"].set_visible(False)
            ax3.xaxis.set_ticks_position("none")
            ax3.grid(False)

            # # Plot bicep and tricep activation on the fourth subplot
            # bicep_activation = np.array(self.action_log)[:, 2]
            # tricep_activation = np.array(self.action_log)[:, 3]

            # # Plot raw and smoothed bicep activation
            # ax4.plot(time_sec, bicep_activation, color="purple", alpha=0.5, linewidth=0.5, label="Raw")
            # smoothed_bicep = np.convolve(bicep_activation, np.ones(100)/100, mode='same')  # Larger smoothing window
            # ax4.plot(time_sec, smoothed_bicep, color="purple", linewidth=2, label="Smoothed")

            # # Plot raw and smoothed tricep activation
            # ax4.plot(time_sec, tricep_activation, color="orange", alpha=0.5, linewidth=0.5, label="Tricep (Raw)")
            # smoothed_tricep = np.convolve(tricep_activation, np.ones(100)/100, mode='same')  # Larger smoothing window
            # ax4.plot(time_sec, smoothed_tricep, color="orange", linewidth=2, label="Tricep (Smoothed)")

            # ax4.set_ylabel("Bicep Activation")
            # ax4.legend(loc="upper right", frameon=False)  # Remove grey box
            # ax4.spines['top'].set_visible(False)
            # ax4.spines['right'].set_visible(False)
            # ax4.spines['bottom'].set_visible(False)
            # ax4.xaxis.set_ticks_position('none')
            # ax4.grid(False)

            # Plot torque on the fifth subplot
            ax5.plot(
                time_sec,
                self.elbow_torque_log,
                color="blue",
                alpha=0.5,
                linewidth=0.5,
                label="Raw",
            )
            smoothed_torque = np.convolve(
                self.elbow_torque_log, np.ones(100) / 100, mode="same"
            )  # Larger smoothing window
            ax5.plot(
                time_sec, smoothed_torque, color="blue", linewidth=2, label="Smoothed"
            )
            ax5.set_xlabel("Time (s)")
            ax5.set_ylabel("Elbow Torque (Nm)", color="blue")
            ax5.tick_params(axis="y", labelcolor="blue")
            ax5.legend(loc="center right", frameon=False)  # Remove grey box
            ax5.spines["top"].set_visible(False)
            ax5.spines["right"].set_visible(False)
            ax5.grid(False)

            session_fig.tight_layout()

            file_name = "elbow_torque_angle_passive"
            png_file = os.path.join(save_path, f"{file_name}.png")

            plt.savefig(
                png_file,
                dpi=900,
            )
            plt.show()

        # Open and update Asatryan and Feldman-style figure
        with open(pickle_path, "rb") as f:
            fig = pickle.load(f)
        plt.figure(fig.number)

        # Add the passive movement points to the existing plot.
        # Join points with straight lines in sorted order
        plt.plot(
            sorted_angles, sorted_torques, color="black", linewidth=2, alpha=1, zorder=2
        )

        # self.elbow_torque_log
        # self.elbow_angle_log
