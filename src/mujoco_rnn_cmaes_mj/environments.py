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
matplotlib.use('Agg')
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
        target_positions = [[-0.33, -0.7, 0.],
                            [-0.15, -0.65, 0],
                            [0, -0.6, 0]]
        print(target_positions)
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
        # elbow_torque_id = self.plant.model.sensor_name2id("elbow_torque")
        
        elbow_angle_log = []
        elbow_torque_log = []
        elbow_torque_sensor_log = []
        

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

            max_pulley_weight = 0.5 * weight_mod
            min_pulley_weight = 1e-3 * weight_mod
            num_unique_weights = 5
            pulley_weights = np.linspace(
                min_pulley_weight, max_pulley_weight, num_unique_weights
            )
            alternating_weights = []
            for i in range(num_unique_weights):
                alternating_weights.append(max_pulley_weight)
                alternating_weights.append(pulley_weights[i])
            num_pulley_weights = len(alternating_weights)

            weight_update_period = target_durations[0] / num_pulley_weights
            target_aligned_time = self.plant.data.time - target_onset_times[target_idx]
            weight_idx = int(np.floor(target_aligned_time / num_pulley_weights))

            # Run the weight update protocol
            if target_aligned_time > weight_update_period * weight_idx:
                weight_mass = alternating_weights[weight_idx]

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
            elbow_torque_log.append(self.plant.data.qfrc_actuator[elbow_joint_id])  # Elbow torque as read out like this.
            elbow_angle_log.append(self.plant.data.qpos[elbow_joint_id])
            elbow_torque_sensor_log.append(self.plant.data.sensordata[-1]) # Elbow torque as read out from a sensor.

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

        self.elbow_torque_log = elbow_torque_log
        self.elbow_angle_log = elbow_angle_log
        self.elbow_torque_sensor_log = elbow_torque_sensor_log
        self.num_pulley_weights = num_pulley_weights
        self.weight_update_period = weight_update_period

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
        #

        timesteps = np.arange(len(self.elbow_torque_log))
        time_sec = timesteps * self.plant.model.opt.timestep

        if hasattr(self, "elbow_torque_log"):
            fig, ax1 = plt.subplots(figsize=(10, 4))

            # Plot torque on the primary y-axis
            ax1.plot(time_sec,self.elbow_torque_log, color='blue', label='Torque')
            ax1.set_xlabel("Timestep")
            ax1.set_ylabel("Torque (Nm)", color='blue')
            ax1.tick_params(axis='y', labelcolor='blue')
            ax1.grid(True)

            # Create secondary y-axis for angle
            ax2 = ax1.twinx()
            ax2.plot(time_sec,self.elbow_angle_log, color='black', label='Angle')
            ax2.set_ylabel("Angle (rad)", color='black')
            ax2.tick_params(axis='y', labelcolor='black')

            plt.title("Elbow")
            fig.tight_layout()
            plt.savefig('/Users/joseph/My Drive/Champalimaud/rotations/Joe/figures/elbow_torque.png', dpi=900)
            plt.show()

            # Split logs based on weight_update_period
            torque_segments = []
            angle_segments = []
            for target_idx in range(self.num_targets):
                for i in range(self.num_pulley_weights):
                    start_time = target_idx * self.num_pulley_weights * self.weight_update_period + i * self.weight_update_period
                    end_time = start_time + self.weight_update_period
                    segment_indices = (time_sec >= start_time) & (time_sec < end_time)
                    torque_segments.append(np.array(self.elbow_torque_log)[segment_indices])
                    angle_segments.append(np.array(self.elbow_angle_log)[segment_indices])

            # Example: Print the number of segments
            print(f"Number of torque segments: {len(torque_segments)}")
            print(f"Number of angle segments: {len(angle_segments)}")
            print(torque_segments)
            
            # Find the average torque and angle for the second half of each segment
            avg_torque = [np.mean(segment[len(segment)//2:]) for segment in torque_segments]
            avg_angle = [np.mean(segment[len(segment)//2:]) for segment in angle_segments]
            avg_angle_degrees = [np.degrees(angle) for angle in avg_angle]
            
            print("Average Torque per Segment:", avg_torque)
            print("Average Angle per Segment:", avg_angle)

            # Scatter plot of avg_torque vs avg_angle
            plt.figure(figsize=(6, 4))
            num_segments_to_plot = len(torque_segments)/self.num_targets
            # plt.scatter(avg_angle_degrees[:int(num_segments_to_plot)], avg_torque[:int(num_segments_to_plot)], color='purple', label='First target')
            # plt.scatter(avg_angle_degrees[int(num_segments_to_plot):], avg_torque[int(num_segments_to_plot):], color='green', label='Second target')
            # plt.scatter(avg_angle_degrees[int(num_segments_to_plot):], avg_torque[int(num_segments_to_plot):], color='red', label='Third target')
            plt.xlabel("Average Angle (degrees)")
            colors = ['purple', 'green', 'red', 'blue', 'orange']  # Add more colors if needed
            for target_idx in range(self.num_targets):
                start_idx = target_idx * int(num_segments_to_plot)
                end_idx = (target_idx + 1) * int(num_segments_to_plot)
                odd_indices = list(range(start_idx + 1, end_idx, 2))
                indices_to_plot = [start_idx] + odd_indices
                plt.scatter([avg_angle_degrees[i] for i in indices_to_plot], 
                            [avg_torque[i] for i in indices_to_plot], 
                            color=colors[target_idx % len(colors)], 
                            label=f'Target {target_idx + 1}',alpha=0.7)
            plt.ylabel("Average Torque (Nm)")
            plt.title("Average Torque vs Average Angle")
            plt.grid(False)
            plt.legend()

            print('angles')
            print(avg_angle_degrees[:int(num_segments_to_plot)])
            print(avg_angle_degrees[int(num_segments_to_plot):])

            # Option to save the figure
            save_path = '/Users/joseph/My Drive/Champalimaud/rotations/Joe/figures/avg_torque_vs_avg_angle_2.png'
            plt.savefig(save_path, dpi=900)
            print(f"Scatter plot saved to {save_path}")

            plt.show()

















        if hasattr(self, "elbow_torque_sensor_log"):
            fig, ax1 = plt.subplots(figsize=(10, 4))

            # Plot torque on the primary y-axis
            ax1.plot(time_sec, self.elbow_torque_sensor_log, color='blue', label='Torque')
            ax1.set_xlabel("Timestep")
            ax1.set_ylabel("Torque (Nm)", color='blue')
            ax1.tick_params(axis='y', labelcolor='blue')
            ax1.grid(True)

            # Create secondary y-axis for angle
            ax2 = ax1.twinx()
            ax2.plot(time_sec, self.elbow_angle_log, color='black', label='Angle')
            ax2.set_ylabel("Angle (rad)", color='black')
            ax2.tick_params(axis='y', labelcolor='black')

            plt.title("Elbow")
            fig.tight_layout()
            plt.savefig('/Users/joseph/My Drive/Champalimaud/rotations/Joe/figures/elbow_torque.png', dpi=900)
            plt.show()

            # Split logs based on weight_update_period
            torque_segments = []
            angle_segments = []
            for target_idx in range(self.num_targets):
                for i in range(self.num_pulley_weights):
                    start_time = target_idx * self.num_pulley_weights * self.weight_update_period + i * self.weight_update_period
                    end_time = start_time + self.weight_update_period
                    segment_indices = (time_sec >= start_time) & (time_sec < end_time)
                    torque_segments.append(np.array(self.elbow_torque_sensor_log)[segment_indices])
                    angle_segments.append(np.array(self.elbow_angle_log)[segment_indices])

            # Example: Print the number of segments
            print(f"Number of torque segments: {len(torque_segments)}")
            print(f"Number of angle segments: {len(angle_segments)}")
            print(torque_segments)

            # Find the average torque and angle for the second half of each segment
            avg_torque = [np.mean(segment[len(segment)//2:]) for segment in torque_segments]
            avg_angle = [np.mean(segment[len(segment)//2:]) for segment in angle_segments]
            avg_angle_degrees = [np.degrees(angle) for angle in avg_angle]

            print("Average Torque per Segment:", avg_torque)
            print("Average Angle per Segment:", avg_angle)

            # Scatter plot of avg_torque vs avg_angle
            plt.figure(figsize=(6, 4))
            num_segments_to_plot = len(torque_segments) / self.num_targets
            plt.xlabel("Average Angle (degrees)")
            colors = ['purple', 'green', 'red', 'blue', 'orange']  # Add more colors if needed
            for target_idx in range(self.num_targets):
                start_idx = target_idx * int(num_segments_to_plot)
                end_idx = (target_idx + 1) * int(num_segments_to_plot)
                odd_indices = list(range(start_idx + 1, end_idx, 2))
                indices_to_plot = [start_idx] + odd_indices
                plt.scatter([avg_angle_degrees[i] for i in indices_to_plot],
                            [avg_torque[i] for i in indices_to_plot],
                            color=colors[target_idx % len(colors)],
                            label=f'Target {target_idx + 1}', alpha=0.7)
            plt.ylabel("Average Torque (Nm)")
            plt.title("Average Torque vs Average Angle")
            plt.grid(False)
            plt.legend()

            print('angles')
            print(avg_angle_degrees[:int(num_segments_to_plot)])
            print(avg_angle_degrees[int(num_segments_to_plot):])

            # Option to save the figure
            save_path = '/Users/joseph/My Drive/Champalimaud/rotations/Joe/figures/avg_torque_vs_avg_angle_3.png'
            plt.savefig(save_path, dpi=900)
            print(f"Scatter plot saved to {save_path}")

            plt.show()



        # log = self.logger

        # _, axes = plt.subplots(3, 2, figsize=(10, 10))

        # # Targets
        # target_onset_idcs = np.where(
        #     np.any(np.diff(np.array(log["target_position"]), axis=0) != 0, axis=1)
        # )[0]
        # target_onset_idcs = np.insert(target_onset_idcs, 0, 0)
        # target_onset_times = np.array(
        #     [log["time"][target_onset_idx] for target_onset_idx in target_onset_idcs]
        # )
        # for t in target_onset_times:
        #     axes[0, 0].axvline(x=t, color="gray", linestyle="--", linewidth=0.5)
        #     axes[0, 1].axvline(x=t, color="gray", linestyle="--", linewidth=0.5)
        #     axes[1, 0].axvline(x=t, color="gray", linestyle="--", linewidth=0.5)
        #     axes[1, 1].axvline(x=t, color="gray", linestyle="--", linewidth=0.5)
        #     axes[2, 0].axvline(x=t, color="gray", linestyle="--", linewidth=0.5)
        #     axes[2, 1].axvline(x=t, color="gray", linestyle="--", linewidth=0.5)

        # linewidth = 1

        # # Length
        # axes[0, 0].plot(log["time"], log["sensors"]["deltoid_len"], label="Deltoid")
        # axes[0, 0].plot(
        #     log["time"],
        #     log["sensors"]["latissimus_len"],
        #     linewidth=linewidth,
        #     label="Latissimus",
        # )
        # axes[0, 0].plot(
        #     log["time"],
        #     log["sensors"]["biceps_len"],
        #     linewidth=linewidth,
        #     label="Biceps",
        # )
        # axes[0, 0].plot(
        #     log["time"],
        #     log["sensors"]["triceps_len"],
        #     linewidth=linewidth,
        #     label="Triceps",
        # )
        # axes[0, 0].set_title("Length")

        # # Velocity
        # axes[0, 1].plot(
        #     log["time"],
        #     log["sensors"]["deltoid_vel"],
        #     linewidth=linewidth,
        #     label="Deltoid",
        # )
        # axes[0, 1].plot(
        #     log["time"],
        #     log["sensors"]["latissimus_vel"],
        #     linewidth=linewidth,
        #     label="Latissimus",
        # )
        # axes[0, 1].plot(
        #     log["time"],
        #     log["sensors"]["biceps_vel"],
        #     linewidth=linewidth,
        #     label="Biceps",
        # )
        # axes[0, 1].plot(
        #     log["time"],
        #     log["sensors"]["triceps_vel"],
        #     linewidth=linewidth,
        #     label="Triceps",
        # )
        # axes[0, 1].set_title("Velocity")
        # axes[0, 1].legend(loc="center left", bbox_to_anchor=(1, 0.5))

        # # Force
        # axes[1, 0].plot(
        #     log["time"],
        #     log["sensors"]["deltoid_frc"],
        #     linewidth=linewidth,
        #     label="Deltoid",
        # )
        # axes[1, 0].plot(
        #     log["time"],
        #     log["sensors"]["latissimus_frc"],
        #     linewidth=linewidth,
        #     label="Latissimus",
        # )
        # axes[1, 0].plot(
        #     log["time"],
        #     log["sensors"]["biceps_frc"],
        #     linewidth=linewidth,
        #     label="Biceps",
        # )
        # axes[1, 0].plot(
        #     log["time"],
        #     log["sensors"]["triceps_frc"],
        #     linewidth=linewidth,
        #     label="Triceps",
        # )
        # axes[1, 0].set_title("Force")

        # # Distance
        # axes[1, 1].plot(
        #     log["time"],
        #     log["manhattan_distance"],
        #     linewidth=linewidth,
        #     label="Manhattan",
        # )
        # axes[1, 1].plot(
        #     log["time"],
        #     log["euclidean_distance"],
        #     linewidth=linewidth,
        #     label="Euclidean",
        # )
        # axes[1, 1].set_title("Distance")
        # axes[1, 1].set_ylim([-0.05, 2.05])
        # axes[1, 1].legend()

        # # Energy
        # axes[2, 0].plot(log["time"], log["entropy"], linewidth=0.1, label="Entropy")
        # axes[2, 0].plot(log["time"], log["energy"], linewidth=0.1, label="Energy")
        # axes[2, 0].set_title("Energy")
        # axes[2, 0].set_ylim([-0.05, 2.05])
        # axes[2, 0].legend()

        # # Fitness
        # axes[2, 1].plot(log["time"], log["reward"], linewidth=linewidth, label="Reward")
        # axes[2, 1].set_title("Loss")
        # axes[2, 1].set_ylim([-2.05, 0.05])

        # # Create a twin axis (right y-axis)
        # r, g, b = np.array([1, 1, 1]) * 0.25
        # fitness_clr = (r, g, b)
        # ax_right = axes[2, 1].twinx()
        # ax_right.plot(log["time"], log["fitness"], color=fitness_clr)
        # ax_right.set_ylabel("Cumulative Reward", color=fitness_clr)
        # ax_right.tick_params(axis="y", labelcolor=fitness_clr)

        # # Set axis labels
        # for ax in axes.flat:
        #     ax.set_xlabel("Time (s)")
        #     ax.set_ylabel("Arb.")

        # plt.tight_layout()
        # plt.show()

        # # Hand Velocity
        # plt.figure(figsize=(10, 1))
        # # Annotate target change times with vertical lines
        # for idx in target_onset_idcs:
        #     plt.axvline(
        #         x=log["time"][idx],
        #         color="blue",
        #         linestyle="--",
        #         linewidth=0.5,
        #         label="Target Change" if idx == target_onset_idcs[0] else None,
        #     )
        # hand_positions = np.array(log["hand_position"])
        # hand_velocities = np.linalg.norm(np.diff(hand_positions, axis=0), axis=1)
        # time = np.array(
        #     log["time"][:-1]
        # )  # Exclude the last time step to match velocity array length
        # plt.plot(
        #     time,
        #     hand_velocities,
        #     linewidth=linewidth,
        #     label="Hand Velocity",
        #     color="black",
        # )
        # plt.xlabel("Time (s)")
        # plt.ylabel("Hand velocity (a.u.)")
        # ax_right = plt.gca().twinx()  # Create a twin axis (right y-axis)
        # ax_right.plot(
        #     log["time"][:-1],
        #     log["euclidean_distance"][:-1],
        #     linewidth=linewidth,
        #     label="Euclidean Distance",
        #     color="red",
        # )
        # ax_right.set_ylabel("Euclidean Distance", color="red")
        # ax_right.tick_params(axis="y", labelcolor="red")

        # self.logger = None
