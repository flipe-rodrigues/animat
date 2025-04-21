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
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import mujoco
from functools import partial
from tqdm import tqdm
from plants import SequentialReacher, get_obs_jit, get_obs_batch
from networks import RNN, rnn_step, parallel_rnn_step
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
        
        # Precompute JAX arrays for normalization
        self.target_mean = jnp.array(self.plant.hand_position_stats["mean"].values)
        self.target_std = jnp.array(self.plant.hand_position_stats["std"].values)
        self.sensor_mean = jnp.array(self.plant.sensor_stats["mean"].values)
        self.sensor_std = jnp.array(self.plant.sensor_stats["std"].values)

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
        mjx_data = self.plant.reset()

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
        mjx_data = self.plant.update_target_mjx(mjx_data, target_positions[target_idx])

        while target_idx < self.num_targets:
            if render:
                self.plant.render()

            # Get observation
            obs = get_obs_jit(
                mjx_data, 
                self.target_mean,
                self.target_std,
                self.sensor_mean,
                self.sensor_std
            )
            
            # RNN step
            rnn.h, rnn.out = rnn_step(
                rnn.get_params(),
                rnn.h,
                rnn.out,
                obs,
                rnn.alpha,
                rnn.activation
            )
            action = rnn.out
            
            # Environment step
            mjx_data = self.plant.step_mjx(mjx_data, action)
            
            hand_position = self.plant.get_hand_pos(mjx_data)
            target_position = target_positions[target_idx]
            manhattan_distance = l1_norm(target_position - hand_position)
            euclidean_distance = l2_norm(target_position - hand_position)
            energy = jnp.mean(action)
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
                    time=mjx_data.time,
                    sensors=mjx_data.sensordata,
                    target_position=target_position,
                    hand_position=hand_position,
                    manhattan_distance=manhattan_distance,
                    euclidean_distance=euclidean_distance,
                    energy=energy,
                    entropy=entropy,
                    reward=reward,
                    fitness=total_reward / trial_duration,
                )

            if mjx_data.time > target_offset_times[target_idx]:
                target_idx += 1
                if target_idx < self.num_targets:
                    mjx_data = self.plant.update_target_mjx(mjx_data, target_positions[target_idx])

        self.plant.close()

        return total_reward / trial_duration

    """
    .########.....###....########.....###....##.......##.......########.##......
    .##.....##...##.##...##.....##...##.##...##.......##.......##.......##......
    .##.....##..##...##..##.....##..##...##..##.......##.......##.......##......
    .########..##.....##.########..##.....##.##.......##.......######...##......
    .##........#########.##...##...#########.##.......##.......##.......##......
    .##........##.....##.##....##..##.....##.##.......##.......##.......##......
    .##........##.....##.##.....##.##.....##.########.########.########.########
    """

    def evaluate_parallel(self, rnn_params_batch, num_envs, seeds=None):
        """Evaluate fitness of multiple RNN policies in parallel"""
        if seeds is None:
            seeds = jnp.arange(num_envs)
        
        # Get parameters for the RNN setup from the first policy
        # We need these dimensions to set up the network states
        first_params = rnn_params_batch[0]
        # Calculate hidden and output sizes based on parameters size
        input_size = 3 + self.plant.num_sensors  # Consistent with RNN initialization
        total_params = first_params.shape[0]
        
        # Determine size of intermediate matrices
        # This is based on RNN param layout in networks.py
        param_size = total_params
        hidden_size = int(jnp.sqrt(
            -input_size + jnp.sqrt(input_size**2 + 4*param_size) 
        ) / 2)
        output_size = self.plant.num_actuators
        
        # Get alpha value (same for all environments)
        alpha = self.plant.mj_model.opt.timestep / 10e-3  # Same as in main.py
        
        # Initialize RNN states - shape (num_envs, hidden_size)
        hidden_states = jnp.zeros((num_envs, hidden_size))
        output_states = jnp.zeros((num_envs, output_size))
        
        # Reset environments
        reset_fn = jax.vmap(lambda _: self.plant.reset_mjx_data(self.plant.mjx_model))
        mjx_data_batch = reset_fn(jnp.arange(num_envs))
        
        # Prepare target positions and durations
        target_positions_batch = self.plant.get_targets_batch(num_envs, self.num_targets)
        
        # Use the same random key for all environments
        key = jax.random.PRNGKey(42)
        
        # Generate target durations for all environments - use JAX for consistency
        def get_durations(idx):
            # Create a different key for each environment using idx as the seed
            subkey = jax.random.fold_in(key, idx)
            durations = jax.random.exponential(
                subkey, 
                shape=(self.num_targets,)
            ) * self.target_duration["mean"]
            # Clip to min/max bounds
            return jnp.clip(
                durations, 
                self.target_duration["min"], 
                self.target_duration["max"]
            )
        
        target_durations_batch = jax.vmap(get_durations)(jnp.arange(num_envs))
        target_offset_times_batch = jnp.cumsum(target_durations_batch, axis=1)
        trial_durations = jnp.sum(target_durations_batch, axis=1)
        
        # Initialize target indices and rewards
        target_indices = jnp.zeros(num_envs, dtype=jnp.int32)
        total_rewards = jnp.zeros(num_envs)
        
        # Set initial targets
        initial_targets = target_positions_batch[:, 0]
        mjx_data_batch = self.plant.batch_update_target(mjx_data_batch, initial_targets)
        
        # Pre-broadcast the normalization arrays to match batch size
        # This ensures all arrays have the same batch dimension
        target_mean_batch = jnp.tile(self.target_mean, (num_envs, 1))
        target_std_batch = jnp.tile(self.target_std, (num_envs, 1))
        sensor_mean_batch = jnp.tile(self.sensor_mean, (num_envs, 1))
        sensor_std_batch = jnp.tile(self.sensor_std, (num_envs, 1))
        
        # Define evaluation loop
        def eval_step(carry, _):
            mjx_data_b, hidden_s, output_s, target_idx, total_r = carry
            
            # Get observations - Now using properly broadcasted arrays
            observations = get_obs_batch(
                mjx_data_b,
                target_mean_batch,
                target_std_batch,
                sensor_mean_batch,
                sensor_std_batch
            )
            
            # RNN step - must use tanh for activation as in main.py
            new_hidden_s, new_output_s = parallel_rnn_step(
                rnn_params_batch, 
                hidden_s,
                output_s,
                observations,
                alpha,
                tanh
            )
            
            # Environment step
            new_mjx_data_b = self.plant.batch_step(self.plant.mjx_model, mjx_data_b, new_output_s)
            
            # Calculate rewards
            # Use JAX vmap to get hand positions for all environments
            hand_positions = jax.vmap(lambda data: data.geom_xpos[self.plant.hand_id])(new_mjx_data_b)
            
            # Get current targets for each environment (vectorized)
            # Use advanced indexing to get the current targets
            current_targets = jnp.take_along_axis(
                target_positions_batch, 
                target_idx[:, jnp.newaxis, jnp.newaxis], 
                axis=1
            ).squeeze(axis=1)
            
            # Calculate distances and energies
            manhattan_distances = jax.vmap(l1_norm)(current_targets - hand_positions)
            euclidean_distances = jax.vmap(l2_norm)(current_targets - hand_positions)
            energies = jnp.mean(new_output_s, axis=1)
            entropies = jax.vmap(action_entropy)(new_output_s)
            
            # Calculate rewards
            rewards = -(
                euclidean_distances * self.loss_weights["euclidean"]
                + manhattan_distances * self.loss_weights["manhattan"]
                + energies * entropies * self.loss_weights["energy"]
                + jax.vmap(lambda p: l1_norm(p * self.loss_weights["ridge"]))(rnn_params_batch)
                + jax.vmap(lambda p: l2_norm(p * self.loss_weights["lasso"]))(rnn_params_batch)
            )
            
            # Update total rewards
            new_total_r = total_r + rewards
            
            # Determine if we need to update targets
            # Compare current time to target offset times for each environment
            times = new_mjx_data_b.time
            
            # Check if we need to advance to the next target for each environment
            should_advance = (times > jnp.take_along_axis(
                target_offset_times_batch,
                target_idx[:, jnp.newaxis],
                axis=1
            ).squeeze()) & (target_idx < self.num_targets - 1)
            
            # Increment target indices where needed
            new_target_idx = target_idx + jnp.where(should_advance, 1, 0)
            
            # Update targets where indices have changed
            # Only if targets should be updated
            next_targets = jnp.take_along_axis(
                target_positions_batch,
                new_target_idx[:, jnp.newaxis, jnp.newaxis],
                axis=1
            ).squeeze(axis=1)
            
            # Create new mocap positions with updated targets
            new_mocap_pos = new_mjx_data_b.mocap_pos.at[:, 0].set(
                jnp.where(
                    should_advance[:, jnp.newaxis],
                    next_targets,
                    new_mjx_data_b.mocap_pos[:, 0]
                )
            )
            
            # Update the MJX data with new targets
            new_mjx_data_b = new_mjx_data_b.replace(mocap_pos=new_mocap_pos)
            
            return (new_mjx_data_b, new_hidden_s, new_output_s, new_target_idx, new_total_r), None
        
        # Maximum number of steps
        max_timesteps = int(jnp.max(trial_durations) / self.plant.mj_model.opt.timestep) + 1
        
        # Run evaluation loop
        (final_mjx_data, _, _, _, final_rewards), _ = jax.lax.scan(
            eval_step,
            (mjx_data_batch, hidden_states, output_states, target_indices, total_rewards),
            None,
            length=max_timesteps
        )
        
        # Normalize rewards by trial duration
        normalized_rewards = final_rewards / trial_durations
        
        return normalized_rewards

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
