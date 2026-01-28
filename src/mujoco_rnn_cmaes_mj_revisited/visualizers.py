"""
Visualization utilities for SequentialReachingEnv.

This module contains all plotting functionality, separated from the
environment logic for better maintainability.
"""
import matplotlib.pyplot as plt
import numpy as np

MUSCLES = ["deltoid", "latissimus", "biceps", "triceps"]
SENSORS = ["len", "vel", "frc"]


class SequentialReachingVisualizer:
    """Handles all plotting for SequentialReachingEnv"""
    
    def __init__(self, logger_data):
        """
        Initialize visualizer with logged data.
        
        Args:
            logger_data: Dictionary containing logged evaluation data
        """
        if logger_data is None:
            raise ValueError("No logged data provided")
        self.data = logger_data
    
    def plot_all(self, save_path=None):
        """
        Generate all plots.
        
        Args:
            save_path: Optional path to save figures
        """
        self.plot_target_observations()
        self.plot_sensors()
        self.plot_hand_velocity()
        
        if save_path:
            plt.savefig(save_path)
    
    def plot_target_observations(self, linewidth=1):
        """Plot target observations as heatmap with onset markers"""
        plt.figure(figsize=(10, 2))
        ax = plt.gca()
        target_observations = np.array(self.data["target_observations"])

        # Plot as heatmap
        im = ax.imshow(
            target_observations.T,
            aspect="auto",
            interpolation="nearest",
            extent=[
                self.data["time"][0], 
                self.data["time"][-1], 
                0, 
                target_observations.shape[1]
            ],
            origin="lower",
        )

        # Draw target onset lines on top
        target_onset_times = self._get_target_onset_times()
        for t in target_onset_times:
            ax.axvline(x=t, color="red", linestyle="--", linewidth=0.5)

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Observation Index")
        plt.colorbar(im, ax=ax, label="Observation Value")
        plt.tight_layout()
        plt.show()

    def plot_sensors(self, linewidth=1):
        """Plot all sensor data, distance, energy, and reward"""
        _, axes = plt.subplots(3, 2, figsize=(10, 10))
        self._draw_target_lines(axes)
        
        # Plot sensors
        self._plot_sensor_group(
            axes[0, 0],
            self.data["time"],
            self.data["sensors"],
            ["deltoid_len", "latissimus_len", "biceps_len", "triceps_len"],
            "Length",
        )
        self._plot_sensor_group(
            axes[0, 1],
            self.data["time"],
            self.data["sensors"],
            ["deltoid_vel", "latissimus_vel", "biceps_vel", "triceps_vel"],
            "Velocity",
        )
        axes[0, 1].legend(loc="center left", bbox_to_anchor=(1, 0.5))

        self._plot_sensor_group(
            axes[1, 0],
            self.data["time"],
            self.data["sensors"],
            ["deltoid_frc", "latissimus_frc", "biceps_frc", "triceps_frc"],
            "Force",
        )

        # Distance
        axes[1, 1].plot(
            self.data["time"], 
            self.data["distance"], 
            linewidth=linewidth, 
            label="Distance"
        )
        axes[1, 1].set_title("Distance")
        axes[1, 1].set_ylim([-0.05, 2.05])
        axes[1, 1].legend()

        # Energy
        axes[2, 0].plot(
            self.data["time"], 
            self.data["energy"], 
            linewidth=linewidth, 
            label="Energy"
        )
        axes[2, 0].set_title("Energy")
        axes[2, 0].set_ylim([-0.05, 2.05])
        axes[2, 0].legend()

        # Reward / fitness
        axes[2, 1].plot(
            self.data["time"], 
            self.data["reward"], 
            linewidth=linewidth, 
            label="Reward"
        )
        axes[2, 1].set_title("Loss")
        axes[2, 1].set_ylim([-2.05, 0.05])
        ax_right = axes[2, 1].twinx()
        ax_right.plot(
            self.data["time"], 
            self.data["fitness"], 
            color=(0.25, 0.25, 0.25)
        )
        ax_right.set_ylabel("Cumulative Reward", color=(0.25, 0.25, 0.25))
        ax_right.tick_params(axis="y", labelcolor=(0.25, 0.25, 0.25))

        # Axis labels
        for ax in axes.flat:
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Arb.")

        plt.tight_layout()
        plt.show()

    def plot_hand_velocity(self, linewidth=1):
        """Plot hand velocity and distance over time"""
        plt.figure(figsize=(10, 1))
        ax = plt.gca()
        self._draw_target_lines(ax)
        
        hand_positions = np.array(self.data["hand_position"])
        hand_velocities = np.linalg.norm(np.diff(hand_positions, axis=0), axis=1)
        time = np.array(self.data["time"][:-1])
        
        ax.plot(
            time,
            hand_velocities,
            linewidth=linewidth,
            label="Hand Velocity",
            color="black",
        )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Hand velocity (a.u.)")
        
        ax_right = ax.twinx()
        ax_right.plot(
            time,
            self.data["distance"][:-1],
            linewidth=linewidth,
            color="red",
            label="Distance",
        )
        ax_right.set_ylabel("Distance", color="red")
        ax_right.tick_params(axis="y", labelcolor="red")
        plt.show()

    def _get_target_onset_times(self):
        """Extract target onset times from logged data"""
        target_onset_idcs = np.where(
            np.any(
                np.diff(np.array(self.data["target_position"]), axis=0) != 0, 
                axis=1
            )
        )[0]
        target_onset_idcs = np.insert(target_onset_idcs, 0, 0)
        return [self.data["time"][idx] for idx in target_onset_idcs]

    def _draw_target_lines(self, axs):
        """Draw vertical lines at target onset times"""
        target_onset_times = self._get_target_onset_times()
        
        if hasattr(axs, "__iter__"):
            axs = axs.flatten()
        else:
            axs = [axs]
        
        for t in target_onset_times:
            for ax in axs:
                ax.axvline(x=t, color="gray", linestyle="--", linewidth=0.5)

    def _plot_sensor_group(self, ax, time, sensors, keys, title):
        """Plot a group of related sensors"""
        for key in keys:
            ax.plot(time, sensors[key], label=key.replace("_", " ").title())
        ax.set_title(title)

    def _setup_time_series_plot(self, ax, title, ylabel="Arb."):
        """Common setup for time series plots"""
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        return ax
