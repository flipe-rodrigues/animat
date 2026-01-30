"""
Gymnasium environment for musculoskeletal arm reaching task.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
from mujoco import viewer
import os
from typing import Optional, Tuple, Dict, Any

from utils.place_cells import PlaceCellGrid


class ReachingEnv(gym.Env):
    """
    Environment for reaching task with musculoskeletal arm.

    Task: Reach random targets and hold position for random duration.
    Observations: Muscle proprioception (length, velocity, force) + place cell target encoding
    Actions: Muscle activations (4 muscles)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        model_path: str = "models/arm.xml",
        render_mode: Optional[str] = None,
        hold_time_range: Tuple[float, float] = (0.2, 0.8),
        reach_threshold: float = 0.03,
        hold_threshold: float = 0.04,
        max_episode_steps: int = 1000,
        workspace_bounds: Tuple[Tuple[float, float], Tuple[float, float]] = (
            (-0.5, 0.5),
            (-0.7, 0.3),
        ),
        place_cell_grid_size: Tuple[int, int] = (8, 8),
        place_cell_sigma: float = 0.08,
    ):
        """
        Initialize the environment.

        Args:
            model_path: Path to MuJoCo XML model
            render_mode: Rendering mode
            hold_time_range: (min, max) time to hold at target (seconds)
            reach_threshold: Distance threshold to consider target reached (m)
            hold_threshold: Distance threshold while holding (m)
            max_episode_steps: Maximum steps per episode
            workspace_bounds: ((x_min, x_max), (z_min, z_max))
            place_cell_grid_size: Grid dimensions for place cells
            place_cell_sigma: Gaussian sigma for place cells
        """
        super().__init__()

        self.render_mode = render_mode
        self.hold_time_range = hold_time_range
        self.reach_threshold = reach_threshold
        self.hold_threshold = hold_threshold
        self.max_episode_steps = max_episode_steps
        self.workspace_bounds = workspace_bounds

        # Load MuJoCo model
        if not os.path.isabs(model_path):
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(base_dir, model_path)

        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        self._parse_sensors()
        self._parse_muscles()

        self.target_id = self.model.geom("target").id
        self.hand_id = self.model.geom("hand").id

        # Initialize place cell grid for target encoding
        self.place_cell_grid = PlaceCellGrid(
            workspace_bounds=workspace_bounds,
            grid_size=place_cell_grid_size,
            sigma=place_cell_sigma,
        )

        # Action space: 4 muscle activations [0, 1]
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.num_muscles,), dtype=np.float32
        )

        # Observation space:
        # - Muscle observations: 4 muscles × 3 (length, velocity, force) = 12
        # - Place cell encoding of target: grid_size[0] × grid_size[1]
        # - Time remaining to hold: 1
        obs_dim = self.num_sensors + self.place_cell_grid.num_cells + 1
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Task state
        self.target_position = np.zeros(2)
        self.required_hold_time = 0.0
        self.time_at_target = 0.0
        self.reached_target = False
        self.steps = 0

        # Rendering
        if self.render_mode == "human":
            self.viewer = viewer.launch_passive(self.model, self.data)
            self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
            self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = True
            self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONSTRAINT] = True
            self.viewer.cam.lookat[:] = [0, -0.25, 0]
            self.viewer.cam.azimuth = 90
            self.viewer.cam.elevation = -90
        else:
            self.viewer = None

    def _parse_muscles(self):
        """Parse actuator IDs for muscles."""
        self.num_muscles = self.model.nu

    def _parse_sensors(self):
        """Parse sensor IDs for muscle observations and hand position."""
        self.sensor_ids_len = []
        self.sensor_ids_vel = []
        self.sensor_ids_frc = []
        self.num_sensors = self.model.nsensor
        for i in range(self.num_sensors):
            sensor_type = self.model.sensor_type[i]
            if sensor_type == mujoco.mjtSensor.mjSENS_ACTUATORPOS:
                self.sensor_ids_len.append(i)
            elif sensor_type == mujoco.mjtSensor.mjSENS_ACTUATORVEL:
                self.sensor_ids_vel.append(i)
            elif sensor_type == mujoco.mjtSensor.mjSENS_ACTUATORFRC:
                self.sensor_ids_frc.append(i)
        self.num_sensors_len = len(self.sensor_ids_len)
        self.num_sensors_vel = len(self.sensor_ids_vel)
        self.num_sensors_frc = len(self.sensor_ids_frc)

    def _get_obs(self) -> np.ndarray:
        """Construct observation from current state."""
        muscle_lengths = self.data.sensordata[self.sensor_ids_len]
        muscle_velocities = self.data.sensordata[self.sensor_ids_vel]
        muscle_forces = self.data.sensordata[self.sensor_ids_frc]
        muscle_obs = np.concatenate([muscle_lengths, muscle_velocities, muscle_forces])

        # Get place cell encoding of target
        target_encoding = self.place_cell_grid.encode(self.target_position)

        # Time remaining to hold (normalized)
        if self.reached_target:
            time_remaining = max(0.0, self.required_hold_time - self.time_at_target)
        else:
            time_remaining = self.required_hold_time
        time_remaining_normalized = time_remaining / self.hold_time_range[1]

        # Combine all observations
        obs = np.concatenate(
            [muscle_obs, target_encoding, [time_remaining_normalized]]
        ).astype(np.float32)

        return obs

    def _get_hand_position(self) -> np.ndarray:
        """Get current hand position (x, y)."""
        return self.data.geom_xpos[self.hand_id][:2]

    def _sample_target(self) -> np.ndarray:
        """Sample a random target within workspace."""
        x = np.random.uniform(*self.workspace_bounds[0])
        y = np.random.uniform(*self.workspace_bounds[1])
        return np.array([x, y])

    def _sample_initial_configuration(self) -> np.ndarray:
        """Sample random initial joint configuration."""
        shoulder_angle = np.random.uniform(np.deg2rad(-60), np.deg2rad(60))
        elbow_angle = np.random.uniform(np.deg2rad(-60), np.deg2rad(60))
        return np.array([shoulder_angle, elbow_angle])

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment."""
        super().reset(seed=seed)

        # Reset MuJoCo simulation
        mujoco.mj_resetData(self.model, self.data)

        # Set random initial joint configuration
        initial_config = self._sample_initial_configuration()
        self.data.qpos[:2] = initial_config

        # Sample new target and required hold time
        self.target_position = self._sample_target()
        self.required_hold_time = np.random.uniform(*self.hold_time_range)

        # Update target visualization in MuJoCo
        self.data.mocap_pos[0] = [
            self.target_position[0],
            self.target_position[1],
            0,
        ]

        # Reset task state
        self.time_at_target = 0.0
        self.reached_target = False
        self.steps = 0

        # Forward simulation to update sensors
        mujoco.mj_forward(self.model, self.data)

        obs = self._get_obs()
        info = {
            "target_position": self.target_position.copy(),
            "required_hold_time": self.required_hold_time,
        }

        return obs, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one environment step."""
        # Apply muscle activations
        self.data.ctrl[:] = np.clip(action, 0.0, 1.0)

        # Step simulation
        mujoco.mj_step(self.model, self.data)
        self.steps += 1

        # Get hand position and distance to target
        hand_pos = self._get_hand_position()
        distance_to_target = np.linalg.norm(hand_pos - self.target_position)

        # Check if hand is at target
        if not self.reached_target:
            if distance_to_target < self.reach_threshold:
                self.reached_target = True
                self.time_at_target = 0.0

        # Update hold timer if at target
        if self.reached_target:
            if distance_to_target < self.hold_threshold:
                self.time_at_target += self.model.opt.timestep
            else:
                # Left target region, reset
                self.reached_target = False
                self.time_at_target = 0.0

        # Compute reward
        reward = self._compute_reward(distance_to_target)

        # Check termination conditions
        success = self.reached_target and self.time_at_target >= self.required_hold_time
        truncated = self.steps >= self.max_episode_steps
        terminated = success

        obs = self._get_obs()
        info = {
            "distance_to_target": distance_to_target,
            "time_at_target": self.time_at_target,
            "reached_target": self.reached_target,
            "success": success,
            "hand_position": hand_pos,
            "target_position": self.target_position,
        }

        if self.render_mode == "human" and self.viewer is not None:
            self.render()

        return obs, reward, terminated, truncated, info

    def _compute_reward(self, distance_to_target: float) -> float:
        """Compute reward for current state."""
        reward = 0.0

        # Distance penalty (encourage moving toward target)
        distance_penalty = distance_to_target * 2.0
        reward -= distance_penalty

        # Bonus for being at target
        if distance_to_target < self.reach_threshold:
            reward += 5.0

        # Bonus for holding at target
        if self.reached_target and distance_to_target < self.hold_threshold:
            reward += 10.0 * self.model.opt.timestep

        # Large bonus for successfully completing hold
        if self.reached_target and self.time_at_target >= self.required_hold_time:
            reward += 100.0

        # Energy penalty (discourage excessive muscle activation)
        energy_penalty = 0.01 * np.sum(np.square(self.data.ctrl))
        reward -= energy_penalty

        return reward

    def render(self):
        """Render the environment."""
        if self.render_mode == "human" and self.viewer is not None:
            self.viewer.sync()
        elif self.render_mode == "rgb_array":
            if not hasattr(self, "renderer"):
                self.renderer = mujoco.Renderer(self.model, 480, 640)

            self.renderer.update_scene(self.data)
            return self.renderer.render()

    def close(self):
        """Clean up resources."""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
