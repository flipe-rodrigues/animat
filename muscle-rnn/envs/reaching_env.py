"""
Gymnasium Environment for Muscle-Driven Arm Target Reaching

Features:
- Random target positions within workspace
- Random initial joint configurations
- Random delays before target onset and after hold
- Hold requirement at target position
- Proprioceptive observations from muscle sensors
- Target encoding via grid-based Gaussian tuned units
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.model_parser import (
    parse_mujoco_xml,
    ParsedModel,
    infer_muscle_sensor_mapping,
)


@dataclass
class TrialConfig:
    """Configuration for a single trial."""

    pre_target_delay: float  # seconds before target appears
    hold_duration: float  # seconds to hold at target
    post_hold_delay: float  # seconds after successful hold
    target_position: np.ndarray  # xyz position
    initial_joint_angles: np.ndarray  # initial joint configuration


class ReachingEnv(gym.Env):
    """
    Gymnasium environment for a muscle-driven arm reaching task.

    Observations:
        - Proprioceptive: muscle length, velocity, force (normalized)
        - Target: encoded via grid of Gaussian-tuned spatial units
        - Phase: one-hot encoding of current trial phase
        - Time: normalized time within trial phases

    Actions:
        - Alpha motor neuron activations (direct muscle control) [0, 1]
        
    Note: Gamma modulation of sensory signals is handled internally by the
    controller, not by the environment. The environment provides raw 
    normalized sensor data.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        xml_path: str,
        render_mode: Optional[str] = None,
        # Timing parameters (in seconds)
        pre_delay_range: Tuple[float, float] = (0.2, 0.5),
        hold_duration_range: Tuple[float, float] = (0.3, 0.8),
        post_delay_range: Tuple[float, float] = (0.1, 0.3),
        max_episode_time: float = 3.0,
        # Target encoding parameters
        target_grid_size: int = 5,  # NxN grid in XY plane
        target_grid_sigma: float = 0.1,  # Gaussian width
        workspace_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        # Reward parameters
        reach_threshold: float = 0.05,  # 5cm
        hold_reward_weight: float = 1.0,
        energy_penalty_weight: float = 0.01,
        # Sensor normalization (will be populated during calibration)
        sensor_stats: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        self.xml_path = xml_path
        self.render_mode = render_mode

        # Parse the model
        self.parsed_model = parse_mujoco_xml(xml_path)
        self.sensor_mapping = infer_muscle_sensor_mapping(self.parsed_model)

        # Load MuJoCo model
        self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self.mj_data = mujoco.MjData(self.mj_model)

        # Get timestep
        self.dt = self.mj_model.opt.timestep

        # Store parameters
        self.pre_delay_range = pre_delay_range
        self.hold_duration_range = hold_duration_range
        self.post_delay_range = post_delay_range
        self.max_episode_time = max_episode_time
        self.reach_threshold = reach_threshold
        self.hold_reward_weight = hold_reward_weight
        self.energy_penalty_weight = energy_penalty_weight

        # Get model dimensions
        self.num_muscles = self.parsed_model.n_muscles
        self.num_joints = self.parsed_model.n_joints

        # Setup target encoding grid
        self.target_grid_size = target_grid_size
        self.target_grid_sigma = target_grid_sigma
        self.num_target_units = target_grid_size * target_grid_size

        # Find relevant body indices
        self._parse_task_relevant_ids()

        # Determine workspace bounds
        if workspace_bounds is None:
            self.workspace_bounds = self._estimate_workspace()
        else:
            self.workspace_bounds = workspace_bounds

        # Create target grid positions
        self._setup_target_grid()

        # Sensor normalization stats
        self.sensor_stats = sensor_stats or {
            "length_mean": np.zeros(self.num_muscles),
            "length_std": np.ones(self.num_muscles),
            "velocity_mean": np.zeros(self.num_muscles),
            "velocity_std": np.ones(self.num_muscles),
            "force_mean": np.zeros(self.num_muscles),
            "force_std": np.ones(self.num_muscles),
        }

        # Define observation space
        # Proprioceptive: 3 values per muscle (length, velocity, force) * n_muscles
        # Target: n_target_units (grid encoding)
        # No phase or time - RNN learns temporal dynamics internally
        proprioceptive_dim = self.num_muscles * 3
        target_dim = self.num_target_units

        obs_dim = proprioceptive_dim + target_dim

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Define action space
        # Alpha MNs only: n_muscles (muscle activations in [0, 1])
        # Gamma modulation is handled internally by the controller
        action_dim = self.num_muscles

        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(action_dim,), dtype=np.float32
        )

        # Trial state
        self.trial_config: Optional[TrialConfig] = None
        self.episode_step = 0
        self.episode_time = 0.0
        self.phase = "pre_delay"  # pre_delay, reach, hold, post_delay, done
        self.hold_start_time: Optional[float] = None
        self.target_visible = False

        # Rendering
        self.viewer = None
        self.renderer = None

    def _parse_task_relevant_ids(self):
        """Find indices for hand and target bodies."""
        self.hand_geom_id = self.mj_model.geom("hand").id
        self.target_body_id = self.mj_model.body("target").id
        self.target_mocap_id = self.mj_model.body_mocapid[self.target_body_id]

    def _estimate_workspace(self) -> Dict[str, Tuple[float, float]]:
        """Estimate the workspace bounds by sampling joint configurations."""
        # Get joint limits
        joint_ranges = []
        for i, joint in enumerate(self.parsed_model.joints):
            if joint.range is not None:
                joint_ranges.append(np.deg2rad(joint.range))
            else:
                joint_ranges.append((-np.pi, np.pi))

        # Sample many configurations and find hand positions
        n_samples = 1000
        positions = []

        for _ in range(n_samples):
            # Random joint angles
            qpos = np.array(
                [np.random.uniform(low, high) for low, high in joint_ranges]
            )

            # Set joint positions
            self.mj_data.qpos[: self.num_joints] = qpos
            mujoco.mj_forward(self.mj_model, self.mj_data)

            # Get hand position
            hand_pos = self._get_hand_position()
            positions.append(hand_pos)

        positions = np.array(positions)

        # Compute bounds with some margin
        margin = 0.05
        bounds = {
            "x": (positions[:, 0].min() - margin, positions[:, 0].max() + margin),
            "y": (positions[:, 1].min() - margin, positions[:, 1].max() + margin),
            "z": (positions[:, 2].min() - margin, positions[:, 2].max() + margin),
        }

        return bounds

    def _setup_target_grid(self):
        """Setup the grid of Gaussian-tuned target encoding units."""
        x_range = np.linspace(
            self.workspace_bounds["x"][0],
            self.workspace_bounds["x"][1],
            self.target_grid_size,
        )
        y_range = np.linspace(
            self.workspace_bounds["y"][0],
            self.workspace_bounds["y"][1],
            self.target_grid_size,
        )

        # Create grid positions (XY plane)
        self.grid_positions = []
        for x in x_range:
            for y in y_range:
                self.grid_positions.append(np.array([x, y, 0.0]))
        self.grid_positions = np.array(self.grid_positions)

    def _get_hand_position(self) -> np.ndarray:
        """Get current hand position."""
        return self.mj_data.xpos[self.hand_geom_id].copy()

    def _set_target_position(self, position: np.ndarray):
        """Set the target (mocap) body position."""
        self.mj_data.mocap_pos[self.target_mocap_id] = position

    def _encode_target(self, target_pos: np.ndarray) -> np.ndarray:
        """
        Encode target position using grid of Gaussian-tuned units.

        Each unit fires according to a 2D Gaussian PDF based on distance
        from its grid position to the target (in XY plane).
        """
        if not self.target_visible:
            return np.zeros(self.num_target_units)

        # Compute distances in XY plane
        target_xy = target_pos[:2]
        grid_xy = self.grid_positions[:, :2]

        distances = np.linalg.norm(grid_xy - target_xy, axis=1)

        # Gaussian activation
        activations = np.exp(-0.5 * (distances / self.target_grid_sigma) ** 2)

        return activations

    def _get_proprioceptive_obs(self) -> np.ndarray:
        """
        Get proprioceptive observations from muscle sensors.

        Returns:
            Array of shape (n_muscles * 3,) with normalized length, velocity, force
        """
        lengths = []
        velocities = []
        forces = []

        for i, muscle in enumerate(self.parsed_model.muscles):
            # Get raw sensor values
            sensors = self.sensor_mapping[muscle.name]

            # Length (actuatorpos)
            if sensors["length"]:
                sensor_id = mujoco.mj_name2id(
                    self.mj_model, mujoco.mjtObj.mjOBJ_SENSOR, sensors["length"]
                )
                length = self.mj_data.sensordata[sensor_id]
            else:
                # Estimate from actuator length
                act_id = mujoco.mj_name2id(
                    self.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, muscle.name
                )
                length = self.mj_data.actuator_length[act_id]

            # Velocity (actuatorvel)
            if sensors["velocity"]:
                sensor_id = mujoco.mj_name2id(
                    self.mj_model, mujoco.mjtObj.mjOBJ_SENSOR, sensors["velocity"]
                )
                velocity = self.mj_data.sensordata[sensor_id]
            else:
                act_id = mujoco.mj_name2id(
                    self.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, muscle.name
                )
                velocity = self.mj_data.actuator_velocity[act_id]

            # Force (actuatorfrc)
            if sensors["force"]:
                sensor_id = mujoco.mj_name2id(
                    self.mj_model, mujoco.mjtObj.mjOBJ_SENSOR, sensors["force"]
                )
                force = self.mj_data.sensordata[sensor_id]
            else:
                act_id = mujoco.mj_name2id(
                    self.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, muscle.name
                )
                force = self.mj_data.actuator_force[act_id]

            lengths.append(length)
            velocities.append(velocity)
            forces.append(force)

        lengths = np.array(lengths)
        velocities = np.array(velocities)
        forces = np.array(forces)

        # Normalize
        lengths_norm = (lengths - self.sensor_stats["length_mean"]) / (
            self.sensor_stats["length_std"] + 1e-8
        )
        velocities_norm = (velocities - self.sensor_stats["velocity_mean"]) / (
            self.sensor_stats["velocity_std"] + 1e-8
        )
        forces_norm = (forces - self.sensor_stats["force_mean"]) / (
            self.sensor_stats["force_std"] + 1e-8
        )

        # Return raw normalized data (gamma modulation handled by controller)
        return np.concatenate([lengths_norm, velocities_norm, forces_norm])

    def _get_phase_encoding(self) -> np.ndarray:
        """Get one-hot encoding of current phase."""
        phase_map = {"pre_delay": 0, "reach": 1, "hold": 2, "post_delay": 1, "done": 2}
        phase_idx = phase_map.get(self.phase, 1)
        encoding = np.zeros(3)
        encoding[phase_idx] = 1.0
        return encoding

    def _get_time_encoding(self) -> np.ndarray:
        """Get normalized time within current phase."""
        if self.trial_config is None:
            return np.array([0.0])

        if self.phase == "pre_delay":
            duration = self.trial_config.pre_target_delay
            elapsed = self.episode_time
        elif self.phase == "reach":
            duration = self.max_episode_time - self.trial_config.pre_target_delay
            elapsed = self.episode_time - self.trial_config.pre_target_delay
        elif self.phase == "hold":
            duration = self.trial_config.hold_duration
            elapsed = (
                self.episode_time - self.hold_start_time
                if self.hold_start_time
                else 0.0
            )
        else:
            return np.array([1.0])

        return np.array([min(1.0, elapsed / (duration + 1e-8))])

    def _get_obs(self) -> np.ndarray:
        """Construct full observation."""
        proprioceptive = self._get_proprioceptive_obs()

        target_pos = (
            self.trial_config.target_position if self.trial_config else np.zeros(3)
        )
        target_encoding = self._encode_target(target_pos)

        # No phase or time encoding - RNN learns temporal dynamics
        obs = np.concatenate([proprioceptive, target_encoding])

        # NaN protection - replace NaNs with zeros and clip to reasonable range
        obs = np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)
        obs = np.clip(obs, -10.0, 10.0)

        return obs.astype(np.float32)

    def _sample_random_target(self) -> np.ndarray:
        """Sample a random target position within workspace."""
        x = np.random.uniform(*self.workspace_bounds["x"])
        y = np.random.uniform(*self.workspace_bounds["y"])
        z = np.random.uniform(*self.workspace_bounds["z"])
        return np.array([x, y, z])

    def _sample_random_initial_joints(self) -> np.ndarray:
        """Sample random initial joint configuration."""
        angles = []
        for joint in self.parsed_model.joints:
            if joint.range is not None:
                low, high = np.deg2rad(joint.range)
                angles.append(np.random.uniform(low, high))
            else:
                angles.append(np.random.uniform(-np.pi / 3, np.pi / 3))
        return np.array(angles)

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment."""
        super().reset(seed=seed)

        # Reset MuJoCo
        mujoco.mj_resetData(self.mj_model, self.mj_data)

        # Sample trial configuration
        pre_delay = np.random.uniform(*self.pre_delay_range)
        hold_duration = np.random.uniform(*self.hold_duration_range)
        post_delay = np.random.uniform(*self.post_delay_range)

        # Sample target and initial configuration
        target_pos = self._sample_random_target()
        initial_joints = self._sample_random_initial_joints()

        self.trial_config = TrialConfig(
            pre_target_delay=pre_delay,
            hold_duration=hold_duration,
            post_hold_delay=post_delay,
            target_position=target_pos,
            initial_joint_angles=initial_joints,
        )

        # Set initial joint configuration
        self.mj_data.qpos[: self.num_joints] = initial_joints
        self.mj_data.qvel[:] = 0

        # Forward kinematics
        mujoco.mj_forward(self.mj_model, self.mj_data)

        # Hide target initially (set far away)
        self._set_target_position(np.array([10.0, 10.0, 10.0]))

        # Reset state
        self.episode_step = 0
        self.episode_time = 0.0
        self.phase = "pre_delay"
        self.hold_start_time = None
        self.target_visible = False

        obs = self._get_obs()
        info = {
            "target_position": target_pos,
            "initial_joints": initial_joints,
            "pre_delay": pre_delay,
            "hold_duration": hold_duration,
        }

        return obs, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step."""
        # Action is just alpha MN activations
        alpha_mn = np.clip(action, 0.0, 1.0)

        # Apply muscle activations
        self.mj_data.ctrl[: self.num_muscles] = alpha_mn

        # Step simulation
        mujoco.mj_step(self.mj_model, self.mj_data)

        self.episode_step += 1
        self.episode_time += self.dt

        # Update phase
        self._update_phase()

        # Compute reward
        reward = self._compute_reward(alpha_mn)

        # Check termination
        terminated = self.phase == "done"
        truncated = self.episode_time >= self.max_episode_time

        obs = self._get_obs()

        info = {
            "phase": self.phase,
            "hand_position": self._get_hand_position(),
            "target_position": self.trial_config.target_position,
            "distance_to_target": self._distance_to_target(),
            "target_visible": self.target_visible,
        }

        return obs, reward, terminated, truncated, info

    def _update_phase(self):
        """Update trial phase based on time and performance."""
        if self.trial_config is None:
            return

        if self.phase == "pre_delay":
            if self.episode_time >= self.trial_config.pre_target_delay:
                self.phase = "reach"
                self.target_visible = True
                self._set_target_position(self.trial_config.target_position)

        elif self.phase == "reach":
            distance = self._distance_to_target()
            if distance < self.reach_threshold:
                self.phase = "hold"
                self.hold_start_time = self.episode_time

        elif self.phase == "hold":
            distance = self._distance_to_target()
            if distance >= self.reach_threshold:
                # Left target, go back to reach
                self.phase = "reach"
                self.hold_start_time = None
            elif self.hold_start_time is not None:
                hold_time = self.episode_time - self.hold_start_time
                if hold_time >= self.trial_config.hold_duration:
                    self.phase = "post_delay"
                    self.post_delay_start = self.episode_time

        elif self.phase == "post_delay":
            post_time = self.episode_time - self.post_delay_start
            if post_time >= self.trial_config.post_hold_delay:
                self.phase = "done"

    def _distance_to_target(self) -> float:
        """Compute distance from hand to target."""
        hand_pos = self._get_hand_position()
        target_pos = self.trial_config.target_position
        return np.linalg.norm(hand_pos - target_pos)

    def _compute_reward(self, alpha_mn: np.ndarray) -> float:
        """Compute reward for current state."""
        reward = 0.0

        if self.phase == "pre_delay":
            # Small reward for staying still
            velocity_penalty = -0.01 * np.sum(np.abs(self.mj_data.qvel))
            reward = velocity_penalty

        elif self.phase in ["reach", "hold"]:
            # Distance-based reward
            distance = self._distance_to_target()
            distance_reward = -distance

            # Bonus for reaching
            if distance < self.reach_threshold:
                distance_reward += 0.5

            # Hold bonus
            if self.phase == "hold":
                distance_reward += self.hold_reward_weight * 0.1

            # Energy penalty
            energy_penalty = -self.energy_penalty_weight * np.sum(alpha_mn**2)

            reward = distance_reward + energy_penalty

        elif self.phase == "post_delay" or self.phase == "done":
            # Success reward
            reward = 1.0

        # NaN protection
        if np.isnan(reward) or np.isinf(reward):
            reward = -1.0

        return float(reward)

    def render(self) -> Optional[np.ndarray]:
        """Render the environment."""
        if self.render_mode is None:
            return None

        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.mj_model, 480, 640)

        self.renderer.update_scene(self.mj_data)
        frame = self.renderer.render()

        if self.render_mode == "human":
            import cv2

            cv2.imshow("Muscle Arm", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

        return frame

    def close(self):
        """Clean up resources."""
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


def calibrate_sensors(
    xml_path: str, n_episodes: int = 100, max_steps_per_episode: int = 200
) -> Dict[str, Any]:
    """
    Run calibration episodes to gather sensor statistics.

    Args:
        xml_path: Path to MuJoCo XML
        n_episodes: Number of calibration episodes
        max_steps_per_episode: Max steps per episode

    Returns:
        Dictionary of sensor statistics for normalization
    """
    env = ReachingEnv(xml_path)

    all_lengths = []
    all_velocities = []
    all_forces = []

    for ep in range(n_episodes):
        obs, _ = env.reset()

        for step in range(max_steps_per_episode):
            # Random actions
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            # Collect raw sensor data
            lengths = []
            velocities = []
            forces = []

            for muscle in env.parsed_model.muscles:
                sensors = env.sensor_mapping[muscle.name]

                # Length
                if sensors["length"]:
                    sensor_id = mujoco.mj_name2id(
                        env.mj_model, mujoco.mjtObj.mjOBJ_SENSOR, sensors["length"]
                    )
                    length = env.mj_data.sensordata[sensor_id]
                else:
                    act_id = mujoco.mj_name2id(
                        env.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, muscle.name
                    )
                    length = env.mj_data.actuator_length[act_id]

                # Velocity
                if sensors["velocity"]:
                    sensor_id = mujoco.mj_name2id(
                        env.mj_model, mujoco.mjtObj.mjOBJ_SENSOR, sensors["velocity"]
                    )
                    velocity = env.mj_data.sensordata[sensor_id]
                else:
                    act_id = mujoco.mj_name2id(
                        env.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, muscle.name
                    )
                    velocity = env.mj_data.actuator_velocity[act_id]

                # Force
                if sensors["force"]:
                    sensor_id = mujoco.mj_name2id(
                        env.mj_model, mujoco.mjtObj.mjOBJ_SENSOR, sensors["force"]
                    )
                    force = env.mj_data.sensordata[sensor_id]
                else:
                    act_id = mujoco.mj_name2id(
                        env.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, muscle.name
                    )
                    force = env.mj_data.actuator_force[act_id]

                lengths.append(length)
                velocities.append(velocity)
                forces.append(force)

            all_lengths.append(lengths)
            all_velocities.append(velocities)
            all_forces.append(forces)

            if terminated or truncated:
                break

    env.close()

    # Compute statistics
    all_lengths = np.array(all_lengths)
    all_velocities = np.array(all_velocities)
    all_forces = np.array(all_forces)

    stats = {
        "length_mean": np.mean(all_lengths, axis=0),
        "length_std": np.std(all_lengths, axis=0) + 1e-8,
        "velocity_mean": np.mean(all_velocities, axis=0),
        "velocity_std": np.std(all_velocities, axis=0) + 1e-8,
        "force_mean": np.mean(all_forces, axis=0),
        "force_std": np.std(all_forces, axis=0) + 1e-8,
    }

    return stats


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python arm_reach_env.py <path_to_xml>")
        sys.exit(1)

    xml_path = sys.argv[1]

    # Run calibration
    print("Running sensor calibration...")
    stats = calibrate_sensors(xml_path, n_episodes=50)
    print("Sensor statistics:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # Test environment
    print("\nTesting environment...")
    env = ReachingEnv(xml_path, render_mode="rgb_array", sensor_stats=stats)

    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Action shape: {env.action_space.shape}")
    print(f"Initial info: {info}")

    total_reward = 0
    for step in range(200):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if step % 50 == 0:
            print(
                f"Step {step}: phase={info['phase']}, distance={info['distance_to_target']:.3f}"
            )

        if terminated or truncated:
            break

    print(f"Episode finished after {step+1} steps, total reward: {total_reward:.2f}")
    env.close()
