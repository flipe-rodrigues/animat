import time
import pickle
import os
from typing import Tuple, Optional
import numpy as np
import mujoco
import mujoco.viewer
from utils import zscore, get_root_path


class SequentialReacher:
    """
    MuJoCo-based reaching task environment.

    Optimizations:
    - Caches sensor statistics arrays to avoid repeated slicing
    - Caches workspace bounds (computed once)
    - Optimized distance calculations
    - Context manager support for automatic cleanup
    """

    def __init__(self, plant_xml_file: str = "arm.xml"):
        """
        Initialize MuJoCo simulation.

        Args:
            plant_xml_file: Name of XML file in mujoco/ directory
        """
        # Load model
        mj_dir = os.path.join(get_root_path(), "mujoco")
        xml_path = os.path.join(mj_dir, plant_xml_file)
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # Basic info
        self.num_sensors = self.model.nsensor
        self.num_actuators = self.model.nu
        self.parse_sensors()
        self.viewer = None

        # Get target ID
        self.target_id = self.model.geom("target").id
        self.target_is_active = True

        # Get hand (end effector) ID
        self.hand_id = self.model.geom("hand").id
        self.hand_default_mass = self.model.body_mass[self.hand_id]
        self.hand_force_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SENSOR, "hand_force"
        )

        # Load sensor statistics
        sensor_stats_path = os.path.join(mj_dir, "sensor_stats.pkl")
        with open(sensor_stats_path, "rb") as f:
            self.sensor_stats = pickle.load(f)

        # Cache sensor normalization arrays (avoids repeated slicing)
        self._cache_sensor_stats()

        # Load hand statistics
        hand_position_stats_path = os.path.join(mj_dir, "hand_position_stats.pkl")
        with open(hand_position_stats_path, "rb") as f:
            self.hand_position_stats = pickle.load(f)

        # Cache workspace bounds
        self._workspace_bounds = None

        # Load candidate target positions
        candidate_targets_path = os.path.join(mj_dir, "candidate_targets.pkl")
        with open(candidate_targets_path, "rb") as f:
            self.candidate_targets = pickle.load(f)

        # Load candidate nail positions
        grid_positions_path = os.path.join(mj_dir, "grid_positions.pkl")
        with open(grid_positions_path, "rb") as f:
            self.grid_positions = pickle.load(f)

    def _cache_sensor_stats(self):
        """
        Cache sensor normalization arrays for faster access.

        This avoids repeated dictionary lookups and array slicing
        in the hot path (get_len_obs, get_vel_obs, get_frc_obs).
        """
        # Length sensors
        self._sensor_mean_len = self.sensor_stats["mean"].values[self.sensor_ids_len]
        self._sensor_std_len = self.sensor_stats["std"].values[self.sensor_ids_len]

        # Velocity sensors
        self._sensor_mean_vel = self.sensor_stats["mean"].values[self.sensor_ids_vel]
        self._sensor_std_vel = self.sensor_stats["std"].values[self.sensor_ids_vel]

        # Force sensors (mean is zeroed out)
        self._sensor_mean_frc = (
            self.sensor_stats["mean"].values[self.sensor_ids_frc] * 0.0
        )
        self._sensor_std_frc = self.sensor_stats["std"].values[self.sensor_ids_frc]

    def parse_sensors(self):
        """Parse and categorize sensors by type"""
        self.sensor_ids_len = []
        self.sensor_ids_vel = []
        self.sensor_ids_frc = []

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

    def randomize_gravity_direction(self):
        """Randomize gravity direction while keeping magnitude constant"""
        G = np.linalg.norm(self.model.opt.gravity)
        direction = np.random.randn(3)
        direction /= np.linalg.norm(direction)
        self.model.opt.gravity[:] = G * direction
        mujoco.mj_forward(self.model, self.data)

    def randomize_configuration(self):
        """Randomize the arm configuration"""
        for i in range(self.model.nq):
            self.data.qpos[i] = np.random.uniform(np.deg2rad(-60), np.deg2rad(60))
        mujoco.mj_forward(self.model, self.data)

    def sample_targets(self, num_samples: int = 10) -> np.ndarray:
        """Sample random target positions from candidate set"""
        return self.candidate_targets.sample(num_samples).values

    def update_target(self, position: np.ndarray):
        """
        Update target position.

        Note: mj_forward is called here, but in tight loops you may want
        to batch updates and call forward once.
        """
        self.data.mocap_pos[0] = position
        mujoco.mj_forward(self.model, self.data)

    def disable_target(self):
        """Disable target by making it invisible"""
        self.model.geom_rgba[self.target_id][-1] = 0.0
        self.target_is_active = False
        mujoco.mj_forward(self.model, self.data)

    def enable_target(self):
        """Enable target by making it visible"""
        self.model.geom_rgba[self.target_id][-1] = 1.0
        self.target_is_active = True
        mujoco.mj_forward(self.model, self.data)

    def update_nail(self, position: np.ndarray):
        """Update nail position (for constrained reaching tasks)"""
        self.data.eq_active[0] = 0
        mujoco.mj_forward(self.model, self.data)
        self.data.mocap_pos[1] = position
        mujoco.mj_forward(self.model, self.data)
        self.data.eq_active[0] = 1
        mujoco.mj_forward(self.model, self.data)

    def reset(self):
        """Reset simulation to initial state"""
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.close()

    def get_workspace_bounds(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Get workspace bounds (cached).

        Returns:
            ((x_min, x_max), (y_min, y_max))
        """
        if self._workspace_bounds is None:
            min_vals = self.hand_position_stats["min"].values
            max_vals = self.hand_position_stats["max"].values
            self._workspace_bounds = (
                (min_vals[0], max_vals[0]),
                (min_vals[1], max_vals[1]),
            )
        return self._workspace_bounds

    def get_target_pos(self) -> np.ndarray:
        """Get current target position (copy)"""
        return self.data.mocap_pos[0].copy()

    def get_len_obs(self) -> np.ndarray:
        """
        Get normalized length sensor observations (optimized).

        Uses cached normalization arrays for faster computation.
        """
        return zscore(
            self.data.sensordata[self.sensor_ids_len],
            self._sensor_mean_len,
            self._sensor_std_len,
        )

    def get_vel_obs(self) -> np.ndarray:
        """
        Get normalized velocity sensor observations (optimized).

        Uses cached normalization arrays for faster computation.
        """
        return zscore(
            self.data.sensordata[self.sensor_ids_vel],
            self._sensor_mean_vel,
            self._sensor_std_vel,
        )

    def get_frc_obs(self) -> np.ndarray:
        """
        Get normalized force sensor observations (optimized).

        Returns negative force (convention for muscle models).
        Uses cached normalization arrays for faster computation.
        """
        return -zscore(
            self.data.sensordata[self.sensor_ids_frc],
            self._sensor_mean_frc,
            self._sensor_std_frc,
        )

    def get_hand_pos(self) -> np.ndarray:
        """Get hand (end effector) position (copy)"""
        return self.data.geom_xpos[self.hand_id].copy()

    def get_gravity(self) -> np.ndarray:
        """Get current gravity vector (copy)"""
        return self.model.opt.gravity.copy()

    def get_distance_to_target(self) -> float:
        """
        Get distance from hand to target (optimized).

        Returns 0 if target is inactive.
        Avoids intermediate copies for better performance.
        """
        if not self.target_is_active:
            return 0.0

        # Direct computation without intermediate copies
        hand_pos = self.data.geom_xpos[self.hand_id]
        target_pos = self.data.mocap_pos[0]

        dx = hand_pos[0] - target_pos[0]
        dy = hand_pos[1] - target_pos[1]
        dz = hand_pos[2] - target_pos[2]

        return np.sqrt(dx * dx + dy * dy + dz * dz)

    def is_hand_touching_target(self) -> bool:
        """Check if hand is touching target based on radii"""
        hand_radius = self.model.geom_size[self.hand_id][0]
        target_radius = self.model.geom_size[self.target_id][0]
        distance = self.get_distance_to_target()
        return distance <= (hand_radius + target_radius)

    def step(self, ctrl: np.ndarray):
        """
        Step simulation forward by one timestep.

        Args:
            ctrl: Control signal (actuator commands)
        """
        self.data.ctrl[:] = ctrl
        mujoco.mj_step(self.model, self.data)

    def render(self, render_speed: float = 1.0):
        """Render simulation (creates viewer on first call)"""
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
            self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = True
            self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONSTRAINT] = True
            self.viewer.cam.lookat[:] = [0, -0.25, 0]
            self.viewer.cam.azimuth = 90
            self.viewer.cam.elevation = -90
        else:
            if self.viewer.is_running():
                self.viewer.sync()
                time.sleep(self.model.opt.timestep / render_speed)

    def get_force_at_eq(self, eq_name: str) -> np.ndarray:
        """
        Get constraint force for a named equality constraint.

        Args:
            eq_name: Name of equality constraint

        Returns:
            Force vector at the constraint
        """
        # Find equality constraint ID by name
        eq_id = None
        for i in range(self.model.neq):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, i)
            if name == eq_name:
                eq_id = i
                break

        if eq_id is None:
            raise RuntimeError(f"Equality constraint '{eq_name}' not found.")

        # Get constraint type and size
        eq_type = self.model.eq_type[eq_id]
        eq_sizes = {
            mujoco.mjtEq.mjEQ_CONNECT: 3,
            mujoco.mjtEq.mjEQ_WELD: 6,
            mujoco.mjtEq.mjEQ_JOINT: 1,
            mujoco.mjtEq.mjEQ_TENDON: 1,
            mujoco.mjtEq.mjEQ_DISTANCE: 1,
        }
        constraint_dim = eq_sizes[eq_type]

        # Find start index in efc_force array
        efc_start = 0
        for i in range(eq_id):
            prev_type = self.model.eq_type[i]
            efc_start += eq_sizes[prev_type]

        # Extract force vector
        return self.data.efc_force[efc_start : efc_start + constraint_dim]

    def close(self):
        """Close viewer and cleanup resources"""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    # Context manager support
    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with automatic cleanup"""
        self.close()
        return False

    def __repr__(self):
        return f"SequentialReacher(num_actuators={self.num_actuators}, num_sensors={self.num_sensors})"
