import time
import pickle
import mujoco
import mujoco.viewer
from utils import *


class SequentialReacher:
    def __init__(self, plant_xml_file="arm_model.xml"):
        """Initialize Mujoco simulation"""

        mj_dir = os.path.join(get_root_path(), "mujoco")
        xml_path = os.path.join(mj_dir, plant_xml_file)
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.num_sensors = self.model.nsensor
        self.num_actuators = self.model.nu
        self.viewer = None

        # Get the site ID using the name of your end effector
        self.hand_id = self.model.geom("hand").id

        # Load sensor stats
        sensor_stats_path = os.path.join(mj_dir, "sensor_stats.pkl")
        with open(sensor_stats_path, "rb") as f:
            self.sensor_stats = pickle.load(f)

        # Load target stats
        target_stats_path = os.path.join(mj_dir, "target_stats.pkl")
        with open(target_stats_path, "rb") as f:
            self.target_stats = pickle.load(f)

        # Load valid target positions
        reachable_positions_path = os.path.join(mj_dir, "reachable_positions.pkl")
        with open(reachable_positions_path, "rb") as f:
            self.reachable_positions = pickle.load(f)

    def sample_targets(self, num_samples=10):
        return self.reachable_positions.sample(num_samples).values

    def update_target(self, position):
        self.data.mocap_pos = position
        mujoco.mj_forward(self.model, self.data)

    def reset(self):
        """Reset limb state"""
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def get_obs(self):
        """Return joint angles, velocities, and end-effector position"""
        target_position = self.data.mocap_pos[0].copy()
        sensor_data = self.data.sensordata.copy()
        norm_target_position = zscore(
            target_position,
            self.target_stats["mean"].values,
            self.target_stats["std"].values,
        )
        norm_sensor_data = zscore(
            sensor_data,
            self.sensor_stats["mean"].values,
            self.sensor_stats["std"].values,
        )
        return norm_target_position, norm_sensor_data

    def get_hand_pos(self):
        return self.data.geom_xpos[self.hand_id].copy()

    def step(self, muscle_activations):
        """Apply torques and step simulation"""
        self.data.ctrl[:] = muscle_activations
        mujoco.mj_step(self.model, self.data)

    def render(self):
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
            self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = True
            self.viewer.cam.lookat[:] = [0, -1.5, -0.5]
            self.viewer.cam.azimuth = 90
            self.viewer.cam.elevation = 0
        else:
            if self.viewer.is_running():
                self.viewer.sync()
                time.sleep(self.model.opt.timestep)

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
