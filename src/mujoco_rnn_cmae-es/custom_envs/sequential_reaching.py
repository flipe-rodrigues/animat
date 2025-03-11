import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import mujoco.viewer
import time


class SequentialReachingEnv(gym.Env):
    """Custom 2-Joint Limb with 4 Muscles, 12 Sensors, and a Target Position"""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        xml_path="path/to/your_model.xml",
        max_num_targets=10,
        max_target_duration=3,
        render_mode=None,
    ):
        super().__init__()

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.max_num_targets = max_num_targets
        self.max_target_duration = max_target_duration
        self.render_mode = render_mode
        self.viewer = None

        # Observation space: 12 sensor readings + 3D target position
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32
        )

        # Action space: 4 muscle activations
        self.action_space = spaces.Box(low=0, high=1.0, shape=(4,), dtype=np.float32)

        # Get the site ID using the name of your end effector
        self.hand_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "hand")

        # Parse target positions from CSV file
        self.reachable_points = self.parse_targets("../../src/targets.csv")

    def parse_targets(self, targets_path="path/to/targets.csv", bins=100):
        target_positions = np.loadtxt(
            targets_path, delimiter=",", skiprows=1, usecols=(0, 2)
        )
        x, y = target_positions[:, 0], target_positions[:, 1]
        counts2d, x_edges, y_edges = np.histogram2d(x, y, bins=bins)
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2
        nonzero_indices = np.argwhere(counts2d > 0)
        return [(x_centers[i], 0, y_centers[j]) for i, j in nonzero_indices]

    def sample_targets(self, num_samples=10):
        sampled_points = np.random.choice(
            len(self.reachable_points), num_samples, replace=False
        )
        return [self.reachable_points[i] for i in sampled_points]

    def update_target(self, position):
        self.data.mocap_pos = position
        mujoco.mj_forward(self.model, self.data)

    def step(self, action):
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)

        sensor_data = self.data.sensordata.copy()
        hand_position = self.data.site_xpos[self.hand_id]
        distance = np.linalg.norm(
            hand_position - self.target_positions[self.target_idx]
        )
        reward = -distance

        done = self.data.time > self.max_target_duration * self.max_num_targets
        if distance < 0.05:
            if self.target_idx < self.max_num_targets - 1:
                self.target_idx += 1
                self.update_target(self.target_positions[self.target_idx])
            else:
                done = True

        obs = np.concatenate([sensor_data, self.target_positions[self.target_idx]])
        return obs, reward, done, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        self.target_positions = self.sample_targets(self.max_num_targets)
        self.target_idx = 0
        self.update_target(self.target_positions[self.target_idx])

        sensor_data = self.data.sensordata.copy()
        obs = np.concatenate([sensor_data, self.target_positions[self.target_idx]])
        return obs, {}

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
                self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
                self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = True
                self.viewer.cam.lookat[:] = [0, -1.5, -0.5]
                self.viewer.cam.azimuth = 90
                self.viewer.cam.elevation = 0
            else:
                self.viewer.sync()
                # step_start = time.time()
                # time_until_next_step = self.model.opt.timestep - (
                #     time.time() - step_start
                # )
                # if time_until_next_step > 0:
                #     time.sleep(time_until_next_step)
        elif self.render_mode == "rgb_array":
            width, height = 640, 480
            img = np.zeros((height, width, 3), dtype=np.uint8)
            mujoco.mjr_render(
                self.viewer.viewport, self.viewer.scene, self.viewer.context
            )
            return img

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
