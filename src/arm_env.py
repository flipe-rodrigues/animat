import mujoco
import mujoco.viewer
import numpy as np

class ArmEnv:
    def __init__(self, model_path):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.target = np.array([0.5, 0.5])  # Example target position

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        # Randomize initial joint positions within their range
        self.data.qpos[:] = np.random.uniform(self.model.jnt_range[:, 0], self.model.jnt_range[:, 1])
        self.target = np.random.uniform(-1, 1, size=2)  # Random target position within reach
        return self._get_obs()

    def step(self, action):
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)
        obs = self._get_obs()
        reward = self._compute_reward()
        done = self._is_done()
        return obs, reward, done

    def _get_obs(self):
        return np.concatenate([
            self.data.qpos[:],  # Joint positions
            self.data.qvel[:],  # Joint velocities
            self.data.actuator_force[:],  # Actuator forces
            self.target  # Target position
        ])

    def _compute_reward(self):
        hand_pos = self.data.geom('hand').xpos[:2]
        distance = np.linalg.norm(hand_pos - self.target)
        return -distance

    def _is_done(self):
        hand_pos = self.data.geom('hand').xpos[:2]
        distance = np.linalg.norm(hand_pos - self.target)
        return distance < 0.1  # Considered done if within 0.1 units of the target

    def render(self):
        self.viewer.sync()