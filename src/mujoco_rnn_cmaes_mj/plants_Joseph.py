import time
import pickle
import mujoco
import mujoco.viewer
from utils import *


class SequentialReacher:
    def __init__(self, plant_xml_file="arm.xml"):
        """Initialize Mujoco simulation"""

        mj_dir = os.path.join(get_root_path(), "mujoco")
        xml_path = os.path.join(mj_dir, plant_xml_file)
        self.xml = plant_xml_file
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.num_sensors = self.model.nu * 3  # Assuming 3 sensors per actuator
        self.num_actuators = self.model.nu
        self.viewer = None

        # Get the site ID using the name of your end effector
        self.hand_id = self.model.geom("hand").id

        # Get the hand's default mass value
        self.hand_default_mass = self.model.body_mass[self.hand_id]
        self.hand_force_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SENSOR, "hand_force"
        )

        # Find the index of the body whose mass you want to change
        self.weight_id = self.model.geom(name="weight").id

        # Load sensor stats
        sensor_stats_path = os.path.join(mj_dir, "sensor_stats.pkl")
        with open(sensor_stats_path, "rb") as f:
            self.sensor_stats = pickle.load(f)

        # Load hand stats
        hand_position_stats_path = os.path.join(mj_dir, "hand_position_stats.pkl")
        with open(hand_position_stats_path, "rb") as f:
            self.hand_position_stats = pickle.load(f)

        # Load candidate target positions
        candidate_targets_path = os.path.join(mj_dir, "candidate_targets.pkl")
        with open(candidate_targets_path, "rb") as f:
            self.candidate_targets = pickle.load(f)

        # Load candidate nail positions
        grid_positions_path = os.path.join(mj_dir, "grid_positions.pkl")
        with open(grid_positions_path, "rb") as f:
            self.grid_positions = pickle.load(f)

    def randomize_configuration(self):
        """Randomize the configuration of the arm"""
        for i in range(self.model.nq):
            self.data.qpos[i] = np.random.uniform(np.deg2rad(-60), np.deg2rad(60))
        mujoco.mj_forward(self.model, self.data)

    def sample_targets(self, num_samples=10):
        return self.candidate_targets.sample(num_samples).values

    def update_target(self, position):
        """Update the position of the target"""
        self.data.mocap_pos[0] = position
        mujoco.mj_forward(self.model, self.data)

    def update_nail(self, position):
        """Update the position of the nail"""
        self.data.eq_active[0] = 0
        mujoco.mj_forward(self.model, self.data)
        self.data.mocap_pos[1] = position  # self.get_hand_pos()
        mujoco.mj_forward(self.model, self.data)
        self.data.eq_active[0] = 1
        mujoco.mj_forward(self.model, self.data)

    def reset(self):
        """Reset limb state"""
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def get_obs(self):
        target_position = self.data.mocap_pos[0].copy()
        sensor_data = self.data.sensordata[0 : self.num_sensors].copy()
        norm_target_position = zscore(
            target_position,
            self.hand_position_stats["mean"].values,
            self.hand_position_stats["std"].values,
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
        self.data.ctrl[:] = muscle_activations
        mujoco.mj_step(self.model, self.data)

    def render(self):
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
                time.sleep(self.model.opt.timestep)

    def get_force_at_eq(self, eq_name):

        # Step 1: Find the equality constraint ID by name
        eq_id = None
        for i in range(self.model.neq):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, i)
            if name == eq_name:
                eq_id = i
                break
        if eq_id is None:
            raise RuntimeError(f"Equality constraint '{eq_name}' not found.")

        # Step 2: Get the constraint type to determine its size
        eq_type = self.model.eq_type[eq_id]
        eq_sizes = {
            mujoco.mjtEq.mjEQ_CONNECT: 3,
            mujoco.mjtEq.mjEQ_WELD: 6,
            mujoco.mjtEq.mjEQ_JOINT: 1,
            mujoco.mjtEq.mjEQ_TENDON: 1,
            mujoco.mjtEq.mjEQ_DISTANCE: 1,
        }
        constraint_dim = eq_sizes[eq_type]

        # Step 3: Sum dimensions of all previous equality constraints to find start index
        efc_start = 0
        for i in range(eq_id):
            prev_type = self.model.eq_type[i]
            efc_start += eq_sizes[prev_type]

        # Step 4: Extract the force vector (usually length 3 for CONNECT)
        force_vec = self.data.efc_force[efc_start : efc_start + constraint_dim]

        return force_vec

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
