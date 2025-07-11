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
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.num_sensors = self.model.nsensor
        self.num_actuators = self.model.nu
        self.viewer = None

        # Get the site ID using the name of your end effector
        self.hand_id = self.model.geom("hand").id

        # Get the hand's default mass value
        self.hand_default_mass = self.model.body_mass[self.hand_id]

        self.hand_force_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SENSOR, "hand_force"
        )

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

    # def fabrik(self, position):
    #     """FABRIK algorithm to solve inverse kinematics for a 2D arm"""
    #     # Initialize the arm configuration
    #     arm_length = 0.1  # Length of each arm segment
    #     num_segments = 2  # Number of segments in the arm
    #     arm_positions = np.zeros((num_segments + 1, 2))  # (x, y) positions of each segment
    #     arm_positions[0] = self.data.mocap_pos[0][:2]  # Start from the current position

    #     pass

    def solve_ik(self, position, max_iters=100, tol=1e-4, alpha=0.5
    ):
        dof_idxs = [self.model.jnt_dofadr[j] for j in [0, 1]]

        for i in range(max_iters):
            mujoco.mj_forward(self.model, self.data)

            current_pos = self.data.site_xpos[self.hand_id].copy()
            error = position - current_pos

            if np.linalg.norm(error) < tol:
                break

            # Compute Jacobian: J is (3 x nv), but we only care about 2 columns (our 2 joints)
            J = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, self.data, J, None, self.hand_id)

            # Extract Jacobian columns for our joints
            J_reduced = J[:, dof_idxs]  # shape (3, 2)

            # Least squares update (pseudo-inverse)
            dq = alpha * np.linalg.pinv(J_reduced) @ error

            # Apply update and clip to joint limits
            for i, dof in enumerate(dof_idxs):
                new_angle = self.data.qpos[dof] + dq[i]
                low, high = np.deg2rad(-60), np.deg2rad(60)
                self.data.qpos[dof] = np.clip(new_angle, low, high)

        mujoco.mj_forward(self.model, self.data)

    def sample_targets(self, num_samples):
        # If candidate_targets is a numpy array, use numpy's random choice
        if isinstance(self.candidate_targets, np.ndarray):
            indices = np.random.choice(len(self.candidate_targets), size=num_samples, replace=True)
            return self.candidate_targets[indices]
        else:
            # If it's a pandas DataFrame, use the sample method
            return self.candidate_targets.sample(num_samples).values

    def update_target(self, position):
        """Update the position of the target"""
        self.data.mocap_pos[0] = position
        mujoco.mj_forward(self.model, self.data)

    def update_nail(self, position):
        """Update the position of the nail"""
        # self.randomize_configuration()
        # self.solve_ik(position)
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
        sensor_data = self.data.sensordata.copy()
        
        # Handle both numpy arrays and pandas objects
        mean_target = self.hand_position_stats["mean"]
        std_target = self.hand_position_stats["std"]
        mean_sensor = self.sensor_stats["mean"]
        std_sensor = self.sensor_stats["std"]
        
        # Extract values if they are pandas objects, otherwise use directly
        if hasattr(mean_target, 'values'):
            mean_target = mean_target.values
        if hasattr(std_target, 'values'):
            std_target = std_target.values
        if hasattr(mean_sensor, 'values'):
            mean_sensor = mean_sensor.values
        if hasattr(std_sensor, 'values'):
            std_sensor = std_sensor.values
        
        norm_target_position = zscore(target_position, mean_target, std_target)
        norm_sensor_data = zscore(sensor_data, mean_sensor, std_sensor)
        
        return norm_target_position, norm_sensor_data

    def get_raw_obs(self):
        """Return raw target position and muscle sensors, no z‐scoring."""
        target_position = self.data.mocap_pos[0].copy()
        sensor_data     = self.data.sensordata.copy()
        return target_position, sensor_data

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
