import time
import pickle
import mujoco
import mujoco.mjx as mjx
import mujoco.viewer
import jax
import jax.numpy as jnp
from functools import partial
from utils import *


class SequentialReacher:
    """
    .####.##....##.####.########
    ..##..###...##..##.....##...
    ..##..####..##..##.....##...
    ..##..##.##.##..##.....##...
    ..##..##..####..##.....##...
    ..##..##...###..##.....##...
    .####.##....##.####....##...
    """

    def __init__(self, plant_xml_file="arm_model.xml"):
        """Initialize Mujoco simulation"""

        mj_dir = os.path.join(get_root_path(), "mujoco")
        xml_path = os.path.join(mj_dir, plant_xml_file)
        self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self.mj_data = mujoco.MjData(self.mj_model)

        self.mjx_model = mjx.put_model(self.mj_model)
        self.mjx_data = mjx.make_data(self.mjx_model)

        self.num_sensors = self.mj_model.nsensor
        self.num_actuators = self.mj_model.nu
        self.viewer = None

        # Get the site ID using the name of your end effector
        self.hand_id = self.mj_model.geom("hand").id

        # Get the hand's default mass value
        self.hand_default_mass = self.mj_model.body_mass[self.hand_id]

        self.hand_force_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "hand_force"
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

    """
    .########..########..######..########.########
    .##.....##.##.......##....##.##..........##...
    .##.....##.##.......##.......##..........##...
    .########..######....######..######......##...
    .##...##...##.............##.##..........##...
    .##....##..##.......##....##.##..........##...
    .##.....##.########..######..########....##...
    """

    def reset(self):
        """Reset limb state"""
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        mujoco.mj_forward(self.mj_model, self.mj_data)
        self.mjx_data = mjx.make_data(self.mjx_model)  # Reset MJX data as well
        return self.mjx_data

    @staticmethod
    def reset_mjx_data(mjx_model):
        """Static method to reset MJX data for parallel environments"""
        return mjx.make_data(mjx_model)

    """
    ..######...########.########.....#######..########...######.
    .##....##..##..........##.......##.....##.##.....##.##....##
    .##........##..........##.......##.....##.##.....##.##......
    .##...####.######......##.......##.....##.########...######.
    .##....##..##..........##.......##.....##.##.....##.......##
    .##....##..##..........##.......##.....##.##.....##.##....##
    ..######...########....##........#######..########...######.
    """

    def get_obs(self, data=None):
        """Get observation vector, normalized"""
        if data is None:
            data = self.mjx_data
            
        target_position = data.mocap_pos[0].copy()
        sensor_data = data.sensordata.copy()
        
        # Convert stats to JAX arrays if needed
        mean_target = jnp.array(self.hand_position_stats["mean"].values)
        std_target = jnp.array(self.hand_position_stats["std"].values)
        mean_sensor = jnp.array(self.sensor_stats["mean"].values)
        std_sensor = jnp.array(self.sensor_stats["std"].values)
        
        norm_target_position = zscore(
            target_position,
            mean_target,
            std_target,
        )
        norm_sensor_data = zscore(
            sensor_data,
            mean_sensor,
            std_sensor,
        )
        
        return jnp.concatenate(
            [
                norm_target_position,
                norm_sensor_data,
            ]
        )

    """
    ..######..########.########.########.
    .##....##....##....##.......##.....##
    .##..........##....##.......##.....##
    ..######.....##....######...########.
    .......##....##....##.......##.......
    .##....##....##....##.......##.......
    ..######.....##....########.##.......
    """

    def step(self, muscle_activations):
        """Step the mujoco simulation"""
        self.mj_data.ctrl[:] = muscle_activations
        mujoco.mj_step(self.mj_model, self.mj_data)

    def step_mjx(self, mjx_data, muscle_activations):
        """Step the MJX simulation"""
        mjx_data = mjx_data.replace(ctrl=muscle_activations)
        mjx_data = mjx.step(self.mjx_model, mjx_data)
        return mjx_data

    @staticmethod
    def batch_step(mjx_model, mjx_data_batch, muscle_activations_batch):
        """Batch step for parallel MJX simulations"""
        mjx_data_batch = mjx_data_batch.replace(ctrl=muscle_activations_batch)
        new_mjx_data_batch = mjx.step(mjx_model, mjx_data_batch)
        return new_mjx_data_batch

    """
    ..#######..########.##.....##.########.########.
    .##.....##....##....##.....##.##.......##.....##
    .##.....##....##....##.....##.##.......##.....##
    .##.....##....##....#########.######...########.
    .##.....##....##....##.....##.##.......##...##..
    .##.....##....##....##.....##.##.......##....##.
    ..#######.....##....##.....##.########.##.....##
    """

    def randomize_configuration(self):
        """Randomize the configuration of the arm"""
        for i in range(self.mj_model.nq):
            self.mj_data.qpos[i] = np.random.uniform(np.deg2rad(-60), np.deg2rad(60))
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def sample_targets(self, num_samples=10):
        """Sample target positions from the candidate targets"""
        return self.candidate_targets.sample(num_samples).values

    def get_targets_batch(self, num_envs, num_targets):
        """Get a batch of target positions for parallel environments"""
        targets_list = []
        for _ in range(num_envs):
            targets_list.append(self.sample_targets(num_targets))
        return jnp.array(targets_list)

    def update_target(self, position):
        """Update the position of the target"""
        self.mj_data.mocap_pos[0] = position
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def update_target_mjx(self, mjx_data, position):
        """Update the target position in MJX data"""
        mocap_pos = mjx_data.mocap_pos.at[0].set(position)
        return mjx_data.replace(mocap_pos=mocap_pos)

    @staticmethod
    def batch_update_target(mjx_data_batch, positions_batch):
        """Update targets for a batch of MJX data"""
        new_mocap_pos = mjx_data_batch.mocap_pos.at[:, 0].set(positions_batch)
        return mjx_data_batch.replace(mocap_pos=new_mocap_pos)

    def update_nail(self, position):
        """Update the position of the nail"""
        self.mj_data.eq_active[0] = 0
        mujoco.mj_forward(self.mj_model, self.mj_data)
        self.mj_data.mocap_pos[1] = position
        mujoco.mj_forward(self.mj_model, self.mj_data)
        self.mj_data.eq_active[0] = 1
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def get_hand_pos(self, data=None):
        """Get the position of the hand"""
        if data is None:
            return self.mj_data.geom_xpos[self.hand_id].copy()
        return data.geom_xpos[self.hand_id].copy()

    def render(self):
        """Render the simulation using the mujoco viewer"""
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)
            self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
            self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = True
            self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONSTRAINT] = True
            self.viewer.cam.lookat[:] = [0, -0.25, 0]
            self.viewer.cam.azimuth = 90
            self.viewer.cam.elevation = -90
        else:
            if self.viewer.is_running():
                self.viewer.sync()
                time.sleep(self.mj_model.opt.timestep)

    def close(self):
        """Close the viewer"""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


# JAX-compiled get_observation function
@jax.jit
def get_obs_jit(mjx_data, target_mean, target_std, sensor_mean, sensor_std):
    """JAX-compiled function to get observations"""
    target_position = mjx_data.mocap_pos[0]
    sensor_data = mjx_data.sensordata
    
    norm_target_position = zscore(
        target_position,
        target_mean,
        target_std,
    )
    norm_sensor_data = zscore(
        sensor_data,
        sensor_mean,
        sensor_std,
    )
    
    return jnp.concatenate([norm_target_position, norm_sensor_data])

# Vectorized version for batch processing
@jax.vmap
def get_obs_batch(mjx_data_batch, target_mean, target_std, sensor_mean, sensor_std):
    """Vectorized function to get observations from multiple environments"""
    return get_obs_jit(mjx_data_batch, target_mean, target_std, sensor_mean, sensor_std)
