import os
import pickle
from etils import epath

import mujoco
import mujoco.mjx as mjx

import jax
from jax import numpy as jnp

from brax import envs
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf


MJCF_ROOT_PATH = epath.Path("../../mujoco")


def zscore(x, xmean, xstd, default=0):
    valid = jnp.greater(xstd, 0)
    return jnp.where(valid, (x - xmean) / xstd, default)


def l1_norm(x):
    return jnp.sum(jnp.abs(x))


def l2_norm(x):
    return jnp.sqrt(jnp.sum(x**2))


class SequentialReacher(PipelineEnv):

    def __init__(
        self,
        target_duration=3,
        num_targets=10,
        **kwargs,
    ):
        self.mj_model = mujoco.MjModel.from_xml_path(
            (MJCF_ROOT_PATH / "arm.xml").as_posix()
        )

        sys = mjcf.load_model(
            self.mj_model
        )  # system defining the kinematic tree and other properties
        kwargs["backend"] = "mjx"  # string specifying the physics pipeline
        kwargs["n_frames"] = (
            1  # the number of times to step the physics pipeline for each environment step
        )

        super().__init__(sys, **kwargs)

        # Get the site ID using the name of your end effector
        self.hand_id = self.mj_model.geom("hand").id
        self.target_id = self.mj_model.body_mocapid[
            mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, b"target")
        ]

        # Load sensor stats
        sensor_stats_path = os.path.join(MJCF_ROOT_PATH, "sensor_stats.pkl")
        with open(sensor_stats_path, "rb") as f:
            self.sensor_stats = pickle.load(f)

        # Load hand stats
        hand_position_stats_path = os.path.join(
            MJCF_ROOT_PATH, "hand_position_stats.pkl"
        )
        with open(hand_position_stats_path, "rb") as f:
            self.hand_position_stats = pickle.load(f)

        # Load candidate target positions
        candidate_targets_path = os.path.join(MJCF_ROOT_PATH, "candidate_targets.pkl")
        with open(candidate_targets_path, "rb") as f:
            self.candidate_targets = pickle.load(f)

        # Load candidate nail positions
        grid_positions_path = os.path.join(MJCF_ROOT_PATH, "grid_positions.pkl")
        with open(grid_positions_path, "rb") as f:
            self.grid_positions = pickle.load(f)

        # Convert stats to JAX arrays
        self.target_means = jnp.array(self.hand_position_stats["mean"].values)
        self.target_stds = jnp.array(self.hand_position_stats["std"].values)
        self.sensor_means = jnp.array(self.sensor_stats["mean"].values)
        self.sensor_stds = jnp.array(self.sensor_stats["std"].values)

        # Convert candidate_targets to JAX array
        self.candidate_target_positions = jnp.array(self.candidate_targets.values)

        self.target_duration = target_duration
        self.num_targets = num_targets

    def reset(self, rng: jnp.ndarray) -> State:
        """Resets the environment to an initial state."""
        qpos = jnp.zeros(self.sys.nq)
        qvel = jnp.zeros(self.sys.nv)
        data = self.pipeline_init(qpos, qvel)

        target_positions = self._sample_target_positions(rng)
        data = self._update_target(data, target_positions)

        obs = self._get_obs(data)
        reward, done = jnp.zeros(2)

        return State(
            data, obs, reward, done, info={"target_positions": target_positions}
        )

    def step(self, state: State, action: jnp.ndarray) -> State:
        """Runs one timestep of the environment's dynamics."""
        data = self.pipeline_step(state.pipeline_state, action)
        data = self._update_target(data, state.info["target_positions"])

        hand_position = self._get_hand_pos(data)
        target_position = self._get_target_pos(data)
        euclidean_distance = l2_norm(target_position - hand_position)

        obs = self._get_obs(data)
        reward = -euclidean_distance

        done = jnp.where(data.time > self.target_duration * self.num_targets, 1.0, 0.0)

        return state.replace(pipeline_state=data, obs=obs, reward=reward, done=done)

    def _get_obs(self, data: mjx.Data) -> jnp.ndarray:
        target_position = self._get_target_pos(data)
        sensor_data = data.sensordata.copy()
        norm_target_position = zscore(
            target_position,
            self.target_means,
            self.target_stds,
        )
        norm_sensor_data = zscore(
            sensor_data,
            self.sensor_means,
            self.sensor_stds,
        )
        obs = jnp.concatenate(
            [
                norm_target_position,
                norm_sensor_data,
            ]
        )
        obs = jnp.reshape(obs, (1, -1))
        return obs

    def _sample_target_positions(self, key: jnp.ndarray):
        """Sample target positions (w/o replacement) from the candidate targets"""
        sample_idcs = jax.random.choice(
            key,
            self.candidate_target_positions.shape[0],
            shape=(self.num_targets,),
            replace=False,
        )
        return self.candidate_target_positions[sample_idcs]

    def _update_target(self, data: mjx.Data, target_positions) -> jnp.ndarray:
        """Update the target position"""
        target_idx = jnp.floor_divide(data.time, self.target_duration).astype(jnp.int32)
        mocap_position = data.mocap_pos.at[self.target_id].set(
            target_positions[target_idx]
        )
        return data.replace(mocap_pos=mocap_position)

    def _get_hand_pos(self, data: mjx.Data) -> jnp.ndarray:
        """Get the position of the end effector (hand)"""
        hand_position = data.geom_xpos[self.hand_id].copy()
        return hand_position

    def _get_target_pos(self, data: mjx.Data) -> jnp.ndarray:
        """Get the position of the target"""
        target_position = data.mocap_pos[self.target_id].copy()
        return target_position


envs.register_environment("sequential_reacher", SequentialReacher)