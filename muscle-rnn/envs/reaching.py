"""Reaching Task Environment - Gymnasium environment for reaching tasks."""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass

from envs.plant import MuJoCoPlant
from core.types import Proprioception, Observation
from core.constants import (
    DEFAULT_PRE_DELAY_RANGE,
    DEFAULT_HOLD_DURATION_RANGE,
    DEFAULT_POST_DELAY_RANGE,
    DEFAULT_MAX_EPISODE_TIME,
    DEFAULT_REACH_THRESHOLD,
    DEFAULT_HOLD_REWARD_WEIGHT,
    DEFAULT_ENERGY_PENALTY_WEIGHT,
    DEFAULT_REACH_BONUS,
    DEFAULT_SUCCESS_REWARD,
    DEFAULT_RIDGE_PENALTY_WEIGHT,
    DEFAULT_LASSO_PENALTY_WEIGHT,
    DEFAULT_RENDER_MODE,
    DEFAULT_VIDEO_FPS,
    OBS_CLIP_MIN,
    OBS_CLIP_MAX,
)


@dataclass
class TrialConfig:
    """Configuration for a single trial."""

    pre_target_delay: float
    hold_duration: float
    post_hold_delay: float
    target_position: np.ndarray
    initial_joint_angles: np.ndarray


class ReachingEnv(gym.Env):
    """Reaching task environment with structured observations."""

    metadata = {
        "render_modes": [DEFAULT_RENDER_MODE, "rgb_array"],
        "render_fps": DEFAULT_VIDEO_FPS,
    }

    def __init__(
        self,
        xml_path: str,
        render_mode: Optional[str] = DEFAULT_RENDER_MODE,
        pre_delay_range: Tuple[float, float] = DEFAULT_PRE_DELAY_RANGE,
        hold_duration_range: Tuple[float, float] = DEFAULT_HOLD_DURATION_RANGE,
        post_delay_range: Tuple[float, float] = DEFAULT_POST_DELAY_RANGE,
        max_episode_time: float = DEFAULT_MAX_EPISODE_TIME,
        workspace_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        reach_threshold: float = DEFAULT_REACH_THRESHOLD,
        hold_reward_weight: float = DEFAULT_HOLD_REWARD_WEIGHT,
        energy_penalty_weight: float = DEFAULT_ENERGY_PENALTY_WEIGHT,
        ridge_penalty_weight: float = DEFAULT_RIDGE_PENALTY_WEIGHT,
        lasso_penalty_weight: float = DEFAULT_LASSO_PENALTY_WEIGHT,
        sensor_stats: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        self.plant = MuJoCoPlant(xml_path, render_mode)
        self.render_mode = render_mode

        # Store parameters
        self.pre_delay_range = pre_delay_range
        self.hold_duration_range = hold_duration_range
        self.post_delay_range = post_delay_range
        self.max_episode_time = max_episode_time
        self.reach_threshold = reach_threshold
        self.hold_reward_weight = hold_reward_weight
        self.energy_penalty_weight = energy_penalty_weight
        self.ridge_penalty_weight = ridge_penalty_weight
        self.lasso_penalty_weight = lasso_penalty_weight

        # Network parameters for regularization (set externally, computed once per trial)
        self._network_params: Optional[np.ndarray] = None
        self._reg_penalty: float = 0.0

        # Dimensions
        self.num_muscles = self.plant.num_muscles
        self.num_joints = self.plant.num_joints
        self.num_sensors = self.plant.num_sensors
        self.dt = self.plant.dt

        # Workspace
        if workspace_bounds is not None:
            self.workspace_bounds = workspace_bounds
            self._reachable_positions = None
        else:
            workspace_data = self.plant.estimate_workspace()
            self.workspace_bounds = {k: workspace_data[k] for k in ["x", "y", "z"]}
            self._reachable_positions = workspace_data.get("positions")

        # Sensor normalization
        self.sensor_stats = sensor_stats or self._default_sensor_stats()

        # Gym spaces
        obs_dim = self.num_muscles * 3 + 3
        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32)
        self.action_space = spaces.Box(0.0, 1.0, (self.num_muscles,), np.float32)

        # Trial state
        self.trial_config: Optional[TrialConfig] = None
        self.episode_step = 0
        self.episode_time = 0.0
        self.phase = "pre_delay"
        self.hold_start_time: Optional[float] = None
        self.target_visible = False

    def _default_sensor_stats(self) -> Dict[str, np.ndarray]:
        return {
            "length_mean": np.zeros(self.num_muscles),
            "length_std": np.ones(self.num_muscles),
            "velocity_mean": np.zeros(self.num_muscles),
            "velocity_std": np.ones(self.num_muscles),
            "force_mean": np.zeros(self.num_muscles),
            "force_std": np.ones(self.num_muscles),
        }

    @property
    def target_position(self) -> np.ndarray:
        return (
            self.trial_config.target_position.copy()
            if self.trial_config
            else np.zeros(3)
        )

    def set_network_params(self, params: np.ndarray) -> None:
        """Set network parameters and compute regularization penalty (once per trial)."""
        self._network_params = params
        ridge = self.ridge_penalty_weight * np.sum(params**2)
        lasso = self.lasso_penalty_weight * np.sum(np.abs(params))
        self._reg_penalty = ridge + lasso

    def _sample_target(self) -> np.ndarray:
        if self._reachable_positions is not None and len(self._reachable_positions) > 0:
            return self._reachable_positions[
                np.random.randint(len(self._reachable_positions))
            ].copy()
        return np.array(
            [np.random.uniform(*self.workspace_bounds[k]) for k in ["x", "y", "z"]]
        )

    def _sample_initial_joints(self) -> np.ndarray:
        angles = []
        for joint in self.plant.parsed_model.joints:
            if joint.range is not None:
                low, high = np.deg2rad(joint.range)
                angles.append(np.random.uniform(low, high))
            else:
                angles.append(np.random.uniform(-np.pi / 2, np.pi / 2))
        return np.array(angles)

    def get_observation(self) -> Observation:
        """Get structured observation with named fields."""
        state = self.plant.get_state()
        stats = self.sensor_stats

        proprio = Proprioception(
            lengths=(state.muscle_lengths - stats["length_mean"]) / stats["length_std"],
            velocities=(state.muscle_velocities - stats["velocity_mean"])
            / stats["velocity_std"],
            forces=(state.muscle_forces - stats["force_mean"]) / stats["force_std"],
        )
        target = (
            self.trial_config.target_position.copy()
            if self.target_visible and self.trial_config
            else np.zeros(3)
        )
        return Observation(proprio=proprio, target=target)

    def _get_flat_obs(self) -> np.ndarray:
        obs = self.get_observation()
        flat = obs.to_flat()
        flat = np.nan_to_num(flat, nan=0.0, posinf=OBS_CLIP_MAX, neginf=OBS_CLIP_MIN)
        return np.clip(flat, OBS_CLIP_MIN, OBS_CLIP_MAX).astype(np.float32)

    def _distance_to_target(self) -> float:
        return np.linalg.norm(
            self.plant.get_hand_position() - self.trial_config.target_position
        )

    def _update_phase(self) -> None:
        if self.trial_config is None:
            return

        tc = self.trial_config
        if self.phase == "pre_delay" and self.episode_time >= tc.pre_target_delay:
            self.phase = "reach"
            self.target_visible = True
            self.plant.set_target_position(tc.target_position)

        elif (
            self.phase == "reach" and self._distance_to_target() < self.reach_threshold
        ):
            self.phase = "hold"
            self.hold_start_time = self.episode_time

        elif self.phase == "hold":
            if self._distance_to_target() >= self.reach_threshold:
                self.phase = "reach"
                self.hold_start_time = None
            elif self.episode_time - self.hold_start_time >= tc.hold_duration:
                self.phase = "post_delay"
                self.target_visible = False

        elif self.phase == "post_delay":
            hold_end = self.hold_start_time + tc.hold_duration
            if self.episode_time - hold_end >= tc.post_hold_delay:
                self.phase = "done"

    def _compute_reward(self, action: np.ndarray) -> float:

        reward = 0.0

        # Regularization penalty
        reward -= self._reg_penalty

        # Energy penalty
        energy = np.sum(action**2)
        reward -= self.energy_penalty_weight * energy

        if self.phase in ["reach", "hold"]:

            # Distance penalty
            distance = self._distance_to_target()
            reward -= distance

            # Reach bonus
            if distance < self.reach_threshold:
                reward += DEFAULT_REACH_BONUS

        if self.phase in ["post_delay", "done"]:
            reward += DEFAULT_SUCCESS_REWARD

        return reward

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        self.trial_config = TrialConfig(
            pre_target_delay=np.random.uniform(*self.pre_delay_range),
            hold_duration=np.random.uniform(*self.hold_duration_range),
            post_hold_delay=np.random.uniform(*self.post_delay_range),
            target_position=self._sample_target(),
            initial_joint_angles=self._sample_initial_joints(),
        )

        self.plant.reset(self.trial_config.initial_joint_angles)

        self.episode_step = 0
        self.episode_time = 0.0
        self.phase = "pre_delay"
        self.hold_start_time = None
        self.target_visible = False

        return self._get_flat_obs(), {
            "target_position": self.trial_config.target_position,
            "observation": self.get_observation(),
        }

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action = np.clip(action, 0.0, 1.0)
        self.plant.step(action)

        self.episode_step += 1
        self.episode_time += self.dt
        self._update_phase()

        reward = self._compute_reward(action)
        reward = -1.0 if np.isnan(reward) or np.isinf(reward) else reward

        terminated = self.phase == "done"
        truncated = self.episode_time >= self.max_episode_time

        info = {
            "phase": self.phase,
            "hand_position": self.plant.get_hand_position(),
            "target_position": self.trial_config.target_position,
            "distance_to_target": self._distance_to_target(),
            "target_visible": self.target_visible,
            "observation": self.get_observation(),
        }

        return self._get_flat_obs(), reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        return self.plant.render()

    def close(self) -> None:
        self.plant.close()
