"""
Reaching Task Environment

A Gymnasium environment for reaching tasks with muscle-driven arms.
Produces structured observations instead of implicit flat arrays.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
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
    """
    Reaching task environment with structured observations.
    
    The environment produces Observation objects with named fields.
    Flat arrays are only created at the gym API boundary.
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}
    
    def __init__(
        self,
        xml_path: str,
        render_mode: Optional[str] = None,
        pre_delay_range: Tuple[float, float] = DEFAULT_PRE_DELAY_RANGE,
        hold_duration_range: Tuple[float, float] = DEFAULT_HOLD_DURATION_RANGE,
        post_delay_range: Tuple[float, float] = DEFAULT_POST_DELAY_RANGE,
        max_episode_time: float = DEFAULT_MAX_EPISODE_TIME,
        workspace_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        reach_threshold: float = DEFAULT_REACH_THRESHOLD,
        hold_reward_weight: float = DEFAULT_HOLD_REWARD_WEIGHT,
        energy_penalty_weight: float = DEFAULT_ENERGY_PENALTY_WEIGHT,
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
        
        # Dimensions
        self.num_muscles = self.plant.num_muscles
        self.num_joints = self.plant.num_joints
        self.num_sensors = self.plant.num_sensors
        self.dt = self.plant.dt
        
        # Workspace - get both bounds and reachable positions
        if workspace_bounds is not None:
            self.workspace_bounds = workspace_bounds
            self._reachable_positions = None  # Use bounds-based sampling
        else:
            workspace_data = self.plant.estimate_workspace()
            self.workspace_bounds = {
                'x': workspace_data['x'],
                'y': workspace_data['y'],
                'z': workspace_data['z'],
            }
            self._reachable_positions = workspace_data.get('positions', None)
        
        # Sensor normalization stats
        self.sensor_stats = sensor_stats or self._default_sensor_stats()
        
        # Gym spaces (flat for API compatibility)
        obs_dim = self.num_muscles * 3 + 3  # proprio + target xyz
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.num_muscles,), dtype=np.float32
        )
        
        # Trial state
        self.trial_config: Optional[TrialConfig] = None
        self.episode_step = 0
        self.episode_time = 0.0
        self.phase = "pre_delay"
        self.hold_start_time: Optional[float] = None
        self.target_visible = False
    
    def _default_sensor_stats(self) -> Dict[str, np.ndarray]:
        """Default normalization stats (identity transform)."""
        return {
            'length_mean': np.zeros(self.num_muscles),
            'length_std': np.ones(self.num_muscles),
            'velocity_mean': np.zeros(self.num_muscles),
            'velocity_std': np.ones(self.num_muscles),
            'force_mean': np.zeros(self.num_muscles),
            'force_std': np.ones(self.num_muscles),
        }
    
    @property
    def target_position(self) -> np.ndarray:
        if self.trial_config is not None:
            return self.trial_config.target_position.copy()
        return np.zeros(3)
    
    def _sample_target(self) -> np.ndarray:
        """
        Sample random target from reachable workspace.
        
        If reachable positions were computed during workspace estimation,
        samples from those positions (with small noise). Otherwise falls
        back to sampling from bounding box.
        """
        if self._reachable_positions is not None and len(self._reachable_positions) > 0:
            # Sample from actual reachable positions
            idx = np.random.randint(len(self._reachable_positions))
            target = self._reachable_positions[idx].copy()
            return target
        else:
            # Fallback to bounding box (less accurate but backward compatible)
            return np.array([
                np.random.uniform(*self.workspace_bounds['x']),
                np.random.uniform(*self.workspace_bounds['y']),
                np.random.uniform(*self.workspace_bounds['z']),
            ])
    
    def _sample_initial_joints(self) -> np.ndarray:
        """Sample random initial joint configuration."""
        angles = []
        for joint in self.plant.parsed_model.joints:
            if joint.range is not None:
                low, high = np.deg2rad(joint.range)
                angles.append(np.random.uniform(low, high))
            else:
                angles.append(np.random.uniform(-np.pi/2, np.pi/2))
        return np.array(angles)
    
    def get_observation(self) -> Observation:
        """
        Get structured observation with named fields.
        
        This is the preferred way to access observations.
        """
        state = self.plant.get_state()
        stats = self.sensor_stats
        
        proprio = Proprioception(
            lengths=(state.muscle_lengths - stats['length_mean']) / stats['length_std'],
            velocities=(state.muscle_velocities - stats['velocity_mean']) / stats['velocity_std'],
            forces=(state.muscle_forces - stats['force_mean']) / stats['force_std'],
        )
        
        if self.target_visible and self.trial_config is not None:
            target = self.trial_config.target_position.copy()
        else:
            target = np.zeros(3)
        
        return Observation(proprio=proprio, target=target)
    
    def _get_flat_obs(self) -> np.ndarray:
        """Get flat observation for gym API."""
        obs = self.get_observation()
        flat = obs.to_flat()
        flat = np.nan_to_num(flat, nan=0.0, posinf=OBS_CLIP_MAX, neginf=OBS_CLIP_MIN)
        return np.clip(flat, OBS_CLIP_MIN, OBS_CLIP_MAX).astype(np.float32)
    
    def _distance_to_target(self) -> float:
        hand_pos = self.plant.get_hand_position()
        return np.linalg.norm(hand_pos - self.trial_config.target_position)
    
    def _update_phase(self):
        if self.trial_config is None:
            return
        
        if self.phase == "pre_delay":
            if self.episode_time >= self.trial_config.pre_target_delay:
                self.phase = "reach"
                self.target_visible = True
                self.plant.set_target_position(self.trial_config.target_position)
        
        elif self.phase == "reach":
            if self._distance_to_target() < self.reach_threshold:
                self.phase = "hold"
                self.hold_start_time = self.episode_time
        
        elif self.phase == "hold":
            if self._distance_to_target() >= self.reach_threshold:
                self.phase = "reach"
                self.hold_start_time = None
            elif self.episode_time - self.hold_start_time >= self.trial_config.hold_duration:
                self.phase = "post_delay"
                self.target_visible = False
        
        elif self.phase == "post_delay":
            hold_end = self.hold_start_time + self.trial_config.hold_duration
            if self.episode_time - hold_end >= self.trial_config.post_hold_delay:
                self.phase = "done"
    
    def _compute_reward(self, action: np.ndarray) -> float:
        if self.phase == "pre_delay":
            return -DEFAULT_ENERGY_PENALTY_WEIGHT * np.sum(np.abs(self.plant.get_joint_velocities()))
        
        if self.phase in ["reach", "hold"]:
            dist = self._distance_to_target()
            reward = -dist
            if dist < self.reach_threshold:
                reward += DEFAULT_REACH_BONUS
            if self.phase == "hold":
                reward += self.hold_reward_weight * 0.1
            reward -= self.energy_penalty_weight * np.sum(action ** 2)
            return reward
        
        if self.phase in ["post_delay", "done"]:
            return DEFAULT_SUCCESS_REWARD
        
        return 0.0
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
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
        self.plant.set_target_position(np.array([10.0, 10.0, 10.0]))  # Hide
        
        self.episode_step = 0
        self.episode_time = 0.0
        self.phase = "pre_delay"
        self.hold_start_time = None
        self.target_visible = False
        
        info = {
            'target_position': self.trial_config.target_position,
            'observation': self.get_observation(),
        }
        return self._get_flat_obs(), info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action = np.clip(action, 0.0, 1.0)
        self.plant.step(action)
        
        self.episode_step += 1
        self.episode_time += self.dt
        self._update_phase()
        
        reward = self._compute_reward(action)
        if np.isnan(reward) or np.isinf(reward):
            reward = -1.0
        
        terminated = self.phase == "done"
        truncated = self.episode_time >= self.max_episode_time
        
        obs = self.get_observation()
        info = {
            'phase': self.phase,
            'hand_position': self.plant.get_hand_position(),
            'target_position': self.trial_config.target_position,
            'distance_to_target': self._distance_to_target(),
            'target_visible': self.target_visible,
            'observation': obs,
        }
        
        return self._get_flat_obs(), reward, terminated, truncated, info
    
    def render(self) -> Optional[np.ndarray]:
        return self.plant.render()
    
    def close(self):
        self.plant.close()
