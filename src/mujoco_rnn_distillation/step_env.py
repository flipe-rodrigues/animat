"""
Environment wrapper for PPO training.

Provides a step-based interface compatible with RL training loops.
"""

import numpy as np
from typing import Tuple, Dict, Optional


class StepBasedReachingEnv:
    """
    Step-based wrapper around SequentialReachingEnv for RL training.
    
    Converts the episodic evaluate() interface into a step-by-step
    interface compatible with standard RL training loops.
    """
    
    def __init__(
        self,
        plant,
        target_encoder,
        env_config: Dict,
        max_steps_per_episode: Optional[int] = None,
    ):
        """
        Initialize step-based environment.
        
        Args:
            plant: SequentialReacher instance
            target_encoder: TargetEncoder instance
            env_config: Environment configuration dict
            max_steps_per_episode: Maximum steps per episode (for truncation)
        """
        from environments import SequentialReachingEnv, TargetSchedule, RewardCalculator
        from utils import truncated_exponential
        
        self.plant = plant
        self.encoder = target_encoder
        self.env_config = env_config
        
        # Create environment (lightweight, just stores config)
        self.env = SequentialReachingEnv(**env_config['env'])
        self.reward_calculator = RewardCalculator(env_config['env']['loss_weights'])
        
        # Compute observation dimension
        self.obs_dim = (
            self.encoder.size +  # target encoding
            self.plant.num_sensors_len +  # length sensors
            self.plant.num_sensors_vel +  # velocity sensors
            self.plant.num_sensors_frc    # force sensors
        )
        self.action_dim = self.plant.num_actuators
        
        # Episode state
        self.schedule = None
        self.target_idx = 0
        self.step_count = 0
        self.episode_reward = 0
        self.max_steps = max_steps_per_episode
        
        # For target schedule creation
        self.target_duration_distro = env_config['env']['target_duration_distro']
        self.iti_distro = env_config['env']['iti_distro']
        self.num_targets = env_config['env']['num_targets']
        self.randomize_gravity = env_config['env']['randomize_gravity']
    
    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """
        Reset environment for new episode.
        
        Args:
            seed: Random seed
        
        Returns:
            Initial observation
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Reset plant
        self.plant.reset()
        self.plant.disable_target()
        
        # Create new target schedule
        self.schedule = self._create_target_schedule()
        
        # Reset episode state
        self.target_idx = 0
        self.step_count = 0
        self.episode_reward = 0
        
        # Get initial observation
        obs = self._get_observation()
        
        return obs
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take one step in the environment.
        
        Args:
            action: Action to take (muscle activations)
        
        Returns:
            obs: Next observation
            reward: Reward for this step
            done: Whether episode is done
            info: Additional information
        """
        # Update target state (enable/disable as needed)
        self.target_idx = self._update_target_state(self.target_idx)
        
        # Step the physics simulation
        self.plant.step(action)
        self.step_count += 1
        
        # Compute reward
        reward = self._compute_reward(action)
        self.episode_reward += reward
        
        # Check if episode is done
        done = (
            self.target_idx >= self.schedule.num_targets or
            (self.max_steps is not None and self.step_count >= self.max_steps)
        )
        
        # Get next observation
        obs = self._get_observation()
        
        # Additional info
        info = {
            'episode_reward': self.episode_reward if done else None,
            'target_idx': self.target_idx,
            'distance': self.plant.get_distance_to_target(),
            'step_count': self.step_count,
        }
        
        return obs, reward, done, info
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation"""
        # Target encoding
        if self.target_idx < self.schedule.num_targets and self.plant.target_is_active:
            target_pos = self.schedule.positions[self.target_idx]
            tgt_obs = self.encoder.encode(x=target_pos[0], y=target_pos[1]).flatten()
        else:
            tgt_obs = np.zeros(self.encoder.size)
        
        # Proprioceptive observations
        len_obs = self.plant.get_len_obs()
        vel_obs = self.plant.get_vel_obs()
        frc_obs = self.plant.get_frc_obs()
        
        # Concatenate all observations
        obs = np.concatenate([tgt_obs, len_obs, vel_obs, frc_obs])
        
        return obs
    
    def _compute_reward(self, action: np.ndarray) -> float:
        """Compute reward for current state and action"""
        distance = self.plant.get_distance_to_target()
        energy = np.sum(np.square(action))
        
        # No regularization during training (only distance and energy)
        reward = self.reward_calculator.compute(
            distance=distance,
            energy=energy,
            ridge=0.0,
            lasso=0.0,
        )
        
        return reward
    
    def _create_target_schedule(self):
        """Create target schedule for this episode"""
        from environments import TargetSchedule
        from utils import truncated_exponential
        
        # Sample target positions
        positions = self.plant.sample_targets(self.num_targets)
        
        # Sample durations
        durations = truncated_exponential(
            mu=self.target_duration_distro['mean'],
            a=self.target_duration_distro['min'],
            b=self.target_duration_distro['max'],
            size=self.num_targets,
        )
        
        # Sample ITIs
        itis = truncated_exponential(
            mu=self.iti_distro['mean'],
            a=self.iti_distro['min'],
            b=self.iti_distro['max'],
            size=self.num_targets,
        )
        
        return TargetSchedule(positions, durations, itis)
    
    def _update_target_state(self, target_idx: int) -> int:
        """Update target enable/disable state"""
        current_time = self.plant.data.time
        
        # Disable target if past offset
        if self.schedule.should_disable_target(current_time, target_idx):
            self.plant.disable_target()
            target_idx += 1
        
        # Enable target if past onset
        if (
            target_idx < self.schedule.num_targets
            and self.schedule.should_enable_target(current_time, target_idx)
            and not self.plant.target_is_active
        ):
            if self.randomize_gravity:
                self.plant.randomize_gravity_direction()
            
            self.plant.update_target(self.schedule.positions[target_idx])
            self.plant.enable_target()
        
        return target_idx
    
    def render(self, render_speed: float = 1.0):
        """Render the environment"""
        self.plant.render(render_speed)
    
    def close(self):
        """Close the environment"""
        self.plant.close()
