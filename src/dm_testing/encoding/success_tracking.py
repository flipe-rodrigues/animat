import numpy as np
from gymnasium.core import Wrapper
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import EvalCallback

class SuccessInfoWrapper(Wrapper):
    """Add success information to the info dictionary."""
    
    def __init__(self, env):
        super().__init__(env)
        self.episode_success = False
        self.closest_distance = float('inf')
        
    def reset(self, **kwargs):
        self.episode_success = False
        self.closest_distance = float('inf')
        obs, info = super().reset(**kwargs)
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        try:
            # Robustly unwrap to base env
            base = self.env
            while hasattr(base, 'env'):
                base = base.env
            physics = getattr(base, 'physics', None)
            dm_env = base

            if physics and hasattr(dm_env, '_task') and hasattr(dm_env._task, '_arm'):
                hand_pos = physics.bind(dm_env._task._arm.hand).xpos
                target_pos = physics.bind(dm_env._task._arm.target).mocap_pos
                distance = np.linalg.norm(hand_pos - target_pos)

                self.closest_distance = min(self.closest_distance, distance)
                if distance < 0.08:
                    self.episode_success = True

                # Consistent info keys
                info['success'] = self.episode_success
                info['closest_distance'] = self.closest_distance
                info['current_distance'] = distance

                if terminated or truncated:
                    info['terminal_distance'] = distance
                    info['success'] = self.episode_success
                    info['closest_distance'] = self.closest_distance
        except Exception as e:
            print(f"[SuccessInfoWrapper] Error: {e}")

        return obs, reward, terminated, truncated, info

# This simple callback just logs the info dict metrics to TensorBoard
class SimpleMetricsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.total_episodes = 0
        self.successful_episodes = 0
        # Add flag to detect first call
        self.first_call = True
    
    def _on_training_start(self) -> None:
        """Log initial values when training starts"""
        # Log initial metrics so they appear at step 0 in TensorBoard
        self.logger.record("metrics/success_rate", 0.0)
        self.logger.record("metrics/terminal_distance", 1.0)  # Starting with a default value
        self.logger.record("metrics/closest_distance", 1.0)
        self.logger.record("training/learning_rate", self.model.learning_rate)
        
        # Dump initial metrics to disk
        self.logger.dump(self.num_timesteps)
        print("Initial metrics logged to TensorBoard")
    
    def _on_step(self) -> bool:
        # Force metrics dump at the start of training
        if self.first_call:
            self.first_call = False
            # Right after first step, log metrics again to ensure they show up
            self.logger.record("metrics/initial_step", 1.0)
            self.logger.dump(self.num_timesteps)
        
        # Your existing episode completion code
        for env_idx, done in enumerate(self.locals['dones']):
            if done:
                info = self.locals['infos'][env_idx]
                self.total_episodes += 1
                
                # Log any metrics found in info dict directly to TensorBoard
                if 'terminal_distance' in info:
                    self.logger.record("metrics/terminal_distance", info['terminal_distance'])
                
                if 'closest_distance' in info:  # Changed from 'final_closest_distance'
                    self.logger.record("metrics/closest_distance", info['closest_distance'])
                
                if 'success' in info and info['success']:  # Changed from 'episode_success'
                    self.successful_episodes += 1
                
                # Calculate and log success rate
                success_rate = self.successful_episodes / self.total_episodes
                self.logger.record("metrics/success_rate", success_rate)
                
                # Print occasional updates
                if self.total_episodes % 20 == 0:
                    print(f"Success rate: {success_rate:.2%} ({self.successful_episodes}/{self.total_episodes})")
        
        return True

