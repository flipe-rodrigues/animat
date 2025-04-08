import gymnasium as gym
from gymnasium import spaces
import jax
import jax.numpy as jnp
import numpy as np
import time

class MJXVecGymEnv:
    """Vectorized environment wrapper for MJX environments optimized for performance."""
    
    def __init__(self, mjx_env):
        self.env = mjx_env
        
        # Set up spaces for actions and observations
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0,
            shape=(self.env.num_sensors + 3,),
            dtype=np.float32
        )
        
        self.action_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(self.env.num_actuators,),
            dtype=np.float32
        )
        
        # Set required attributes for VectorEnv API
        self.num_envs = self.env.num_envs
        self.metadata = {"render_modes": ["human"]}
        self.render_mode = "human"
        
        # For SB3 compatibility 
        self.single_observation_space = self.observation_space
        self.single_action_space = self.action_space
        
        # For step_async/step_wait API compatibility
        self._actions = None
        
        # Episode tracking (for rewards logging)
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.recent_dones = np.zeros(self.num_envs, dtype=bool)
        self.recent_episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        
        # Info buffer for TensorBoard callbacks
        self.info_buffer = []
    
    def reset(self, *, seed=None, options=None):
        """Reset all environments."""
        # Reset episode tracking
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        
        # Reset with seed if provided
        if seed is not None:
            key = jax.random.PRNGKey(seed)
            return np.array(self.env.reset(key=key))
        
        return np.array(self.env.reset())
    
    def step_async(self, actions):
        """Store actions for asynchronous stepping (SB3 compatibility)."""
        self._actions = actions
    
    def step_wait(self):
        """Execute stored actions and wait for results (SB3 compatibility)."""
        return self.step(self._actions)
    
    def step(self, actions):
        """Step all environments with the given actions."""
        # Move actions to device efficiently
        actions_jax = jnp.asarray(actions, device=jax.devices()[0])
        
        # Step the MJX environments (keeps computation on GPU)
        obs, rewards, dones, info = self.env.step(actions_jax)
        
        # Store original info dict for metrics access
        self.info_buffer = [info]
        
        # Fast conversion to numpy (only once when needed)
        np_obs = np.array(obs)
        np_rewards = np.array(rewards)
        np_dones = np.array(dones)
        
        # Update episode tracking
        self.episode_returns += np_rewards
        self.episode_lengths += 1
        self.recent_dones = np_dones
        self.recent_episode_returns = np.copy(self.episode_returns)
        
        # Create info dictionaries with episode data for SB3
        info_list = [{} for _ in range(self.num_envs)]
        for i in range(self.num_envs):
            if np_dones[i]:
                # CRITICAL: Format exactly as SB3 expects
                info_list[i]['episode'] = {
                    'r': float(self.episode_returns[i]),
                    'l': int(self.episode_lengths[i]),
                    't': time.time()
                }
                
                # Copy over environment info keys that might be needed
                for key in ["target_reached", "current_target_idx", "euclidean_distances"]:
                    if key in info and i < len(info[key]):
                        info_list[i][key] = float(info[key][i]) if hasattr(info[key][i], "item") else info[key][i]
                
                # Reset tracking for this environment
                self.episode_returns[i] = 0
                self.episode_lengths[i] = 0
        
        return np_obs, np_rewards, np_dones, info_list
    
    def seed(self, seed=None):
        """Set random seed."""
        return [seed] * self.num_envs if seed is not None else None
    
    def close(self):
        """Close all environments."""
        pass
        
    def render(self):
        """Render one of the environments."""
        return self.env.render(0)
    
    def get_attr(self, attr_name, indices=None):
        """Return attribute from vectorized environment (SB3 compatibility)."""
        if attr_name in ["render_mode", "metadata"]:
            return [getattr(self, attr_name)] * self.num_envs
        return [getattr(self.env, attr_name)] * self.num_envs if hasattr(self.env, attr_name) else None