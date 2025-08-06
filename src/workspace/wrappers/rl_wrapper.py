# Enable MKL optimizations for NumPy
import os
import torch

os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_DYNAMIC'] = 'FALSE'

import numpy as np
from shimmy.dm_control_compatibility import DmControlCompatibilityV0
import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import FlattenObservation, RescaleAction
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import sync_envs_normalization

# Clean imports
from envs.dm_env import make_arm_env
from .success_tracking import SuccessInfoWrapper


def set_seeds(seed=42):
    """Set seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    return seed


class EncoderWrapper(gym.ObservationWrapper):
    """Wrapper that applies a provided encoder to observations."""
    
    def __init__(self, env, encoder):  # Removed type hint
        super().__init__(env)
        self.encoder = encoder
        
        # Validate encoder has required attributes
        if not hasattr(encoder, 'output_dim'):
            raise AttributeError(f"Encoder {type(encoder).__name__} must have 'output_dim' property")
        
        # Update observation space based on encoder output
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(encoder.output_dim,), dtype=np.float32
        )
        #print(f"Encoder wrapper: {env.observation_space.shape} -> {self.observation_space.shape}")
    
    def observation(self, obs):
        """Apply encoder to observation."""
        # Try the new forward() method first, fall back to encode() for compatibility
        if hasattr(self.encoder, 'forward') and callable(self.encoder.forward):
            # PyTorch-style encoder
            import torch
            with torch.no_grad():
                if isinstance(obs, np.ndarray):
                    obs_tensor = torch.FloatTensor(obs)
                    encoded = self.encoder.forward(obs_tensor)
                    return encoded.numpy().astype(np.float32)
                else:
                    return self.encoder.forward(obs).astype(np.float32)
        elif hasattr(self.encoder, 'encode') and callable(self.encoder.encode):
            # Legacy encode() method
            return self.encoder.encode(obs).astype(np.float32)
        else:
            raise AttributeError(f"Encoder {type(self.encoder).__name__} must have either 'forward()' or 'encode()' method")


def create_env(random_seed=None, base_env=None, encoder=None):  # Removed type hint
    """Create environment with optional encoder - fully modular."""
    dm_env = base_env if base_env is not None else make_arm_env(random_seed=random_seed)
    env = DmControlCompatibilityV0(dm_env)
    env = FlattenObservation(env)
    
    # Apply encoder if provided
    if encoder is not None:
        env = EncoderWrapper(env, encoder)
    
    # Rescale actions to [0, 1]
    env = RescaleAction(env, min_action=0.0, max_action=1.0)
    
    # Add success tracking wrapper
    env = SuccessInfoWrapper(env)
    
    # Add monitoring
    env = Monitor(env)

    # Ensure action space is float32
    env.action_space = spaces.Box(
        low=env.action_space.low.astype(np.float32),
        high=env.action_space.high.astype(np.float32),
        dtype=np.float32
    )
    
    return env


def create_training_env(num_envs=4, encoder=None):  # Removed type hint
    """Create vectorized environment for training - now accepts encoder parameter."""
    base_seed = 12345
    
    def _init_env(rank):
        # Each subprocess needs to set its own env vars
        import os
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_DYNAMIC'] = 'FALSE'
        return create_env(random_seed=base_seed + rank, encoder=encoder)
    
    # Create vectorized environment
    vec_env = SubprocVecEnv([
        lambda rank=i: _init_env(rank=i) 
        for i in range(num_envs)
    ])
    
    # Apply standard normalization (for RL training)
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        gamma=0.99,
    )
    
    return vec_env


def create_eval_env(training_env=None, encoder=None):  # Removed type hint
    """Create evaluation environment with same normalization as training."""
    def _make_env():
        env = create_env(random_seed=42, encoder=encoder)
        return env
    
    vec_env = DummyVecEnv([_make_env])
    
    if training_env is not None:
        # Use same normalization as training
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, training=False)
        sync_envs_normalization(training_env, vec_env)
    
    return vec_env