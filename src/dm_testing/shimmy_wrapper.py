# Enable MKL optimizations for NumPy
import os
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_DYNAMIC'] = 'FALSE'

import numpy as np
from shimmy.dm_control_compatibility import DmControlCompatibilityV0
from gymnasium.wrappers import FlattenObservation, RescaleAction
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from gymnasium import spaces
from stable_baselines3.common.vec_env import sync_envs_normalization

from environment import make_arm_env
from success_tracking import SuccessInfoWrapper

import multiprocessing

def set_seeds(seed=42):
    """Set seeds for reproducibility."""
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    return seed

def create_env(random_seed=None, base_env=None):
    """Create a single environment instance."""
    # Use provided base_env or create new one
    dm_env = base_env if base_env is not None else make_arm_env(random_seed=random_seed)
    
    # Apply Shimmy compatibility wrapper
    env = DmControlCompatibilityV0(dm_env)
    
    # Flatten observations
    env = FlattenObservation(env)
    
    # Continue with normal setup
    env = RescaleAction(env, min_action=-1.0, max_action=1.0)
    
    # Add success tracking wrapper
    env = SuccessInfoWrapper(env)
    
    # Add monitoring
    env = Monitor(env)

    # Adjust action space dtype
    env.action_space = spaces.Box(
        low=env.action_space.low,
        high=env.action_space.high,
        dtype=np.float32
    )
    
    return env

def create_training_env(num_envs=4):
    """Create vectorized environment for training with true parallelism."""
    base_seed = 12345
    
    # Create worker initialization function that sets env variables
    def _init_env(rank):
        # Each subprocess needs to set its own env vars
        import os
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_DYNAMIC'] = 'FALSE'
        # Create and return environment with appropriate seed
        return create_env(random_seed=base_seed + rank)
    
    # Create vectorized environment using the worker init function
    vec_env = SubprocVecEnv([
        lambda rank=i: _init_env(rank=i) 
        for i in range(num_envs)
    ])
    
    # Only normalize observations for PPO
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        clip_reward=10.0
    )
    return vec_env

def create_eval_env(vec_normalize):
    """Create an evaluation environment with synced normalization."""
    # Create a function that returns a new environment
    def _make_env():
        env = create_env(random_seed=42)
        return Monitor(env)
    
    # Create the vectorized environment
    vec_env = DummyVecEnv([_make_env])
    
    # Add normalization with same parameters as training env
    normalized_env = VecNormalize(
        vec_env,
        norm_obs=vec_normalize.norm_obs,
        norm_reward=vec_normalize.norm_reward,
        clip_obs=vec_normalize.clip_obs,
        clip_reward=vec_normalize.clip_reward,
        gamma=vec_normalize.gamma,
        epsilon=vec_normalize.epsilon,
        training=False
    )
    
    # Sync the normalization statistics
    sync_envs_normalization(vec_normalize, normalized_env)
    
    return normalized_env

