# Enable MKL optimizations for NumPy
import os
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_DYNAMIC'] = 'FALSE'

import numpy as np
from shimmy.dm_control_compatibility import DmControlCompatibilityV0
from gymnasium.wrappers import FlattenObservation, RecordEpisodeStatistics, TransformObservation
from gymnasium import spaces
import gymnasium as gym
import pickle
import torch

from tianshou.env import DummyVectorEnv, SubprocVectorEnv

from environment import make_arm_env
from networks import ModalitySpecificEncoder


def set_seeds(seed=42):
    """Set seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    return seed

class BiologicalEncodingWrapper(gym.ObservationWrapper):
    def __init__(self, env, encoder):
        super().__init__(env)
        self.encoder = encoder
        
        # Since we explicitly move encoder to CPU, just use CPU device
        self.encoder_device = torch.device('cpu')
        
        # Update observation space
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(encoder.output_size,), 
            dtype=np.float32
        )
    
    def observation(self, obs):
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.encoder_device)
            
            # Ensure we have a batch dimension - the encoder expects 2D input [batch_size, features]
            if obs_tensor.dim() == 1:
                obs_tensor = obs_tensor.unsqueeze(0)  # Add batch dimension
            
            encoded = self.encoder(obs_tensor)
            
            # Remove batch dimension if we added it
            if encoded.dim() == 2 and encoded.shape[0] == 1:
                encoded = encoded.squeeze(0)
            
            return encoded.cpu().numpy()

def create_env(random_seed=None, base_env=None):
    """Create a single environment instance."""
    # Use provided base_env or create new one
    dm_env = base_env if base_env is not None else make_arm_env(random_seed=random_seed)
    
    # Apply Shimmy compatibility wrapper
    env = DmControlCompatibilityV0(dm_env)
    
    # Flatten observations
    env = FlattenObservation(env)
    
    # Add normalization using stats files
    stats_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'mujoco')
    
    # Load stats once here
    with open(os.path.join(stats_dir, 'sensor_stats.pkl'), 'rb') as f:
        sensor_stats = pickle.load(f)
    with open(os.path.join(stats_dir, 'hand_position_stats.pkl'), 'rb') as f:
        target_stats = pickle.load(f)
    
    # Create normalization function with proper debugging
    def normalize_obs(obs):
        result = obs.copy()
        
        # Debug: Check input sensory data
        sensory_raw = obs[:12]
        sensory_mean = np.array(sensor_stats['mean'])
        sensory_std = np.array(sensor_stats['std'])
        
        # Apply z-score normalization with small epsilon to prevent division by zero
        result[:12] = (sensory_raw - sensory_mean) / np.maximum(sensory_std, 1e-8)
        
        # Keep target coordinates unchanged (they'll be encoded later)
        result[12:15] = obs[12:15]
        
        return result
    
    # Updated: Remove the observation_space parameter for compatibility with newer Gymnasium
    env = TransformObservation(env, normalize_obs)
    
    # Add record keeping
    env = RecordEpisodeStatistics(env)

    # Fix precision warnings by explicitly creating spaces with float32
    env.observation_space = spaces.Box(
        low=env.observation_space.low.astype(np.float32),
        high=env.observation_space.high.astype(np.float32),
        dtype=np.float32
    )
    
    env.action_space = spaces.Box(
        low=env.action_space.low.astype(np.float32),
        high=env.action_space.high.astype(np.float32),
        dtype=np.float32
    )
    
    # Add a seed method to the wrapped environment
    def seed_env(seed=None):
        # Store the seed for reset operations
        env.np_random = np.random.RandomState(seed)
        return [seed]
    
    # Add the method to the environment
    env.seed = seed_env
    
    # If a seed was provided, apply it
    if random_seed is not None:
        env.seed(random_seed)
    
    # Add biological encoding at environment level
    # Create encoder with explicit device setting
    encoder = ModalitySpecificEncoder(target_size=40)
    encoder = encoder.to('cpu')  # Explicitly move to CPU for environment processing
    env = BiologicalEncodingWrapper(env, encoder)
    
    return env

def create_training_env(num_envs: int = 4, base_seed: int = 12345):
    """Return a Tianshou vectorized env with subprocess workers."""
    def _make(rank: int):
        # Each worker gets its own seed
        return create_env(random_seed=base_seed + rank)
    
    # SubprocVectorEnv batches calls to make_env across processes
    return SubprocVectorEnv([lambda i=i: _make(i) for i in range(num_envs)])

def create_training_env_good(num_envs: int = 4, base_seed: int = 12345):
    """Return a Tianshou vectorized env with subprocess workers."""
    
    # Create environment factory functions with different seeds
    env_fns = []
    for i in range(num_envs):
        def make_env(seed=base_seed + i):
            # Explicitly set different seeds for each env
            env = create_env(random_seed=seed)
            env.seed(seed)  # Explicitly call seed method
            return env
        env_fns.append(make_env)
    
    # Create the vectorized environment
    return SubprocVectorEnv(
        env_fns,
        wait_num=num_envs,  # Process all environments in parallel
    )

def create_eval_env(num_envs: int = 1, base_seed: int = 42):
    """Return a Tianshou DummyVectorEnv for evaluation."""
    def _make(rank: int):
        env = create_env(random_seed=base_seed + rank)
        return env
    
    return DummyVectorEnv([lambda i=i: _make(i) for i in range(num_envs)])

def create_eval_env_good(num_envs: int = 1, base_seed: int = 42):
    """Return a Tianshou DummyVectorEnv for evaluation."""
    
    # Create environment factory functions with different seeds
    env_fns = []
    for i in range(num_envs):
        def make_env(seed=base_seed + i):
            # Explicitly set different seeds for each env
            env = create_env(random_seed=seed)
            env.seed(seed)  # Explicitly call seed method
            return env
        env_fns.append(make_env)
    
    # Create the vectorized environment using DummyVectorEnv for sequential evaluation
    return DummyVectorEnv(env_fns)


