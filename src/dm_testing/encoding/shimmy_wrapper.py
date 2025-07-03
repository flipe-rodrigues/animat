# Enable MKL optimizations for NumPy
import os
import torch
import torch.nn as nn

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
import gymnasium as gym
import torch

from environment import make_arm_env
from success_tracking import SuccessInfoWrapper


class GridEncoder(nn.Module):
    """Grid-based Gaussian encoding for spatial coordinates."""
    def __init__(self, grid_size=5, min_val=[-0.65, -0.90], max_val=[0.90, 0.35], sigma_scale=0.8):
        super().__init__()
        self.grid_size = grid_size
        
        # Handle scalar or vector ranges
        if not isinstance(min_val, (list, tuple, np.ndarray, torch.Tensor)):
            min_val = [min_val, min_val]
        if not isinstance(max_val, (list, tuple, np.ndarray, torch.Tensor)):
            max_val = [max_val, max_val]
            
        self.register_buffer('min_val', torch.tensor(min_val, dtype=torch.float32))
        self.register_buffer('max_val', torch.tensor(max_val, dtype=torch.float32))
        
        # Create grid of Gaussian centers
        x_centers = torch.linspace(min_val[0], max_val[0], grid_size)
        y_centers = torch.linspace(min_val[1], max_val[1], grid_size)
        
        # Create meshgrid and flatten to get all 25 (x,y) center positions
        X_grid, Y_grid = torch.meshgrid(x_centers, y_centers, indexing='ij')
        grid_centers = torch.stack([X_grid.flatten(), Y_grid.flatten()], dim=1)  # [25, 2]
        
        self.register_buffer('grid_centers', grid_centers)
        
        # Calculate per-axis sigma for Gaussians (based on grid spacing)
        grid_spacing_x = (max_val[0] - min_val[0]) / (grid_size - 1)
        grid_spacing_y = (max_val[1] - min_val[1]) / (grid_size - 1)
        
        # Per-axis sigma values for uniform overlap
        sigma_x = sigma_scale * grid_spacing_x
        sigma_y = sigma_scale * grid_spacing_y
        
        self.register_buffer('sigma_x', torch.tensor(sigma_x, dtype=torch.float32))
        self.register_buffer('sigma_y', torch.tensor(sigma_y, dtype=torch.float32))
        
        
    def forward(self, x):
        """
        Encode 2D coordinates using grid of Gaussians with per-axis sigma.
        
        Args:
            x: [batch_size, 2] - (x, y) coordinates
            
        Returns:
            activations: [batch_size, grid_size²] - Gaussian activations [0,1]
        """
        batch_size = x.shape[0]
        
        # Expand for broadcasting
        x_expanded = x.unsqueeze(1)  # [batch_size, 1, 2]
        centers_expanded = self.grid_centers.unsqueeze(0)  # [1, 25, 2]
        
        # Calculate distances along each axis
        diff = x_expanded - centers_expanded  # [batch_size, 25, 2]
        dx = diff[:, :, 0]  # [batch_size, 25] - x differences
        dy = diff[:, :, 1]  # [batch_size, 25] - y differences
        
        # Apply per-axis Gaussian: exp(-dx²/(2σx²) - dy²/(2σy²))
        activations = torch.exp(
            -(dx ** 2) / (2 * self.sigma_x ** 2) - 
            (dy ** 2) / (2 * self.sigma_y ** 2)
        )
        
        return activations


class ModalitySpecificEncoder(nn.Module):
    """
    Encode 15D input using grid-based spatial encoding.
    - Direct muscle sensors: 12D -> 12D (pass through)
    - Target position: 3D -> 5×5 grid (25D) + Z passthrough (1D)
    """
    def __init__(self, grid_size=5):
        super().__init__()
        
        # Grid encoder for target XY coordinates
        self.target_encoder = GridEncoder(
            grid_size=grid_size,
            min_val=[-0.65, -0.90],
            max_val=[0.90, 0.35],
            sigma_scale=0.8  # Controls overlap between Gaussians
        )
        
        # Calculate output size: muscle(12) + grid_encoded_xy(grid_size²) + target_z(1)
        self.output_size = 12 + (grid_size * grid_size) + 1
        self.grid_size = grid_size
        
        #print(f"Modality encoder: 15D -> {self.output_size}D")
        #print(f"Components: muscle(12) + grid_xy({grid_size*grid_size}) + target_z(1)")
    
    def forward(self, x):
        """
        Encode 15D input to grid-encoded representation.
        
        Args:
            x: [batch_size, 15] - [muscle_sensors(12), target_pos(3)]
        
        Returns:
            encoded: [batch_size, output_size] - [muscle(12), grid_xy(25), target_z(1)]
        """
        # Split input
        muscle_data = x[:, :12]      # Muscle sensors: direct passthrough
        target_pos = x[:, 12:15]     # Target position [x, y, z]
        
        # Encode XY coordinates with grid Gaussians
        target_xy = target_pos[:, :2]
        encoded_target = self.target_encoder(target_xy)
        
        # Keep Z coordinate as direct input
        target_z = target_pos[:, 2:3]
        
        # Concatenate: [muscle(12), grid_encoded_xy(25), target_z(1)]
        return torch.cat([muscle_data, encoded_target, target_z], dim=1)


class ModalityEncodingWrapper(gym.ObservationWrapper):
    """Wrapper that applies grid-based encoding at environment level."""
    
    def __init__(self, env, grid_size=5):
        super().__init__(env)
        
        # Create the encoder
        self.encoder = ModalitySpecificEncoder(grid_size=grid_size)
        
        # Update observation space to match encoded output
        encoded_size = self.encoder.output_size  # 12 + 25 + 1 = 38
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(encoded_size,),
            dtype=np.float32
        )
        
        #print(f"Observation space: {env.observation_space.shape} -> {self.observation_space.shape}")
    
    def observation(self, obs):
        """Apply grid encoding to raw 15D observation."""
        # Convert to torch tensor
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)  # Add batch dim
        
        # Apply encoding
        with torch.no_grad():
            encoded = self.encoder(obs_tensor)
        
        # Convert back to numpy and remove batch dimension
        return encoded.squeeze(0).numpy().astype(np.float32)


# Update create_env to use grid encoding
def create_env(random_seed=None, base_env=None):
    """Create a single environment instance with grid encoding."""
    # Use provided base_env or create new one
    dm_env = base_env if base_env is not None else make_arm_env(random_seed=random_seed)
    
    # Apply Shimmy compatibility wrapper
    env = DmControlCompatibilityV0(dm_env)
    
    # Flatten observations
    env = FlattenObservation(env)
    
    # Apply grid-based encoding (15D -> 38D)
    env = ModalityEncodingWrapper(env, grid_size=4)  # 5×5 = 25D for XY
    
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


# Update SelectiveVecNormalize for the new observation size
class SelectiveVecNormalize(VecNormalize):
    """VecNormalize that only normalizes the first 12 dimensions (muscle sensors)."""
    
    def __init__(self, venv, norm_dims=12, **kwargs):
        self.norm_dims = norm_dims
        
        # Initialize parent class first
        super().__init__(venv, **kwargs)
        
        # Now override the obs_rms to only track norm_dims
        from stable_baselines3.common.running_mean_std import RunningMeanStd
        self.obs_rms = RunningMeanStd(shape=(norm_dims,))
        
        print(f"Selective normalization: normalizing first {norm_dims} dimensions only")
    
    def reset(self):
        """Reset environments and update selective normalization."""
        obs = self.venv.reset()
        if self.training and self.norm_obs:
            # Only update with first norm_dims dimensions
            self.obs_rms.update(obs[:, :self.norm_dims])
        return self.normalize_obs(obs)
    
    def step_wait(self):
        """Override step_wait to handle selective normalization."""
        obs, rewards, dones, infos = self.venv.step_wait()
        
        if self.training and self.norm_obs:
            # Only update with first norm_dims dimensions
            self.obs_rms.update(obs[:, :self.norm_dims])
        
        if self.norm_reward:
            rewards = self.normalize_reward(rewards)
        
        obs = self.normalize_obs(obs)
        
        return obs, rewards, dones, infos
        
    def normalize_obs(self, obs):
        """Override normalize_obs to handle selective normalization."""
        if self.norm_obs:
            obs_to_norm = obs[:, :self.norm_dims]
            obs_rest = obs[:, self.norm_dims:]
            
            # Normalize only first part
            normalized_part = (obs_to_norm - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon)
            normalized_part = np.clip(normalized_part, -self.clip_obs, self.clip_obs)
            
            # Concatenate with unchanged part
            return np.concatenate([normalized_part, obs_rest], axis=1)
        return obs
    
    def _update_obs(self, obs):
        """Update statistics only for first norm_dims dimensions."""
        if self.training and self.norm_obs:
            self.obs_rms.update(obs[:, :self.norm_dims])

def set_seeds(seed=42):
    """Set seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    return seed



def create_training_env(num_envs=4):
    """Create vectorized environment for training WITH VecNormalize."""
    base_seed = 12345
    
    def _init_env(rank):
        # Each subprocess needs to set its own env vars
        import os
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_DYNAMIC'] = 'FALSE'
        return create_env(random_seed=base_seed + rank)
    
    # Create vectorized environment
    vec_env = SubprocVecEnv([
        lambda rank=i: _init_env(rank=i) 
        for i in range(num_envs)
    ])
    
    # Apply VecNormalize to normalize the encoded observations
    vec_env = SelectiveVecNormalize(
        vec_env,
        norm_dims=12,  # Normalize only the first 12 dimensions (muscle sensors)
        norm_obs=True,
        norm_reward=False,  # Don't normalize rewards
        clip_obs=10.0,
        gamma=0.99
    )

    """vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=False,  # Don't normalize rewards
        clip_obs=10.0,
        gamma=0.99,
    )"""
    
    return vec_env


def create_eval_env(training_env=None):
    """Create evaluation environment with same normalization as training."""
    def _make_env():
        env = create_env(random_seed=42)
        return env
    
    vec_env = DummyVecEnv([_make_env])
    
    if training_env is not None:
        # Use same normalization as training
        vec_env = SelectiveVecNormalize(vec_env, norm_dims=12, norm_obs=True, training=False)
        #vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, training=False)
        sync_envs_normalization(training_env, vec_env)
    
    return vec_env




