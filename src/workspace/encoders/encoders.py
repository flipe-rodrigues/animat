import torch
import torch.nn as nn
import numpy as np

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
        
        X_grid, Y_grid = torch.meshgrid(x_centers, y_centers, indexing='ij')
        grid_centers = torch.stack([X_grid.flatten(), Y_grid.flatten()], dim=1)
        self.register_buffer('grid_centers', grid_centers)
        
        # Calculate per-axis sigma
        grid_spacing_x = (max_val[0] - min_val[0]) / (grid_size - 1)
        grid_spacing_y = (max_val[1] - min_val[1]) / (grid_size - 1)
        
        sigma_x = sigma_scale * grid_spacing_x
        sigma_y = sigma_scale * grid_spacing_y
        
        self.register_buffer('sigma_x', torch.tensor(sigma_x, dtype=torch.float32))
        self.register_buffer('sigma_y', torch.tensor(sigma_y, dtype=torch.float32))
    
    @property
    def input_dim(self) -> int:
        return 2
    
    @property
    def output_dim(self) -> int:
        return self.grid_size * self.grid_size
    
    def forward(self, x):
        """
        Encode 2D coordinates using grid of Gaussians.
        
        Args:
            x: [batch_size, 2] or [2] - (x, y) coordinates
            
        Returns:
            activations: [batch_size, grid_size²] or [grid_size²]
        """
        # Handle single sample input
        single_sample = x.dim() == 1
        if single_sample:
            x = x.unsqueeze(0)
        
        # Expand for broadcasting
        x_expanded = x.unsqueeze(1)  # [batch_size, 1, 2]
        centers_expanded = self.grid_centers.unsqueeze(0)  # [1, 25, 2]
        
        # Calculate distances and apply Gaussian
        diff = x_expanded - centers_expanded  # [batch_size, 25, 2]
        dx, dy = diff[:, :, 0], diff[:, :, 1]
        
        activations = torch.exp(
            -(dx ** 2) / (2 * self.sigma_x ** 2) - 
            (dy ** 2) / (2 * self.sigma_y ** 2)
        )
        
        # Remove batch dim if input was single sample
        if single_sample:
            activations = activations.squeeze(0)
            
        return activations

class ModalitySpecificEncoder(nn.Module):
    """Encode observation using grid-based spatial encoding."""
    
    def __init__(self, grid_size=5, raw_obs_dim=15):
        super().__init__()
        self._raw_obs_dim = raw_obs_dim
        self.target_encoder = GridEncoder(
            grid_size=grid_size,
            min_val=[-0.65, -0.90],
            max_val=[0.90, 0.35],
            sigma_scale=0.8
        )
        self.grid_size = grid_size
    
    @property
    def input_dim(self) -> int:
        return self._raw_obs_dim
    
    @property
    def output_dim(self) -> int:
        return 12 + (self.grid_size * self.grid_size) + 1
    
    def forward(self, x):
        """
        Encode observation: [muscle(12), target_pos(3)] → [muscle(12), grid_xy(25), target_z(1)]
        """
        # Handle single sample
        single_sample = x.dim() == 1
        if single_sample:
            x = x.unsqueeze(0)
        
        # Split input
        muscle_data = x[:, :12]      # Muscle sensors: passthrough
        target_pos = x[:, 12:15]     # Target position [x, y, z]
        
        # Encode XY with grid, keep Z direct
        target_xy = target_pos[:, :2]
        encoded_target = self.target_encoder(target_xy)
        target_z = target_pos[:, 2:3]
        
        # Concatenate
        result = torch.cat([muscle_data, encoded_target, target_z], dim=1)
        
        # Remove batch dim if input was single sample
        if single_sample:
            result = result.squeeze(0)
            
        return result

class IdentityEncoder(nn.Module):  # Make it a PyTorch module for consistency
    """Pass-through encoder."""
    
    def __init__(self, obs_dim: int):
        super().__init__()
        self._obs_dim = obs_dim
    
    @property
    def input_dim(self) -> int:
        return self._obs_dim
    
    @property
    def output_dim(self) -> int:
        return self._obs_dim
    
    def forward(self, x):
        """Identity encoding - just return input."""
        return x