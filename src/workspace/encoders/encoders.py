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