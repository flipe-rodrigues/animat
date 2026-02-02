"""
Target Encoding Module - Spatial target representation

Encodes 3D target positions as activations in a grid of Gaussian-tuned spatial units.
This is a neural population code representation of target location.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional

from core.constants import DEFAULT_TARGET_GRID_SIZE, DEFAULT_TARGET_SIGMA


class TargetEncoder(nn.Module):
    """
    Encodes target position using grid of Gaussian-tuned spatial units.
    
    The grid covers the 2D workspace (XY plane) with Gaussian receptive fields.
    Each unit fires according to its distance from the target.
    """
    
    def __init__(
        self,
        grid_size: int = DEFAULT_TARGET_GRID_SIZE,
        sigma: float = DEFAULT_TARGET_SIGMA,
        workspace_bounds: Optional[dict] = None,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.num_units = grid_size * grid_size
        self.sigma = sigma
        
        # Default workspace bounds
        if workspace_bounds is None:
            workspace_bounds = {
                'x': (-0.5, 0.5),
                'y': (-0.5, 0.5),
            }
        self.workspace_bounds = workspace_bounds
        
        # Create grid positions (registered as buffer, not parameter)
        grid_positions = self._create_grid()
        self.register_buffer('grid_positions', grid_positions)
    
    def _create_grid(self) -> torch.Tensor:
        """Create the 2D grid of unit positions."""
        x_range = np.linspace(
            self.workspace_bounds['x'][0],
            self.workspace_bounds['x'][1],
            self.grid_size
        )
        y_range = np.linspace(
            self.workspace_bounds['y'][0],
            self.workspace_bounds['y'][1],
            self.grid_size
        )
        
        xx, yy = np.meshgrid(x_range, y_range)
        positions = np.stack([xx.flatten(), yy.flatten()], axis=1)
        
        return torch.tensor(positions, dtype=torch.float32)
    
    def encode(self, target_position: torch.Tensor) -> torch.Tensor:
        """
        Encode target position as Gaussian activations.
        
        Args:
            target_position: Target XYZ position [batch, 3] or [3]
            
        Returns:
            activations: Gaussian activations [batch, num_units]
        """
        if target_position.dim() == 1:
            target_position = target_position.unsqueeze(0)
        
        # Get XY components
        target_xy = target_position[:, :2]  # [batch, 2]
        
        # Compute distances to all grid positions
        # grid_positions: [num_units, 2]
        # target_xy: [batch, 2]
        distances = torch.cdist(target_xy, self.grid_positions)  # [batch, num_units]
        
        # Gaussian activation based on distance
        activations = torch.exp(-0.5 * (distances / self.sigma) ** 2)
        
        return activations
    
    def forward(self, target_position: torch.Tensor) -> torch.Tensor:
        """Forward pass - encode target position."""
        return self.encode(target_position)
