"""
Place cell grid for encoding target positions.
Uses 2D Gaussian tuning curves centered on a grid.
"""

import numpy as np
from typing import Tuple


class PlaceCellGrid:
    """Grid of place cells with Gaussian tuning curves for spatial encoding."""

    def __init__(
        self,
        workspace_bounds: Tuple[Tuple[float, float], Tuple[float, float]],
        grid_size: Tuple[int, int] = (8, 8),
        sigma: float = 0.1,
    ):
        """
        Initialize place cell grid.

        Args:
            workspace_bounds: ((x_min, x_max), (y_min, y_max))
            grid_size: (n_cells_x, n_cells_y) number of cells along each dimension
            sigma: Standard deviation of Gaussian tuning curve
        """
        self.workspace_bounds = workspace_bounds
        self.grid_size = grid_size
        self.sigma = sigma
        self.num_cells = grid_size[0] * grid_size[1]
        self._init_grid()

    def _init_grid(self):
        """Initialize the grid of place cell centers."""
        x_centers = np.linspace(
            self.workspace_bounds[0][0], self.workspace_bounds[0][1], self.grid_size[0]
        )
        y_centers = np.linspace(
            self.workspace_bounds[1][0], self.workspace_bounds[1][1], self.grid_size[1]
        )
        X, Y = np.meshgrid(x_centers, y_centers)
        self.cell_centers = np.column_stack([X.ravel(), Y.ravel()])  # (num_cells, 2)

    def encode(self, position: np.ndarray) -> np.ndarray:
        """
        Encode a 2D position using place cell activities.

        Args:
            position: (x, y) coordinates in workspace

        Returns:
            activities: (num_cells,) array of cell activities
        """
        distances = np.linalg.norm(self.cell_centers - position.reshape(1, 2), axis=1)
        activities = np.exp(-(distances**2) / (2 * self.sigma**2))
        return activities

    def decode(self, activities: np.ndarray) -> np.ndarray:
        """
        Decode position from place cell activities using population vector.

        Args:
            activities: (num_cells,) array of cell activities

        Returns:
            position: (x, y) decoded coordinates
        """
        total_activity = np.sum(activities) + 1e-8
        position = (
            np.sum(self.cell_centers * activities.reshape(-1, 1), axis=0)
            / total_activity
        )
        return position

    def get_observation_size(self) -> int:
        """Return the dimensionality of the place cell encoding."""
        return self.num_cells
