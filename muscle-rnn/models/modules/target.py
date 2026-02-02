"""Target Encoding Module - Spatial target representation."""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional

from core.constants import (
    DEFAULT_TARGET_GRID_SIZE,
    DEFAULT_TARGET_GRID_PADDING,
    DEFAULT_TARGET_SIGMA,
)


class TargetEncoder(nn.Module):
    """
    Encodes target position using grid of Gaussian-tuned spatial units.

    The grid covers the 2D workspace (XY plane) with Gaussian receptive fields.
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

        if workspace_bounds is None:
            workspace_bounds = {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}
        self.workspace_bounds = workspace_bounds

        # Create grid (non-trainable buffer)
        self.register_buffer("grid_positions", self._create_grid())

    def _create_grid(self) -> torch.Tensor:
        xrange = self.workspace_bounds["x"][1] - self.workspace_bounds["x"][0]
        yrange = self.workspace_bounds["y"][1] - self.workspace_bounds["y"][0]
        xpadding = DEFAULT_TARGET_GRID_PADDING * xrange / self.grid_size
        ypadding = DEFAULT_TARGET_GRID_PADDING * yrange / self.grid_size
        xbounds = (
            self.workspace_bounds["x"][0] - xpadding,
            self.workspace_bounds["x"][1] + xpadding,
        )
        ybounds = (
            self.workspace_bounds["y"][0] - ypadding,
            self.workspace_bounds["y"][1] + ypadding,
        )
        x = np.linspace(*xbounds, self.grid_size)
        y = np.linspace(*ybounds, self.grid_size)
        xx, yy = np.meshgrid(x, y)
        return torch.tensor(
            np.stack([xx.flatten(), yy.flatten()], axis=1), dtype=torch.float32
        )

    def encode(self, target_position: torch.Tensor) -> torch.Tensor:
        """
        Encode target position as Gaussian activations.

        Args:
            target_position: [batch, 3] or [3] XYZ position

        Returns:
            activations: [batch, num_units]
        """
        if target_position.dim() == 1:
            target_position = target_position.unsqueeze(0)

        target_xy = target_position[:, :2]
        distances = torch.cdist(target_xy, self.grid_positions)
        return torch.exp(-0.5 * (distances / self.sigma) ** 2)

    def forward(self, target_position: torch.Tensor) -> torch.Tensor:
        return self.encode(target_position)
