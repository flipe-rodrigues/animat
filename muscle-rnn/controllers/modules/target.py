"""Target encoding modules."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config import WorkspaceBounds


class TargetEncoder(nn.Module):
    """
    Encodes 3D target positions as 2D Gaussian activation maps.

    Creates a grid of spatial receptive fields and computes soft
    assignments based on Euclidean distance to grid centers.
    """

    def __init__(
        self,
        grid_size: int = 4,
        sigma: float = 0.5,
        bounds: WorkspaceBounds = WorkspaceBounds(),
    ):
        super().__init__()
        self.sigma = sigma
        self.grid_size = grid_size
        self._create_grid(grid_size, bounds)

    @property
    def output_size(self) -> int:
        return self.grid_size**2

    def _create_grid(self, grid_size: int, bounds: WorkspaceBounds) -> None:
        x = torch.linspace(*bounds.x, grid_size)
        y = torch.linspace(*bounds.y, grid_size)
        xx, yy = torch.meshgrid(x, y, indexing="xy")
        self.register_buffer(
            "grid_centers", torch.stack([xx.flatten(), yy.flatten()], dim=1)
        )

    def forward(self, target_xyz: torch.Tensor) -> torch.Tensor:
        target_xy = target_xyz[:, :2]
        diff = target_xy.unsqueeze(1) - self.grid_centers.unsqueeze(0)
        dist_sq = (diff**2).sum(dim=2)
        activation = torch.exp(-dist_sq / (2 * self.sigma**2))
        return (
            activation / (activation.sum(dim=1, keepdim=True) + 1e-8) * self.grid_size
        )
