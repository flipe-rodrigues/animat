from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod


class TargetEncoder(ABC):
    """Abstract base class for target position encoders"""

    def __init__(
        self,
        size: int,
        x_bounds: Tuple[float, float] = (0, 1),
        y_bounds: Tuple[float, float] = (0, 1),
    ):
        self.size = size
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds

    @abstractmethod
    def encode(self, x: float, y: float):
        """Encode target position into neural representation"""
        pass


class GridTargetEncoder(TargetEncoder):
    """
    Grid-based population code encoder using Gaussian receptive fields.

    Optimizations:
    - Pre-computes 1/(2*sigma^2) to avoid repeated division
    - Caches grid structure
    - Uses multiplication instead of division in hot path
    """

    def __init__(
        self,
        grid_size: int = 10,
        x_bounds: Tuple[float, float] = (0, 1),
        y_bounds: Tuple[float, float] = (0, 1),
        sigma: float = 0.1,
    ):
        """
        Initialize grid encoder.

        Args:
            grid_size: Number of neurons per dimension (total = grid_size^2)
            x_bounds: (min, max) bounds for x dimension
            y_bounds: (min, max) bounds for y dimension
            sigma: Standard deviation of Gaussian receptive fields
        """
        # Validation
        if grid_size < 1:
            raise ValueError("grid_size must be >= 1")
        if sigma <= 0:
            raise ValueError("sigma must be > 0")
        if x_bounds[0] >= x_bounds[1]:
            raise ValueError("x_bounds must be (min, max) with min < max")
        if y_bounds[0] >= y_bounds[1]:
            raise ValueError("y_bounds must be (min, max) with min < max")

        super().__init__(
            size=grid_size * grid_size,
            x_bounds=x_bounds,
            y_bounds=y_bounds,
        )
        self.grid_size = grid_size
        self.sigma = sigma

        # Pre-compute constant for faster encoding
        # exp(-dist^2 / (2*sigma^2)) = exp(-dist^2 * inv_2sigma_sq)
        self._inv_2sigma_sq = 1.0 / (2 * sigma**2)

        self._initialize_grid()

    def _initialize_grid(self):
        """Pre-compute grid centers"""
        x_centers = np.linspace(self.x_bounds[0], self.x_bounds[1], self.grid_size)
        y_centers = np.linspace(self.y_bounds[0], self.y_bounds[1], self.grid_size)
        self.x_grid, self.y_grid = np.meshgrid(x_centers, y_centers)

    def encode(self, x: float, y: float) -> np.ndarray:
        """
        Encode target position as population code.

        Optimized with pre-computed constant to avoid division in hot path.

        Args:
            x: Target x position
            y: Target y position

        Returns:
            Grid of activation values (grid_size x grid_size)
        """
        dist_sq = (self.x_grid - x) ** 2 + (self.y_grid - y) ** 2
        return np.exp(-dist_sq * self._inv_2sigma_sq)  # Multiply instead of divide

    def visualize(self, x: float, y: float, figsize: Tuple[int, int] = (12, 5)):
        """
        Visualize the receptive field response for a target position.

        Args:
            x: Target x position
            y: Target y position
            figsize: Figure size (width, height)
        """
        h = self.encode(x, y)
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # 2D heatmap
        im = axes[0].imshow(
            h,
            origin="lower",
            cmap="hot",
            extent=[
                self.x_bounds[0],
                self.x_bounds[1],
                self.y_bounds[0],
                self.y_bounds[1],
            ],
        )
        axes[0].plot(x, y, "c*", markersize=15, label="Target")
        axes[0].set_xlabel("X")
        axes[0].set_ylabel("Y")
        axes[0].set_title("Firing Rate Heatmap")
        axes[0].legend()
        plt.colorbar(im, ax=axes[0], label="Firing Rate")

        # 3D surface
        ax = fig.add_subplot(122, projection="3d")
        ax.plot_surface(self.x_grid, self.y_grid, h, cmap="hot", alpha=0.8)
        ax.scatter([x], [y], [h.max()], c="cyan", s=100, marker="*", label="Target")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Firing Rate")
        ax.set_title("3D Firing Rate Surface")

        plt.tight_layout()
        plt.show()

    def __repr__(self):
        return (
            f"GridTargetEncoder(grid_size={self.grid_size}, "
            f"sigma={self.sigma:.3f}, "
            f"x_bounds={self.x_bounds}, y_bounds={self.y_bounds})"
        )
